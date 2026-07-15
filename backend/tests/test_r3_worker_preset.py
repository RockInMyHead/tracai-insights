"""Regression tests for the R3 release-compatible worker command."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3_worker_wrapper import (
    R3_POSE_GRAPH_EXPORT_ANCHOR,
    _build_r3_infer_cmd,
    _ensure_r3_pose_graph_export,
    _patch_r3_infer_source,
    _probe_video_frame_timestamps,
    collect_results,
)
from r3_pose_graph import (
    R3_ABSOLUTE_POSE_SPACE,
    R3_CONFIDENCE_SEMANTICS,
    R3_POSE_ENCODING,
    R3_POSE_GRAPH_SCHEMA_VERSION,
    R3_RELATIVE_TRANSFORM_CONVENTION,
)


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


class R3WorkerPresetTests(unittest.TestCase):
    def test_pose_graph_export_patch_is_valid_and_idempotent(self) -> None:
        source = (
            "import json, os\nimport numpy as np\n"
            "def export(output_dir, edge_records, edges, frame_id_to_output_idx):\n"
            "    if True:\n"
            "        if edge_records:\n"
            f"{R3_POSE_GRAPH_EXPORT_ANCHOR}"
        )

        patched, diagnostics = _patch_r3_infer_source(source)
        patched_again, repeated = _patch_r3_infer_source(patched)

        self.assertTrue(diagnostics["changed"])
        self.assertIn('"pose_graph_edges.npz"', patched)
        self.assertIn('"rel_pose_enc"', patched)
        self.assertIn('"confidence_t"', patched)
        self.assertIn('"confidence_r"', patched)
        self.assertEqual(patched_again, patched)
        self.assertEqual(repeated["status"], "already_available")

    def test_patched_export_writes_full_relative_edge_payload(self) -> None:
        source = (
            "import json, os\nimport numpy as np\n"
            "def export(output_dir, edge_records, edges, frame_id_to_output_idx):\n"
            "    if True:\n"
            "        if edge_records:\n"
            f"{R3_POSE_GRAPH_EXPORT_ANCHOR}"
        )
        patched, diagnostics = _patch_r3_infer_source(source)
        namespace: dict = {}
        exec(patched, namespace)

        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def float(self):
                return self

            def numpy(self):
                return np.asarray(
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 0.7],
                    dtype=np.float32,
                )

            def reshape(self, _size):
                return self

        edge = SimpleNamespace(
            frame_i=10,
            frame_j=20,
            rel_pose_enc=FakeTensor(),
            confidence=2.0,
            confidence_t=2.1,
            confidence_r=1.9,
            edge_type="normal",
        )
        with tempfile.TemporaryDirectory() as directory:
            namespace["export"](
                directory,
                [{"frame_i": 0, "frame_j": 1}],
                [edge],
                {10: 0, 20: 1},
            )
            with np.load(Path(directory) / "pose_graph_edges.npz", allow_pickle=False) as archive:
                exported = {key: archive[key].copy() for key in archive.files}

        self.assertTrue(diagnostics["changed"])
        self.assertEqual(str(exported["absolute_pose_space"]), "world_to_camera")
        self.assertEqual(str(exported["confidence_semantics"]), "softplus_positive_weight_not_covariance")
        self.assertEqual(exported["rel_pose_enc"][0, :3].tolist(), [1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(exported["confidence_t"][0]), 2.1, places=5)
        self.assertAlmostEqual(float(exported["confidence_r"][0]), 1.9, places=5)

    def test_pose_graph_export_patches_external_infer_atomically(self) -> None:
        source = (
            "import json, os\nimport numpy as np\n"
            "def export(output_dir, edge_records, edges, frame_id_to_output_idx):\n"
            "    if True:\n"
            "        if edge_records:\n"
            f"{R3_POSE_GRAPH_EXPORT_ANCHOR}"
        )
        with tempfile.TemporaryDirectory() as directory:
            infer_path = Path(directory) / "infer.py"
            infer_path.write_text(source, encoding="utf-8")

            first = _ensure_r3_pose_graph_export(directory)
            second = _ensure_r3_pose_graph_export(directory)
            persisted = infer_path.read_text(encoding="utf-8")

        self.assertEqual(first["status"], "patched")
        self.assertEqual(second["status"], "already_available")
        self.assertIn('"pose_graph_edges.npz"', persisted)

    def test_collect_results_runs_guarded_shadow_optimizer(self) -> None:
        point_count = 8
        truth = np.broadcast_to(np.eye(4), (point_count, 4, 4)).copy()
        truth[:, 0, 3] = np.arange(point_count, dtype=float)
        initial = truth.copy()
        initial[:, 0, 3] *= 1.2
        rel_pose = np.tile(
            # The R3 sidecar stores world-to-camera transforms.  Increasing
            # camera centers therefore produces a negative relative t_x.
            np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.8, 0.8]),
            (point_count - 1, 1),
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "run_params.json").write_text("{}", encoding="utf-8")
            camera_dir = output / "camera"
            camera_dir.mkdir()
            for index, pose in enumerate(initial):
                np.savez_compressed(
                    camera_dir / f"{index:06d}.npz",
                    pose=pose,
                    intrinsics=np.eye(3),
                )
            np.savez_compressed(
                output / "pose_graph_edges.npz",
                schema_version=np.asarray([R3_POSE_GRAPH_SCHEMA_VERSION], dtype=np.int32),
                pose_encoding=np.asarray(R3_POSE_ENCODING),
                transform_convention=np.asarray(R3_RELATIVE_TRANSFORM_CONVENTION),
                frame_index_space=np.asarray("exported_camera_index"),
                absolute_pose_space=np.asarray(R3_ABSOLUTE_POSE_SPACE),
                confidence_semantics=np.asarray(R3_CONFIDENCE_SEMANTICS),
                frame_i=np.arange(point_count - 1, dtype=np.int32),
                frame_j=np.arange(1, point_count, dtype=np.int32),
                rel_pose_enc=rel_pose.astype(np.float32),
                confidence=np.full(point_count - 1, 2.0, dtype=np.float32),
                confidence_t=np.full(point_count - 1, 2.0, dtype=np.float32),
                confidence_r=np.full(point_count - 1, 2.0, dtype=np.float32),
                edge_type=np.zeros(point_count - 1, dtype=np.uint8),
            )

            with patch.dict(
                os.environ,
                {"R3_POSE_GRAPH_OPTIMIZER_MODE": "shadow"},
                clear=False,
            ):
                result = collect_results(directory, export_pointcloud=False)
            raw_last_pose = np.load(camera_dir / "000007.npz")["pose"]
            with np.load(output / "pose_graph_candidate.npz") as archive:
                candidate_last_pose = archive["c2w"][-1]

        self.assertTrue(result["pose_graph"]["optimizer_ready"])
        self.assertTrue(result["pose_graph_candidate"]["available"])
        self.assertTrue(result["pose_graph_candidate"]["accepted"])
        self.assertEqual(result["run_params"]["pose_graph_optimizer_mode"], "shadow")
        np.testing.assert_allclose(raw_last_pose, initial[-1])
        self.assertAlmostEqual(float(candidate_last_pose[0, 3]), 7.0, delta=0.05)

    @patch("r3_worker_wrapper.subprocess.run")
    def test_probes_exact_video_presentation_timestamps(self, run_mock) -> None:
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps({
                "frames": [
                    {"best_effort_timestamp_time": "0.000000"},
                    {"best_effort_timestamp_time": "0.033367"},
                    {"best_effort_timestamp_time": "0.101000"},
                ],
            }),
        )

        timestamps, diagnostics = _probe_video_frame_timestamps("/tmp/input.mp4")

        self.assertEqual(timestamps, [0.0, 0.033367, 0.101])
        self.assertTrue(diagnostics["available"])
        self.assertEqual(diagnostics["finite_timestamps"], 3)
        command = run_mock.call_args.args[0]
        self.assertIn("frame=best_effort_timestamp_time", command)

    def test_release_preset_neutralizes_regressed_pgo_settings(self) -> None:
        stale_environment = {
            "R3_REL_POSE_METHOD": "pgo",
            "R3_KEYFRAME_MAX_INTERVAL": "15",
            "R3_KEYFRAME_MAX_KEYFRAMES": "160",
            "R3_ENABLE_SEGMENT_PGO": "true",
            "R3_ENABLE_METRIC_SCALE": "true",
            "R3_FALLBACK_MIN_BRIDGE_BASELINE_RATIO": "0",
            "R3_FALLBACK_MAX_BRIDGE_LOOKBACK": "10",
        }
        with patch.dict(os.environ, stale_environment, clear=True):
            command, mode, checkpoint = _build_r3_infer_cmd(
                "/tmp/frames",
                "/tmp/output",
                "r3.safetensors",
                "long",
                392,
                0,
            )

        self.assertEqual(mode, "long")
        self.assertEqual(checkpoint, "r3_long.safetensors")
        self.assertEqual(option_value(command, "--online_kv_cache_mode"), "dynamic")
        self.assertEqual(option_value(command, "--rel_pose_reconstruction_method"), "greedy")
        self.assertEqual(option_value(command, "--keyframe_max_interval"), "30")
        self.assertEqual(option_value(command, "--keyframe_max_keyframes"), "100")
        self.assertIn("--disable_segment_pgo", command)
        self.assertNotIn("--metric_scale_enabled", command)
        self.assertEqual(option_value(command, "--fallback_min_bridge_baseline_ratio"), "0.35")
        self.assertEqual(option_value(command, "--fallback_max_bridge_lookback"), "40")

    def test_custom_experimental_preset_remains_opt_in(self) -> None:
        custom_environment = {
            "R3_USE_RELEASE_PRESET": "false",
            "R3_REL_POSE_METHOD": "pgo",
            "R3_KEYFRAME_MAX_INTERVAL": "15",
            "R3_KEYFRAME_MAX_KEYFRAMES": "160",
            "R3_ENABLE_SEGMENT_PGO": "true",
            "R3_FALLBACK_MIN_BRIDGE_BASELINE_RATIO": "0.5",
            "R3_FALLBACK_MAX_BRIDGE_LOOKBACK": "60",
        }
        with patch.dict(os.environ, custom_environment, clear=True):
            command, _, _ = _build_r3_infer_cmd(
                "/tmp/frames",
                "/tmp/output",
                "r3_long.safetensors",
                "long",
                392,
                0,
            )

        self.assertEqual(option_value(command, "--rel_pose_reconstruction_method"), "pgo")
        self.assertEqual(option_value(command, "--keyframe_max_interval"), "15")
        self.assertEqual(option_value(command, "--keyframe_max_keyframes"), "160")
        self.assertEqual(option_value(command, "--fallback_min_bridge_baseline_ratio"), "0.5")
        self.assertEqual(option_value(command, "--fallback_max_bridge_lookback"), "60")
        self.assertNotIn("--disable_segment_pgo", command)

    def test_metric_reanchor_requires_new_explicit_scale_policy(self) -> None:
        with patch.dict(os.environ, {"R3_SCALE_POLICY": "metric_reanchor"}, clear=True):
            command, _, _ = _build_r3_infer_cmd(
                "/tmp/frames",
                "/tmp/output",
                "r3_long.safetensors",
                "long",
                392,
                0,
            )

        self.assertIn("--metric_scale_enabled", command)
        self.assertEqual(option_value(command, "--metric_bootstrap_frames"), "5")


if __name__ == "__main__":
    unittest.main()
