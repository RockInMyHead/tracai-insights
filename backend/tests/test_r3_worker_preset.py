"""Regression tests for the R3 release-compatible worker command."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3_worker_wrapper import _build_r3_infer_cmd, _probe_video_frame_timestamps


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


class R3WorkerPresetTests(unittest.TestCase):
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
