"""Tests for reproducible trajectory baseline capture."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from r3_pose_graph_optimizer import save_pose_graph_candidate  # noqa: E402
from r3_scale_aware import save_scale_aware_candidate  # noqa: E402
from trajectory_geometry import (  # noqa: E402
    compare_trajectories,
    deviation_from_reference,
    trajectory_acceptance,
    trajectory_metrics,
)


def poses_from_xy(points: list[tuple[float, float]]) -> np.ndarray:
    poses = np.broadcast_to(np.eye(4), (len(points), 4, 4)).copy()
    poses[:, 0, 3] = [point[0] for point in points]
    poses[:, 2, 3] = [point[1] for point in points]
    return poses


class TrajectoryGeometryTests(unittest.TestCase):
    def test_reports_requested_geometry_metrics(self) -> None:
        points = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]
        metrics = trajectory_metrics(points)

        self.assertEqual(metrics["path_length"], 4.0)
        self.assertAlmostEqual(
            metrics["endpoint_displacement"], 2**0.5 * 2, places=6
        )
        self.assertEqual(metrics["bbox_width"], 2.0)
        self.assertEqual(metrics["bbox_height"], 2.0)
        self.assertEqual(metrics["bbox_area"], 4.0)
        self.assertIn("1", metrics["near_start_fraction"])
        self.assertGreaterEqual(metrics["turn_count"], 1)

    def test_deviation_is_similarity_invariant_but_not_reflection_invariant(self) -> None:
        raw = [[0, 0], [2, 0], [2, 1], [3, 1]]
        transformed = [[10, 5], [10, 9], [8, 9], [8, 11]]
        reflected = [[0, 0], [2, 0], [2, -1], [3, -1]]

        equivalent = deviation_from_reference(raw, transformed)
        mirror = deviation_from_reference(raw, reflected)

        self.assertLess(equivalent["p95"], 1e-6)
        self.assertGreater(mirror["p95"], 0.1)

    def test_three_dimensional_metrics_use_all_axes(self) -> None:
        metrics = trajectory_metrics([[0, 0, 0], [1, 0, 2], [1, 3, 2]])

        self.assertEqual(metrics["dimensions"], 3)
        self.assertAlmostEqual(metrics["path_length"], 5.236068, places=6)
        self.assertEqual(metrics["bbox_extents"], [1.0, 3.0, 2.0])
        self.assertEqual(metrics["bbox_depth"], 2.0)

    def test_comparison_is_invariant_to_non_reflecting_similarity(self) -> None:
        reference = [[0, 0], [3, 0], [3, 2], [5, 2], [5, 5]]
        candidate = [[10, -4], [10, 2], [6, 2], [6, 6], [0, 6]]

        comparison = compare_trajectories(reference, candidate)

        self.assertTrue(comparison["available"])
        self.assertAlmostEqual(comparison["rotation_determinant"], 1.0)
        self.assertLess(comparison["normalized_frechet_distance"], 1e-6)
        self.assertLess(comparison["normalized_chamfer_distance"], 1e-6)
        self.assertGreater(comparison["turn_sequence_agreement"], 0.99)
        self.assertGreater(comparison["local_direction_agreement"], 0.99)

    def test_comparison_reports_requested_shape_components(self) -> None:
        reference = [[0, 0], [4, 0], [4, 3], [8, 3]]
        candidate = [[0, 0], [2, 0], [2, 6], [8, 6]]

        comparison = compare_trajectories(reference, candidate)

        for key in (
            "normalized_frechet_distance",
            "normalized_chamfer_distance",
            "turn_sequence_agreement",
            "local_direction_agreement",
            "segment_length_log_rmse",
            "endpoint_progress_ratio",
            "spatial_span_ratio",
            "curvature_distribution_distance",
        ):
            self.assertIn(key, comparison)
        self.assertGreater(comparison["segment_length_log_rmse"], 0.1)

    def test_comparison_penalizes_reversed_temporal_direction(self) -> None:
        reference = [[0, 0], [4, 0], [4, 2], [8, 2], [8, 5]]

        comparison = compare_trajectories(reference, list(reversed(reference)))

        self.assertLess(comparison["local_direction_agreement"], 0.75)
        self.assertGreater(comparison["normalized_frechet_distance"], 0.05)

    def test_acceptance_rejects_collapsed_candidate(self) -> None:
        reference = [[0, 0], [10, 0], [10, 8], [20, 8]]
        collapsed = [[0, 0], [2, 0], [0, 1], [2, 1], [0.2, 0.1]]

        result = trajectory_acceptance(reference, collapsed)

        self.assertFalse(result["accepted"])
        self.assertTrue(
            {"endpoint_progress_ratio_out_of_bounds", "spatial_span_ratio_out_of_bounds"}
            .intersection(result["rejection_reasons"])
        )

    def test_acceptance_thresholds_are_context_overridable(self) -> None:
        reference = [[0, 0], [5, 0], [5, 5]]
        candidate = [[0, 0], [4, 0], [4, 6]]
        strict = trajectory_acceptance(reference, candidate, {
            "thresholds": {"maximum_normalized_frechet": 0.0}
        })
        permissive = trajectory_acceptance(reference, candidate, {
            "thresholds": {
                "maximum_normalized_frechet": 1.0,
                "maximum_normalized_chamfer": 1.0,
                "minimum_turn_sequence_agreement": 0.0,
                "minimum_local_direction_agreement": 0.0,
                "maximum_segment_length_log_rmse": 10.0,
                "minimum_endpoint_ratio": 0.0,
                "maximum_endpoint_ratio": 10.0,
                "minimum_span_ratio": 0.0,
                "maximum_span_ratio": 10.0,
                "maximum_curvature_distribution_distance": 1.0,
            }
        })

        self.assertFalse(strict["accepted"])
        self.assertTrue(permissive["accepted"])


class BaselineCaptureTests(unittest.TestCase):
    def test_five_minute_problem_video_golden_metric_ranges(self) -> None:
        baseline = (
            BACKEND_ROOT.parent
            / "baselines"
            / "five_minute_2026-07-23"
        )
        ground_truth = json.loads(
            (baseline / "ground_truth.json").read_text(encoding="utf-8")
        )
        expected = json.loads(
            (baseline / "expected_metric_ranges.json").read_text(
                encoding="utf-8"
            )
        )
        trajectory = ground_truth["trajectory"]
        metrics = trajectory_metrics(trajectory)

        self.assertTrue((baseline / "reference_route.png").is_file())
        self.assertGreaterEqual(
            len(trajectory),
            expected["required"]["minimum_point_count"],
        )
        self.assertEqual(
            trajectory[0],
            expected["required"]["start_point"],
        )
        self.assertEqual(
            ground_truth["initial_direction_point"],
            {
                "x": expected["required"]["initial_direction_point"][0],
                "y": expected["required"]["initial_direction_point"][1],
            },
        )
        self.assertEqual(
            trajectory[-1],
            expected["required"]["end_point"],
        )
        for name, bounds in expected["ranges"].items():
            value = float(metrics[name])
            self.assertGreaterEqual(value, float(bounds[0]), name)
            self.assertLessEqual(value, float(bounds[1]), name)

    def test_cli_copies_artifacts_and_writes_hashed_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            r3 = root / "r3"
            camera = r3 / "camera"
            camera.mkdir(parents=True)
            raw = poses_from_xy([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)])
            robust = raw.copy()
            robust[:, 0, 3] *= 1.05
            scale = robust.copy()
            scale[:, 2, 3] *= 0.95
            for index, pose in enumerate(raw):
                np.savez_compressed(camera / f"{index:06d}.npz", pose=pose)
            graph = r3 / "pose_graph_edges.npz"
            np.savez_compressed(graph, placeholder=np.asarray([1]))
            save_pose_graph_candidate(r3, {
                "c2w": robust,
                "diagnostics": {
                    "schema_version": 1,
                    "accepted": True,
                    "rejection_reasons": [],
                    "source_graph_mtime_ns": graph.stat().st_mtime_ns,
                },
            })
            save_scale_aware_candidate(r3, {
                "c2w": scale,
                "scale": np.ones(len(scale)),
                "diagnostics": {
                    "schema_version": 1,
                    "accepted": True,
                    "rejection_reasons": [],
                },
            })
            (r3 / "frame_selection.json").write_text(
                json.dumps({"source_indices": list(range(len(raw)))}),
                encoding="utf-8",
            )
            (r3 / "run_params.json").write_text("{}", encoding="utf-8")
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps({
                "analysis_result": {
                    "map_trajectory": [[100, 100, 0], [120, 100, 0], [120, 120, 0]],
                    "map_metadata": {"meters_per_pixel": 0.05},
                    "floorplan_constraint": {
                        "accepted": True,
                        "correction_median_meters": 0.1,
                        "correction_p95_meters": 0.4,
                        "maximum_correction_meters": 0.8,
                    },
                    "processing_stats": {"map_matching_applied": True},
                }
            }), encoding="utf-8")
            context = root / "map_context.json"
            context.write_text(json.dumps({
                "floorplan_id": "test",
                "reference_point": {"x": 10, "y": 20},
                "direction_point": {"x": 5, "y": 20},
                "drawn_plan": [{"id": "truth", "type": "line", "points": [
                    {"x": 0, "y": 0}, {"x": 10, "y": 0}
                ]}],
            }), encoding="utf-8")
            ground_truth = root / "ground_truth.json"
            ground_truth.write_text(
                json.dumps({"trajectory": [[0, 0], [10, 0], [10, 10]]}),
                encoding="utf-8",
            )
            output = root / "baseline"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(BACKEND_ROOT / "tools" / "capture_trajectory_baseline.py"),
                    "--r3-output", str(r3),
                    "--analysis", str(analysis),
                    "--map-context", str(context),
                    "--ground-truth", str(ground_truth),
                    "--output", str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((output / "trajectory_report.json").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                set(report["sources"]),
                {"raw", "robust_candidate", "scale_aware_candidate", "final_map"},
            )
            self.assertEqual(
                report["sources"]["final_map"]["map_repair_meters"]["p95"], 0.4
            )
            self.assertTrue(report["ground_truth"]["metrics"]["available"])
            self.assertTrue((output / "artifacts" / "camera" / "000000.npz").is_file())
            self.assertTrue(all(item["sha256"] for item in manifest["files"]))

    def test_cli_refuses_to_overwrite_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing"
            existing.mkdir()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(BACKEND_ROOT / "tools" / "capture_trajectory_baseline.py"),
                    "--r3-output", directory,
                    "--analysis", str(existing / "missing.json"),
                    "--map-context", str(existing / "missing.json"),
                    "--ground-truth", str(existing / "missing.json"),
                    "--output", str(existing),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("never overwrites", completed.stderr)


if __name__ == "__main__":
    unittest.main()
