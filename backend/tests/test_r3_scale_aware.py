"""Tests for floor-height anchored R3 scale correction."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3_scale_aware import (
    _estimate_world_up,
    build_scale_aware_candidate,
    estimate_floor_height_observations,
    load_scale_aware_candidate_c2w,
    save_scale_aware_candidate,
)


def pitched_rotation(heading: float, pitch_degrees: float = 55.0) -> np.ndarray:
    horizontal = np.array([np.cos(heading), 0.0, np.sin(heading)])
    camera_down_world = np.array([0.0, 1.0, 0.0])
    pitch = np.radians(pitch_degrees)
    forward = np.cos(pitch) * horizontal + np.sin(pitch) * camera_down_world
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, -camera_down_world)
    right /= np.linalg.norm(right)
    camera_down = np.cross(forward, right)
    return np.column_stack((right, camera_down, forward))


def variable_scale_poses(point_count: int = 120) -> tuple[np.ndarray, np.ndarray]:
    local_scale = np.ones(point_count, dtype=np.float64)
    local_scale[point_count // 2:] = 2.0
    poses = np.broadcast_to(np.eye(4), (point_count, 4, 4)).copy()
    poses[1:, 0, 3] = np.cumsum(local_scale[1:])
    return poses, local_scale


def floor_observations(local_scale: np.ndarray, stride: int = 3) -> list[dict]:
    return [
        {
            "trajectory_index": index,
            "height": 1.6 * float(local_scale[index]),
            "inlier_fraction": 0.55,
            "normal_alignment": 0.98,
            "residual_ratio": 0.01,
        }
        for index in range(0, len(local_scale), stride)
    ]


class R3ScaleAwareTests(unittest.TestCase):
    def test_turn_axis_recovers_floor_normal_for_downward_camera(self) -> None:
        poses = np.broadcast_to(np.eye(4), (80, 4, 4)).copy()
        for index in range(len(poses)):
            heading = 0.5 * np.pi * index / (len(poses) - 1)
            poses[index, :3, :3] = pitched_rotation(heading)
            poses[index, 0, 3] = 20.0 * np.sin(heading)
            poses[index, 2, 3] = 20.0 * (1.0 - np.cos(heading))

        up, method = _estimate_world_up(poses)

        self.assertEqual(method, "camera_rotation_axis")
        self.assertGreater(float(np.dot(up, np.array([0.0, -1.0, 0.0]))), 0.99)

    def test_log_scale_graph_equalizes_piecewise_scale_drift(self) -> None:
        poses, local_scale = variable_scale_poses()
        result = build_scale_aware_candidate(poses, floor_observations(local_scale))
        candidate = result["c2w"]
        steps = np.linalg.norm(np.diff(candidate[:, :3, 3], axis=0), axis=1)
        before = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)

        self.assertTrue(result["diagnostics"]["accepted"])
        self.assertAlmostEqual(float(np.median(before[75:110]) / np.median(before[10:45])), 2.0, delta=0.01)
        self.assertAlmostEqual(float(np.median(steps[75:110]) / np.median(steps[10:45])), 1.0, delta=0.08)
        self.assertGreater(result["diagnostics"]["height_consistency_improvement"], 0.8)
        np.testing.assert_allclose(candidate[0, :3, 3], poses[0, :3, 3])

    def test_sparse_floor_support_is_rejected_without_mutating_input(self) -> None:
        poses, local_scale = variable_scale_poses()
        result = build_scale_aware_candidate(poses, floor_observations(local_scale, stride=30))

        self.assertFalse(result["diagnostics"]["accepted"])
        self.assertIn("insufficient_floor_observations", result["diagnostics"]["rejection_reasons"])

    def test_floor_plane_observer_recovers_synthetic_camera_height(self) -> None:
        point_count = 24
        poses = np.broadcast_to(np.eye(4), (point_count, 4, 4)).copy()
        poses[:, 0, 3] = np.arange(point_count, dtype=np.float64)
        height_px = width_px = 96
        fx = fy = 90.0
        cx = cy = 47.5
        intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "camera").mkdir()
            (base / "depth").mkdir()
            (base / "conf").mkdir()
            rows = np.arange(height_px, dtype=np.float64)[:, None]
            for index in range(point_count):
                camera_height = 1.4 + 0.2 * index / (point_count - 1)
                depth = np.full((height_px, width_px), np.nan, dtype=np.float32)
                valid_rows = rows[:, 0] > cy + 1
                depth[valid_rows, :] = (
                    camera_height * fy / (rows[valid_rows, 0] - cy)
                )[:, None]
                np.savez_compressed(
                    base / "camera" / f"{index:06d}.npz",
                    pose=poses[index],
                    intrinsics=intrinsics,
                )
                np.save(base / "depth" / f"{index:06d}.npy", depth)
                np.save(base / "conf" / f"{index:06d}.npy", np.full_like(depth, 2.0))

            observations, diagnostics = estimate_floor_height_observations(
                base, poses, maximum_frames=point_count
            )

        heights = np.asarray([item["height"] for item in observations])
        self.assertTrue(diagnostics["available"])
        self.assertGreaterEqual(len(observations), 20)
        self.assertAlmostEqual(float(heights[0]), 1.4, delta=0.05)
        self.assertAlmostEqual(float(heights[-1]), 1.6, delta=0.05)

    def test_candidate_artifact_round_trip(self) -> None:
        poses, local_scale = variable_scale_poses()
        result = build_scale_aware_candidate(poses, floor_observations(local_scale))
        with tempfile.TemporaryDirectory() as directory:
            summary = save_scale_aware_candidate(directory, result)
            loaded = load_scale_aware_candidate_c2w(
                directory, expected_count=len(poses), accepted_only=True
            )
        self.assertTrue(summary["available"])
        self.assertIsNotNone(loaded)
        np.testing.assert_allclose(loaded, result["c2w"])


if __name__ == "__main__":
    unittest.main()
