"""Regression tests for R3 plan-space trajectory conversion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3_trajectory import build_r3_trajectory


def make_pose(frame: int, x: float, y: float, z: float) -> dict:
    return {
        "frame": frame,
        "pose": [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


class R3TrajectoryTests(unittest.TestCase):
    def test_projects_floor_path_and_maps_turn_to_source_frame(self) -> None:
        poses = [make_pose(i, float(i), 0.0, 0.0) for i in range(12)]
        poses.extend(make_pose(i + 12, 11.0, 0.0, float(i + 1)) for i in range(11))

        result = build_r3_trajectory(
            poses,
            [2.0] * len(poses),
            {"source_indices": [i * 5 for i in range(len(poses))]},
        )

        self.assertEqual(len(result["plan_trajectory"]), len(poses))
        self.assertEqual(len(result["raw_trajectory_3d"]), len(poses))
        self.assertGreaterEqual(len(result["turn_points"]), 1)
        turn = result["turn_points"][0]
        self.assertIsNotNone(turn["source_frame_index"])
        self.assertEqual(turn["position"], result["plan_trajectory"][turn["trajectory_index"]])

    def test_low_confidence_isolated_reverse_pose_is_repaired(self) -> None:
        coordinates = [[float(i), 0.0, 0.0] for i in range(16)]
        coordinates[8] = [2.0, 0.0, 0.0]
        poses = [make_pose(i, *point) for i, point in enumerate(coordinates)]
        confidence = [2.0] * len(poses)
        confidence[8] = 0.01

        result = build_r3_trajectory(poses, confidence)

        self.assertGreater(result["raw_trajectory_3d"][8][0], 7.0)
        self.assertEqual(result["turn_points"], [])


if __name__ == "__main__":
    unittest.main()
