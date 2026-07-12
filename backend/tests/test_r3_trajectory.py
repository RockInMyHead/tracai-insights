"""Regression tests for R3 plan-space trajectory conversion."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3_trajectory import build_r3_trajectory


def make_pose(frame: int, x: float, y: float, z: float, rotation: list[list[float]] | None = None) -> dict:
    rotation = rotation or [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return {
        "frame": frame,
        "pose": [
            [*rotation[0], x],
            [*rotation[1], y],
            [*rotation[2], z],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def rotation_for_forward(x: float, y: float, z: float) -> list[list[float]]:
    forward = np.array([x, y, z], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(up, forward)
    return np.column_stack((right, up, forward)).tolist()


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

    def test_detects_full_angle_of_gradual_ninety_degree_turn(self) -> None:
        # A rounded 90-degree turn spread across 24 R3 poses used to produce
        # no event because no short local window exceeded the threshold.
        poses = [make_pose(i, float(i), 0.0, 0.0) for i in range(21)]
        for index in range(1, 25):
            theta = -math.pi / 2.0 + (math.pi / 2.0) * (index / 24.0)
            poses.append(make_pose(
                len(poses),
                20.0 + 10.0 * math.cos(theta),
                0.0,
                10.0 + 10.0 * math.sin(theta),
            ))
        for index in range(1, 25):
            poses.append(make_pose(len(poses), 30.0, 0.0, 10.0 + float(index)))

        result = build_r3_trajectory(poses, [2.0] * len(poses))

        self.assertEqual(len(result["turn_points"]), 1)
        turn = result["turn_points"][0]
        self.assertAlmostEqual(abs(turn["angle_degrees"]), 90.0, delta=2.0)
        self.assertEqual(turn["angle_source"], "trajectory_multiscale")
        self.assertGreater(turn["span_points"], 20)

    def test_camera_heading_corrects_underestimated_position_turn(self) -> None:
        # Positions drift into a 45-degree bend, while c2w rotations still
        # align with movement and show the physical 90-degree turn.
        poses = [
            make_pose(i, float(i), 0.0, 0.0, rotation_for_forward(1.0, 0.0, 0.0))
            for i in range(17)
        ]
        for index in range(1, 20):
            poses.append(make_pose(
                len(poses),
                16.0 + index / math.sqrt(2.0),
                0.0,
                index / math.sqrt(2.0),
                rotation_for_forward(0.0, 0.0, 1.0),
            ))

        result = build_r3_trajectory(poses, [2.0] * len(poses))

        self.assertEqual(len(result["turn_points"]), 1)
        turn = result["turn_points"][0]
        self.assertEqual(turn["angle_source"], "camera_orientation")
        self.assertAlmostEqual(abs(turn["trajectory_angle_degrees"]), 45.0, delta=2.0)
        self.assertAlmostEqual(abs(turn["angle_degrees"]), 90.0, delta=2.0)

    def test_camera_pan_without_path_turn_is_not_reported(self) -> None:
        # The orientation cue must stay disabled if it is not consistently
        # aligned with the trajectory (for example, an operator pans sideways
        # while continuing straight).
        poses = [
            make_pose(
                i,
                float(i),
                0.0,
                0.0,
                rotation_for_forward(1.0, 0.0, 0.0) if i < 15 else rotation_for_forward(0.0, 0.0, 1.0),
            )
            for i in range(30)
        ]

        result = build_r3_trajectory(poses, [2.0] * len(poses))

        self.assertEqual(result["turn_points"], [])
        orientation = result["trajectory_quality"]["turn_detection"]["camera_orientation"]
        self.assertFalse(orientation["reliable"])


if __name__ == "__main__":
    unittest.main()
