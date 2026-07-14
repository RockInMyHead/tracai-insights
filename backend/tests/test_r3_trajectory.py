"""Regression tests for R3 plan-space trajectory conversion."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3_trajectory import build_r3_trajectory, summarize_fallback_edges


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
    physical_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, physical_up)
    right /= np.linalg.norm(right)
    camera_down = np.cross(forward, right)
    camera_down /= np.linalg.norm(camera_down)
    return np.column_stack((right, camera_down, forward)).tolist()


def rotation_for_pitched_heading(x: float, z: float, pitch_degrees: float) -> list[list[float]]:
    horizontal = np.array([x, 0.0, z], dtype=np.float64)
    horizontal /= np.linalg.norm(horizontal)
    pitch = math.radians(pitch_degrees)
    camera_down_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward = math.cos(pitch) * horizontal + math.sin(pitch) * camera_down_world
    forward /= np.linalg.norm(forward)
    physical_up = -camera_down_world
    right = np.cross(forward, physical_up)
    right /= np.linalg.norm(right)
    camera_down = np.cross(forward, right)
    camera_down /= np.linalg.norm(camera_down)
    return np.column_stack((right, camera_down, forward)).tolist()


class R3TrajectoryTests(unittest.TestCase):
    def test_opencv_left_turn_has_canonical_left_handedness(self) -> None:
        # R3 exports OpenCV c2w: local +Y is camera-down.  With a +Z initial
        # heading, a physical left turn goes toward -X and must become +Y in
        # canonical plan space.
        poses = [
            make_pose(i, 0.0, 0.0, float(i), rotation_for_forward(0.0, 0.0, 1.0))
            for i in range(18)
        ]
        poses.extend(
            make_pose(
                len(poses),
                -float(index),
                0.0,
                17.0,
                rotation_for_forward(-1.0, 0.0, 0.0),
            )
            for index in range(1, 19)
        )

        result = build_r3_trajectory(poses, [2.0] * len(poses))

        self.assertEqual(result["trajectory_quality"]["projection"]["plan_coordinate_convention"], "x_forward_y_left_z_up")
        self.assertGreater(result["plan_trajectory"][-1][1], 15.0)
        self.assertEqual(len(result["turn_points"]), 1)
        self.assertEqual(result["turn_points"][0]["turn_type"], "left")
        self.assertAlmostEqual(result["turn_points"][0]["angle_degrees"], 90.0, delta=2.0)

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
        self.assertEqual(turn["angle_source"], "trajectory_curvature")
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

    def test_separates_repeated_turns_and_repairs_underestimated_loop(self) -> None:
        # This reproduces the field regression: the reconstructed translation
        # bends only 20 degrees at each physical 90-degree corner.  The old
        # largest-span candidate merged all four same-sign turns into one and
        # globally disabled the otherwise correct camera-yaw signal.
        poses = []
        x = 0.0
        z = 0.0
        trajectory_heading = 0.0
        for side in range(4):
            camera_heading = side * math.pi / 2.0
            for _ in range(25):
                x += math.cos(trajectory_heading)
                z += math.sin(trajectory_heading)
                poses.append(make_pose(
                    len(poses),
                    x,
                    0.0,
                    z,
                    rotation_for_forward(math.cos(camera_heading), 0.0, math.sin(camera_heading)),
                ))
            for index in range(1, 21):
                position_yaw = trajectory_heading + math.radians(20.0) * index / 20.0
                camera_yaw = camera_heading + math.pi / 2.0 * index / 20.0
                x += math.cos(position_yaw)
                z += math.sin(position_yaw)
                poses.append(make_pose(
                    len(poses),
                    x,
                    0.0,
                    z,
                    rotation_for_forward(math.cos(camera_yaw), 0.0, math.sin(camera_yaw)),
                ))
            trajectory_heading += math.radians(20.0)

        result = build_r3_trajectory(poses, [2.0] * len(poses))

        self.assertEqual(len(result["turn_points"]), 4)
        for turn in result["turn_points"]:
            self.assertEqual(turn["angle_source"], "camera_orientation")
            self.assertAlmostEqual(abs(turn["trajectory_angle_degrees"]), 20.0, delta=2.0)
            self.assertAlmostEqual(abs(turn["angle_degrees"]), 90.0, delta=2.0)

        raw_plan = np.asarray(result["raw_plan_trajectory"], dtype=np.float64)
        corrected_plan = np.asarray(result["plan_trajectory"], dtype=np.float64)
        raw_closure_error = float(np.linalg.norm(raw_plan[-1, :2] - raw_plan[0, :2]))
        corrected_closure_error = float(np.linalg.norm(corrected_plan[-1, :2] - corrected_plan[0, :2]))
        self.assertGreater(raw_closure_error, 100.0)
        self.assertLess(corrected_closure_error, 12.0)
        correction = result["trajectory_quality"]["heading_correction"]
        self.assertEqual(correction["applied_count"], 4)

    def test_trajectory_plane_prevents_pitch_dependent_scale(self) -> None:
        # A downward-looking camera makes camera-local up tilt backward.  Using
        # that vector as the floor normal compresses the long first leg and
        # makes the post-turn leg look larger.  A planar L-route has enough 2D
        # support to recover the true floor with trajectory PCA.
        poses = [
            make_pose(
                i,
                0.0,
                0.0,
                float(i),
                rotation_for_pitched_heading(0.0, 1.0, 55.0),
            )
            for i in range(70)
        ]
        poses.extend(
            make_pose(
                len(poses),
                -float(index),
                0.0,
                69.0,
                rotation_for_pitched_heading(-1.0, 0.0, 55.0),
            )
            for index in range(1, 31)
        )

        result = build_r3_trajectory(poses, [2.0] * len(poses))
        plan = np.asarray(result["raw_plan_trajectory"], dtype=np.float64)
        steps = np.linalg.norm(np.diff(plan[:, :2], axis=0), axis=1)
        before = float(np.median(steps[10:60]))
        after = float(np.median(steps[75:95]))

        projection = result["trajectory_quality"]["projection"]
        self.assertEqual(projection["method"], "trajectory_plane_pca")
        self.assertTrue(projection["trajectory_plane"]["eligible"])
        self.assertAlmostEqual(after / before, 1.0, delta=0.03)

    def test_camera_turn_axis_recovers_gravity_from_downward_camera(self) -> None:
        # A production camera is pointed steeply at the floor, so its local
        # up vector is not gravity. Gradual yaw still has one common world
        # rotation axis and recovers the physical floor normal.
        poses = []
        for index in range(100):
            angle = (math.pi / 2.0) * index / 99.0
            poses.append(make_pose(
                index,
                40.0 * math.sin(angle),
                0.0,
                40.0 * (1.0 - math.cos(angle)),
                rotation_for_pitched_heading(math.cos(angle), math.sin(angle), 55.0),
            ))

        result = build_r3_trajectory(poses, [2.0] * len(poses))

        projection = result["trajectory_quality"]["projection"]
        self.assertEqual(projection["method"], "camera_rotation_axis")
        self.assertTrue(projection["camera_rotation_axis"]["reliable"])
        normal = np.asarray(projection["normal"], dtype=np.float64)
        self.assertGreater(float(np.dot(normal, np.array([0.0, -1.0, 0.0]))), 0.99)

    def test_repairs_persistent_scale_reset_at_forced_fallback(self) -> None:
        poses = []
        z = 0.0
        poses.append(make_pose(0, 0.0, 0.0, z))
        for _ in range(60):
            z += 1.0
            poses.append(make_pose(len(poses), 0.0, 0.0, z))
        for _ in range(80):
            z += 3.0
            poses.append(make_pose(len(poses), 0.0, 0.0, z))

        result = build_r3_trajectory(
            poses,
            [2.0] * len(poses),
            run_params={
                "online_fallback_enabled": True,
                "metric_scale_enabled": True,
                "fallback_boundaries": [61],
                "fallback_boundary_source": "pose_edge_log",
            },
        )

        plan = np.asarray(result["plan_trajectory"], dtype=np.float64)
        distance = float(np.linalg.norm(np.diff(plan[:, :2], axis=0), axis=1).sum())
        scale = result["trajectory_quality"]["scale_stability"]
        self.assertTrue(scale["applied"])
        self.assertEqual(scale["applied_count"], 1)
        self.assertAlmostEqual(scale["regime_changes"][0]["raw_velocity_ratio"], 3.0, delta=0.05)
        self.assertAlmostEqual(distance, 142.0, delta=3.0)

    def test_repeated_fallbacks_do_not_compound_one_scale_epoch(self) -> None:
        poses = []
        z = 0.0
        poses.append(make_pose(0, 0.0, 0.0, z))
        for _ in range(50):
            z += 1.0
            poses.append(make_pose(len(poses), 0.0, 0.0, z))
        for _ in range(160):
            z += 0.4
            poses.append(make_pose(len(poses), 0.0, 0.0, z))

        result = build_r3_trajectory(
            poses,
            [2.0] * len(poses),
            {"source_indices": [index * 5 for index in range(len(poses))]},
            run_params={
                "online_fallback_enabled": True,
                "fallback_boundaries": [51, 91, 131, 171],
                "fallback_boundary_source": "pose_edge_log",
            },
        )

        scale = result["trajectory_quality"]["scale_stability"]
        distance = float(np.linalg.norm(
            np.diff(np.asarray(result["plan_trajectory"])[:, :2], axis=0),
            axis=1,
        ).sum())
        self.assertTrue(scale["applied"])
        self.assertEqual(scale["applied_count"], 1)
        self.assertFalse(scale["cumulative_scaling"])
        self.assertEqual(len(scale["regimes"]), 2)
        self.assertAlmostEqual(scale["regime_changes"][0]["applied_scale"], 2.5, delta=0.05)
        self.assertAlmostEqual(distance, 208.0, delta=4.0)

    def test_pose_edge_log_yields_only_explicit_fallback_boundaries(self) -> None:
        summary = summarize_fallback_edges(
            [
                {"frame_i": 0, "frame_j": 1, "edge_type": "normal"},
                {"frame_i": 40, "frame_j": 45, "edge_type": "bridge"},
                {"frame_i": 45, "frame_j": 49, "edge_type": "bridge"},
                {"frame_i": 180, "frame_j": 190, "edge_type": "bridge"},
                {"frame_i": 190, "frame_j": 195, "edge_type": "bridge"},
            ],
            point_count=220,
            bridge_window=10,
        )

        self.assertEqual(summary["bridge_edge_count"], 4)
        self.assertEqual(summary["boundaries"], [50, 196])
        self.assertEqual([event["bridge_end"] for event in summary["events"]], [49, 195])

    def test_does_not_normalize_unconfirmed_walking_speed_change(self) -> None:
        poses = []
        z = 0.0
        poses.append(make_pose(0, 0.0, 0.0, z))
        for _ in range(70):
            z += 1.0
            poses.append(make_pose(len(poses), 0.0, 0.0, z))
        for _ in range(70):
            z += 2.0
            poses.append(make_pose(len(poses), 0.0, 0.0, z))

        result = build_r3_trajectory(poses, [2.0] * len(poses))

        scale = result["trajectory_quality"]["scale_stability"]
        distance = float(np.linalg.norm(
            np.diff(np.asarray(result["plan_trajectory"])[:, :2], axis=0),
            axis=1,
        ).sum())
        self.assertFalse(scale["applied"])
        self.assertAlmostEqual(distance, 210.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
