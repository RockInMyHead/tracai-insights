"""Tests for segmented long-video pose stitching."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3_long_video import (
    align_segment_poses,
    estimate_pose_similarity,
    plan_segment_windows,
    transform_camera_pose,
)


def yaw_pose(x: float, z: float, yaw_degrees: float) -> np.ndarray:
    yaw = math.radians(yaw_degrees)
    rotation = np.array([
        [math.cos(yaw), 0.0, math.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-math.sin(yaw), 0.0, math.cos(yaw)],
    ])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = [x, 0.0, z]
    return pose


class R3LongVideoTests(unittest.TestCase):
    def test_segment_windows_cover_all_frames_with_overlap(self) -> None:
        windows = plan_segment_windows(4_500, segment_frames=1_500, overlap_frames=90)

        self.assertEqual(windows[0].start, 0)
        self.assertEqual(windows[-1].end, 4_500)
        self.assertTrue(all(window.frame_count <= 1_500 for window in windows))
        for previous, current in zip(windows, windows[1:]):
            self.assertEqual(previous.end - current.start, 90)

    def test_similarity_recovers_scale_rotation_and_translation(self) -> None:
        local = [yaw_pose(float(i), float(i % 2), i * 5.0) for i in range(6)]
        world_rotation = yaw_pose(0.0, 0.0, 35.0)[:3, :3]
        world_scale = 1.7
        world_translation = np.array([4.0, -0.3, 8.0])
        global_poses = [
            transform_camera_pose(pose, world_rotation, world_scale, world_translation)
            for pose in local
        ]

        rotation, scale, translation, diagnostics = estimate_pose_similarity(global_poses, local)

        self.assertTrue(np.allclose(rotation, world_rotation, atol=1e-7))
        self.assertAlmostEqual(scale, world_scale, places=7)
        self.assertTrue(np.allclose(translation, world_translation, atol=1e-7))
        self.assertLess(float(diagnostics["max_residual"]), 1e-6)

    def test_segment_alignment_preserves_overlap_and_extends_path(self) -> None:
        merged = {index: yaw_pose(float(index), 0.0, 0.0) for index in range(8)}
        # Local segment has a different origin, yaw and scale. Its first three
        # entries correspond to global frames 5, 6 and 7.
        inverse_rotation = yaw_pose(0.0, 0.0, -25.0)[:3, :3]
        local_scale = 0.5
        local_translation = np.array([-3.0, 0.0, 2.0])
        local: dict[int, np.ndarray] = {}
        global_indices = list(range(5, 13))
        for local_index, global_index in enumerate(global_indices):
            world_pose = yaw_pose(float(global_index), 0.0, 0.0)
            local_pose = np.eye(4)
            local_pose[:3, :3] = inverse_rotation @ world_pose[:3, :3]
            local_pose[:3, 3] = local_scale * (inverse_rotation @ world_pose[:3, 3]) + local_translation
            local[local_index] = local_pose

        aligned, _, diagnostics = align_segment_poses(local, global_indices, merged)

        for global_index in (5, 6, 7):
            self.assertTrue(np.allclose(aligned[global_index], merged[global_index], atol=1e-6))
        self.assertAlmostEqual(float(aligned[12][0, 3]), 12.0, places=5)
        self.assertEqual(diagnostics["overlap_pairs"], 3)


if __name__ == "__main__":
    unittest.main()
