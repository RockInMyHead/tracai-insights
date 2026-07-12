"""Tests for bounded R3 production point-cloud generation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3_pointcloud import build_sampled_pointcloud


class R3PointCloudTests(unittest.TestCase):
    def _make_output(self, root: Path, frame_count: int = 10) -> None:
        camera_dir = root / "camera"
        depth_dir = root / "depth"
        conf_dir = root / "conf"
        camera_dir.mkdir()
        depth_dir.mkdir()
        conf_dir.mkdir()
        intrinsics = np.array([
            [24.0, 0.0, 16.0],
            [0.0, 24.0, 16.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        for frame in range(frame_count):
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = frame * 0.25
            np.savez(camera_dir / f"{frame:06d}.npz", pose=pose, intrinsics=intrinsics)
            np.save(depth_dir / f"{frame:06d}.npy", np.full((32, 32), 2.0, dtype=np.float32))
            np.save(conf_dir / f"{frame:06d}.npy", np.full((32, 32), 2.0, dtype=np.float32))

    def test_production_cloud_is_bounded_and_preserves_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            self._make_output(output)

            result = build_sampled_pointcloud(
                output,
                stride=1,
                max_points=1_000,
                save_full_debug=False,
            )

            points = np.load(output / "pointcloud.npz")["points"]
            self.assertEqual(result["num_points"], 1_000)
            self.assertEqual(points.shape, (1_000, 8))
            self.assertEqual(set(points[:, 7].astype(int)), set(range(10)))
            self.assertFalse((output / "pointcloud_full_debug.npz").exists())

    def test_full_debug_is_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            self._make_output(output, frame_count=3)

            result = build_sampled_pointcloud(
                output,
                stride=1,
                max_points=1_000,
                save_full_debug=True,
                return_points=False,
            )

            production = np.load(output / "pointcloud.npz")["points"]
            debug = np.load(output / "pointcloud_full_debug.npz")["points"]
            self.assertIsNone(result["points"])
            self.assertEqual(len(production), 1_000)
            self.assertEqual(len(debug), 3 * 32 * 32)
            self.assertTrue(result["full_debug_saved"])


if __name__ == "__main__":
    unittest.main()
