"""Tests for immutable raw/robust R3 trajectory source selection."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3_pose_graph_optimizer import save_pose_graph_candidate
from r3_trajectory_sources import select_r3_trajectory_camera_poses


def line_poses(point_count: int, scale: float) -> np.ndarray:
    poses = np.broadcast_to(np.eye(4), (point_count, 4, 4)).copy()
    poses[:, 0, 3] = np.arange(point_count, dtype=float) * scale
    return poses


def write_camera_artifacts(base: Path, poses: np.ndarray) -> None:
    camera_dir = base / "camera"
    camera_dir.mkdir()
    for index, pose in enumerate(poses):
        np.savez_compressed(
            camera_dir / f"{index:06d}.npz",
            pose=pose,
            intrinsics=np.eye(3),
        )


def write_candidate(
    base: Path,
    poses: np.ndarray,
    *,
    accepted: bool = True,
) -> None:
    graph_path = base / "pose_graph_edges.npz"
    np.savez_compressed(graph_path, placeholder=np.asarray([1]))
    save_pose_graph_candidate(base, {
        "c2w": poses,
        "diagnostics": {
            "schema_version": 1,
            "accepted": accepted,
            "rejection_reasons": [] if accepted else ["quality_gate"],
            "objective_improvement": 0.5,
            "runtime_seconds": 0.1,
            "source_graph_mtime_ns": graph_path.stat().st_mtime_ns,
        },
    })


class R3CandidateSelectionTests(unittest.TestCase):
    def test_accepted_current_candidate_is_selected_without_touching_raw(self) -> None:
        raw = line_poses(8, 1.2)
        candidate = line_poses(8, 1.0)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            write_camera_artifacts(base, raw)
            write_candidate(base, candidate)

            selected, selection = select_r3_trajectory_camera_poses(
                base,
                [
                    {"frame": index, "pose": pose.tolist(), "intrinsics": None}
                    for index, pose in enumerate(raw)
                ],
                "robust_candidate",
            )
            persisted_raw = np.load(base / "camera" / "000007.npz")["pose"]

        self.assertEqual(selection["selected"], "robust_candidate")
        self.assertIsNone(selection["fallback_reason"])
        self.assertAlmostEqual(selected[-1]["pose"][0][3], 7.0)
        np.testing.assert_allclose(persisted_raw, raw[-1])

    def test_stale_candidate_falls_back_to_raw(self) -> None:
        raw = line_poses(8, 1.2)
        candidate = line_poses(8, 1.0)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            write_camera_artifacts(base, raw)
            write_candidate(base, candidate)
            graph_path = base / "pose_graph_edges.npz"
            current = graph_path.stat().st_mtime_ns
            os.utime(graph_path, ns=(current + 1_000_000, current + 1_000_000))

            selected, selection = select_r3_trajectory_camera_poses(
                base,
                [
                    {"frame": index, "pose": pose.tolist(), "intrinsics": None}
                    for index, pose in enumerate(raw)
                ],
                "robust_candidate",
            )

        self.assertEqual(selection["selected"], "raw")
        self.assertEqual(selection["fallback_reason"], "candidate_stale")
        self.assertAlmostEqual(selected[-1]["pose"][0][3], 8.4)

    def test_rejected_candidate_falls_back_to_raw(self) -> None:
        raw = line_poses(8, 1.2)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            write_camera_artifacts(base, raw)
            write_candidate(base, line_poses(8, 1.0), accepted=False)

            _, selection = select_r3_trajectory_camera_poses(
                base,
                [
                    {"frame": index, "pose": pose.tolist(), "intrinsics": None}
                    for index, pose in enumerate(raw)
                ],
                "robust_candidate",
            )

        self.assertEqual(selection["selected"], "raw")
        self.assertEqual(selection["fallback_reason"], "candidate_rejected")


if __name__ == "__main__":
    unittest.main()
