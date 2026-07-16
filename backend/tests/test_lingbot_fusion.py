import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.lingbot_fusion import build_lingbot_fusion_candidate
from backend.lingbot_worker.lingbot_adapter import LingBotMapAdapter


class LingBotFusionTests(unittest.TestCase):
    def _r3_path(self) -> np.ndarray:
        return np.asarray(
            [[float(x), 0.0] for x in range(8)]
            + [[7.0, float(y)] for y in range(1, 7)]
            + [[float(x), 6.0] for x in range(8, 14)],
            dtype=np.float64,
        )

    def test_similarity_aligns_lingbot_without_reflection_and_builds_candidate(self) -> None:
        r3 = self._r3_path()
        angle = np.deg2rad(31.0)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )
        scale = 2.4
        translation = np.asarray([13.0, -8.0])
        lingbot = ((r3 - translation) @ rotation.T) / scale
        lingbot[:, 1] += np.sin(np.linspace(0.0, np.pi, len(lingbot))) * 0.03

        result = build_lingbot_fusion_candidate(
            {
                "plan_trajectory": np.column_stack((r3, np.zeros(len(r3)))).tolist(),
                "r3_pose_confidence": np.linspace(0.1, 1.0, len(r3)).tolist(),
            },
            {"trajectory": np.column_stack((lingbot, np.zeros(len(lingbot)))).tolist()},
        )

        self.assertTrue(result["accepted"], result["diagnostics"])
        self.assertEqual(len(result["plan_trajectory"]), len(r3))
        self.assertLess(result["diagnostics"]["alignment_p95_ratio"], 0.02)
        self.assertAlmostEqual(result["plan_trajectory"][0][0], r3[0, 0], places=6)
        self.assertFalse(result["diagnostics"]["chirality_conflict"])

    def test_incompatible_trajectory_remains_shadow_only(self) -> None:
        r3 = self._r3_path()
        lingbot = r3.copy()
        lingbot[:, 1] = np.linspace(0.0, 40.0, len(lingbot)) ** 1.2
        result = build_lingbot_fusion_candidate(
            {"plan_trajectory": np.column_stack((r3, np.zeros(len(r3)))).tolist()},
            {"trajectory": np.column_stack((lingbot, np.zeros(len(lingbot)))).tolist()},
        )
        self.assertFalse(result["accepted"])
        self.assertIn(
            result["diagnostics"]["reason"],
            {"trajectory_disagreement_too_large", "turn_chirality_conflict"},
        )
        self.assertTrue(result["independent_accepted"])
        self.assertEqual(len(result["independent_plan_trajectory"]), len(lingbot))

    def test_raw_lingbot_xyz_is_projected_to_its_motion_plane(self) -> None:
        r3 = self._r3_path()
        # Motion lies in world X/Z; raw X/Y projection would collapse it.
        xyz = np.column_stack((r3[:, 0], np.full(len(r3), 3.0), r3[:, 1]))
        result = build_lingbot_fusion_candidate(
            {"plan_trajectory": np.column_stack((r3, np.zeros(len(r3)))).tolist()},
            {"trajectory": xyz.tolist()},
        )
        independent = np.asarray(result["independent_plan_trajectory"])
        self.assertGreater(float(np.ptp(independent[:, 0])), 4.0)
        self.assertGreater(float(np.ptp(independent[:, 1])), 4.0)
        self.assertEqual(
            result["diagnostics"]["lingbot_projection"]["method"],
            "pca_motion_plane",
        )

    def test_saved_upstream_extrinsic_is_kept_as_c2w(self) -> None:
        adapter = LingBotMapAdapter(repo_path=Path("."), model_path=Path("model.pt"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame_000000.npz"
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 3] = [1.0, 2.0, 3.0]
            np.savez_compressed(
                path,
                extrinsic=c2w[:3, :4],
                depth_conf=np.full((4, 5), 2.5, dtype=np.float32),
            )
            trajectory = adapter._discover_trajectory([path])

        self.assertEqual(trajectory["source"], "lingbot_per_frame_npz")
        self.assertEqual(trajectory["poses"][0]["position"], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(trajectory["poses"][0]["confidence"], 2.5)


if __name__ == "__main__":
    unittest.main()
