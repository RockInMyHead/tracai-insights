import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.lingbot_fusion import (
    build_lingbot_fusion_candidate,
    should_restore_lingbot_fusion_candidate,
)
from backend.lingbot_worker.lingbot_adapter import LingBotMapAdapter


class LingBotFusionTests(unittest.TestCase):
    def _r3_path(self) -> np.ndarray:
        return np.asarray(
            [[float(x), 0.0] for x in range(8)]
            + [[7.0, float(y)] for y in range(1, 7)]
            + [[float(x), 6.0] for x in range(8, 14)],
            dtype=np.float64,
        )

    def _r3_left_hook(self) -> np.ndarray:
        """Single left turn with large net signed rotation for chirality tests."""
        return np.asarray(
            [[float(x), 0.0] for x in range(12)]
            + [[11.0, float(y)] for y in range(1, 12)],
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
        self.assertAlmostEqual(result["plan_trajectory"][-1][0], r3[-1, 0], places=6)
        self.assertAlmostEqual(result["plan_trajectory"][-1][1], r3[-1, 1], places=6)
        self.assertEqual(result["diagnostics"]["endpoint_displacement"], 0.0)
        self.assertFalse(result["diagnostics"]["chirality_conflict"])
        self.assertIn("v3_gip", result["diagnostics"]["method"])

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
            {"trajectory": xyz.tolist(), "raw_trajectory_3d": xyz.tolist()},
        )
        independent = np.asarray(result["independent_plan_trajectory"])
        self.assertGreater(float(np.ptp(independent[:, 0])), 4.0)
        self.assertGreater(float(np.ptp(independent[:, 1])), 4.0)
        self.assertEqual(
            result["diagnostics"]["lingbot_projection"]["method"],
            "pca_motion_plane",
        )

    def test_raw_trajectory_3d_is_preferred_over_adapter_xz_plan(self) -> None:
        """Adapter XZ in plan_trajectory must not disable PCA gauge freedom."""
        r3 = self._r3_path()
        xyz = np.column_stack((r3[:, 0], np.full(len(r3), 1.5), r3[:, 1]))
        # Deliberately mirrored adapter plane — must be ignored when raw 3-D exists.
        mirrored = r3.copy()
        mirrored[:, 1] *= -1.0
        result = build_lingbot_fusion_candidate(
            {"plan_trajectory": np.column_stack((r3, np.zeros(len(r3)))).tolist()},
            {
                "plan_trajectory": np.column_stack((mirrored, np.zeros(len(mirrored)))).tolist(),
                "trajectory": np.column_stack((mirrored, np.zeros(len(mirrored)))).tolist(),
                "raw_trajectory_3d": xyz.tolist(),
            },
        )
        self.assertEqual(
            result["diagnostics"]["lingbot_projection"]["method"],
            "pca_motion_plane",
        )
        self.assertTrue(result["independent_accepted"], result["diagnostics"])

    def test_pca_chirality_conflict_keeps_independent_observer(self) -> None:
        r3 = self._r3_left_hook()
        # Motion in X/Z with opposite Z so residual-best PCA sign fights chirality.
        xyz = np.column_stack((r3[:, 0], np.full(len(r3), 1.2), -r3[:, 1]))
        result = build_lingbot_fusion_candidate(
            {"plan_trajectory": np.column_stack((r3, np.zeros(len(r3)))).tolist()},
            {"raw_trajectory_3d": xyz.tolist(), "trajectory": xyz.tolist()},
        )
        self.assertEqual(
            result["diagnostics"]["lingbot_projection"]["method"],
            "pca_motion_plane",
        )
        # Noisy cumulative-turn chirality must not revoke the independent observer
        # or hard-reject residual-gated PCA fusion on its own.
        self.assertTrue(result["independent_accepted"], result["diagnostics"])
        self.assertGreater(len(result["independent_plan_trajectory"]), 5)
        if result["diagnostics"].get("chirality_conflict"):
            self.assertTrue(result["diagnostics"].get("chirality_soft_conflict"))
            self.assertNotEqual(
                result["diagnostics"].get("reason"),
                "turn_chirality_conflict",
            )


    def test_explicit_opposite_chirality_rejects_fusion_and_independent(self) -> None:
        r3 = self._r3_left_hook()
        lingbot = r3.copy()
        lingbot[:, 1] *= -1.0
        result = build_lingbot_fusion_candidate(
            {"plan_trajectory": np.column_stack((r3, np.zeros(len(r3)))).tolist()},
            {"plan_trajectory": np.column_stack((lingbot, np.zeros(len(lingbot)))).tolist()},
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["diagnostics"]["reason"], "turn_chirality_conflict")
        self.assertFalse(result["independent_accepted"])
        self.assertFalse(result["diagnostics"]["reflection_applied"])

    def test_explicit_path_does_not_silently_accept_y_flip_to_fake_chirality(self) -> None:
        r3 = self._r3_left_hook()
        lingbot = r3.copy()
        lingbot[:, 1] *= -1.0
        result = build_lingbot_fusion_candidate(
            {"plan_trajectory": np.column_stack((r3, np.zeros(len(r3)))).tolist()},
            {"plan_trajectory": np.column_stack((lingbot, np.zeros(len(lingbot)))).tolist()},
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["diagnostics"]["reason"], "turn_chirality_conflict")
        self.assertIn("adapter_y_flip_median_residual", result["diagnostics"])
        self.assertNotEqual(
            result["diagnostics"].get("selected_hypothesis"),
            "coordinate_adapter_y_flip",
        )

    def test_pca_selected_sign_is_applied_to_independent_trajectory(self) -> None:
        r3 = self._r3_path()
        # Build a PCA path whose native Y-sign disagrees, but the flipped sign fits.
        xyz = np.column_stack((r3[:, 0], np.full(len(r3), 1.5), -r3[:, 1]))
        result = build_lingbot_fusion_candidate(
            {"plan_trajectory": np.column_stack((r3, np.zeros(len(r3)))).tolist()},
            {"trajectory": xyz.tolist()},
        )
        self.assertIn(result["diagnostics"]["selected_sign"], (1.0, -1.0))
        independent = np.asarray(result["independent_plan_trajectory"])
        self.assertGreater(len(independent), 5)
        if result["diagnostics"]["selected_sign"] < 0.0:
            # Independent polarity matches the selected PCA sign gauge.
            self.assertLess(float(independent[-1, 1]), 0.0)

    def test_stationary_lingbot_is_not_independent_accepted(self) -> None:
        r3 = self._r3_path()
        lingbot = np.repeat([[1.0, 2.0]], 12, axis=0)
        result = build_lingbot_fusion_candidate(
            {"plan_trajectory": np.column_stack((r3, np.zeros(len(r3)))).tolist()},
            {"plan_trajectory": np.column_stack((lingbot, np.zeros(len(lingbot)))).tolist()},
        )
        self.assertFalse(result["accepted"])
        self.assertFalse(result["independent_accepted"])
        self.assertIn("degenerate_span", result["diagnostics"]["independent_quality"]["reasons"])

    def test_timestamp_correspondence_is_preferred_over_arc_length(self) -> None:
        r3 = self._r3_path()
        lingbot = r3.copy()
        # Dense early samples then sparse late samples: arc-length still works,
        # but timestamp mode should be selected when both clocks exist.
        r3_t = np.linspace(0.0, 10.0, len(r3)).tolist()
        lb_t = np.linspace(0.0, 10.0, len(lingbot)).tolist()
        result = build_lingbot_fusion_candidate(
            {
                "plan_trajectory": np.column_stack((r3, np.zeros(len(r3)))).tolist(),
                "r3_source_timestamps_seconds": r3_t,
            },
            {
                "plan_trajectory": np.column_stack((lingbot, np.zeros(len(lingbot)))).tolist(),
                "lingbot_source_timestamps_seconds": lb_t,
            },
        )
        self.assertTrue(result["accepted"], result["diagnostics"])
        self.assertEqual(result["diagnostics"]["correspondence_mode"], "timestamp_offset")
        self.assertIn("lingbot_source_timestamps_seconds", result)

    def test_timestamp_correspondence_uses_nonuniform_r3_clock(self) -> None:
        from backend.lingbot_fusion import _correspond
        lingbot = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        mapped, mode = _correspond(
            lingbot, 4, np.asarray([0.0, 5.0, 10.0]), np.asarray([0.0, 1.0, 9.0, 10.0])
        )
        self.assertEqual(mode, "timestamp_offset")
        self.assertTrue(np.allclose(mapped[:, 0], [0.0, 0.2, 1.8, 2.0]))

    def test_divergent_duration_falls_back_to_arc_length_not_stretch(self) -> None:
        """LingBot 250s vs R³ 305s must not invent a 1.22× time stretch."""
        from backend.lingbot_fusion import _correspond

        r3 = self._r3_path()
        lingbot = r3.copy()
        # Same geometry, but clocks cover different video durations (production case).
        lb_t = np.linspace(0.0, 250.2, len(lingbot))
        r3_t = np.linspace(0.0, 305.3, len(r3))
        mapped, mode = _correspond(lingbot, len(r3), lb_t, r3_t)
        self.assertEqual(mode, "arc_length")
        self.assertEqual(len(mapped), len(r3))

        result = build_lingbot_fusion_candidate(
            {
                "plan_trajectory": np.column_stack((r3, np.zeros(len(r3)))).tolist(),
                "r3_source_timestamps_seconds": r3_t.tolist(),
            },
            {
                "plan_trajectory": np.column_stack((lingbot, np.zeros(len(lingbot)))).tolist(),
                "lingbot_source_timestamps_seconds": lb_t.tolist(),
            },
        )
        self.assertEqual(result["diagnostics"]["correspondence_mode"], "arc_length")
        self.assertTrue(result["accepted"], result["diagnostics"])

    def test_restore_helper_keeps_independent_candidates(self) -> None:
        self.assertTrue(
            should_restore_lingbot_fusion_candidate(
                {"accepted": False, "independent_accepted": True},
                requested_source="scale_aware_candidate",
                saved_source="scale_aware_candidate",
            )
        )
        self.assertFalse(
            should_restore_lingbot_fusion_candidate(
                {"accepted": False, "independent_accepted": True},
                requested_source="raw",
                saved_source="scale_aware_candidate",
            )
        )
        self.assertTrue(
            should_restore_lingbot_fusion_candidate(
                {"accepted": True, "independent_accepted": False},
                requested_source="raw",
                saved_source="raw",
            )
        )
        self.assertTrue(
            should_restore_lingbot_fusion_candidate(
                {"accepted": True, "independent_accepted": True},
                requested_source="scale_aware_candidate",
                saved_source="raw",
                saved_requested_source="scale_aware_candidate",
            )
        )
        # Production bug: pose-confidence arrays were stored as saved_source.
        self.assertTrue(
            should_restore_lingbot_fusion_candidate(
                {"accepted": False, "independent_accepted": True},
                requested_source="scale_aware_candidate",
                saved_source=str([34.9, 34.9, 37.0]),
            )
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
