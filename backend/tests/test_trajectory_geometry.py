import unittest

import numpy as np

from backend.trajectory_geometry import (
    align_trajectory_to_anchor,
    compare_trajectories,
    trajectory_acceptance,
    trajectory_metrics,
)


class TrajectoryGeometryTests(unittest.TestCase):
    def test_anchor_alignment_preserves_start_under_translation_rotation_and_scale(self) -> None:
        reference = np.asarray([[20, 30], [30, 30], [30, 40]], dtype=float)
        rotation = np.asarray([[0, -1], [1, 0]], dtype=float)
        candidate = (reference - reference[0]) @ rotation.T * 2.5
        candidate += np.asarray([800.0, -200.0])
        aligned, details = align_trajectory_to_anchor(reference, candidate)
        self.assertTrue(details["available"], details)
        np.testing.assert_allclose(aligned, reference, atol=1e-8)
        np.testing.assert_allclose(aligned[0], reference[0], atol=1e-10)
        self.assertFalse(details["translation_fitted_from_centroid"])

    def test_reflection_is_only_used_when_explicit(self) -> None:
        reference = np.asarray([[0, 0], [10, 0], [10, 5]], dtype=float)
        reflected = reference * np.asarray([1.0, -1.0])
        without_reflection, plain = align_trajectory_to_anchor(reference, reflected)
        with_reflection, explicit = align_trajectory_to_anchor(
            reference, reflected, {"reflect_y": True}
        )
        self.assertFalse(plain["reflection_applied"])
        self.assertTrue(explicit["reflection_applied"])
        self.assertGreater(np.linalg.norm(without_reflection[-1] - reference[-1]), 1.0)
        np.testing.assert_allclose(with_reflection, reference, atol=1e-8)

    def test_similarity_does_not_reflect_turns(self) -> None:
        reference = np.asarray([[0, 0], [5, 0], [5, 5]], dtype=float)
        reflected = np.asarray([[0, 0], [5, 0], [5, -5]], dtype=float)
        comparison = compare_trajectories(reference, reflected)
        self.assertTrue(comparison["available"])
        self.assertAlmostEqual(
            comparison["similarity"]["rotation_determinant"], 1.0, places=6
        )
        self.assertLess(comparison["turn_sequence_agreement"], 0.5)

    def test_collapsed_open_route_is_rejected(self) -> None:
        reference = np.asarray([[0, 0], [10, 0], [20, 0]], dtype=float)
        candidate = np.asarray([[0, 0], [10, 0], [1, 0]], dtype=float)
        result = trajectory_acceptance(reference, candidate)
        self.assertFalse(result["accepted"])
        self.assertIn(
            "endpoint_progress_ratio_out_of_bounds",
            result["rejection_reasons"],
        )

    def test_closed_route_relaxes_endpoint_gate(self) -> None:
        square = np.asarray(
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=float
        )
        result = trajectory_acceptance(
            square,
            square * 1.8 + np.asarray([12.0, -4.0]),
            {"verified_loop_closure": True},
        )
        self.assertTrue(result["accepted"], result)

    def test_metrics_report_span_and_tortuosity(self) -> None:
        metrics = trajectory_metrics([[0, 0], [3, 0], [3, 4]])
        self.assertAlmostEqual(metrics["path_length"], 7.0)
        self.assertAlmostEqual(metrics["endpoint_displacement"], 5.0)
        self.assertAlmostEqual(metrics["tortuosity"], 1.4)


if __name__ == "__main__":
    unittest.main()
