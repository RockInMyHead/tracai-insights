import unittest

import numpy as np

from backend.trajectory_geometry import (
    compare_trajectories,
    trajectory_acceptance,
    trajectory_metrics,
)


class TrajectoryGeometryTests(unittest.TestCase):
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
