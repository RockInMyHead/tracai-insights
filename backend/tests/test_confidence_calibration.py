import tempfile
import unittest
from pathlib import Path

from backend.confidence_calibration import calibrated_probability, fit_isotonic


class ConfidenceCalibrationTests(unittest.TestCase):
    def test_requires_real_positive_and_negative_routes(self) -> None:
        with self.assertRaises(ValueError):
            fit_isotonic([0.5] * 10, [1] * 10)

    def test_isotonic_fit_is_monotone(self) -> None:
        scores = [index / 29 for index in range(30)]
        labels = [0] * 8 + [1, 0, 1, 0] + [1] * 18
        model = fit_isotonic(scores, labels)
        probabilities = model["probabilities"]
        self.assertTrue(all(a <= b for a, b in zip(probabilities, probabilities[1:])))

    def test_missing_model_never_claims_probability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probability, diagnostics = calibrated_probability(0.8, Path(directory) / "missing.json")
        self.assertIsNone(probability)
        self.assertEqual(diagnostics["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
