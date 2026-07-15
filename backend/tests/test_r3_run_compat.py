"""Regression tests for R3 replay cache compatibility."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3_run_compat import sampling_contract_matches


class R3RunCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {"wrapper_input_size": 392}
        self.selection = {
            "requested_frame_stride": 3,
            "requested_max_frames": 3000,
            "long_video_sampling": True,
            "long_target_fps": 8.0,
        }

    def matches(self, **overrides: object) -> bool:
        requested = {
            "frame_stride": 3,
            "max_frames": 3000,
            "size": 392,
            "long_target_fps": 8.0,
            **overrides,
        }
        return sampling_contract_matches(self.params, self.selection, **requested)

    def test_exact_sampling_contract_allows_replay(self) -> None:
        self.assertTrue(self.matches())

    def test_accuracy_inputs_invalidate_old_reconstruction(self) -> None:
        self.assertFalse(self.matches(frame_stride=5))
        self.assertFalse(self.matches(max_frames=1500))
        self.assertFalse(self.matches(size=518))
        self.assertFalse(self.matches(long_target_fps=5.0))

    def test_missing_legacy_fields_force_fresh_inference(self) -> None:
        self.assertFalse(sampling_contract_matches(
            {},
            {},
            frame_stride=3,
            max_frames=3000,
            size=392,
            long_target_fps=8.0,
        ))


if __name__ == "__main__":
    unittest.main()
