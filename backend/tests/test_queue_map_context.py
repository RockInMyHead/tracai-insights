import unittest
from unittest.mock import patch

from backend.main import (
    QUEUE_NORMAL_VIDEO_MIN_DURATION_SEC,
    _build_session_input_queue_context,
    _enqueue_processing_queue_item,
    _get_analysis_video_duration_sec,
    _map_context_has_anchor,
    _resolve_queue_map_context,
)


class QueueMapContextTests(unittest.TestCase):
    def test_map_context_has_anchor(self):
        self.assertTrue(_map_context_has_anchor({
            "reference_point": {"x": 41.9, "y": 18.1},
            "direction_point": {"x": 37.7, "y": 18.7},
        }))
        self.assertFalse(_map_context_has_anchor({"queue_inherit_anchor": True}))

    def test_duration_prefers_video_info(self):
        duration = _get_analysis_video_duration_sec(
            "vid-short",
            {
                "video_info": {"duration": 15.0, "fps": 30.0, "frame_count": 450},
                "r3_source_timestamps_seconds": [0.0, 14.5],
            },
        )
        self.assertEqual(duration, 15.0)

    def test_short_prior_keeps_session_input(self):
        session_anchor = {
            "reference_point": {"x": 41.9, "y": 18.1},
            "direction_point": {"x": 37.7, "y": 18.7},
        }
        queue_items = [
            {
                "video_id": "vid-1",
                "sequence_index": 0,
                "map_context": {
                    "floorplan_id": "kerama_marazzi_2025",
                    **session_anchor,
                },
            },
            {
                "video_id": "vid-2",
                "sequence_index": 1,
                "map_context": {"floorplan_id": "kerama_marazzi_2025", "queue_inherit_anchor": True},
            },
        ]
        short_result = {
            "video_info": {"duration": 15.0},
            "map_trajectory": [[100.0, 100.0], [101.0, 101.0]],
        }

        with patch("backend.main._load_processing_queues", return_value={
            "desktop-batch": {
                "queue_id": "desktop-batch",
                "session_anchor": session_anchor,
                "items": queue_items,
            },
        }), patch("backend.main._save_processing_queues"), patch(
            "backend.main._load_analysis_result",
            return_value=short_result,
        ):
            resolved = _resolve_queue_map_context(
                "desktop-batch",
                "vid-2",
                {"floorplan_id": "kerama_marazzi_2025", "queue_inherit_anchor": True},
                1,
            )

        self.assertEqual(resolved["queue_anchor_source"], "session_input_after_short_segments")
        self.assertEqual(resolved["reference_point"], session_anchor["reference_point"])
        self.assertEqual(resolved["direction_point"], session_anchor["direction_point"])

    def test_long_prior_uses_tail(self):
        session_anchor = {
            "reference_point": {"x": 41.9, "y": 18.1},
            "direction_point": {"x": 37.7, "y": 18.7},
        }
        queue_items = [
            {
                "video_id": "vid-1",
                "sequence_index": 0,
                "map_context": {
                    "floorplan_id": "kerama_marazzi_2025",
                    **session_anchor,
                },
            },
            {
                "video_id": "vid-2",
                "sequence_index": 1,
                "map_context": {"floorplan_id": "kerama_marazzi_2025", "queue_inherit_anchor": True},
            },
        ]
        long_result = {
            "video_info": {"duration": 180.0},
            "map_trajectory": [[800.0, 600.0], [820.0, 610.0], [840.0, 620.0]],
            "map_metadata": {"plan_width": 800.0, "plan_height": 600.0},
        }

        with patch("backend.main._load_processing_queues", return_value={
            "desktop-batch": {
                "queue_id": "desktop-batch",
                "session_anchor": session_anchor,
                "items": queue_items,
            },
        }), patch("backend.main._save_processing_queues"), patch(
            "backend.main._load_analysis_result",
            return_value=long_result,
        ):
            resolved = _resolve_queue_map_context(
                "desktop-batch",
                "vid-2",
                {"floorplan_id": "kerama_marazzi_2025", "queue_inherit_anchor": True},
                1,
            )

        self.assertEqual(resolved["queue_anchor_source"], "previous_long_video_tail")
        self.assertEqual(resolved["queue_anchor_video_id"], "vid-1")
        self.assertGreaterEqual(
            resolved["queue_anchor_video_duration_sec"],
            QUEUE_NORMAL_VIDEO_MIN_DURATION_SEC,
        )
        self.assertNotEqual(resolved["reference_point"], session_anchor["reference_point"])

    def test_build_session_input_queue_context(self):
        anchor = {
            "reference_point": {"x": 10.0, "y": 20.0},
            "direction_point": {"x": 12.0, "y": 22.0},
        }
        resolved = _build_session_input_queue_context(
            {"floorplan_id": "kerama_marazzi_2025", "queue_inherit_anchor": True},
            anchor,
        )
        self.assertEqual(resolved["reference_point"], anchor["reference_point"])
        self.assertNotIn("queue_inherit_anchor", resolved)

    def test_enqueue_does_not_deadlock_on_nested_queue_lock(self):
        queue_id = "desktop-batch-deadlock"
        with patch("backend.main._load_processing_queues", return_value={}), patch(
            "backend.main._save_processing_queues",
        ) as save_mock, patch("backend.main._any_gpu_pipeline_active", return_value=False):
            item, start_now = _enqueue_processing_queue_item(
                queue_id,
                "vid-first",
                original_filename="VID00000.AVI",
                analysis_method="r3",
                sequence_index=0,
                run_params={"frame_stride": 5},
                map_context={
                    "floorplan_id": "kerama_marazzi_2025",
                    "reference_point": {"x": 41.9, "y": 18.1},
                    "direction_point": {"x": 37.7, "y": 18.7},
                },
            )
        self.assertTrue(start_now)
        self.assertEqual(item.get("status"), "running")
        save_mock.assert_called()


if __name__ == "__main__":
    unittest.main()
