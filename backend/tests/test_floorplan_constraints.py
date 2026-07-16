import unittest

import numpy as np

from backend.floorplan_constraints import (
    FloorplanConstraintEngine,
    apply_floorplan_constraints,
    get_floorplan_engine,
)


class FloorplanConstraintEngineTests(unittest.TestCase):
    def test_r3_left_turn_keeps_physical_chirality_in_svg_coordinates(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((140, 180), dtype=bool), meters_per_pixel=0.1
        )
        trajectory = [[0, 0, 0], [10, 0, 0], [20, 0, 0], [20, 10, 0], [20, 20, 0]]
        result = engine.align(
            trajectory,
            {"x": 10, "y": 70},
            {"x": 30, "y": 70},
            scale_candidates=[2.0],
            yaw_offsets_degrees=[0.0],
        )
        self.assertTrue(result["accepted"], result["diagnostics"])
        points = np.asarray(result["trajectory"])
        self.assertAlmostEqual(points[0, 1], 98.0, delta=1.0)
        self.assertLess(points[-1, 1], points[2, 1])

    def test_collision_segment_is_rerouted_without_moving_safe_prefix(self) -> None:
        mask = np.zeros((120, 180), dtype=bool)
        mask[42:78, 78:102] = True
        engine = FloorplanConstraintEngine.from_mask(
            mask, meters_per_pixel=0.1, person_radius_meters=0.0
        )
        trajectory = [[0, 0, 0], [20, 0, 0], [40, 0, 0], [60, 0, 0], [80, 0, 0]]
        result = engine.align(
            trajectory,
            {"x": 10, "y": 50},
            {"x": 30, "y": 50},
            scale_candidates=[2.0],
            yaw_offsets_degrees=[0.0],
        )
        self.assertTrue(result["accepted"], result["diagnostics"])
        points = np.asarray(result["trajectory"])
        self.assertGreater(result["diagnostics"]["rerouted_segments"], 0)
        self.assertEqual(result["diagnostics"]["corrected_collision_ratio"], 0.0)
        self.assertAlmostEqual(points[0, 0], 18.0, delta=1.0)
        self.assertAlmostEqual(points[0, 1], 60.0, delta=1.0)
        self.assertTrue(np.any(np.abs(points[:, 1] - 60.0) > 15.0))

    def test_start_inside_restricted_area_is_rejected(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True
        engine = FloorplanConstraintEngine.from_mask(mask, meters_per_pixel=0.1)
        result = engine.align(
            [[0, 0], [10, 0]],
            {"x": 50, "y": 50},
            {"x": 70, "y": 50},
            scale_candidates=[1.0],
            yaw_offsets_degrees=[0.0],
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["diagnostics"]["reason"], "start_in_restricted_area")

    def test_stationary_time_is_removed_from_scale_prior(self) -> None:
        points = np.asarray([
            [0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01],
            [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [3.01, 0.0],
        ])
        duration = FloorplanConstraintEngine._motion_duration_seconds(
            list(range(len(points))), points
        )
        self.assertIsNotNone(duration)
        self.assertLess(duration, 7.0)
        self.assertGreaterEqual(duration, 3.0)
        partial = FloorplanConstraintEngine._motion_duration_seconds(
            [None, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, None], points
        )
        self.assertIsNotNone(partial)

    def test_fixed_floorplan_uses_office_area_calibration(self) -> None:
        engine = get_floorplan_engine()
        self.assertEqual(engine.config.map_id, "kerama_marazzi_2025")
        self.assertEqual((engine.config.width, engine.config.height), (5298, 3743))
        self.assertAlmostEqual(engine.config.meters_per_pixel, 0.0496291667, places=8)
        self.assertGreater(int(engine.occupied.sum()), 1000)

    def test_fixed_floorplan_routes_around_real_annotated_machine(self) -> None:
        engine = get_floorplan_engine()
        result = engine.align(
            [[0, 0], [150, 0], [300, 0], [450, 0], [600, 0]],
            {"x": 1200 / 5298 * 100, "y": 850 / 3743 * 100},
            {"x": 1100 / 5298 * 100, "y": 850 / 3743 * 100},
            scale_candidates=[1.0],
            yaw_offsets_degrees=[0.0],
        )
        self.assertTrue(result["accepted"], result["diagnostics"])
        self.assertEqual(result["diagnostics"]["corrected_collision_ratio"], 0.0)
        self.assertGreater(result["diagnostics"]["rerouted_segments"], 0)
        self.assertGreater(len(result["trajectory"]), 5)

    def test_wrapper_preserves_visual_trajectory_when_map_context_is_incomplete(self) -> None:
        source = {
            "method": "r3_reconstruction",
            "plan_trajectory": [[0, 0, 0], [1, 0, 0]],
            "turn_points": [],
            "processing_stats": {},
        }
        updated = apply_floorplan_constraints(source, {"floorplan_id": "missing"})
        self.assertEqual(updated["plan_trajectory"], source["plan_trajectory"])
        self.assertNotIn("map_trajectory", updated)
        self.assertFalse(updated["processing_stats"]["map_matching_applied"])

    def test_wrapper_attaches_metric_map_result_without_overwriting_r3(self) -> None:
        source = {
            "success": True,
            "method": "r3_reconstruction_scale_aware",
            "plan_trajectory": [
                [0, 0, 0], [400, 0, 0], [800, 0, 0],
                [1200, 0, 0], [1600, 0, 0], [2000, 0, 0],
            ],
            "turn_points": [{
                "frame_index": 3,
                "trajectory_index": 3,
                "angle_degrees": 90.0,
                "position": [1200, 0, 0],
                "turn_type": "left",
            }],
            "source_timestamps_seconds": [0, 16, 32, 48, 64, 80],
            "trajectory_quality": {
                "projection": {"plan_coordinate_convention": "x_forward_y_left_z_up"}
            },
            "processing_stats": {},
        }
        original = [point[:] for point in source["plan_trajectory"]]
        updated = apply_floorplan_constraints(source, {
            "floorplan_id": "kerama_marazzi_2025",
            "reference_point": {"x": 2600 / 5298 * 100, "y": 1000 / 3743 * 100},
            "direction_point": {"x": 2400 / 5298 * 100, "y": 1000 / 3743 * 100},
        })
        self.assertTrue(updated["processing_stats"]["map_matching_applied"])
        self.assertGreater(len(updated["map_trajectory"]), 1)
        self.assertEqual(updated["plan_trajectory"], original)
        self.assertEqual(updated["map_turn_points"][0]["angle_degrees"], 90.0)
        self.assertEqual(updated["map_turn_points"][0]["turn_type"], "left")
        self.assertTrue(updated["map_turn_points"][0]["map_constrained"])
        self.assertAlmostEqual(updated["processing_stats"]["estimated_distance"], 96.0, delta=2.0)
        self.assertEqual(updated["map_metadata"]["map_id"], "kerama_marazzi_2025")


if __name__ == "__main__":
    unittest.main()
