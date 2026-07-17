import unittest
from unittest.mock import patch

import numpy as np

from backend.floorplan_constraints import (
    FloorplanConstraintEngine,
    _trajectory_fractions,
    apply_floorplan_constraints,
    get_floorplan_engine,
)


class FloorplanConstraintEngineTests(unittest.TestCase):
    def test_long_route_initial_heading_uses_only_early_anchor(self) -> None:
        points = np.zeros((600, 2), dtype=float)
        points[:49, 0] = np.arange(49)
        points[49:, 0] = 48.0
        points[49:, 1] = np.arange(1, 552)
        self.assertAlmostEqual(
            FloorplanConstraintEngine._initial_heading(points), 0.0, places=6
        )

    def test_segment_collision_sampling_matches_final_validation(self) -> None:
        mask = np.zeros((30, 30), dtype=bool)
        mask[10, 10] = True
        engine = FloorplanConstraintEngine.from_mask(mask, grid_cell_pixels=1)
        segment = np.asarray([[2.0, 2.0], [18.0, 18.0]])
        self.assertEqual(
            engine._segment_collides(segment[0], segment[1]),
            engine._path_metrics(segment)["collision_ratio"] > 0.0,
        )

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

    def test_start_inside_restricted_area_is_projected_to_walkable_mask(self) -> None:
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
        self.assertTrue(result["accepted"], result["diagnostics"])
        self.assertEqual(result["diagnostics"]["corrected_collision_ratio"], 0.0)
        self.assertIn(
            "start_projected_to_walkable_area",
            result["diagnostics"]["quality_warnings"],
        )

    def test_route_leaving_plan_is_constrained_inside_instead_of_rejected(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((80, 100), dtype=bool), meters_per_pixel=0.1
        )
        result = engine.align(
            [[0, 0], [40, 0], [80, 0], [120, 0]],
            {"x": 20, "y": 50},
            {"x": 40, "y": 50},
            scale_candidates=[1.0],
            yaw_offsets_degrees=[0.0],
        )
        self.assertTrue(result["accepted"], result["diagnostics"])
        self.assertEqual(result["diagnostics"]["outside_ratio"], 0.0)
        points = np.asarray(result["trajectory"])
        self.assertTrue(np.all(points[:, 0] >= 0.0))
        self.assertTrue(np.all(points[:, 0] < 100.0))

    def test_implausible_speed_is_warning_not_map_rejection(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((100, 160), dtype=bool), meters_per_pixel=0.1
        )
        result = engine.align(
            [[0, 0], [20, 0], [40, 0]],
            {"x": 10, "y": 50},
            {"x": 30, "y": 50},
            timestamps=[0.0, 0.5, 1.0],
            scale_candidates=[1.0],
            yaw_offsets_degrees=[0.0],
        )
        self.assertTrue(result["accepted"], result["diagnostics"])
        self.assertIn(
            "walking_speed_prior_inconsistent",
            result["diagnostics"]["quality_warnings"],
        )

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

    def test_floorplan_can_select_guarded_r3_lingbot_fusion_candidate(self) -> None:
        source_path = [
            [0, 0, 0], [100, 0, 0], [200, 0, 0],
            [300, 0, 0], [400, 0, 0], [500, 0, 0],
        ]
        fused_path = [point[:] for point in source_path]
        source = {
            "method": "r3_reconstruction_scale_aware",
            "plan_trajectory": source_path,
            "turn_points": [],
            "processing_stats": {},
            "lingbot_fusion_candidate": {
                "accepted": True,
                "plan_trajectory": fused_path,
                "diagnostics": {"accepted": True},
            },
        }
        updated = apply_floorplan_constraints(source, {
            "floorplan_id": "kerama_marazzi_2025",
            "reference_point": {"x": 2600 / 5298 * 100, "y": 1000 / 3743 * 100},
            "direction_point": {"x": 2400 / 5298 * 100, "y": 1000 / 3743 * 100},
        })

        self.assertTrue(updated["processing_stats"]["map_matching_applied"])
        self.assertEqual(
            updated["processing_stats"]["map_observation_source"],
            "r3_lingbot_fusion",
        )
        self.assertEqual(
            updated["floorplan_constraint"]["observation_source_selection"]["reason"],
            "fusion_supported_by_floorplan",
        )

    def test_fragmented_r3_selects_independent_lingbot_even_after_fusion_veto(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((120, 180), dtype=bool), meters_per_pixel=0.1
        )
        independent = [[float(x), 0.0, 0.0] for x in range(0, 61, 10)]
        source = {
            "method": "r3_reconstruction_scale_aware",
            "plan_trajectory": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "source_timestamps_seconds": [0.0, 60.0],
            "turn_points": [{"trajectory_index": 1, "angle_degrees": 90.0}],
            "processing_stats": {
                "pose_graph": {
                    "component_count": 25,
                    "largest_component_coverage": 0.014,
                }
            },
            "lingbot_fusion_candidate": {
                "accepted": False,
                "independent_accepted": True,
                "independent_plan_trajectory": independent,
                "lingbot_source_timestamps_seconds": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "diagnostics": {
                    "reason": "trajectory_disagreement_too_large",
                    "independent_quality": {"accepted": True, "reasons": []},
                },
            },
        }
        with patch(
            "backend.floorplan_constraints.get_floorplan_engine", return_value=engine
        ):
            updated = apply_floorplan_constraints(source, {
                "floorplan_id": "test",
                "reference_point": {"x": 10, "y": 50},
                "direction_point": {"x": 30, "y": 50},
            })
        self.assertTrue(updated["processing_stats"]["map_matching_applied"])
        self.assertEqual(
            updated["processing_stats"]["map_observation_source"],
            "lingbot_independent",
        )
        self.assertEqual(
            updated["floorplan_constraint"]["observation_source_selection"]["reason"],
            "fragmented_r3_uses_independent_lingbot",
        )
        self.assertEqual(updated["map_turn_points"], [])
        self.assertEqual(updated["final_turn_points"], [])

    def test_fragmented_r3_refuses_low_quality_independent(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((120, 180), dtype=bool), meters_per_pixel=0.1
        )
        source = {
            "method": "r3_reconstruction_scale_aware",
            "plan_trajectory": [
                [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0],
                [30.0, 0.0, 0.0], [40.0, 0.0, 0.0], [50.0, 0.0, 0.0],
            ],
            "processing_stats": {
                "pose_graph": {
                    "component_count": 12,
                    "largest_component_coverage": 0.2,
                }
            },
            "lingbot_fusion_candidate": {
                "accepted": False,
                "independent_accepted": True,
                "independent_plan_trajectory": [[0.0, 0.0, 0.0]] * 8,
                "diagnostics": {
                    "reason": "turn_chirality_conflict",
                    "independent_quality": {
                        "accepted": False,
                        "reasons": ["turn_chirality_conflict"],
                    },
                },
            },
        }
        with patch(
            "backend.floorplan_constraints.get_floorplan_engine", return_value=engine
        ):
            updated = apply_floorplan_constraints(source, {
                "floorplan_id": "test",
                "reference_point": {"x": 10, "y": 50},
                "direction_point": {"x": 30, "y": 50},
            })
        self.assertNotEqual(
            updated["processing_stats"].get("map_observation_source"),
            "lingbot_independent",
        )
        self.assertEqual(
            updated["floorplan_constraint"]["observation_source_selection"].get(
                "independent_rejected_reason"
            ),
            "independent_quality_failed",
        )

    def test_scale_prior_uses_walkable_extent_not_annotation_bbox(self) -> None:
        mask = np.zeros((100, 200), dtype=bool)
        # Tiny annotation island vs large walkable free space.
        mask[10:12, 10:12] = True
        engine = FloorplanConstraintEngine.from_mask(
            mask, meters_per_pixel=0.1, grid_cell_pixels=1
        )
        relative = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        scales = engine._scale_candidates(relative, duration=None)
        walk_width = engine.walkable_bbox[2] - engine.walkable_bbox[0]
        ann_width = engine.annotation_bbox[2] - engine.annotation_bbox[0]
        self.assertGreater(walk_width, ann_width * 5)
        expected_base = max(walk_width, engine.walkable_bbox[3] - engine.walkable_bbox[1]) * 0.72 / 20.0
        self.assertTrue(any(abs(scale - expected_base) < 1e-6 for scale in scales))

    def test_diverse_beam_includes_multiple_yaw_bins(self) -> None:
        hypotheses = [
            {"score": float(index), "scale": 1.0 + index * 0.01, "yaw": yaw}
            for index, yaw in enumerate([-10.0, -10.0, -5.0, 0.0, 0.0, 5.0, 10.0, 10.0])
        ]
        beam = FloorplanConstraintEngine._select_diverse_beam(
            sorted(hypotheses, key=lambda item: item["score"]),
            per_yaw=1,
            global_top=3,
        )
        yaws = {item["yaw"] for item in beam}
        self.assertGreaterEqual(len(yaws), 4)

    def test_malformed_points_are_dropped_not_zeroed(self) -> None:
        from backend.floorplan_constraints import _normalise_points
        points = _normalise_points([[1.0, 2.0], [float("nan"), 3.0], {"x": 4.0, "y": 5.0}, "bad"])
        self.assertEqual(len(points), 2)
        self.assertTrue(np.allclose(points[0], [1.0, 2.0]))
        self.assertTrue(np.allclose(points[1], [4.0, 5.0]))

    def test_partial_grid_padding_outside_pdf_is_occupied(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((5, 5), dtype=bool), grid_cell_pixels=4, meters_per_pixel=1.0
        )
        self.assertTrue(engine._point_occupied([6.0, 2.0]))
        self.assertGreater(engine._path_metrics(np.asarray([[2.0, 2.0], [6.0, 2.0]]))["outside_ratio"], 0.0)

    def test_adaptive_anchors_retain_corner(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(np.zeros((100, 100), dtype=bool))
        points = np.asarray([[float(x), 20.0] for x in range(10, 60)] + [[59.0, float(y)] for y in range(21, 80)])
        fractions = engine._adaptive_anchor_fractions(points, maximum=12)
        corner_fraction = float(_trajectory_fractions(points)[49])
        self.assertLess(float(np.min(np.abs(fractions - corner_fraction))), 0.02)

    def test_multilevel_viterbi_uses_doorway(self) -> None:
        mask = np.zeros((120, 180), dtype=bool)
        mask[:, 88:94] = True
        mask[16:34, 88:94] = False
        engine = FloorplanConstraintEngine.from_mask(mask, meters_per_pixel=0.1)
        observed = np.asarray([[20.0, 60.0], [50.0, 60.0], [80.0, 60.0], [110.0, 60.0], [140.0, 60.0], [160.0, 60.0]])
        baseline, _ = engine._repair_collisions(observed)
        matched, diagnostics = engine._multilevel_viterbi_map_match(observed, baseline)
        self.assertIsNotNone(matched, diagnostics)
        self.assertEqual(engine._collision_runs(matched), [])
        self.assertGreater(diagnostics["corridor_graph_nodes"], 0)

    def test_experimental_hmm_is_disabled_in_production_by_default(self) -> None:
        mask = np.zeros((120, 180), dtype=bool)
        mask[42:78, 78:102] = True
        engine = FloorplanConstraintEngine.from_mask(mask, meters_per_pixel=0.1)
        result = engine.align(
            [[0, 0], [20, 0], [40, 0], [60, 0], [80, 0]],
            {"x": 10, "y": 50}, {"x": 30, "y": 50},
            scale_candidates=[2.0], yaw_offsets_degrees=[0.0],
        )
        nonlinear = result["diagnostics"]["nonlinear_map_matching"]
        self.assertFalse(nonlinear["attempted"])
        self.assertFalse(nonlinear["production_enabled"])
        self.assertEqual(nonlinear["reason"], "disabled_pending_production_validation")


if __name__ == "__main__":
    unittest.main()
