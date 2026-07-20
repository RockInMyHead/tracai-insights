import unittest
from unittest.mock import patch

import numpy as np

from backend.floorplan_constraints import (
    FloorplanConfig,
    FloorplanConstraintEngine,
    _densify_polyline,
    _polyline_progress_metrics,
    _polyline_sharp_reverse_ratio,
    _stabilize_independent_observation,
    _trajectory_fractions,
    apply_floorplan_constraints,
    get_floorplan_engine,
)
from backend.kerama_reference_route import load_reference_route


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

    def test_physically_impossible_authoritative_speed_is_rejected(self) -> None:
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
        self.assertFalse(result["accepted"], result["diagnostics"])
        self.assertEqual(result["diagnostics"]["reason"], "metric_prior_inconsistent")

    def test_compressed_independent_scale_is_hard_rejected(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((300, 1000), dtype=bool),
            meters_per_pixel=0.1,
            walking_speed_mps=1.2,
        )
        trajectory = [[float(x), 0.0] for x in range(0, 101, 10)]
        timestamps = [float(x) for x in range(0, 101, 10)]
        # Production regression: a collision-free 0.79 m/s hypothesis must
        # not beat the 1.2 m/s metric prior merely because it fits a narrow
        # office strip.
        result = engine.align(
            trajectory,
            {"x": 10, "y": 50},
            {"x": 20, "y": 50},
            timestamps=timestamps,
            scale_candidates=[7.9],
            yaw_offsets_degrees=[0.0],
            observation_policy="independent",
        )
        self.assertFalse(result["accepted"], result["diagnostics"])
        self.assertEqual(result["diagnostics"]["reason"], "metric_prior_inconsistent")
        self.assertIn(
            "implausible_metric_scale", result["diagnostics"]["rejection_reasons"]
        )

    def test_metric_scale_beats_collision_free_compressed_scale(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((300, 1600), dtype=bool),
            meters_per_pixel=0.1,
            walking_speed_mps=1.2,
        )
        trajectory = [[float(x), 0.0] for x in range(0, 101, 10)]
        result = engine.align(
            trajectory,
            {"x": 10, "y": 50},
            {"x": 20, "y": 50},
            timestamps=[float(x) for x in range(0, 101, 10)],
            scale_candidates=[7.9, 12.0],
            yaw_offsets_degrees=[0.0],
            observation_policy="independent",
        )
        self.assertTrue(result["accepted"], result["diagnostics"])
        self.assertAlmostEqual(
            result["diagnostics"]["selected_scale_pixels_per_unit"], 12.0
        )
        self.assertAlmostEqual(result["diagnostics"]["estimated_speed_mps"], 1.2)
        self.assertLessEqual(result["diagnostics"]["confidence"], 0.55)

    def test_independent_metric_prior_requires_monotonic_time(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((100, 200), dtype=bool), meters_per_pixel=0.1
        )
        result = engine.align(
            [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
            {"x": 10, "y": 50},
            {"x": 30, "y": 50},
            timestamps=[0.0, 2.0, 1.0, 3.0],
            scale_candidates=[1.2],
            yaw_offsets_degrees=[0.0],
            observation_policy="independent",
        )
        self.assertFalse(result["accepted"], result["diagnostics"])
        self.assertEqual(result["diagnostics"]["reason"], "metric_prior_unavailable")

    def test_start_is_not_silently_snapped_across_large_obstacle(self) -> None:
        mask = np.zeros((200, 200), dtype=bool)
        mask[40:160, 40:160] = True
        engine = FloorplanConstraintEngine.from_mask(mask, meters_per_pixel=0.1)
        result = engine.align(
            [[0.0, 0.0], [10.0, 0.0]],
            {"x": 50, "y": 50},
            {"x": 70, "y": 50},
            scale_candidates=[1.0],
            yaw_offsets_degrees=[0.0],
        )
        self.assertFalse(result["accepted"], result["diagnostics"])
        self.assertEqual(
            result["diagnostics"]["reason"], "start_too_far_from_walkable_area"
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

    def test_fixed_floorplan_excludes_blank_space_above_north_roof(self) -> None:
        engine = get_floorplan_engine()
        self.assertTrue(engine._support_mask[705, 2240])
        # Regression for the production route that escaped above the long
        # green roof line and then travelled left through blank PDF canvas.
        self.assertFalse(engine._support_mask[492, 1800])
        start_cell = engine._pixel_to_cell([2240.0, 705.0])
        snapped_start = engine._nearest_free(start_cell)
        outside_component = engine._component_ids[
            engine._pixel_to_cell([1800.0, 492.0])[1],
            engine._pixel_to_cell([1800.0, 492.0])[0],
        ]
        self.assertIsNotNone(snapped_start)
        self.assertLess(
            np.linalg.norm(np.asarray(snapped_start) - np.asarray(start_cell))
            * engine.cell_meters,
            1.0,
        )
        self.assertEqual(int(outside_component), 0)

    def test_astar_spike_detour_is_rejected_as_topology_break(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((80, 120), dtype=bool), meters_per_pixel=0.1
        )
        start = np.asarray([10.0, 40.0])
        end = np.asarray([40.0, 40.0])
        raw = np.asarray([[10.0, 40.0], [25.0, 40.0], [40.0, 40.0]])
        # 3 m chord, ~35 m invented loop — classic mask-legal spike.
        spike = np.asarray([
            [10.0, 40.0],
            [10.0, 10.0],
            [70.0, 10.0],
            [70.0, 70.0],
            [40.0, 70.0],
            [40.0, 40.0],
        ])
        self.assertTrue(engine._detour_is_spike(spike, start, end, raw))
        local = np.asarray([[10.0, 40.0], [25.0, 48.0], [40.0, 40.0]])
        self.assertFalse(engine._detour_is_spike(local, start, end, raw))

    def test_sharp_reverse_ratio_flags_triangular_spike(self) -> None:
        # Straight walk with one large triangular detour (classic bad A* spike).
        points = np.asarray([
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [22.0, 18.0],
            [24.0, 0.0],
            [40.0, 0.0],
            [50.0, 0.0],
        ], dtype=float)
        ratio = _polyline_sharp_reverse_ratio(points, meters_per_pixel=1.0)
        self.assertGreater(ratio, 0.08)
        straight = np.asarray([[float(x), 0.0] for x in range(0, 51, 5)], dtype=float)
        self.assertLess(
            _polyline_sharp_reverse_ratio(straight, meters_per_pixel=1.0),
            0.01,
        )

    def test_positive_support_rejects_topology_destroying_repair(self) -> None:
        height, width = 100, 220
        support = np.zeros((height, width), dtype=bool)
        support[44:57, 5:215] = True
        engine = FloorplanConstraintEngine(
            FloorplanConfig(
                map_id="supported_test",
                width=width,
                height=height,
                meters_per_pixel=1.0,
                grid_cell_pixels=1,
                person_radius_meters=0.0,
                obstacle_mask_file="",
            ),
            np.zeros_like(support),
            support,
        )
        trajectory = []
        trajectory.extend([[float(x), 0.0] for x in range(20)])
        trajectory.extend([[19.0, float(y)] for y in range(1, 31)])
        trajectory.extend([[float(x), 30.0] for x in range(20, 101)])
        result = engine.align(
            trajectory,
            {"x": 10.0, "y": 50.0},
            {"x": 30.0, "y": 50.0},
            scale_candidates=[1.0],
            yaw_offsets_degrees=[0.0],
        )
        self.assertFalse(result["accepted"], result["diagnostics"])
        self.assertIn(
            result["diagnostics"]["reason"],
            {
                "map_correction_exceeds_observation_budget",
                "constraint_solution_not_found",
            },
        )

    def test_authoritative_safe_fallback_returns_bounded_collision_free_route(self) -> None:
        height, width = 100, 220
        support = np.zeros((height, width), dtype=bool)
        support[44:57, 5:215] = True
        engine = FloorplanConstraintEngine(
            FloorplanConfig(
                map_id="supported_fallback_test",
                width=width,
                height=height,
                meters_per_pixel=1.0,
                grid_cell_pixels=1,
                person_radius_meters=0.0,
                obstacle_mask_file="",
            ),
            np.zeros_like(support),
            support,
        )
        trajectory = []
        trajectory.extend([[float(x), 0.0] for x in range(20)])
        trajectory.extend([[19.0, float(y)] for y in range(1, 16)])
        trajectory.extend([[float(x), 15.0] for x in range(20, 101)])
        result = engine.align(
            trajectory,
            {"x": 10.0, "y": 50.0},
            {"x": 30.0, "y": 50.0},
            scale_candidates=[1.0],
            yaw_offsets_degrees=[0.0],
            allow_safe_shape_fallback=True,
        )
        self.assertTrue(result["accepted"], result["diagnostics"])
        self.assertTrue(result["diagnostics"]["shape_fallback_used"])
        self.assertEqual(
            result["diagnostics"]["shape_fallback_policy"],
            "authoritative_plan_connectivity_v2",
        )
        self.assertEqual(result["diagnostics"]["corrected_collision_ratio"], 0.0)
        self.assertIn(
            "authoritative_safe_map_fallback",
            result["diagnostics"]["quality_warnings"],
        )

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
        self.assertGreaterEqual(len(result["trajectory"]), 5)

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
        self.assertEqual(
            updated["floorplan_constraint"]["constraint_revision"],
            "kerama_reference_topology_v7",
        )
        self.assertEqual(
            len(updated["map_trajectory_timestamps_seconds"]),
            len(updated["map_trajectory"]),
        )
        mapped = np.asarray(updated["map_trajectory"], dtype=float)
        max_step_meters = float(
            np.max(np.linalg.norm(np.diff(mapped[:, :2], axis=0), axis=1))
        ) * updated["map_metadata"]["meters_per_pixel"]
        self.assertLessEqual(max_step_meters, 0.751)

    def test_reference_mask_blocks_false_north_corridor(self) -> None:
        engine = get_floorplan_engine()

        self.assertTrue(engine._point_occupied([1700, 575]))
        self.assertFalse(engine._point_occupied([1700, 705]))
        self.assertFalse(engine._point_occupied([1700, 850]))
        self.assertFalse(engine._point_occupied([2200, 850]))

    def test_operator_reference_route_is_one_collision_free_component(self) -> None:
        engine = get_floorplan_engine()
        payload = load_reference_route()
        route = np.asarray(payload["points"], dtype=np.float64)
        metrics = engine._path_metrics(route)

        self.assertEqual(metrics["collision_ratio"], 0.0)
        self.assertEqual(metrics["outside_ratio"], 0.0)
        self.assertEqual(engine._collision_runs(route), [])
        self.assertEqual(engine._path_component_count(route), 1)
        length_meters = (
            np.linalg.norm(np.diff(route, axis=0), axis=1).sum()
            * engine.config.meters_per_pixel
        )
        self.assertAlmostEqual(
            length_meters,
            float(payload["expected_length_meters"]),
            delta=float(payload["length_tolerance_meters"]),
        )

    def test_operator_reference_route_aligns_end_to_end_at_metric_scale(self) -> None:
        engine = get_floorplan_engine()
        payload = load_reference_route()
        route = np.asarray(payload["points"], dtype=np.float64)
        dense = _densify_polyline(route, 2.0)
        fractions = _trajectory_fractions(dense)
        duration = float(payload["expected_length_meters"]) / 1.5
        reference = payload["reference_point"]
        direction = payload["direction_point"]

        result = engine.align(
            dense.tolist(),
            {
                "x": reference[0] / engine.config.width * 100.0,
                "y": reference[1] / engine.config.height * 100.0,
            },
            {
                "x": direction[0] / engine.config.width * 100.0,
                "y": direction[1] / engine.config.height * 100.0,
            },
            timestamps=(fractions * duration).tolist(),
            coordinate_convention="x_right_y_down",
            scale_candidates=[1.0],
            yaw_offsets_degrees=[0.0],
        )

        self.assertTrue(result["accepted"], result["diagnostics"])
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["corrected_collision_ratio"], 0.0)
        self.assertAlmostEqual(
            diagnostics["published_length_meters"],
            float(payload["expected_length_meters"]),
            delta=float(payload["length_tolerance_meters"]),
        )
        endpoint = np.asarray(result["trajectory"][-1][:2], dtype=np.float64)
        endpoint_error_meters = (
            float(np.linalg.norm(endpoint - np.asarray(payload["expected_end_point"])))
            * engine.config.meters_per_pixel
        )
        self.assertLessEqual(
            endpoint_error_meters, float(payload["endpoint_tolerance_meters"])
        )
        self.assertTrue(diagnostics["nonlinear_map_matching"]["attempted"])

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
                "independent_accepted": True,
                "independent_plan_trajectory": [
                    [0, 0, 0], [120, 20, 0], [250, -20, 0], [500, 0, 0],
                ],
                "diagnostics": {
                    "accepted": True,
                    "independent_quality": {"accepted": True, "reasons": []},
                },
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
            "authoritative_candidate_accepted",
        )
        selection = updated["floorplan_constraint"]["observation_source_selection"]
        independent = next(
            item for item in selection["candidate_results"]
            if item["source"] == "lingbot_independent"
        )
        self.assertTrue(independent["skipped"])
        self.assertEqual(independent["reason"], "authoritative_candidate_accepted")

    def test_fragmented_r3_selects_independent_lingbot_even_after_fusion_veto(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((120, 300), dtype=bool), meters_per_pixel=0.1
        )
        independent = [[float(x), 0.0, 0.0] for x in range(0, 61, 10)]
        source = {
            "method": "r3_reconstruction_scale_aware",
            "plan_trajectory": [[0.0, 0.0, 0.0]],
            "source_timestamps_seconds": [0.0],
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
                "lingbot_source_timestamps_seconds": [
                    0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0,
                ],
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
            "independent_fallback_after_authoritative_rejection",
        )
        self.assertEqual(updated["map_turn_points"], [])
        self.assertEqual(updated["final_turn_points"], [])

    def test_fusion_support_cannot_weaken_independent_shape_gate(self) -> None:
        class StubConfig:
            meters_per_pixel = 0.1
            width = 100
            height = 100
            person_radius_meters = 0.0

        class StubEngine:
            config = StubConfig()

            def align(self, trajectory, *args, **kwargs):
                points = np.asarray(trajectory, dtype=float)
                is_independent = len(points) >= 21
                if is_independent and kwargs.get("allow_safe_shape_fallback"):
                    return {
                        "accepted": True,
                        "trajectory": [[10.0, 50.0], [20.0, 50.0]],
                        "diagnostics": {
                            "accepted": True,
                            "reason": None,
                            "constrained_score": 1.0,
                            "correction_p95_meters": 6.6,
                            "corrected_collision_ratio": 0.0,
                            "length_ratio": 1.0,
                            "estimated_length_meters": 1.0,
                            "plan_width": 100,
                            "plan_height": 100,
                            "meters_per_pixel": 0.1,
                            "person_radius_meters": 0.0,
                            "confidence": 0.5,
                        },
                    }
                return {
                    "accepted": False,
                    "trajectory": [],
                    "diagnostics": {
                        "accepted": False,
                        "reason": "constraint_solution_not_found",
                        "rejection_reasons": ["different_walkable_components"],
                    },
                }

        independent = [
            [float(index), float(index % 3), 0.0]
            for index in range(25)
        ]
        source = {
            "method": "r3_reconstruction_scale_aware",
            "plan_trajectory": [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            "turn_points": [],
            "processing_stats": {},
            "lingbot_fusion_candidate": {
                "accepted": True,
                "plan_trajectory": [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
                "independent_accepted": True,
                "independent_plan_trajectory": independent,
                "diagnostics": {
                    "independent_quality": {"accepted": True, "reasons": []},
                },
            },
        }
        with patch(
            "backend.floorplan_constraints.get_floorplan_engine",
            return_value=StubEngine(),
        ):
            updated = apply_floorplan_constraints(source, {
                "floorplan_id": "test",
                "reference_point": {"x": 10, "y": 50},
                "direction_point": {"x": 30, "y": 50},
            })
        self.assertFalse(updated["processing_stats"]["map_matching_applied"])
        self.assertNotIn("map_observation_source", updated["processing_stats"])
        selection = updated["floorplan_constraint"]["observation_source_selection"]
        independent_result = next(
            item for item in selection["candidate_results"]
            if item["source"] == "lingbot_independent"
        )
        self.assertTrue(independent_result["fusion_supported"])
        self.assertFalse(independent_result["accepted"])

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
        selection = updated["floorplan_constraint"]["observation_source_selection"]
        self.assertFalse(any(
            item["source"] == "lingbot_independent"
            for item in selection["candidate_results"]
        ))

    def test_fragmentation_is_soft_prior_and_cannot_veto_valid_r3(self) -> None:
        class StubConfig:
            meters_per_pixel = 0.1
            plan_width = 100
            plan_height = 100
            person_radius_meters = 0.0

        class StubEngine:
            config = StubConfig()

            def align(self, trajectory, *args, **kwargs):
                points = np.asarray(trajectory, dtype=float)
                is_primary = float(np.ptp(points[:, 0])) < 5.0
                if is_primary:
                    return {
                        "accepted": True,
                        "trajectory": [[10.0, 50.0], [20.0, 50.0]],
                        "diagnostics": {
                            "accepted": True,
                            "reason": None,
                            "constrained_score": 1.0,
                            "correction_p95_meters": 0.1,
                            "length_ratio": 1.0,
                            "estimated_length_meters": 1.0,
                            "plan_width": 100,
                            "plan_height": 100,
                            "meters_per_pixel": 0.1,
                            "person_radius_meters": 0.0,
                            "confidence": 0.9,
                        },
                    }
                return {
                    "accepted": False,
                    "trajectory": [],
                    "diagnostics": {
                        "accepted": False,
                        "reason": "constraint_solution_not_found",
                        "rejection_reasons": ["different_walkable_components"],
                    },
                }

        source = {
            "method": "r3_reconstruction_scale_aware",
            "plan_trajectory": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "turn_points": [],
            "processing_stats": {
                "pose_graph": {
                    "component_count": 25,
                    "largest_component_coverage": 0.01,
                }
            },
            "lingbot_fusion_candidate": {
                "accepted": False,
                "independent_accepted": True,
                "independent_plan_trajectory": [
                    [0.0, 0.0, 0.0], [50.0, 0.0, 0.0],
                ],
                "diagnostics": {
                    "independent_quality": {"accepted": True, "reasons": []},
                },
            },
        }
        with patch(
            "backend.floorplan_constraints.get_floorplan_engine",
            return_value=StubEngine(),
        ):
            updated = apply_floorplan_constraints(source, {
                "floorplan_id": "test",
                "reference_point": {"x": 10, "y": 50},
                "direction_point": {"x": 30, "y": 50},
            })

        self.assertTrue(updated["processing_stats"]["map_matching_applied"])
        self.assertEqual(updated["processing_stats"]["map_observation_source"], "r3")
        selection = updated["floorplan_constraint"]["observation_source_selection"]
        self.assertTrue(selection["r3_severely_fragmented"])
        self.assertEqual(selection["fragmentation_policy"], "soft_prior_not_veto")
        self.assertEqual(selection["selected"], "r3")
        independent_result = next(
            item for item in selection["candidate_results"]
            if item["source"] == "lingbot_independent" and item["variant"] == "native"
        )
        self.assertFalse(independent_result["accepted"])

    def test_rejected_independent_is_never_published_as_map_source(self) -> None:
        class StubEngine:
            def align(self, *args, **kwargs):
                return {
                    "accepted": False,
                    "trajectory": [],
                    "diagnostics": {
                        "accepted": False,
                        "reason": "constraint_solution_not_found",
                        "rejection_reasons": ["different_walkable_components"],
                    },
                }

        source = {
            "method": "r3_reconstruction_scale_aware",
            "plan_trajectory": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            "processing_stats": {
                "pose_graph": {
                    "component_count": 20,
                    "largest_component_coverage": 0.02,
                }
            },
            "lingbot_fusion_candidate": {
                "accepted": True,
                "plan_trajectory": [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
                "independent_accepted": True,
                "independent_plan_trajectory": [[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
                "diagnostics": {
                    "independent_quality": {"accepted": True, "reasons": []},
                },
            },
        }
        with patch(
            "backend.floorplan_constraints.get_floorplan_engine",
            return_value=StubEngine(),
        ):
            updated = apply_floorplan_constraints(source, {
                "floorplan_id": "test",
                "reference_point": {"x": 10, "y": 50},
                "direction_point": {"x": 30, "y": 50},
            })

        self.assertFalse(updated["processing_stats"]["map_matching_applied"])
        self.assertNotIn("map_observation_source", updated["processing_stats"])
        self.assertNotIn("map_trajectory", updated)
        selection = updated["floorplan_constraint"]["observation_source_selection"]
        self.assertIsNone(selection["selected"])
        self.assertEqual(selection["reason"], "no_candidate_satisfied_floorplan")
        self.assertIsNone(
            updated["floorplan_constraint"]["trajectory_observation_source"]
        )

    def test_unrepairable_segment_reports_disconnected_mask_components(self) -> None:
        mask = np.zeros((80, 120), dtype=bool)
        mask[:, 58:62] = True
        engine = FloorplanConstraintEngine.from_mask(
            mask,
            meters_per_pixel=0.1,
            grid_cell_pixels=1,
            person_radius_meters=0.0,
        )
        failures: list[str] = []
        repaired, _ = engine._repair_collisions(
            np.asarray([[20.0, 40.0], [100.0, 40.0]]),
            failure_reasons=failures,
        )

        self.assertIsNone(repaired)
        self.assertIn("different_walkable_components", failures)

    def test_three_meter_collision_is_never_kept_as_micro_collision(self) -> None:
        mask = np.zeros((80, 120), dtype=bool)
        mask[:, 58:62] = True
        engine = FloorplanConstraintEngine.from_mask(
            mask,
            meters_per_pixel=0.1,
            grid_cell_pixels=1,
            person_radius_meters=0.0,
        )
        failures: list[str] = []
        repaired, _ = engine._repair_collisions(
            np.asarray([[42.0, 40.0], [76.0, 40.0]]),
            failure_reasons=failures,
        )

        self.assertIsNone(repaired)
        self.assertIn("different_walkable_components", failures)

    def test_independent_stabilization_removes_length_inflating_jitter(self) -> None:
        count = 600
        x = np.linspace(0.0, 120.0, count)
        y = np.sin(np.linspace(0.0, np.pi, count)) * 20.0
        jitter = np.where(np.arange(count) % 2 == 0, -0.45, 0.45)
        noisy = np.column_stack((x + jitter, y - jitter, np.zeros(count))).tolist()

        stabilized, diagnostics = _stabilize_independent_observation(noisy)
        raw = np.asarray(noisy)[:, :2]
        stable = np.asarray(stabilized)[:, :2]
        raw_length = np.linalg.norm(np.diff(raw, axis=0), axis=1).sum()
        stable_length = np.linalg.norm(np.diff(stable, axis=0), axis=1).sum()

        self.assertTrue(diagnostics["applied"], diagnostics)
        self.assertEqual(len(stabilized), count)
        self.assertTrue(np.allclose(stable[[0, -1]], raw[[0, -1]]))
        self.assertLess(stable_length, raw_length * 0.65)

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

    def test_diverse_beam_retains_distant_metric_scale_strata(self) -> None:
        hypotheses = [
            {"score": float(index), "scale": float(index + 1), "yaw": 0.0}
            for index in range(10)
        ]
        beam = FloorplanConstraintEngine._select_diverse_beam(
            hypotheses, per_yaw=4, global_top=0
        )
        scales = [float(item["scale"]) for item in beam]
        self.assertLessEqual(min(scales), 1.0)
        self.assertGreaterEqual(max(scales), 8.0)

    def test_walking_speed_prior_is_flat_across_normal_human_range(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((500, 500), dtype=bool), meters_per_pixel=0.1
        )
        _, normal = engine._score_hypothesis(
            np.asarray([[50.0, 250.0], [170.0, 250.0]]),
            10.0,
            0.0,
            observation_policy="authoritative",
        )
        _, brisk = engine._score_hypothesis(
            np.asarray([[50.0, 250.0], [242.0, 250.0]]),
            10.0,
            0.0,
            observation_policy="authoritative",
        )
        self.assertEqual(normal["speed_prior_penalty"], 0.0)
        self.assertEqual(brisk["speed_prior_penalty"], 0.0)

    def test_04e39cf_compressed_independent_shape_is_rejected_before_scale_search(self) -> None:
        # RDP landmarks from the bad 04e39cf output.  The trajectory doubles
        # back through the office strip, so its endpoint covers only 63.5% of
        # its travelled arc even before monocular scale is chosen.
        bad_route = np.asarray([
            [2226.0, 678.0], [1794.2, 656.9], [1742.5, 691.9],
            [1646.0, 678.0], [1602.0, 734.0], [1498.0, 718.0],
            [1490.0, 834.0], [1456.8, 808.4], [1466.0, 858.0],
            [1498.0, 910.0], [1560.1, 894.7],
        ])
        progress = _polyline_progress_metrics(bad_route)
        self.assertLess(progress["net_progress_ratio"], 0.64)
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((1200, 2600), dtype=bool), meters_per_pixel=0.05
        )
        result = engine.align(
            bad_route.tolist(),
            {"x": 2226.0 / 2600.0 * 100.0, "y": 678.0 / 1200.0 * 100.0},
            {"x": 2145.0 / 2600.0 * 100.0, "y": 705.0 / 1200.0 * 100.0},
            timestamps=np.linspace(0.0, 54.0, len(bad_route)).tolist(),
            coordinate_convention="x_right_y_down",
            scale_candidates=[0.7, 1.0, 1.4],
            observation_policy="independent",
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(
            result["diagnostics"]["reason"],
            "insufficient_independent_net_progress",
        )

    def test_independent_alignment_fails_closed_when_two_scales_are_tied(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((500, 500), dtype=bool), meters_per_pixel=0.1
        )
        result = engine.align(
            [[0.0, 0.0], [40.0, 0.0], [80.0, 0.0], [120.0, 0.0]],
            {"x": 20.0, "y": 50.0},
            {"x": 30.0, "y": 50.0},
            timestamps=[0.0, 3.0, 6.0, 10.0],
            coordinate_convention="x_right_y_down",
            scale_candidates=[1.0, 1.5],
            yaw_offsets_degrees=[0.0],
            observation_policy="independent",
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(
            result["diagnostics"]["reason"],
            "ambiguous_independent_map_alignment",
        )
        self.assertGreater(result["diagnostics"]["ambiguous_scale_ratio"], 1.18)

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

    def test_second_order_hmm_is_enabled_in_production_by_default(self) -> None:
        mask = np.zeros((120, 180), dtype=bool)
        mask[42:78, 78:102] = True
        engine = FloorplanConstraintEngine.from_mask(mask, meters_per_pixel=0.1)
        result = engine.align(
            [[0, 0], [20, 0], [40, 0], [60, 0], [80, 0]],
            {"x": 10, "y": 50}, {"x": 30, "y": 50},
            scale_candidates=[2.0], yaw_offsets_degrees=[0.0],
        )
        nonlinear = result["diagnostics"]["nonlinear_map_matching"]
        self.assertTrue(nonlinear["attempted"])
        self.assertTrue(nonlinear["production_enabled"])
        self.assertEqual(nonlinear["coarse"]["order"], 2)
        self.assertEqual(nonlinear["fine"]["order"], 2)


if __name__ == "__main__":
    unittest.main()
