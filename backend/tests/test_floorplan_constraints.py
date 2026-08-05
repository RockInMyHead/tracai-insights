import os
import unittest
from unittest.mock import patch

import numpy as np

from backend.floorplan_constraints import (
    FloorplanConfig,
    FloorplanConstraintEngine,
    _polyline_progress_metrics,
    _polyline_sharp_reverse_ratio,
    _speed_prior_penalty,
    _stabilize_authoritative_map_observation,
    _stabilize_independent_observation,
    _trajectory_fractions,
    _turn_topology_metrics,
    apply_floorplan_constraints,
    get_floorplan_engine,
)
from backend.kerama_reference_route import load_reference_route


class FloorplanConstraintEngineTests(unittest.TestCase):
    def test_graph_initial_direction_is_soft_until_clearly_opposite(self) -> None:
        thirty_degrees = np.cos(np.radians(30.0))
        sixty_degrees = np.cos(np.radians(60.0))
        seventy_one_degrees = np.cos(np.radians(71.0))

        thirty_cost, thirty_rejected = (
            FloorplanConstraintEngine._initial_direction_cost(thirty_degrees)
        )
        sixty_cost, sixty_rejected = (
            FloorplanConstraintEngine._initial_direction_cost(sixty_degrees)
        )
        opposite_cost, opposite_rejected = (
            FloorplanConstraintEngine._initial_direction_cost(
                seventy_one_degrees
            )
        )

        self.assertFalse(thirty_rejected)
        self.assertFalse(sixty_rejected)
        self.assertGreater(sixty_cost, thirty_cost)
        self.assertTrue(opposite_rejected)
        self.assertTrue(np.isinf(opposite_cost))

    def test_graph_initial_direction_uses_corridor_lookahead(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((100, 100), dtype=bool),
            meters_per_pixel=0.1,
        )
        # The first one-metre editor segment is diagonal, while the next four
        # metres establish the actual eastbound corridor.
        route = np.asarray([
            [10.0, 10.0],
            [18.0, 16.0],
            [60.0, 10.0],
            [90.0, 10.0],
        ])
        direction = engine._topology_route_initial_direction(
            route, lookahead_meters=5.0
        )
        angle = abs(np.degrees(np.arctan2(direction[1], direction[0])))
        first_segment_angle = abs(np.degrees(np.arctan2(6.0, 8.0)))

        self.assertLess(angle, 2.0)
        self.assertGreater(first_segment_angle, 30.0)

    def test_viterbi_prefix_stops_at_uncertainty_without_graph_tail(self) -> None:
        candidate = FloorplanConstraintEngine._viterbi_prefix_candidate({
            "_confirmed_prefix": [[10.0, 20.0], [30.0, 20.0]],
            "_confirmed_edge_ids": ["edge_confirmed"],
            "_confirmed_fraction_end": 0.25,
            "_uncertainty_marker": [30.0, 20.0],
            "_competing_next_edges": [
                {
                    "edge_id": "edge_left",
                    "points": [[30.0, 20.0], [40.0, 10.0]],
                },
                {
                    "edge_id": "edge_right",
                    "points": [[30.0, 20.0], [40.0, 30.0]],
                },
                {
                    "edge_id": "edge_forbidden_third",
                    "points": [[30.0, 20.0], [20.0, 20.0]],
                },
            ],
        })
        self.assertTrue(candidate["available"])
        self.assertEqual(
            candidate["trajectory"], [[10.0, 20.0], [30.0, 20.0]]
        )
        self.assertEqual(candidate["segments"][0]["status"], "confirmed")
        self.assertEqual(candidate["uncertain_segments"], 0)
        self.assertEqual(candidate["uncertainty_marker"], [30.0, 20.0])
        self.assertEqual(len(candidate["competing_next_edges"]), 2)
        self.assertEqual(
            candidate["policy"], "stop_at_last_confirmed_viterbi_node_v2"
        )

    def test_directed_edge_match_rejects_opposite_turn_branch(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((80, 80), dtype=bool),
            meters_per_pixel=1.0,
        )
        engine._topology_node_points = np.asarray([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, -10.0],
            [0.0, -10.0],
        ], dtype=np.float64)
        engine._topology_authored_node_count = 4
        engine._topology_node_ids = [
            "node_start",
            "node_branch",
            "node_first_turn",
            "node_wrong_left",
        ]
        engine._topology_adjacency = {
            0: [(1, 10.0)],
            1: [(2, 10.0)],
            2: [(3, 10.0)],
            3: [],
        }
        engine._topology_segment_edges = {
            (0, 1): "edge_start",
            (1, 2): "edge_first_turn",
            (2, 3): "edge_wrong_left",
        }

        observation = np.asarray([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, -10.0],
            [20.0, -10.0],
        ], dtype=np.float64)

        route, diagnostics = engine._directed_edge_event_match(
            observation,
            reject_committed_inversions=True,
        )

        self.assertIsNone(route)
        self.assertEqual(
            diagnostics["reason"],
            "directed_edge_event_sequence_not_found",
        )
        self.assertGreater(diagnostics["rejected_inverted_turns"], 0)

    def test_rejected_independent_candidate_does_not_hide_r3_graph_recovery(self) -> None:
        class FakeEngine:
            config = type("Config", (), {
                "default_anchor_reference_pixels": (10.0, 10.0),
                "default_anchor_direction_pixels": (20.0, 10.0),
                "default_anchor_source": "test",
                "width": 100,
                "height": 100,
                "meters_per_pixel": 0.1,
                "person_radius_meters": 0.0,
            })()

            def align(self, *args, **kwargs):
                authoritative = kwargs.get("observation_policy") == "authoritative"
                return {
                    "accepted": False,
                    "trajectory": [],
                    "diagnostics": {
                        "accepted": False,
                        "reason": (
                            "constraint_solution_not_found"
                            if authoritative
                            else "map_correction_exceeds_observation_budget"
                        ),
                        "rejection_reasons": [],
                        "graph_map_matching_configured": True,
                        "topology_recovery_enabled": authoritative,
                        "topology_recovery_attempted": 3 if authoritative else 0,
                        "topology_recovery_accepted": 0,
                        "graph_first_candidate": (
                            {
                                "available": True,
                                "trajectory": [[10.0, 10.0], [20.0, 10.0]],
                                "segments": [{
                                    "start_index": 0,
                                    "end_index": 1,
                                    "status": "uncertain",
                                    "edge_ids": ["edge_test"],
                                }],
                                "confirmed_segments": 0,
                                "uncertain_segments": 1,
                                "confirmed_source_fraction": 0.0,
                                "uncertainty_source_fraction_start": 0.0,
                                "policy": "graph_edges_only_no_raw_chords_v1",
                            }
                            if authoritative else None
                        ),
                    },
                }

        payload = {
            "method": "r3_reconstruction",
            "trajectory": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
            "r3_source_timestamps_seconds": [0.0, 1.0, 2.0, 3.0],
            "lingbot_fusion_candidate": {
                "accepted": False,
                "independent_accepted": True,
                "independent_plan_trajectory": [
                    [0, 0], [0, 1], [0, 2], [0, 3],
                ],
                "diagnostics": {
                    "independent_quality": {"accepted": True},
                },
            },
        }
        with patch(
            "backend.floorplan_constraints.get_floorplan_engine",
            return_value=FakeEngine(),
        ):
            result = apply_floorplan_constraints(
                payload, {"floorplan_id": "kerama_marazzi_2025"}
            )

        diagnostics = result["floorplan_constraint"]
        self.assertFalse(diagnostics["reported_candidate_topology_recovery_enabled"])
        self.assertTrue(diagnostics["topology_recovery_enabled"])
        self.assertEqual(
            diagnostics["authoritative_topology_recovery_attempted"], 6
        )
        self.assertTrue(diagnostics["authoritative_graph_matching"]["enabled"])
        self.assertEqual(
            diagnostics["observation_source_selection"]["candidate_results"][0][
                "topology_recovery_attempted"
            ],
            3,
        )
        self.assertTrue(diagnostics["graph_first_output_published"])
        self.assertEqual(
            result["graph_first_trajectory"],
            [[10.0, 10.0], [20.0, 10.0]],
        )
        self.assertNotIn("map_trajectory", result)
        self.assertEqual(
            result["graph_first_segments"][0]["status"], "uncertain"
        )

    def test_production_engine_uses_explicit_operator_topology_graph(self) -> None:
        engine = FloorplanConstraintEngine.load("kerama_marazzi_2025")
        self.assertEqual(engine._topology_authored_node_count, 147)
        self.assertGreater(len(engine._topology_node_points), 147)
        self.assertEqual(
            len(engine._topology_adjacency), len(engine._topology_node_points)
        )
        connected_pair = next(
            (left, neighbours[0][0])
            for left, neighbours in engine._topology_adjacency.items()
            if neighbours
        )
        route = engine._topology_route(
            engine._topology_node_points[connected_pair[0]],
            engine._topology_node_points[connected_pair[1]],
        )
        self.assertIsNotNone(route)
        self.assertGreaterEqual(len(route), 2)

    def test_red_obstacle_has_absolute_priority_over_green_support(self) -> None:
        support = np.ones((20, 20), dtype=bool)
        obstacle = np.zeros_like(support)
        obstacle[8:12, 8:12] = True
        engine = FloorplanConstraintEngine(
            FloorplanConfig(
                map_id="red_priority",
                width=20,
                height=20,
                meters_per_pixel=0.1,
                grid_cell_pixels=1,
                person_radius_meters=0.0,
                obstacle_mask_file="",
            ),
            obstacle,
            support,
        )
        self.assertTrue(engine._point_occupied([9, 9]))
        self.assertFalse(engine._point_occupied([2, 2]))

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

    def test_distant_parallel_branch_is_not_a_valid_r3_repair(self) -> None:
        support = np.zeros((100, 180), dtype=bool)
        support[28:33, 10:170] = True
        support[68:73, 10:170] = True
        support[28:73, 160:165] = True
        mask = np.zeros_like(support)
        mask[28:33, 55:95] = True
        engine = FloorplanConstraintEngine(
            FloorplanConfig(
                map_id="parallel_branch_test",
                width=180,
                height=100,
                meters_per_pixel=0.1,
                grid_cell_pixels=1,
                person_radius_meters=0.0,
                obstacle_mask_file="",
            ),
            mask,
            support,
        )
        observed = [[float(x), 30.0] for x in range(20, 141, 10)]
        result = engine.align(
            observed,
            {"x": 20 / 180 * 100, "y": 30},
            {"x": 40 / 180 * 100, "y": 30},
            coordinate_convention="x_right_y_down",
            scale_candidates=[1.0],
            yaw_offsets_degrees=[0.0],
            allow_safe_shape_fallback=True,
        )
        self.assertFalse(result["accepted"], result["diagnostics"])
        self.assertIn(
            result["diagnostics"]["reason"],
            {
                "constraint_solution_not_found",
                "map_correction_exceeds_observation_budget",
            },
        )
        if "shape_gate_details" in result["diagnostics"]:
            self.assertFalse(
                all(result["diagnostics"]["shape_gate_details"].values()),
                result["diagnostics"]["shape_gate_details"],
            )

    def test_start_inside_restricted_area_is_rejected_without_hidden_projection(self) -> None:
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
        self.assertFalse(result["accepted"], result["diagnostics"])
        self.assertTrue(result["diagnostics"]["start_anchor_locked"])
        self.assertGreater(result["diagnostics"]["start_snap_meters"], 0.05)

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
        self.assertEqual(result["diagnostics"]["reason"], "constraint_solution_not_found")
        self.assertTrue(result["diagnostics"]["start_anchor_locked"])

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

    def test_long_detour_supported_by_observed_arc_is_not_a_false_spike(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((100, 120), dtype=bool), meters_per_pixel=0.1
        )
        observed_arc = np.asarray([
            [10.0, 60.0], [10.0, -10.0], [40.0, -10.0], [40.0, 60.0]
        ])
        metrics = engine._detour_metrics(
            observed_arc, observed_arc[0], observed_arc[-1], observed_arc
        )
        self.assertGreater(metrics["route_meters"], metrics["chord_meters"] + 12.0)
        self.assertAlmostEqual(metrics["route_observed_length_ratio"], 1.0)
        self.assertFalse(metrics["spike"], metrics)

    def test_short_authoritative_spike_is_not_globally_rewritten(self) -> None:
        mask = np.zeros((120, 120), dtype=bool)
        mask[10:110, 55:65] = True
        engine = FloorplanConstraintEngine.from_mask(
            mask,
            grid_cell_pixels=1,
            meters_per_pixel=0.1,
            person_radius_meters=0.0,
        )
        observed = np.asarray([
            [45.5, 60.5], [53.0, 60.5], [60.5, 60.5],
            [68.0, 60.5], [75.5, 60.5],
        ])
        result = engine.align(
            observed.tolist(),
            {"x": 45.5 / 120.0 * 100.0, "y": 60.5 / 120.0 * 100.0},
            {"x": 53.0 / 120.0 * 100.0, "y": 60.5 / 120.0 * 100.0},
            coordinate_convention="x_right_y_down",
            scale_candidates=[1.0],
            yaw_offsets_degrees=[0.0],
        )
        self.assertFalse(result["accepted"], result["diagnostics"])
        self.assertEqual(result["diagnostics"]["topology_recovery_attempted"], 0)
        self.assertIn(
            "topology_destroying_map_correction",
            result["diagnostics"]["rejection_reasons"],
        )

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

    def test_authoritative_safe_fallback_rejects_topology_destroying_route(self) -> None:
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
        self.assertFalse(result["accepted"], result["diagnostics"])
        self.assertEqual(
            result["diagnostics"]["reason"],
            "map_correction_exceeds_observation_budget",
        )
        self.assertFalse(
            result["diagnostics"]["shape_gate_details"]["p95_within_budget"]
        )
        # The fallback preserves the two measured turns, but still requires a
        # non-local translation and is rejected by the correction budget.
        self.assertTrue(
            result["diagnostics"]["shape_gate_details"]["turn_topology_preserved"]
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
        self.assertFalse(result["accepted"], result["diagnostics"])
        self.assertTrue(
            {
                "different_walkable_components",
                "topology_destroying_map_correction",
            }.intersection(result["diagnostics"]["rejection_reasons"]),
            result["diagnostics"],
        )

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
                [0, 0, 0], [100, 0, 0], [200, 0, 0],
                [300, 0, 0], [400, 0, 0], [500, 0, 0],
            ],
            "turn_points": [{
                "frame_index": 3,
                "trajectory_index": 3,
                "angle_degrees": 90.0,
                "position": [300, 0, 0],
                "turn_type": "left",
            }],
            "source_timestamps_seconds": [0, 2, 4, 6, 8, 10],
            "trajectory_quality": {
                "projection": {"plan_coordinate_convention": "x_forward_y_left_z_up"}
            },
            "processing_stats": {},
        }
        original = [point[:] for point in source["plan_trajectory"]]
        updated = apply_floorplan_constraints(source, {
            "floorplan_id": "kerama_marazzi_2025",
            "reference_point": {"x": 2222.623 / 5298 * 100, "y": 684.183 / 3743 * 100},
            "direction_point": {"x": 2000 / 5298 * 100, "y": 703 / 3743 * 100},
        })
        self.assertTrue(updated["processing_stats"]["map_matching_applied"])
        self.assertIn("map_trajectory", updated)
        self.assertEqual(updated["plan_trajectory"], original)
        self.assertTrue(updated["floorplan_constraint"]["start_anchor_locked"])

    def test_operator_anchor_snaps_locally_and_heading_points_left(self) -> None:
        engine = get_floorplan_engine()
        anchor = np.asarray([2222.623, 684.183])
        direction = np.asarray([2000.0, 703.0])
        cell = engine._nearest_free(engine._pixel_to_cell(anchor))
        self.assertIsNotNone(cell)
        snapped = engine._cell_to_pixel(cell)
        self.assertLessEqual(
            float(np.linalg.norm(snapped - anchor)) * engine.config.meters_per_pixel,
            1.0,
        )
        self.assertLess(direction[0], anchor[0])
        self.assertLess(abs(direction[1] - anchor[1]), 20.0)
        self.assertTrue(engine._point_occupied([100, 100]))

    def test_operator_reference_route_repairs_locally_end_to_end(self) -> None:
        engine = get_floorplan_engine()
        reference = load_reference_route()
        points = reference["points"]

        def percent(point: list[float]) -> dict[str, float]:
            return {
                "x": point[0] / engine.config.width * 100.0,
                "y": point[1] / engine.config.height * 100.0,
            }

        result = engine.align(
            points,
            percent(reference["reference_point"]),
            percent(reference["direction_point"]),
            coordinate_convention="x_right_y_down",
            scale_candidates=[1.0],
            yaw_offsets_degrees=[0.0],
            allow_safe_shape_fallback=True,
        )
        self.assertTrue(result["accepted"], result["diagnostics"])
        self.assertTrue(result["diagnostics"]["start_anchor_locked"])
        self.assertLessEqual(result["diagnostics"]["correction_p95_meters"], 1.05)

    def test_guarded_fusion_does_not_promote_untrusted_independent(self) -> None:
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
        with patch.dict(os.environ, {"TRACKAI_ENABLE_FUSION_MAP_CANDIDATE": "1"}):
            updated = apply_floorplan_constraints(source, {
                "floorplan_id": "kerama_marazzi_2025",
                "reference_point": {"x": 2222.623 / 5298 * 100, "y": 684.183 / 3743 * 100},
                "direction_point": {"x": 2000 / 5298 * 100, "y": 703 / 3743 * 100},
            })

        self.assertTrue(updated["processing_stats"]["map_matching_applied"])
        selection = updated["floorplan_constraint"]["observation_source_selection"]
        independent = next(
            item for item in selection["candidate_results"]
            if item["source"] == "lingbot_independent"
        )
        self.assertFalse(independent["accepted"])

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

    def test_authoritative_macro_motion_removes_pose_jitter_and_locks_endpoints(
        self,
    ) -> None:
        count = 600
        x = np.linspace(0.0, 100.0, count)
        jitter = np.where(np.arange(count) % 2 == 0, -0.5, 0.5)
        raw = np.column_stack((x, jitter))

        macro, diagnostics = _stabilize_authoritative_map_observation(raw)

        self.assertTrue(diagnostics["applied"], diagnostics)
        np.testing.assert_allclose(macro[[0, -1]], raw[[0, -1]], atol=1e-9)
        self.assertLess(
            np.linalg.norm(np.diff(macro, axis=0), axis=1).sum(),
            np.linalg.norm(np.diff(raw, axis=0), axis=1).sum() * 0.7,
        )

    def test_opposite_turn_sequence_cannot_be_overridden_by_graph_score(self) -> None:
        source = np.asarray([
            [0.0, 0.0], [20.0, 0.0], [40.0, 0.0],
            [40.0, 20.0], [40.0, 40.0],
            [60.0, 40.0], [80.0, 40.0],
        ])
        opposite = source.copy()
        opposite[:, 1] *= -1.0

        metrics = _turn_topology_metrics(source, opposite)

        self.assertFalse(metrics["event_sequence_preserved"], metrics)
        self.assertGreater(metrics["sign_mismatch_ratio"], 0.0)

    def test_turn_event_count_change_is_not_topology_preserving(self) -> None:
        source = np.asarray([
            [0.0, 0.0], [30.0, 0.0], [30.0, 30.0],
            [60.0, 30.0], [60.0, 60.0],
        ])
        corrected = np.asarray([
            [0.0, 0.0], [15.0, 0.0], [30.0, 0.0],
            [45.0, 0.0], [60.0, 0.0],
        ])

        metrics = _turn_topology_metrics(source, corrected)

        self.assertFalse(metrics["event_sequence_preserved"], metrics)
        self.assertGreater(metrics["event_count_delta"], 0)

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

    def test_diverse_beam_retains_metric_scale_strata(self) -> None:
        hypotheses = [
            {"score": abs(index - 10), "scale": float(2 ** (index / 4)), "yaw": 0.0}
            for index in range(20)
        ]
        beam = FloorplanConstraintEngine._select_diverse_beam(
            hypotheses, per_yaw=1, global_top=1
        )
        scales = {item["scale"] for item in beam}
        self.assertIn(min(item["scale"] for item in hypotheses), scales)
        self.assertIn(max(item["scale"] for item in hypotheses), scales)
        self.assertGreaterEqual(len(scales), 6)

    def test_speed_prior_is_flat_across_normal_walking_speeds(self) -> None:
        self.assertEqual(_speed_prior_penalty(1.0, "authoritative"), 0.0)
        self.assertEqual(_speed_prior_penalty(1.5, "authoritative"), 0.0)
        self.assertEqual(_speed_prior_penalty(1.0, "independent"), 0.0)
        self.assertGreater(_speed_prior_penalty(0.65, "independent"), 4.0)

    def test_low_progress_independent_is_rejected_before_scale_search(self) -> None:
        engine = FloorplanConstraintEngine.from_mask(
            np.zeros((200, 200), dtype=bool), meters_per_pixel=0.1
        )
        observation = np.asarray([
            [0.0, 0.0], [15.0, 0.0], [30.0, 0.0], [30.0, 15.0],
            [30.0, 30.0], [19.0, 30.0], [8.0, 30.0],
        ])
        self.assertLess(
            _polyline_progress_metrics(observation)["net_progress_ratio"], 0.64
        )
        result = engine.align(
            observation.tolist(),
            {"x": 10.0, "y": 50.0},
            {"x": 30.0, "y": 50.0},
            timestamps=[0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0],
            observation_policy="independent",
        )
        self.assertFalse(result["accepted"], result["diagnostics"])
        self.assertEqual(
            result["diagnostics"]["reason"],
            "insufficient_independent_net_progress",
        )

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

    def test_second_order_graph_does_not_replace_better_safe_baseline(self) -> None:
        mask = np.zeros((120, 180), dtype=bool)
        mask[42:78, 78:102] = True
        engine = FloorplanConstraintEngine.from_mask(mask, meters_per_pixel=0.1)
        result = engine.align(
            [[0, 0], [20, 0], [40, 0], [60, 0], [80, 0]],
            {"x": 10, "y": 50}, {"x": 30, "y": 50},
            scale_candidates=[2.0], yaw_offsets_degrees=[0.0],
        )
        self.assertTrue(result["accepted"], result["diagnostics"])
        nonlinear = result["diagnostics"]["nonlinear_map_matching"]
        self.assertFalse(nonlinear["attempted"])
        self.assertTrue(nonlinear["production_enabled"])
        self.assertFalse(nonlinear["accepted"])
        self.assertEqual(nonlinear["reason"], "global_solution_stable")
        self.assertEqual(
            nonlinear["method"],
            "directed_edge_events_then_anchor_fallback_v1",
        )
        if "fine" in nonlinear and nonlinear["fine"].get("reason") is None:
            self.assertEqual(nonlinear["fine"]["order"], 2)

    def test_five_minute_reference_route_uses_operator_graph_over_bad_mask(self) -> None:
        engine = get_floorplan_engine()
        points = np.asarray(load_reference_route()["points"], dtype=np.float64)
        relative = points - points[0]
        observation = np.column_stack((
            relative[:, 0],
            -relative[:, 1],
            np.zeros(len(relative)),
        ))
        start = engine.config.default_anchor_reference_pixels
        direction = engine.config.default_anchor_direction_pixels
        self.assertIsNotNone(start)
        self.assertIsNotNone(direction)
        expected_start = np.asarray([
            engine.config.width * 0.419,
            engine.config.height * 0.181,
        ])
        expected_direction = np.asarray([
            engine.config.width * 0.377,
            engine.config.height * 0.187,
        ])
        np.testing.assert_allclose(start, expected_start, atol=1e-6)
        np.testing.assert_allclose(direction, expected_direction, atol=1e-6)
        result = engine.align(
            observation.tolist(),
            {
                "x": start[0] / engine.config.width * 100.0,
                "y": start[1] / engine.config.height * 100.0,
            },
            {
                "x": direction[0] / engine.config.width * 100.0,
                "y": direction[1] / engine.config.height * 100.0,
            },
            timestamps=np.linspace(0.0, 305.533, len(points)).tolist(),
            coordinate_convention="x_forward_y_left_z_up",
            scale_candidates=[1.0],
            yaw_offsets_degrees=[0.0],
        )
        self.assertTrue(result["accepted"], result["diagnostics"])
        self.assertEqual(
            result["diagnostics"]["turn_topology"]["source_event_count"],
            6,
        )
        self.assertEqual(
            result["diagnostics"]["turn_topology"]["sign_mismatch_ratio"],
            0.0,
        )
        self.assertTrue(
            result["diagnostics"]["turn_topology"][
                "graph_sign_sequence_preserved"
            ]
        )
        self.assertTrue(result["diagnostics"]["start_anchor_locked"])
        # The operator point is exact; the occupancy search works on a
        # four-pixel grid, so its internal walkable cell may be up to one grid
        # cell away without changing the requested/published anchor.
        self.assertLessEqual(
            result["diagnostics"]["start_snap_meters"],
            engine.config.grid_cell_pixels * engine.config.meters_per_pixel,
        )
        np.testing.assert_allclose(
            result["diagnostics"]["requested_start_pixels"],
            expected_start,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
