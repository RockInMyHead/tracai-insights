"""Regression tests for robust R3 pose-graph shadow optimization."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3_pose_graph_optimizer import (
    PoseGraphOptimizerConfig,
    _geometry_rejection_reasons,
    _path_metrics,
    _temporal_backbone_mask,
    _verify_distant_edges,
    load_pose_graph_candidate_c2w,
    load_pose_graph_candidate_summary,
    optimize_pose_graph_arrays,
    run_pose_graph_shadow,
    save_pose_graph_candidate,
)
from r3_pose_graph import (
    R3_ABSOLUTE_POSE_SPACE,
    R3_CONFIDENCE_SEMANTICS,
    R3_POSE_ENCODING,
    R3_POSE_GRAPH_SCHEMA_VERSION,
    R3_RELATIVE_TRANSFORM_CONVENTION,
)


def rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


def l_route() -> np.ndarray:
    count = 20
    centers = np.asarray([
        [index, 0.0, 0.0] if index < 10 else [9.0, index - 9, 0.0]
        for index in range(count)
    ])
    yaws = [0.0] * 10 + [np.pi / 2.0] * 10
    c2w = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    c2w[:, :3, :3] = np.stack([rotation_z(yaw) for yaw in yaws])
    c2w[:, :3, 3] = centers
    return c2w


def square_route(point_count: int = 120) -> np.ndarray:
    perimeter = np.arange(point_count, dtype=float) / (point_count / 4.0)
    centers = np.zeros((point_count, 3), dtype=float)
    yaws = np.zeros(point_count, dtype=float)
    first = perimeter < 1.0
    second = (perimeter >= 1.0) & (perimeter < 2.0)
    third = (perimeter >= 2.0) & (perimeter < 3.0)
    fourth = perimeter >= 3.0
    centers[first, 0] = 10.0 * perimeter[first]
    centers[second, 0] = 10.0
    centers[second, 1] = 10.0 * (perimeter[second] - 1.0)
    centers[third, 0] = 10.0 * (3.0 - perimeter[third])
    centers[third, 1] = 10.0
    centers[fourth, 1] = 10.0 * (4.0 - perimeter[fourth])
    yaws[second] = np.pi / 2.0
    yaws[third] = np.pi
    yaws[fourth] = -np.pi / 2.0
    c2w = np.broadcast_to(np.eye(4), (point_count, 4, 4)).copy()
    c2w[:, :3, :3] = np.stack([rotation_z(yaw) for yaw in yaws])
    c2w[:, :3, 3] = centers
    return c2w


def edge_from_c2w(c2w: np.ndarray, frame_i: int, frame_j: int, confidence=2.0):
    w2c = np.linalg.inv(c2w)
    relative = w2c[frame_j] @ np.linalg.inv(w2c[frame_i])
    quaternion = Rotation.from_matrix(relative[:3, :3]).as_quat()
    return (
        frame_i,
        frame_j,
        np.concatenate([relative[:3, 3], quaternion, [0.8, 0.8]]),
        confidence,
        confidence,
        confidence,
        0,
    )


def graph_arrays(edges: list[tuple]) -> dict[str, np.ndarray]:
    columns = list(zip(*edges))
    return {
        "frame_i": np.asarray(columns[0], dtype=np.int32),
        "frame_j": np.asarray(columns[1], dtype=np.int32),
        "rel_pose_enc": np.asarray(columns[2], dtype=np.float32),
        "confidence": np.asarray(columns[3], dtype=np.float32),
        "confidence_t": np.asarray(columns[4], dtype=np.float32),
        "confidence_r": np.asarray(columns[5], dtype=np.float32),
        "edge_type": np.asarray(columns[6], dtype=np.uint8),
    }


def graph_metadata() -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray([R3_POSE_GRAPH_SCHEMA_VERSION], dtype=np.int32),
        "pose_encoding": np.asarray(R3_POSE_ENCODING),
        "transform_convention": np.asarray(R3_RELATIVE_TRANSFORM_CONVENTION),
        "frame_index_space": np.asarray("exported_camera_index"),
        "absolute_pose_space": np.asarray(R3_ABSOLUTE_POSE_SPACE),
        "confidence_semantics": np.asarray(R3_CONFIDENCE_SEMANTICS),
    }


class R3PoseGraphOptimizerTests(unittest.TestCase):
    def test_residual_improvement_cannot_hide_destroyed_l_turn_sequence(self) -> None:
        raw = l_route()
        folded = raw.copy()
        folded[:, :3, :3] = np.stack([
            rotation_z(0.0 if index < 10 else np.pi)
            for index in range(len(folded))
        ])
        folded[:, 0, 3] = np.concatenate((
            np.arange(10, dtype=float),
            np.arange(8, -2, -1, dtype=float),
        ))
        folded[:, 1, 3] = 0.0
        edges = [
            edge_from_c2w(folded, index, index + 1, confidence=5.0)
            for index in range(len(folded) - 1)
        ]

        result = optimize_pose_graph_arrays(raw, **graph_arrays(edges))
        diagnostics = result["diagnostics"]

        self.assertGreater(diagnostics["objective_improvement"], 0.5)
        self.assertFalse(diagnostics["accepted"])
        self.assertIn(
            "turn_sequence_destroyed",
            diagnostics["rejection_reasons"],
        )
        self.assertLess(
            diagnostics["candidate_path"]["endpoint_displacement"],
            diagnostics["initial_path"]["endpoint_displacement"] * 0.2,
        )

    def test_mutually_supported_false_distant_edges_are_still_rejected(self) -> None:
        poses = np.broadcast_to(np.eye(4), (16, 4, 4)).copy()
        poses[:, 0, 3] = np.arange(len(poses), dtype=float)
        frame_i = np.asarray([0, 1, 2], dtype=np.int32)
        frame_j = np.asarray([10, 11, 12], dtype=np.int32)
        wrong = np.zeros((3, 9), dtype=np.float64)
        wrong[:, 0] = 10.0
        # The W2C convention predicts negative X for this forward C2W motion.
        wrong[:, 6] = 1.0
        edge_type = np.full(3, 3, dtype=np.uint8)

        verified_types, diagnostics = _verify_distant_edges(
            poses,
            frame_i=frame_i,
            frame_j=frame_j,
            rel_pose_enc=wrong,
            confidence_t=np.full(3, 20.0),
            confidence_r=np.full(3, 20.0),
            edge_type=edge_type,
            matching_support=np.full(3, np.nan),
            config=PoseGraphOptimizerConfig(
                distant_edge_minimum_temporal_gap=2,
            ),
        )

        self.assertEqual(verified_types.tolist(), [5, 5, 5])
        self.assertEqual(diagnostics["rejected_count"], 3)
        for decision in diagnostics["decisions"]:
            self.assertGreaterEqual(decision["neighbor_match_count"], 2)
            self.assertNotIn("isolated_match", decision["reasons"])
            self.assertIn(
                "translation_direction_inconsistent",
                decision["reasons"],
            )

    def test_backbone_prefers_gap_one_then_gap_two(self) -> None:
        frame_i = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int32)
        frame_j = np.asarray([1, 2, 2, 3, 3, 4], dtype=np.int32)
        confidence = np.asarray([1, 10, 1, 10, 1, 1], dtype=float)
        edge_type = np.zeros(len(frame_i), dtype=np.uint8)

        mask, diagnostics = _temporal_backbone_mask(
            frame_i,
            frame_j,
            confidence,
            point_count=5,
            edge_type=edge_type,
            config=PoseGraphOptimizerConfig(),
        )

        selected = list(zip(frame_i[mask].tolist(), frame_j[mask].tolist()))
        self.assertEqual(selected, [(0, 1), (1, 2), (2, 3), (3, 4)])
        self.assertEqual(diagnostics["backbone_coverage"], 1.0)
        self.assertEqual(diagnostics["maximum_backbone_gap"], 1)
        self.assertEqual(diagnostics["missing_temporal_links"], [])

    def test_backbone_requires_bridge_at_fallback_boundary(self) -> None:
        frame_i = np.asarray([0, 1, 2, 2, 3, 4], dtype=np.int32)
        frame_j = np.asarray([1, 2, 3, 3, 4, 5], dtype=np.int32)
        confidence = np.ones(len(frame_i), dtype=float)
        edge_type = np.asarray([0, 0, 0, 2, 0, 0], dtype=np.uint8)
        config = PoseGraphOptimizerConfig(fallback_boundaries=(3,))

        mask, diagnostics = _temporal_backbone_mask(
            frame_i, frame_j, confidence, 6, edge_type, config
        )

        selected_types = edge_type[mask].tolist()
        self.assertEqual(diagnostics["backbone_coverage"], 1.0)
        self.assertEqual(diagnostics["bridge_boundary_coverage"], 1.0)
        self.assertEqual(selected_types.count(2), 1)
        # The parallel normal edge 2->3 must not cross the boundary.
        self.assertFalse(mask[2])
        self.assertTrue(mask[3])

    def test_missing_bridge_breaks_backbone_even_if_full_graph_is_connected(self) -> None:
        frame_i = np.asarray([0, 1, 2, 3, 4, 0], dtype=np.int32)
        frame_j = np.asarray([1, 2, 3, 4, 5, 5], dtype=np.int32)
        confidence = np.ones(len(frame_i), dtype=float)
        edge_type = np.zeros(len(frame_i), dtype=np.uint8)
        config = PoseGraphOptimizerConfig(fallback_boundaries=(3,))

        mask, diagnostics = _temporal_backbone_mask(
            frame_i, frame_j, confidence, 6, edge_type, config
        )

        self.assertLess(diagnostics["backbone_coverage"], 0.95)
        self.assertEqual(diagnostics["bridge_boundary_coverage"], 0.0)
        self.assertIn(3, diagnostics["missing_temporal_links"])
        self.assertEqual(diagnostics["long_range_backbone_edges"], 0)
        self.assertEqual(diagnostics["input_long_range_edges"], 1)
        self.assertEqual(int(mask.sum()), 2)

    def test_long_range_edge_is_never_promoted_into_backbone(self) -> None:
        frame_i = np.asarray([0, 0], dtype=np.int32)
        frame_j = np.asarray([1, 4], dtype=np.int32)
        confidence = np.asarray([1.0, 100.0])
        edge_type = np.zeros(2, dtype=np.uint8)

        mask, diagnostics = _temporal_backbone_mask(
            frame_i,
            frame_j,
            confidence,
            5,
            edge_type,
            PoseGraphOptimizerConfig(maximum_backbone_gap=2),
        )

        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertIn(4, diagnostics["missing_temporal_links"])
        self.assertEqual(diagnostics["long_range_backbone_edges"], 0)

    def test_geometry_gate_rejects_same_length_spatial_collapse(self) -> None:
        point_count = 101
        raw = np.broadcast_to(np.eye(4), (point_count, 4, 4)).copy()
        raw[:, 0, 3] = np.linspace(0.0, 100.0, point_count)
        collapsed = raw.copy()
        # Preserve approximately the same travelled length while folding every
        # ten steps back through a narrow strip around the start.
        x = np.arange(point_count, dtype=float) % 10
        direction = (np.arange(point_count) // 10) % 2
        collapsed[:, 0, 3] = np.where(direction == 0, x, 9.0 - x)
        collapsed[:, 1, 3] = np.arange(point_count, dtype=float) // 10 * 0.1
        raw_metrics = _path_metrics(raw)
        candidate_metrics = _path_metrics(
            collapsed, near_start_radius=raw_metrics["near_start_radius"]
        )

        reasons, diagnostics = _geometry_rejection_reasons(
            raw,
            collapsed,
            raw_metrics,
            candidate_metrics,
            PoseGraphOptimizerConfig(),
        )

        self.assertIn("spatial_span_collapsed", reasons)
        self.assertIn("endpoint_displacement_collapsed", reasons)
        self.assertGreater(candidate_metrics["near_start_fraction"], 0.9)
        self.assertIn("comparison", diagnostics)

    def test_verified_loop_mode_only_relaxes_endpoint_semantics(self) -> None:
        raw = np.broadcast_to(np.eye(4), (5, 4, 4)).copy()
        raw[:, :2, 3] = np.asarray([
            [0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [1.0, 0.0]
        ])
        candidate = raw.copy()
        candidate[-1, :2, 3] = [0.0, 0.0]
        raw_metrics = _path_metrics(raw)
        candidate_metrics = _path_metrics(
            candidate, near_start_radius=raw_metrics["near_start_radius"]
        )
        config = PoseGraphOptimizerConfig(
            verified_loop_closure=True,
            minimum_spatial_span_ratio=0.0,
            maximum_tortuosity_factor=1e9,
            maximum_tortuosity_increase=1e9,
            maximum_segment_length_log_rmse=10.0,
            minimum_straight_run_preservation=0.0,
            maximum_sharp_reverse_ratio_increase=1.0,
        )

        reasons, diagnostics = _geometry_rejection_reasons(
            raw, candidate, raw_metrics, candidate_metrics, config
        )

        self.assertNotIn("endpoint_displacement_collapsed", reasons)
        self.assertNotIn("near_start_concentration_regression", reasons)
        self.assertTrue(diagnostics["verified_loop_closure"])

    def test_closed_raw_route_is_not_rejected_for_low_endpoint_alone(self) -> None:
        raw = square_route()
        candidate = raw.copy()
        candidate[:, :3, 3] *= 1.03
        raw_metrics = _path_metrics(raw)
        candidate_metrics = _path_metrics(
            candidate, near_start_radius=raw_metrics["near_start_radius"]
        )

        reasons, _ = _geometry_rejection_reasons(
            raw,
            candidate,
            raw_metrics,
            candidate_metrics,
            PoseGraphOptimizerConfig(),
        )

        self.assertNotIn("endpoint_displacement_collapsed", reasons)

    def test_recovers_left_turn_from_right_turn_with_outliers(self) -> None:
        truth = l_route()
        edges = []
        for gap in (1, 2, 5):
            for index in range(len(truth) - gap):
                edges.append(edge_from_c2w(truth, index, index + gap))

        rng = np.random.default_rng(3)
        for _ in range(5):
            frame_i, frame_j = sorted(rng.choice(len(truth), 2, replace=False))
            wrong = np.eye(4)
            wrong[:3, :3] = rotation_z(float(rng.uniform(-np.pi, np.pi)))
            wrong[:3, 3] = rng.normal(0.0, 5.0, 3)
            quaternion = Rotation.from_matrix(wrong[:3, :3]).as_quat()
            edges.append((
                int(frame_i),
                int(frame_j),
                np.concatenate([wrong[:3, 3], quaternion, [0.8, 0.8]]),
                0.8,
                0.8,
                0.8,
                0,
            ))

        initial = truth.copy()
        initial[10:, :3, 3] = np.asarray([
            [9.0, -(index - 9), 0.0] for index in range(10, len(truth))
        ])
        initial[10:, :3, :3] = np.stack(
            [rotation_z(-np.pi / 2.0)] * (len(truth) - 10)
        )
        arrays = graph_arrays(edges)
        result = optimize_pose_graph_arrays(initial, **arrays)
        candidate = result["c2w"]

        initial_turn_sign = np.cross(
            initial[9, :3, 3] - initial[0, :3, 3],
            initial[-1, :3, 3] - initial[9, :3, 3],
        )[2]
        candidate_turn_sign = np.cross(
            candidate[9, :3, 3] - candidate[0, :3, 3],
            candidate[-1, :3, 3] - candidate[9, :3, 3],
        )[2]
        truth_centers = truth[:, :3, 3]
        initial_rmse = np.sqrt(np.mean(np.sum(
            (initial[:, :3, 3] - truth_centers) ** 2, axis=1
        )))
        candidate_rmse = np.sqrt(np.mean(np.sum(
            (candidate[:, :3, 3] - truth_centers) ** 2, axis=1
        )))

        self.assertLess(initial_turn_sign, 0.0)
        self.assertGreater(candidate_turn_sign, 0.0)
        self.assertLess(candidate_rmse, initial_rmse * 0.2)
        self.assertTrue(result["diagnostics"]["accepted"])
        self.assertGreater(result["diagnostics"]["objective_improvement"], 0.5)

    def test_recovers_four_ninety_degree_turns_from_drifted_square(self) -> None:
        truth = square_route()
        edges = [
            edge_from_c2w(truth, index, index + gap)
            for gap in (1, 2, 5, 12, 30)
            for index in range(len(truth) - gap)
        ]
        rng = np.random.default_rng(17)
        for _ in range(12):
            frame_i, frame_j = sorted(rng.choice(len(truth), 2, replace=False))
            wrong_rotation = rotation_z(float(rng.uniform(-np.pi, np.pi)))
            wrong_translation = rng.normal(0.0, 8.0, 3)
            edges.append((
                int(frame_i),
                int(frame_j),
                np.concatenate([
                    wrong_translation,
                    Rotation.from_matrix(wrong_rotation).as_quat(),
                    [0.8, 0.8],
                ]),
                0.7,
                0.7,
                0.7,
                1,
            ))

        progress = np.linspace(0.0, 1.0, len(truth))
        initial = truth.copy()
        initial[:, :3, 3] *= (1.0 + 0.2 * progress[:, None])
        initial[:, 1, 3] += 2.0 * np.sin(np.pi * progress)
        truth_yaws = np.unwrap(np.arctan2(
            truth[:, 1, 0],
            truth[:, 0, 0],
        ))
        initial[:, :3, :3] = np.stack([
            rotation_z(yaw + math.radians(35.0) * fraction)
            for yaw, fraction in zip(truth_yaws, progress)
        ])

        result = optimize_pose_graph_arrays(initial, **graph_arrays(edges))
        candidate = result["c2w"]
        truth_centers = truth[:, :3, 3]
        initial_rmse = np.sqrt(np.mean(np.sum(
            (initial[:, :3, 3] - truth_centers) ** 2, axis=1
        )))
        candidate_rmse = np.sqrt(np.mean(np.sum(
            (candidate[:, :3, 3] - truth_centers) ** 2, axis=1
        )))
        turn_angles = []
        for corner in (30, 60, 90):
            incoming = candidate[corner, :2, 3] - candidate[corner - 8, :2, 3]
            outgoing = candidate[corner + 8, :2, 3] - candidate[corner, :2, 3]
            cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            turn_angles.append(math.degrees(math.atan2(
                cross,
                np.dot(incoming, outgoing),
            )))

        self.assertTrue(result["diagnostics"]["accepted"])
        self.assertLess(candidate_rmse, initial_rmse * 0.35)
        for angle in turn_angles:
            self.assertGreater(angle, 80.0)
            self.assertLess(angle, 100.0)

    def test_disconnected_graph_never_becomes_authoritative(self) -> None:
        c2w = np.broadcast_to(np.eye(4), (6, 4, 4)).copy()
        c2w[:, 0, 3] = np.arange(6, dtype=float)
        edges = [
            edge_from_c2w(c2w, 0, 1),
            edge_from_c2w(c2w, 2, 3),
            edge_from_c2w(c2w, 4, 5),
        ]
        initial = c2w.copy()
        initial[:, 0, 3] *= 1.2
        result = optimize_pose_graph_arrays(initial, **graph_arrays(edges))

        self.assertFalse(result["diagnostics"]["accepted"])
        self.assertIn(
            "insufficient_graph_coverage",
            result["diagnostics"]["rejection_reasons"],
        )
        self.assertEqual(result["diagnostics"]["graph"]["component_count"], 3)

    def test_connected_full_graph_with_broken_backbone_is_rejected(self) -> None:
        truth = np.broadcast_to(np.eye(4), (5, 4, 4)).copy()
        truth[:, 0, 3] = np.arange(5, dtype=float)
        edges = [
            edge_from_c2w(truth, 0, 1),
            edge_from_c2w(truth, 1, 2),
            edge_from_c2w(truth, 3, 4),
            edge_from_c2w(truth, 0, 4),
        ]
        initial = truth.copy()
        initial[:, 0, 3] *= 1.1

        result = optimize_pose_graph_arrays(initial, **graph_arrays(edges))
        diagnostics = result["diagnostics"]

        self.assertEqual(diagnostics["graph"]["component_count"], 1)
        self.assertFalse(diagnostics["accepted"])
        self.assertIn(
            "insufficient_backbone_coverage",
            diagnostics["rejection_reasons"],
        )
        self.assertLess(diagnostics["backbone"]["backbone_coverage"], 0.95)
        self.assertEqual(diagnostics["backbone"]["long_range_backbone_edges"], 0)

    def test_candidate_artifact_round_trip_does_not_touch_raw_cameras(self) -> None:
        truth = l_route()
        edges = [
            edge_from_c2w(truth, index, index + 1)
            for index in range(len(truth) - 1)
        ]
        initial = truth.copy()
        initial[:, :3, 3] *= 1.1
        result = optimize_pose_graph_arrays(initial, **graph_arrays(edges))

        with tempfile.TemporaryDirectory() as directory:
            saved = save_pose_graph_candidate(directory, result)
            summary = load_pose_graph_candidate_summary(directory)
            loaded = load_pose_graph_candidate_c2w(
                directory,
                expected_count=len(truth),
                accepted_only=False,
            )
            camera_dir = Path(directory) / "camera"

        self.assertIn("candidate_path", saved)
        self.assertTrue(summary["available"])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, (len(truth), 4, 4))
        self.assertFalse(camera_dir.exists())

    def test_file_runner_reads_exported_graph_and_persists_candidate(self) -> None:
        truth = l_route()
        arrays = graph_arrays([
            edge_from_c2w(truth, index, index + gap)
            for gap in (1, 2, 5)
            for index in range(len(truth) - gap)
        ])
        initial = truth.copy()
        initial[:, :3, 3] *= 1.15

        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "pose_graph_edges.npz"
            np.savez_compressed(graph_path, **graph_metadata(), **arrays)

            result = run_pose_graph_shadow(directory, initial)
            summary = load_pose_graph_candidate_summary(directory)
            candidate = load_pose_graph_candidate_c2w(
                directory,
                expected_count=len(truth),
                accepted_only=False,
            )

        self.assertTrue(result["available"])
        self.assertTrue(summary["available"])
        self.assertEqual(summary["source_graph"], str(graph_path))
        self.assertEqual(summary["point_count"], len(truth))
        self.assertIsNotNone(candidate)

    def test_file_runner_rejects_unknown_transform_convention(self) -> None:
        truth = l_route()
        arrays = graph_arrays([
            edge_from_c2w(truth, index, index + 1)
            for index in range(len(truth) - 1)
        ])
        metadata = graph_metadata()
        metadata["transform_convention"] = np.asarray("unknown")

        with tempfile.TemporaryDirectory() as directory:
            np.savez_compressed(
                Path(directory) / "pose_graph_edges.npz",
                **metadata,
                **arrays,
            )
            result = run_pose_graph_shadow(directory, truth)

        self.assertFalse(result["available"])
        self.assertFalse(result["accepted"])
        self.assertIn("unsupported pose graph metadata", result["error"])

    def test_legacy_v1_long_normal_edge_is_excluded_safely(self) -> None:
        truth = square_route()
        edges = [
            edge_from_c2w(truth, index, index + 1)
            for index in range(len(truth) - 1)
        ]
        edges.append(edge_from_c2w(truth, 0, len(truth) - 1, confidence=20.0))
        arrays = graph_arrays(edges)
        metadata = graph_metadata()
        metadata["schema_version"] = np.asarray([1], dtype=np.int32)
        initial = truth.copy()
        initial[:, :3, 3] *= 1.1

        with tempfile.TemporaryDirectory() as directory:
            np.savez_compressed(
                Path(directory) / "pose_graph_edges.npz",
                **metadata,
                **arrays,
            )
            result = run_pose_graph_shadow(directory, initial)

        diagnostics = result
        self.assertEqual(
            diagnostics["edge_classification_mode"],
            "legacy_v1_safe_classification",
        )
        self.assertEqual(diagnostics["excluded_edge_count"], 1)
        self.assertTrue(diagnostics["accepted"])

    def test_unverified_loop_candidate_never_enters_optimizer(self) -> None:
        truth = l_route()
        edges = [
            edge_from_c2w(truth, index, index + 1)
            for index in range(len(truth) - 1)
        ]
        distant = list(edge_from_c2w(truth, 0, len(truth) - 1, confidence=50.0))
        distant[-1] = 3
        edges.append(tuple(distant))
        initial = truth.copy()
        initial[:, :3, 3] *= 1.1

        result = optimize_pose_graph_arrays(initial, **graph_arrays(edges))

        self.assertEqual(result["diagnostics"]["excluded_edge_count"], 1)
        self.assertTrue(result["diagnostics"]["accepted"])

    def test_supported_distant_overlap_is_not_mislabeled_as_loop(self) -> None:
        truth = l_route()
        edges = [
            edge_from_c2w(truth, index, index + 1)
            for index in range(len(truth) - 1)
        ]
        distant = list(edge_from_c2w(truth, 2, len(truth) - 2, confidence=4.0))
        # A source-side verified label must not bypass local verification.
        distant[-1] = 4
        edges.append(tuple(distant))
        arrays = graph_arrays(edges)
        support = np.full(len(edges), np.nan, dtype=np.float32)
        support[-1] = 3.0

        result = optimize_pose_graph_arrays(
            truth.copy(),
            **arrays,
            matching_support=support,
            config=PoseGraphOptimizerConfig(distant_edge_minimum_temporal_gap=2),
        )

        verification = result["diagnostics"]["distant_edge_verification"]
        self.assertEqual(verification["verified_overlap_count"], 1)
        self.assertEqual(verification["verified_loop_count"], 0)
        self.assertEqual(verification["rejected_count"], 0)
        self.assertEqual(
            verification["decisions"][0]["classification"],
            "verified_overlap",
        )

    def test_isolated_distant_match_is_rejected_despite_high_confidence(self) -> None:
        truth = l_route()
        edges = [
            edge_from_c2w(truth, index, index + 1)
            for index in range(len(truth) - 1)
        ]
        distant = list(edge_from_c2w(truth, 1, len(truth) - 1, confidence=20.0))
        distant[-1] = 3
        edges.append(tuple(distant))

        result = optimize_pose_graph_arrays(
            truth.copy(),
            **graph_arrays(edges),
            config=PoseGraphOptimizerConfig(distant_edge_minimum_temporal_gap=2),
        )

        verification = result["diagnostics"]["distant_edge_verification"]
        self.assertEqual(verification["rejected_count"], 1)
        self.assertIn(
            "isolated_match",
            verification["decisions"][0]["reasons"],
        )
        self.assertEqual(result["diagnostics"]["excluded_edge_count"], 1)

    def test_duplicate_pair_uses_strongest_measurement_once(self) -> None:
        truth = l_route()
        edges = [
            edge_from_c2w(truth, index, index + 1, confidence=3.0)
            for index in range(len(truth) - 1)
        ]
        strongest = edge_from_c2w(truth, 4, 12, confidence=4.0)
        edges.append(strongest)

        wrong = np.eye(4)
        wrong[:3, :3] = rotation_z(-np.pi / 2.0)
        wrong[:3, 3] = [20.0, -20.0, 0.0]
        wrong_quaternion = Rotation.from_matrix(wrong[:3, :3]).as_quat()
        weak_duplicate = (
            4,
            12,
            np.concatenate([wrong[:3, 3], wrong_quaternion, [0.8, 0.8]]),
            0.2,
            0.2,
            0.2,
            0,
        )
        edges.extend([weak_duplicate] * 20)

        initial = truth.copy()
        initial[:, :3, 3] *= 1.1
        result = optimize_pose_graph_arrays(initial, **graph_arrays(edges))
        diagnostics = result["diagnostics"]

        self.assertEqual(diagnostics["input_edge_count"], len(edges))
        self.assertEqual(diagnostics["deduplicated_edge_count"], len(truth))
        self.assertTrue(diagnostics["accepted"])
        self.assertLess(diagnostics["after"]["rotation_p90_degrees"], 1.0)

    def test_file_runner_reports_missing_graph_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_pose_graph_shadow(directory, l_route())

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "pose_graph_missing")


if __name__ == "__main__":
    unittest.main()
