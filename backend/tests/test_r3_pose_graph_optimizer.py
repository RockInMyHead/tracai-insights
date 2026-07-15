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
