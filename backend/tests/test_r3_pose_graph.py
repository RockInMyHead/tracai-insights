"""Tests for R3 relative-pose sidecar validation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3_pose_graph import (
    R3_ABSOLUTE_POSE_SPACE,
    R3_CONFIDENCE_SEMANTICS,
    R3_POSE_ENCODING,
    R3_POSE_GRAPH_SCHEMA_VERSION,
    R3_RELATIVE_TRANSFORM_CONVENTION,
    load_pose_graph_summary,
    summarize_pose_graph_edges,
)


def edge(frame_i: int, frame_j: int, *, quaternion=None) -> dict:
    return {
        "frame_i": frame_i,
        "frame_j": frame_j,
        "rel_pose_enc": [
            1.0,
            0.0,
            0.0,
            *(quaternion or [0.0, 0.0, 0.0, 1.0]),
            0.9,
            0.7,
        ],
        "confidence": 2.0,
        "confidence_t": 2.1,
        "confidence_r": 1.9,
        "edge_type": "normal",
    }


def payload(edges: list[dict]) -> dict:
    return {
        "schema_version": R3_POSE_GRAPH_SCHEMA_VERSION,
        "pose_encoding": R3_POSE_ENCODING,
        "transform_convention": R3_RELATIVE_TRANSFORM_CONVENTION,
        "frame_index_space": "exported_camera_index",
        "absolute_pose_space": R3_ABSOLUTE_POSE_SPACE,
        "confidence_semantics": R3_CONFIDENCE_SEMANTICS,
        "edges": edges,
    }


class R3PoseGraphTests(unittest.TestCase):
    def test_complete_connected_graph_is_optimizer_ready(self) -> None:
        summary = summarize_pose_graph_edges(
            payload([edge(0, 1), edge(1, 2), edge(2, 3), edge(0, 3)]),
            point_count=4,
        )

        self.assertTrue(summary["available"])
        self.assertTrue(summary["optimizer_ready"])
        self.assertEqual(summary["relative_pose_coverage"], 1.0)
        self.assertEqual(summary["split_confidence_coverage"], 1.0)
        self.assertEqual(summary["component_count"], 1)
        self.assertEqual(summary["largest_component_coverage"], 1.0)

    def test_bad_edges_are_reported_and_block_optimizer(self) -> None:
        bad_quaternion = edge(0, 1, quaternion=[0.0, 0.0, 0.0, 0.2])
        missing_relative = edge(1, 2)
        missing_relative.pop("rel_pose_enc")
        summary = summarize_pose_graph_edges(
            payload([bad_quaternion, missing_relative, edge(2, 9)]),
            point_count=4,
        )

        self.assertFalse(summary["optimizer_ready"])
        self.assertEqual(summary["quaternion_norm_outliers"], 1)
        self.assertEqual(summary["relative_pose_coverage"], 0.5)
        self.assertEqual(summary["out_of_range_edges"], 1)

    def test_legacy_topology_log_is_visible_but_not_optimizer_ready(self) -> None:
        summary = summarize_pose_graph_edges([
            {"frame_i": 0, "frame_j": 1, "confidence": 2.0, "edge_type": "normal"},
        ], point_count=2)

        self.assertTrue(summary["available"])
        self.assertFalse(summary["optimizer_ready"])
        self.assertEqual(summary["schema_version"], 0)
        self.assertEqual(summary["relative_pose_edges"], 0)

    def test_nonpositive_confidence_is_not_optimizer_ready(self) -> None:
        invalid = edge(0, 1)
        invalid["confidence_r"] = 0.0
        summary = summarize_pose_graph_edges(payload([invalid]), point_count=2)

        self.assertFalse(summary["optimizer_ready"])
        self.assertEqual(summary["split_confidence_coverage"], 0.0)

    def test_load_summary_handles_file_and_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "pose_graph_edges.json"
            graph_path.write_text(json.dumps(payload([edge(0, 1)])), encoding="utf-8")

            loaded = load_pose_graph_summary(graph_path, point_count=2)
            missing = load_pose_graph_summary(Path(directory) / "missing.json", point_count=2)

        self.assertTrue(loaded["optimizer_ready"])
        self.assertFalse(missing["available"])
        self.assertEqual(missing["error"], "missing")

    def test_loads_compact_npz_and_caches_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "pose_graph_edges.npz"
            np.savez_compressed(
                graph_path,
                schema_version=np.asarray([R3_POSE_GRAPH_SCHEMA_VERSION], dtype=np.int32),
                pose_encoding=np.asarray(R3_POSE_ENCODING),
                transform_convention=np.asarray(R3_RELATIVE_TRANSFORM_CONVENTION),
                frame_index_space=np.asarray("exported_camera_index"),
                absolute_pose_space=np.asarray(R3_ABSOLUTE_POSE_SPACE),
                confidence_semantics=np.asarray(R3_CONFIDENCE_SEMANTICS),
                frame_i=np.asarray([0, 1], dtype=np.int32),
                frame_j=np.asarray([1, 2], dtype=np.int32),
                rel_pose_enc=np.asarray([
                    edge(0, 1)["rel_pose_enc"],
                    edge(1, 2)["rel_pose_enc"],
                ], dtype=np.float32),
                confidence=np.asarray([2.0, 2.0], dtype=np.float32),
                confidence_t=np.asarray([2.1, 2.1], dtype=np.float32),
                confidence_r=np.asarray([1.9, 1.9], dtype=np.float32),
                edge_type=np.asarray([0, 0], dtype=np.uint8),
            )

            first = load_pose_graph_summary(graph_path, point_count=3)
            second = load_pose_graph_summary(graph_path, point_count=3)
            cache_path = graph_path.with_suffix(".summary.json")
            cache_exists = cache_path.exists()

        self.assertTrue(first["optimizer_ready"])
        self.assertEqual(first["storage"], "compressed_npz")
        self.assertEqual(second["edge_count"], 2)
        self.assertTrue(cache_exists)


if __name__ == "__main__":
    unittest.main()
