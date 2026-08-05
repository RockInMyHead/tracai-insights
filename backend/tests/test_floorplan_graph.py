import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backend.floorplan_graph import (
    SCHEMA_VERSION,
    build_floorplan_graph,
    floorplan_graph_geometry_sha256,
    normalize_floorplan_graph,
    validate_floorplan_graph,
)


class FloorplanGraphTests(unittest.TestCase):
    def test_normalization_turns_hidden_bends_into_explicit_nodes(self) -> None:
        graph = {
            "schema_version": SCHEMA_VERSION,
            "map_id": "manual",
            "width": 100,
            "height": 100,
            "meters_per_pixel": 0.1,
            "source": {},
            "nodes": [
                {"id": "node_0001", "kind": "manual", "x": 5, "y": 5, "enabled": True},
                {"id": "node_0002", "kind": "manual", "x": 80, "y": 80, "enabled": True},
            ],
            "edges": [{
                "id": "edge_00001",
                "from": "node_0001",
                "to": "node_0002",
                "points": [[0, 0], [40, 5], [40, 80], [90, 90]],
                "enabled": True,
                "bidirectional": True,
            }],
        }

        normalized = normalize_floorplan_graph(graph)
        nodes = {node["id"]: node for node in normalized["nodes"]}

        self.assertEqual(len(normalized["nodes"]), 4)
        self.assertEqual(len(normalized["edges"]), 3)
        self.assertEqual(
            sum(node["kind"] == "turn" for node in normalized["nodes"]), 2
        )
        self.assertIn("node_0001", nodes)
        self.assertIn("node_0002", nodes)
        self.assertEqual(normalized["edges"][0]["id"], "edge_00001")
        self.assertTrue(all(
            edge["source_edge_id"] == "edge_00001"
            for edge in normalized["edges"]
        ))
        self.assertEqual(
            normalized["source"]["normalization"],
            "preserve_ids_explicit_turn_nodes_v3",
        )
        self.assertEqual(
            normalized["source"]["geometry_sha256"],
            floorplan_graph_geometry_sha256(normalized),
        )
        for edge in normalized["edges"]:
            self.assertEqual(len(edge["points"]), 2)
            self.assertEqual(
                edge["points"][0],
                [nodes[edge["from"]]["x"], nodes[edge["from"]]["y"]],
            )
            self.assertEqual(
                edge["points"][1],
                [nodes[edge["to"]]["x"], nodes[edge["to"]]["y"]],
            )
        validation = validate_floorplan_graph(normalized)
        self.assertEqual(validation["endpoint_mismatches"], 0)
        self.assertEqual(validation["invalid_edges"], 0)

    def test_validation_rejects_unsaved_endpoint_geometry(self) -> None:
        graph = {
            "schema_version": SCHEMA_VERSION,
            "map_id": "manual",
            "nodes": [
                {"id": "a", "kind": "manual", "x": 0, "y": 0},
                {"id": "b", "kind": "manual", "x": 10, "y": 0},
            ],
            "edges": [{
                "id": "edge",
                "from": "a",
                "to": "b",
                "points": [[1, 0], [10, 0]],
            }],
        }
        with self.assertRaisesRegex(ValueError, "endpoint mismatches"):
            validate_floorplan_graph(graph)

    def test_builds_exportable_graph_from_support_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            support = np.zeros((128, 128), dtype=np.uint8)
            support[56:72, 12:116] = 255
            support[12:116, 56:72] = 255
            obstacle = np.zeros_like(support)
            Image.fromarray(support).save(root / "support.png")
            Image.fromarray(obstacle).save(root / "obstacle.png")
            metadata = {
                "map_id": "synthetic_cross",
                "width": 128,
                "height": 128,
                "meters_per_pixel": 0.1,
                "support_mask_file": "support.png",
                "obstacle_mask_file": "obstacle.png",
            }
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            graph = build_floorplan_graph(
                metadata_path,
                minimum_edge_meters=0.5,
                graph_scale_pixels=4,
            )

            self.assertEqual(graph["schema_version"], SCHEMA_VERSION)
            self.assertEqual(graph["map_id"], "synthetic_cross")
            self.assertGreaterEqual(len(graph["nodes"]), 5)
            self.assertGreaterEqual(len(graph["edges"]), 4)
            node_ids = {node["id"] for node in graph["nodes"]}
            for edge in graph["edges"]:
                self.assertIn(edge["from"], node_ids)
                self.assertIn(edge["to"], node_ids)
                self.assertEqual(len(edge["points"]), 2)
                self.assertGreater(edge["length_meters"], 0)
                from_node = next(node for node in graph["nodes"] if node["id"] == edge["from"])
                to_node = next(node for node in graph["nodes"] if node["id"] == edge["to"])
                self.assertEqual(edge["points"][0], [from_node["x"], from_node["y"]])
                self.assertEqual(edge["points"][1], [to_node["x"], to_node["y"]])
            self.assertEqual(
                graph["validation"]["node_count"], len(graph["nodes"])
            )
            self.assertEqual(
                graph["validation"]["edge_count"], len(graph["edges"])
            )


if __name__ == "__main__":
    unittest.main()
