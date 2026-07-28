import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backend.floorplan_graph import SCHEMA_VERSION, build_floorplan_graph


class FloorplanGraphTests(unittest.TestCase):
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
                self.assertGreaterEqual(len(edge["points"]), 2)
                self.assertGreater(edge["length_meters"], 0)
            self.assertEqual(
                graph["validation"]["node_count"], len(graph["nodes"])
            )
            self.assertEqual(
                graph["validation"]["edge_count"], len(graph["edges"])
            )


if __name__ == "__main__":
    unittest.main()
