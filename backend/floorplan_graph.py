"""Build an editable topological corridor graph from a walkable support mask."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize


SCHEMA_VERSION = "trackai.floorplan_graph.v1"


def floorplan_graph_geometry_sha256(graph: dict[str, Any]) -> str:
    """Hash the exact enabled production geometry and stable identifiers."""
    payload = {
        "schema_version": graph.get("schema_version"),
        "map_id": graph.get("map_id"),
        "nodes": [
            {
                "id": str(node.get("id")),
                "kind": str(node.get("kind")),
                "x": float(node.get("x")),
                "y": float(node.get("y")),
                "enabled": bool(node.get("enabled", True)),
            }
            for node in graph.get("nodes", [])
        ],
        "edges": [
            {
                "id": str(edge.get("id")),
                "source_edge_id": str(
                    edge.get("source_edge_id") or edge.get("id")
                ),
                "from": str(edge.get("from")),
                "to": str(edge.get("to")),
                "points": [
                    [float(point[0]), float(point[1])]
                    for point in edge.get("points", [])
                ],
                "bidirectional": bool(edge.get("bidirectional", True)),
                "enabled": bool(edge.get("enabled", True)),
            }
            for edge in graph.get("edges", [])
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_floorplan_graph(graph: dict[str, Any]) -> dict[str, int]:
    """Validate an editor payload before it can replace production geometry."""
    if graph.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported floorplan graph schema")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Floorplan graph nodes and edges must be arrays")
    node_ids = [str(node.get("id") or "") for node in nodes]
    edge_ids = [str(edge.get("id") or "") for edge in edges]
    if any(not value for value in node_ids) or len(node_ids) != len(set(node_ids)):
        raise ValueError("Floorplan graph contains missing or duplicate node IDs")
    if any(not value for value in edge_ids) or len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Floorplan graph contains missing or duplicate edge IDs")
    node_by_id = {str(node["id"]): node for node in nodes}
    endpoint_mismatches = 0
    invalid_edges = 0
    for edge in edges:
        start = node_by_id.get(str(edge.get("from") or ""))
        end = node_by_id.get(str(edge.get("to") or ""))
        points = edge.get("points")
        if (
            start is None
            or end is None
            or not isinstance(points, list)
            or len(points) != 2
        ):
            invalid_edges += 1
            continue
        expected = (
            [float(start["x"]), float(start["y"])],
            [float(end["x"]), float(end["y"])],
        )
        actual = (
            [float(points[0][0]), float(points[0][1])],
            [float(points[1][0]), float(points[1][1])],
        )
        if any(
            math.hypot(
                actual[index][0] - expected[index][0],
                actual[index][1] - expected[index][1],
            ) > 1e-3
            for index in (0, 1)
        ):
            endpoint_mismatches += 1
    if invalid_edges:
        raise ValueError(f"Floorplan graph contains {invalid_edges} invalid edges")
    if endpoint_mismatches:
        raise ValueError(
            f"Floorplan graph contains {endpoint_mismatches} endpoint mismatches"
        )
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "invalid_edges": invalid_edges,
        "endpoint_mismatches": endpoint_mismatches,
    }


def normalize_floorplan_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Make bends explicit without changing authored node or edge identities."""
    result = deepcopy(graph)
    nodes = [dict(node) for node in result.get("nodes", [])]
    node_by_id = {str(node["id"]): node for node in nodes}
    existing_node_ids = set(node_by_id)
    existing_edge_ids = {
        str(edge.get("id")) for edge in result.get("edges", [])
    }
    edges: list[dict[str, Any]] = []

    for source_edge in result.get("edges", []):
        source_edge_id = str(source_edge.get("id") or "")
        start_id = str(source_edge.get("from") or "")
        end_id = str(source_edge.get("to") or "")
        if (
            not source_edge_id
            or start_id not in node_by_id
            or end_id not in node_by_id
        ):
            continue
        start_node = node_by_id[start_id]
        end_node = node_by_id[end_id]
        start_point = [float(start_node["x"]), float(start_node["y"])]
        end_point = [float(end_node["x"]), float(end_node["y"])]
        source_points = [
            [float(point[0]), float(point[1])]
            for point in source_edge.get("points", [])
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if len(source_points) < 2:
            source_points = [start_point, end_point]
        else:
            # Edge connectivity is authoritative at its endpoints. Interior
            # authored geometry is preserved exactly and promoted to nodes.
            source_points[0] = start_point
            source_points[-1] = end_point
        source_points = [
            point for index, point in enumerate(source_points)
            if index == 0 or point != source_points[index - 1]
        ]
        if len(source_points) < 2:
            continue

        chain_points = [source_points[0]]
        source_point_indices = [0]
        for point_index in range(1, len(source_points) - 1):
            left = np.asarray(source_points[point_index - 1], dtype=np.float64)
            centre = np.asarray(source_points[point_index], dtype=np.float64)
            right = np.asarray(source_points[point_index + 1], dtype=np.float64)
            before = centre - left
            after = right - centre
            denominator = max(
                float(np.linalg.norm(before) * np.linalg.norm(after)), 1e-9
            )
            cosine = float(np.clip(np.dot(before, after) / denominator, -1, 1))
            bend_degrees = math.degrees(math.acos(cosine))
            if bend_degrees >= 1.0:
                chain_points.append(source_points[point_index])
                source_point_indices.append(point_index)
        chain_points.append(source_points[-1])
        source_point_indices.append(len(source_points) - 1)

        chain_node_ids = [start_id]
        for point, source_point_index in zip(
            chain_points[1:-1], source_point_indices[1:-1]
        ):
            turn_id = (
                f"turn__{source_edge_id}__p{source_point_index:03d}"
            )
            suffix = 1
            while turn_id in existing_node_ids:
                suffix += 1
                turn_id = (
                    f"turn__{source_edge_id}__p{source_point_index:03d}"
                    f"__{suffix}"
                )
            turn_node = {
                "id": turn_id,
                "kind": "turn",
                "x": point[0],
                "y": point[1],
                "degree": 0,
                "enabled": bool(source_edge.get("enabled", True)),
                "source_edge_id": source_edge_id,
                "source_point_index": source_point_index,
            }
            nodes.append(turn_node)
            node_by_id[turn_id] = turn_node
            existing_node_ids.add(turn_id)
            chain_node_ids.append(turn_id)
        chain_node_ids.append(end_id)

        for segment_index, (left, right) in enumerate(
            zip(chain_points, chain_points[1:])
        ):
            segment_pixels = math.hypot(
                right[0] - left[0], right[1] - left[1]
            )
            if segment_pixels <= 1e-9:
                continue
            edge_id = (
                source_edge_id
                if segment_index == 0
                else f"{source_edge_id}__s{segment_index + 1:03d}"
            )
            suffix = 1
            while (
                edge_id in existing_edge_ids
                and edge_id != source_edge_id
            ):
                suffix += 1
                edge_id = (
                    f"{source_edge_id}__s{segment_index + 1:03d}"
                    f"__{suffix}"
                )
            existing_edge_ids.add(edge_id)
            edge = dict(source_edge)
            edge.update({
                "id": edge_id,
                "source_edge_id": source_edge_id,
                "source_segment_index": segment_index,
                "from": chain_node_ids[segment_index],
                "to": chain_node_ids[segment_index + 1],
                "points": [left, right],
                "length_meters": round(
                    segment_pixels * float(result["meters_per_pixel"]), 4
                ),
            })
            edges.append(edge)

    degrees: dict[str, int] = {}
    for edge in edges:
        if not edge.get("enabled", True):
            continue
        degrees[edge["from"]] = degrees.get(edge["from"], 0) + 1
        degrees[edge["to"]] = degrees.get(edge["to"], 0) + 1
    for node in nodes:
        node["degree"] = degrees.get(str(node["id"]), 0)

    result["nodes"] = nodes
    result["edges"] = edges
    result["validation"] = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "disabled_nodes": sum(not node.get("enabled", True) for node in nodes),
        "disabled_edges": sum(not edge.get("enabled", True) for edge in edges),
    }
    result.setdefault("source", {})["normalization"] = (
        "preserve_ids_explicit_turn_nodes_v3"
    )
    result["source"]["geometry_sha256"] = floorplan_graph_geometry_sha256(
        result
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _simplify(points: list[tuple[int, int]], scale: int) -> list[list[float]]:
    if len(points) <= 2:
        return [[float(x * scale), float(y * scale)] for x, y in points]
    curve = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    simplified = cv2.approxPolyDP(curve, epsilon=1.25, closed=False)
    return [
        [round(float(point[0]) * scale, 3), round(float(point[1]) * scale, 3)]
        for point in simplified[:, 0, :]
    ]


def _polyline_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    array = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def build_floorplan_graph(
    metadata_path: Path,
    *,
    minimum_edge_meters: float = 1.5,
    graph_scale_pixels: int = 8,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    root = metadata_path.parent
    support_path = root / metadata["support_mask_file"]
    obstacle_path = root / metadata["obstacle_mask_file"]
    support = np.asarray(Image.open(support_path).convert("L")) > 127
    obstacle = np.asarray(Image.open(obstacle_path).convert("L")) > 127
    walkable = support & ~obstacle

    scale = max(2, int(graph_scale_pixels))
    reduced_width = int(math.ceil(walkable.shape[1] / scale))
    reduced_height = int(math.ceil(walkable.shape[0] / scale))
    reduced = cv2.resize(
        walkable.astype(np.uint8),
        (reduced_width, reduced_height),
        interpolation=cv2.INTER_AREA,
    ) >= 0.35
    reduced = ndimage.binary_closing(reduced, iterations=1)
    reduced = ndimage.binary_opening(reduced, iterations=1)
    skeleton = skeletonize(reduced)
    clearance = ndimage.distance_transform_edt(reduced) * scale

    kernel = np.ones((3, 3), dtype=np.uint8)
    degree = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode="constant")
    degree = degree - skeleton.astype(np.uint8)
    terminals = skeleton & (degree == 1)
    junction_pixels = skeleton & (degree >= 3)
    junction_labels, junction_count = ndimage.label(
        junction_pixels, structure=np.ones((3, 3), dtype=np.uint8)
    )

    node_cells: list[tuple[int, int, str]] = []
    claimed: dict[tuple[int, int], int] = {}
    for label in range(1, junction_count + 1):
        ys, xs = np.where(junction_labels == label)
        if not len(xs):
            continue
        x = int(round(float(np.mean(xs))))
        y = int(round(float(np.mean(ys))))
        node_index = len(node_cells)
        node_cells.append((x, y, "junction"))
        for px, py in zip(xs, ys):
            claimed[(int(px), int(py))] = node_index
    for y, x in np.argwhere(terminals):
        claimed[(int(x), int(y))] = len(node_cells)
        node_cells.append((int(x), int(y), "endpoint"))

    neighbours = (
        (-1, -1), (0, -1), (1, -1), (-1, 0),
        (1, 0), (-1, 1), (0, 1), (1, 1),
    )
    skeleton_cells = {
        (int(x), int(y)) for y, x in np.argwhere(skeleton)
    }

    def adjacent(cell: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = cell
        return [
            (x + dx, y + dy)
            for dx, dy in neighbours
            if (x + dx, y + dy) in skeleton_cells
        ]

    visited_segments: set[frozenset[tuple[int, int]]] = set()
    raw_edges: list[tuple[int, int, list[tuple[int, int]]]] = []
    for start_cell, start_node in list(claimed.items()):
        for next_cell in adjacent(start_cell):
            segment = frozenset((start_cell, next_cell))
            if segment in visited_segments:
                continue
            visited_segments.add(segment)
            path = [start_cell, next_cell]
            previous, current = start_cell, next_cell
            while current not in claimed or claimed[current] == start_node:
                candidates = [
                    cell for cell in adjacent(current)
                    if cell != previous
                    and frozenset((current, cell)) not in visited_segments
                ]
                if not candidates:
                    break
                next_step = min(
                    candidates,
                    key=lambda cell: (
                        0 if cell in claimed else 1,
                        cell[1],
                        cell[0],
                    ),
                )
                visited_segments.add(frozenset((current, next_step)))
                previous, current = current, next_step
                path.append(current)
            end_node = claimed.get(current)
            if end_node is None or end_node == start_node:
                continue
            raw_edges.append((start_node, end_node, path))

    meters_per_pixel = float(metadata["meters_per_pixel"])
    nodes = []
    for index, (x, y, kind) in enumerate(node_cells):
        nodes.append({
            "id": f"node_{index + 1:04d}",
            "kind": kind,
            "x": round(float(x * scale), 3),
            "y": round(float(y * scale), 3),
            "degree": 0,
            "enabled": True,
        })

    edges = []
    seen_pairs: set[tuple[int, int, tuple[tuple[float, float], ...]]] = set()
    for start_index, end_index, cells in raw_edges:
        points = _simplify(cells, scale)
        points[0] = [nodes[start_index]["x"], nodes[start_index]["y"]]
        points[-1] = [nodes[end_index]["x"], nodes[end_index]["y"]]
        points = [
            point for index, point in enumerate(points)
            if index == 0 or point != points[index - 1]
        ]
        if len(points) < 2:
            continue
        length_pixels = _polyline_length(points)
        length_meters = length_pixels * meters_per_pixel
        if length_meters < minimum_edge_meters:
            continue
        key = (
            min(start_index, end_index),
            max(start_index, end_index),
            tuple((point[0], point[1]) for point in points),
        )
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        widths = [
            float(clearance[y, x] * 2.0 * meters_per_pixel)
            for x, y in cells
            if 0 <= y < clearance.shape[0] and 0 <= x < clearance.shape[1]
        ]
        chain_node_ids = [nodes[start_index]["id"]]
        for point in points[1:-1]:
            turn_node = {
                "id": f"node_{len(nodes) + 1:04d}",
                "kind": "turn",
                "x": point[0],
                "y": point[1],
                "degree": 0,
                "enabled": True,
            }
            nodes.append(turn_node)
            chain_node_ids.append(turn_node["id"])
        chain_node_ids.append(nodes[end_index]["id"])

        minimum_width = round(min(widths), 3) if widths else None
        median_width = (
            round(float(np.median(widths)), 3) if widths else None
        )
        for segment_index, (start_point, end_point) in enumerate(
            zip(points, points[1:])
        ):
            segment_length_meters = (
                math.hypot(
                    end_point[0] - start_point[0],
                    end_point[1] - start_point[1],
                )
                * meters_per_pixel
            )
            if segment_length_meters <= 0:
                continue
            edge = {
                "id": f"edge_{len(edges) + 1:05d}",
                "from": chain_node_ids[segment_index],
                "to": chain_node_ids[segment_index + 1],
                "points": [start_point, end_point],
                "length_meters": round(segment_length_meters, 4),
                "minimum_width_meters": minimum_width,
                "median_width_meters": median_width,
                "bidirectional": True,
                "enabled": True,
            }
            edges.append(edge)

    degrees: dict[str, int] = {}
    for edge in edges:
        degrees[edge["from"]] = degrees.get(edge["from"], 0) + 1
        degrees[edge["to"]] = degrees.get(edge["to"], 0) + 1
    for node in nodes:
        node["degree"] = degrees.get(node["id"], 0)

    used_node_ids = {
        edge["from"] for edge in edges
    } | {
        edge["to"] for edge in edges
    }
    nodes = [node for node in nodes if node["id"] in used_node_ids]

    return {
        "schema_version": SCHEMA_VERSION,
        "map_id": metadata["map_id"],
        "coordinate_system": "plan_pixels_x_right_y_down",
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
        "meters_per_pixel": meters_per_pixel,
        "source": {
            "support_mask_file": support_path.name,
            "support_mask_sha256": _sha256(support_path),
            "obstacle_mask_file": obstacle_path.name,
            "obstacle_mask_sha256": _sha256(obstacle_path),
            "graph_scale_pixels": scale,
            "minimum_edge_meters": minimum_edge_meters,
        },
        "nodes": nodes,
        "edges": edges,
        "validation": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "disabled_nodes": 0,
            "disabled_edges": 0,
        },
    }
