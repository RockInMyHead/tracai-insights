"""Build an editable topological corridor graph from a walkable support mask."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize


SCHEMA_VERSION = "trackai.floorplan_graph.v1"


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
        edge = {
            "id": f"edge_{len(edges) + 1:05d}",
            "from": nodes[start_index]["id"],
            "to": nodes[end_index]["id"],
            "points": points,
            "length_meters": round(length_meters, 4),
            "minimum_width_meters": round(min(widths), 3) if widths else None,
            "median_width_meters": round(float(np.median(widths)), 3)
            if widths else None,
            "bidirectional": True,
            "enabled": True,
        }
        edges.append(edge)
        nodes[start_index]["degree"] += 1
        nodes[end_index]["degree"] += 1

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
