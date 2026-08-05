#!/usr/bin/env python3
"""Render early R3 kinks, graph node IDs, and oracle route for inspection."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.floorplan_constraints import _resample_polyline, _turn_events  # noqa: E402


PLAN = ROOT / "public/floorplans/kerama-marazzi-2025.png"
GRAPH = ROOT / "backend/assets/floorplans/kerama_marazzi_2025.topology-graph.v1.json"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def anchored_r3(payload: dict[str, Any], graph: dict[str, Any]) -> np.ndarray:
    points = np.asarray(payload["trajectory"], dtype=np.float64)[:, :2]
    points -= points[0]
    quality = payload.get("trajectory_quality") or {}
    projection = quality.get("projection") if isinstance(quality, dict) else {}
    projection = projection if isinstance(projection, dict) else {}
    convention = str(projection.get("plan_coordinate_convention") or "")
    if convention == "x_forward_y_left_z_up":
        points[:, 1] *= -1.0
    displacement = np.linalg.norm(points, axis=1)
    threshold = max(2.0, float(np.max(displacement)) * 0.05)
    heading_index = int(np.argmax(displacement >= threshold))
    if heading_index <= 0:
        heading_index = min(len(points) - 1, 20)
    source_heading = math.atan2(points[heading_index, 1], points[heading_index, 0])
    metadata = graph.get("metadata") or {}
    start = np.asarray(
        metadata.get("default_anchor_reference_pixels") or [2219.862, 677.483],
        dtype=np.float64,
    )
    direction = np.asarray(
        metadata.get("default_anchor_direction_pixels") or [1997.346, 699.941],
        dtype=np.float64,
    )
    target_heading = math.atan2(direction[1] - start[1], direction[0] - start[0])
    angle = target_heading - source_heading
    rotation = np.asarray([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    meters_per_pixel = float(
        metadata.get("meters_per_pixel") or 0.049629166698546515
    )
    return points @ rotation.T / meters_per_pixel + start


def first_directed(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if isinstance(payload.get("directed_edge_search"), dict):
            return payload["directed_edge_search"]
        for value in payload.values():
            found = first_directed(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = first_directed(value)
            if found:
                return found
    return {}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    saved = json.loads(args.analysis.read_text(encoding="utf-8"))
    payload = saved.get("analysis_result", saved)
    oracle_payload = json.loads(args.oracle.read_text(encoding="utf-8"))
    directed = first_directed(oracle_payload)
    r3 = anchored_r3(payload, graph)
    events = _turn_events(
        _resample_polyline(r3, np.linspace(0.0, 1.0, 128)),
        samples=128,
    )[:4]
    event_points = _resample_polyline(
        r3,
        np.asarray([float(item[0]) for item in events], dtype=np.float64),
    )
    route = np.asarray(
        directed.get("best_terminal_trajectory") or [],
        dtype=np.float64,
    )
    focus = event_points if len(event_points) else r3[:1]
    min_xy = np.min(focus, axis=0) - np.asarray([260.0, 220.0])
    max_xy = np.max(focus, axis=0) + np.asarray([260.0, 220.0])
    crop = (
        max(0, int(min_xy[0])),
        max(0, int(min_xy[1])),
        min(int(graph["width"]), int(max_xy[0])),
        min(int(graph["height"]), int(max_xy[1])),
    )
    scale = 2.0
    header = 170
    plan = Image.open(PLAN).convert("RGB").crop(crop)
    plan = plan.resize(
        (round(plan.width * scale), round(plan.height * scale)),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (plan.width, plan.height + header), "white")
    canvas.paste(plan, (0, header))
    draw = ImageDraw.Draw(canvas)

    def pt(value: Any) -> tuple[float, float]:
        return (
            (float(value[0]) - crop[0]) * scale,
            (float(value[1]) - crop[1]) * scale + header,
        )

    for edge in graph["edges"]:
        if not edge.get("enabled", True):
            continue
        points = [pt(value) for value in edge.get("points", [])]
        if len(points) >= 2:
            draw.line(points, fill="#2563eb", width=3)
            mid = points[len(points) // 2]
            if crop[0] <= edge["points"][0][0] <= crop[2]:
                draw.text(mid, str(edge.get("id", "")), font=font(12), fill="#1d4ed8")

    for node in graph["nodes"]:
        if not node.get("enabled", True):
            continue
        x = float(node["x"])
        y = float(node["y"])
        if not (crop[0] <= x <= crop[2] and crop[1] <= y <= crop[3]):
            continue
        px, py = pt([x, y])
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill="#ef4444", outline="white")
        draw.text((px + 7, py - 7), str(node.get("id", "")), font=font(13, True), fill="#111827")

    draw.line([pt(value) for value in r3], fill="#06b6d4", width=7, joint="curve")
    if len(route) >= 2:
        draw.line([pt(value) for value in route], fill="#dc2626", width=5, joint="curve")

    for index, (fraction, _angle) in enumerate(events):
        ex, ey = pt(event_points[index])
        draw.ellipse((ex - 11, ey - 11, ex + 11, ey + 11), fill="#facc15", outline="#111827", width=3)
        draw.text((ex + 14, ey + 10), f"R3 event {index} f={fraction:.3f}", font=font(18, True), fill="#111827")

    draw.text((22, 18), "Early R3 kink sanity check", font=font(30, True), fill="#0f172a")
    draw.text((22, 62), "Cyan: anchored R3 | Red: oracle best terminal | Yellow: first turn events | Labels: graph IDs", font=font(18), fill="#334155")
    progress = directed.get("best_terminal_event_progress") or {}
    draw.text((22, 102), json.dumps({
        "reason": directed.get("reason"),
        "errors_0_3": [
            item.get("abs_error")
            for item in (progress.get("per_event") or [])[:4]
        ],
    }, ensure_ascii=False), font=font(16), fill="#7f1d1d")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    print(args.output)


if __name__ == "__main__":
    main()
