#!/usr/bin/env python3
"""Render authored graph turns responsible for the five-minute gate failure."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "public/floorplans/kerama-marazzi-2025.png"
GRAPH = ROOT / "backend/assets/floorplans/kerama_marazzi_2025.topology-graph.v1.json"
ANALYSIS = ROOT / "artifacts/golden5-preserved-graph-v33-20260728/analysis.json"
OUTPUT = ROOT / "artifacts/golden5-preserved-graph-v33-20260728"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", size
    )


def selected_route() -> tuple[np.ndarray, list[str]]:
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    attempts = analysis["floorplan_constraint"]["topology_recovery_diagnostics"]
    edge_ids = attempts[2]["competing_branches"]["selected_edge_ids"]
    edges = {edge["id"]: edge for edge in graph["edges"]}
    current = "node_0008"
    points: list[list[float]] = []
    for edge_id in edge_ids:
        edge = edges[edge_id]
        if edge["from"] == current:
            following = edge["to"]
            pair = edge["points"]
        elif edge["to"] == current:
            following = edge["from"]
            pair = list(reversed(edge["points"]))
        else:
            raise ValueError(f"Disconnected route at {edge_id}")
        if not points:
            points.append(pair[0])
        points.append(pair[-1])
        current = following
    return np.asarray(points, dtype=np.float64), edge_ids


def turns(points: np.ndarray) -> list[tuple[int, float, float]]:
    vectors = np.diff(points, axis=0)
    headings = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    angles = (np.diff(headings) + 180.0) % 360.0 - 180.0
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(vectors, axis=1))]
    return [
        (index, float(angle), float(distances[index]))
        for index, angle in enumerate(angles, start=1)
        if abs(float(angle)) >= 12.0
    ]


def render(
    points: np.ndarray,
    events: list[tuple[int, float, float]],
    crop: tuple[int, int, int, int],
    output: Path,
    title: str,
    *,
    scale: float,
) -> None:
    plan = Image.open(PLAN).convert("RGB").crop(crop)
    plan = plan.resize(
        (round(plan.width * scale), round(plan.height * scale)),
        Image.Resampling.LANCZOS,
    )
    header = 150
    canvas = Image.new("RGB", (plan.width, plan.height + header), "white")
    canvas.paste(plan, (0, header))
    draw = ImageDraw.Draw(canvas)

    def p(value: np.ndarray) -> tuple[float, float]:
        return (
            (float(value[0]) - crop[0]) * scale,
            (float(value[1]) - crop[1]) * scale + header,
        )

    draw.line(
        [p(point) for point in points],
        fill="#2563eb",
        width=max(8, round(9 * scale)),
        joint="curve",
    )
    for event_number, (index, angle, _) in enumerate(events, start=1):
        x, y = p(points[index])
        colour = "#16a34a" if angle > 0 else "#dc2626"
        radius = max(13, round(14 * scale))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=colour,
            outline="white",
            width=max(3, round(4 * scale)),
        )
        draw.text(
            (x + radius + 3, y - radius - 8),
            f"{event_number}: {angle:+.0f}°",
            font=font(max(18, round(18 * scale))),
            fill=colour,
            stroke_width=max(2, round(3 * scale)),
            stroke_fill="white",
        )
    draw.text((24, 14), title, font=font(34), fill="#0f172a")
    draw.text(
        (24, 66),
        "Blue = exact production graph route; green = left; red = right",
        font=font(24),
        fill="#334155",
    )
    draw.text(
        (24, 104),
        "R3 has 6 stable events; this graph route contains 13 >=12° vertices",
        font=font(24),
        fill="#7c2d12",
    )
    canvas.save(output, optimize=True)


def main() -> None:
    points, _ = selected_route()
    events = turns(points)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    render(
        points,
        events,
        (620, 500, 2320, 850),
        OUTPUT / "01-full-route-turn-events.png",
        "Five-minute route: authored graph turn events",
        scale=1.1,
    )
    render(
        points,
        events,
        (900, 520, 1440, 760),
        OUTPUT / "02-extra-turns-zoom.png",
        "Root cause: alternating bends inside one corridor section",
        scale=2.2,
    )
    print(OUTPUT / "01-full-route-turn-events.png")
    print(OUTPUT / "02-extra-turns-zoom.png")


if __name__ == "__main__":
    main()
