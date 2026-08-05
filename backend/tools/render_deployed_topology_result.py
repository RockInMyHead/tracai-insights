#!/usr/bin/env python3
"""Render the deployed five-minute R3 result against the production graph."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "public/floorplans/kerama-marazzi-2025.png"
GRAPH = ROOT / "backend/assets/floorplans/kerama_marazzi_2025.topology-graph.v1.json"
ANALYSIS = ROOT / "artifacts/golden5-deployed-topology-fix-20260728/analysis.json"
OUTPUT = ROOT / "artifacts/golden5-deployed-topology-fix-20260728"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def anchored_r3(payload: dict, graph: dict) -> np.ndarray:
    points = np.asarray(payload["trajectory"], dtype=np.float64)[:, :2]
    points -= points[0]
    method = str(payload.get("method") or "").lower()
    quality = payload.get("trajectory_quality") or {}
    projection = quality.get("projection") if isinstance(quality, dict) else {}
    convention = (
        str(projection.get("plan_coordinate_convention"))
        if isinstance(projection, dict)
        and projection.get("plan_coordinate_convention")
        else ("x_forward_y_left_z_up" if method.startswith("r3") else "x_right_y_down")
    )
    if convention == "x_forward_y_left_z_up":
        points[:, 1] *= -1.0
    # Robust early heading: the first metre is noisy, so use the first point
    # whose displacement reaches 5% of the total span.
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


def render(
    graph: dict,
    r3: np.ndarray,
    diagnostics: dict,
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
    header = 210
    canvas = Image.new("RGB", (plan.width, plan.height + header), "white")
    canvas.paste(plan, (0, header))
    draw = ImageDraw.Draw(canvas)

    def point(value: list[float] | np.ndarray) -> tuple[float, float]:
        return (
            (float(value[0]) - crop[0]) * scale,
            (float(value[1]) - crop[1]) * scale + header,
        )

    for edge in graph["edges"]:
        if edge.get("enabled", True):
            draw.line(
                [point(value) for value in edge["points"]],
                fill="#2563eb",
                width=max(3, round(6 * scale)),
                joint="curve",
            )
    for node in graph["nodes"]:
        if not node.get("enabled", True):
            continue
        x, y = point([node["x"], node["y"]])
        radius = max(3, round(5 * scale))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="#f59e0b" if node.get("kind") == "turn" else "#ef4444",
            outline="white",
            width=max(1, round(2 * scale)),
        )

    draw.line(
        [point(value) for value in r3],
        fill="#06b6d4",
        width=max(5, round(10 * scale)),
        joint="curve",
    )
    sx, sy = point(r3[0])
    radius = max(9, round(14 * scale))
    draw.ellipse(
        (sx - radius, sy - radius, sx + radius, sy + radius),
        fill="#fbbf24",
        outline="#111827",
        width=max(2, round(4 * scale)),
    )

    matching = diagnostics.get("authoritative_graph_matching") or {}
    draw.text((24, 14), title, font=font(34, True), fill="#0f172a")
    draw.text(
        (24, 62),
        "Cyan: R3 anchored by start + heading   Blue: production graph   Yellow: START",
        font=font(22),
        fill="#334155",
    )
    draw.text(
        (24, 105),
        (
            f"REJECTED: {diagnostics.get('reason')} | "
            f"graph attempts {matching.get('attempted', 0)}, accepted "
            f"{matching.get('accepted', 0)}"
        ),
        font=font(24, True),
        fill="#b91c1c",
    )
    draw.text(
        (24, 150),
        "R3 branches cannot form one connected edge sequence; no map_trajectory published",
        font=font(22, True),
        fill="#7f1d1d",
    )
    canvas.save(output, optimize=True)


def main() -> None:
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    saved = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    payload = saved.get("analysis_result", saved)
    diagnostics = payload["floorplan_constraint"]
    r3 = anchored_r3(payload, graph)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    render(
        graph,
        r3,
        diagnostics,
        (0, 0, int(graph["width"]), int(graph["height"])),
        OUTPUT / "01-full-result-overlay.png",
        "Deployed 5-minute test: R3 vs production corridor graph",
        scale=0.38,
    )
    render(
        graph,
        r3,
        diagnostics,
        (450, 430, 2700, 1150),
        OUTPUT / "02-start-and-route-detail.png",
        "Start area and first graph-matching conflict",
        scale=1.0,
    )
    print(OUTPUT / "01-full-result-overlay.png")
    print(OUTPUT / "02-start-and-route-detail.png")


if __name__ == "__main__":
    main()
