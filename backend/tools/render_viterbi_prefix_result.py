#!/usr/bin/env python3
"""Render the honest Viterbi-prefix result: plan, marker and short branches."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "public/floorplans/kerama-marazzi-2025.png"
ANALYSIS = (
    ROOT / "artifacts/golden5-viterbi-prefix-v2-full-20260728/analysis.json"
)
OUTPUT = ROOT / "artifacts/golden5-viterbi-prefix-v2-full-20260728"


def dashed(
    draw: ImageDraw.ImageDraw,
    left: tuple[float, float],
    right: tuple[float, float],
    *,
    width: int,
    dash: float,
    gap: float,
) -> None:
    dx, dy = right[0] - left[0], right[1] - left[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1e-9)
    offset = 0.0
    while offset < length:
        stop = min(length, offset + dash)
        draw.line(
            (
                left[0] + dx * offset / length,
                left[1] + dy * offset / length,
                left[0] + dx * stop / length,
                left[1] + dy * stop / length,
            ),
            fill="#f59e0b",
            width=width,
        )
        offset += dash + gap


def overlay(plan: Image.Image, *, scale: float = 1.0) -> Image.Image:
    saved = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    payload = saved.get("analysis_result", saved)
    uncertainty = payload["graph_first_uncertainty"]
    marker = uncertainty["marker"]
    image = plan.convert("RGB")
    draw = ImageDraw.Draw(image)
    for edge in uncertainty.get("competing_next_edges", [])[:2]:
        left, right = edge["points"]
        dashed(
            draw,
            (left[0] * scale, left[1] * scale),
            (right[0] * scale, right[1] * scale),
            width=max(9, round(18 * scale)),
            dash=max(10, 22 * scale),
            gap=max(7, 13 * scale),
        )
    x, y = marker[0] * scale, marker[1] * scale
    radius = max(14, round(24 * scale))
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill="#fbbf24",
        outline="#111827",
        width=max(4, round(6 * scale)),
    )
    return image


def main() -> None:
    plan = Image.open(PLAN).convert("RGB")
    full = overlay(plan.copy())
    full.save(OUTPUT / "01-full-plan-result.png", optimize=True)

    crop = (1700, 450, 2500, 950)
    zoom = plan.crop(crop).resize((1600, 1000), Image.Resampling.LANCZOS)
    saved = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    payload = saved.get("analysis_result", saved)
    uncertainty = payload["graph_first_uncertainty"]
    draw = ImageDraw.Draw(zoom)
    factor = 2.0
    for edge in uncertainty.get("competing_next_edges", [])[:2]:
        values = [
            ((point[0] - crop[0]) * factor, (point[1] - crop[1]) * factor)
            for point in edge["points"]
        ]
        dashed(draw, values[0], values[1], width=24, dash=34, gap=20)
    marker = uncertainty["marker"]
    x = (marker[0] - crop[0]) * factor
    y = (marker[1] - crop[1]) * factor
    draw.ellipse(
        (x - 34, y - 34, x + 34, y + 34),
        fill="#fbbf24",
        outline="#111827",
        width=8,
    )
    zoom.save(OUTPUT / "02-start-uncertainty-zoom.png", optimize=True)
    print(OUTPUT / "01-full-plan-result.png")
    print(OUTPUT / "02-start-uncertainty-zoom.png")


if __name__ == "__main__":
    main()
