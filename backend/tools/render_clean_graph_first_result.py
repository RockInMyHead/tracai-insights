#!/usr/bin/env python3
"""Render only the Kerama plan and graph-first trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "public/floorplans/kerama-marazzi-2025.png"
ANALYSIS = (
    ROOT
    / "artifacts/golden5-deployed-topology-fix-20260728"
    / "analysis-graph-first.json"
)
OUTPUT = (
    ROOT
    / "artifacts/golden5-deployed-topology-fix-20260728"
    / "03-clean-plan-and-graph-first-trajectory.png"
)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: str,
    width: int,
    dash: float = 24.0,
    gap: float = 15.0,
) -> None:
    for left, right in zip(points, points[1:]):
        dx, dy = right[0] - left[0], right[1] - left[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        offset = 0.0
        while offset < length:
            start = offset
            stop = min(length, offset + dash)
            draw.line(
                (
                    left[0] + dx * start / length,
                    left[1] + dy * start / length,
                    left[0] + dx * stop / length,
                    left[1] + dy * stop / length,
                ),
                fill=fill,
                width=width,
            )
            offset += dash + gap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    trajectory = payload["graph_first_trajectory"]
    segments = payload["graph_first_segments"]
    plan = Image.open(PLAN).convert("RGB")
    draw = ImageDraw.Draw(plan)

    for segment in segments:
        start = int(segment["start_index"])
        end = int(segment["end_index"])
        points = [
            (float(value[0]), float(value[1]))
            for value in trajectory[start:end + 1]
        ]
        if len(points) < 2:
            continue
        if segment["status"] == "confirmed":
            draw.line(
                points,
                fill="#0877ff",
                width=18,
                joint="curve",
            )
        else:
            dashed_line(
                draw,
                points,
                fill="#ff8a00",
                width=18,
            )

    start_x, start_y = trajectory[0]
    radius = 22
    draw.ellipse(
        (
            start_x - radius,
            start_y - radius,
            start_x + radius,
            start_y + radius,
        ),
        fill="#ffd21f",
        outline="#111827",
        width=6,
    )
    uncertainty = payload.get("graph_first_uncertainty") or {}
    marker = uncertainty.get("marker")
    if marker:
        radius = 22
        draw.ellipse(
            (
                marker[0] - radius,
                marker[1] - radius,
                marker[0] + radius,
                marker[1] + radius,
            ),
            fill="#ff9f0a",
            outline="#111827",
            width=6,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan.save(args.output, optimize=True)
    print(args.output)
    if trajectory:
        xs = [float(point[0]) for point in trajectory]
        ys = [float(point[1]) for point in trajectory]
        padding = 180
        crop = (
            max(0, int(min(xs) - padding)),
            max(0, int(min(ys) - padding)),
            min(plan.width, int(max(xs) + padding)),
            min(plan.height, int(max(ys) + padding)),
        )
        zoom = plan.crop(crop)
        if zoom.width > 0 and zoom.height > 0:
            target_width = 1800
            target_height = max(
                1, round(zoom.height * target_width / zoom.width)
            )
            zoom = zoom.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )
            zoom_output = args.output.with_name(
                f"{args.output.stem}-zoom{args.output.suffix}"
            )
            zoom.save(zoom_output, optimize=True)
            print(zoom_output)


if __name__ == "__main__":
    main()
