#!/usr/bin/env python3
"""Render the exact backend production graph over the Kerama drawing."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "public/floorplans/kerama-marazzi-2025.png"
GRAPH = ROOT / "backend/assets/floorplans/kerama_marazzi_2025.topology-graph.v1.json"
OUTPUT = ROOT / "artifacts/production-graph-overlay-20260728"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", size
    )


def render(
    graph: dict,
    crop: tuple[int, int, int, int],
    output: Path,
    *,
    scale: float,
    title: str,
    labels: bool,
) -> None:
    plan = Image.open(PLAN).convert("RGB").crop(crop)
    plan = plan.resize(
        (round(plan.width * scale), round(plan.height * scale)),
        Image.Resampling.LANCZOS,
    )
    header = round(120 * max(1.0, scale))
    canvas = Image.new("RGB", (plan.width, plan.height + header), "white")
    canvas.paste(plan, (0, header))
    draw = ImageDraw.Draw(canvas)

    def point(x: float, y: float) -> tuple[float, float]:
        return (
            (float(x) - crop[0]) * scale,
            (float(y) - crop[1]) * scale + header,
        )

    for edge in graph["edges"]:
        if not edge.get("enabled", True):
            continue
        coordinates = [point(value[0], value[1]) for value in edge["points"]]
        draw.line(
            coordinates,
            fill="#2563eb",
            width=max(3, round(5 * scale)),
            joint="curve",
        )
    for node in graph["nodes"]:
        if not node.get("enabled", True):
            continue
        x, y = point(node["x"], node["y"])
        is_turn = node.get("kind") == "turn"
        radius = max(3, round((4 if is_turn else 6) * scale))
        colour = "#f59e0b" if is_turn else (
            "#ef4444" if node.get("kind") == "junction" else "#10b981"
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=colour,
            outline="white",
            width=max(1, round(2 * scale)),
        )
        if labels and not is_turn:
            draw.text(
                (x + radius + 2, y - radius - 2),
                str(node["id"]),
                font=font(max(14, round(14 * scale))),
                fill="#0f172a",
                stroke_width=max(2, round(2 * scale)),
                stroke_fill="white",
            )

    draw.text(
        (24, 12),
        title,
        font=font(round(31 * max(1.0, scale))),
        fill="#0f172a",
    )
    draw.text(
        (24, round(57 * max(1.0, scale))),
        (
            f"Blue: {len(graph['edges'])} production segments   "
            f"Red: junctions   Green: endpoints/manual   "
            f"Orange: explicit turn-nodes"
        ),
        font=font(round(20 * max(1.0, scale))),
        fill="#334155",
    )
    canvas.save(output, optimize=True)


def main() -> None:
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    render(
        graph,
        (0, 0, int(graph["width"]), int(graph["height"])),
        OUTPUT / "01-production-graph-full.png",
        scale=0.38,
        title="Exact graph currently loaded by backend over the Kerama drawing",
        labels=False,
    )
    render(
        graph,
        (520, 480, 2650, 1100),
        OUTPUT / "02-production-graph-start-area.png",
        scale=1.0,
        title="Production graph detail: start and left-side corridor network",
        labels=True,
    )
    print(OUTPUT / "01-production-graph-full.png")
    print(OUTPUT / "02-production-graph-start-area.png")


if __name__ == "__main__":
    main()
