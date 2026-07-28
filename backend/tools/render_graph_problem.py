#!/usr/bin/env python3
"""Render the diagnosed Kerama topology conflict as a reviewable PNG."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "public/floorplans/kerama-marazzi-2025.png"
GRAPH = (
    ROOT
    / "backend/assets/floorplans/kerama_marazzi_2025.topology-graph.v1.json"
)
OUTPUT = (
    ROOT
    / "artifacts/golden5-directed-edge-debug-20260728/problem-node-0014.png"
)

CROP = (520, 500, 1180, 900)
SCALE = 2
HEADER = 210


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    locations = (
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/Library/Fonts") / name,
    )
    for location in locations:
        if location.exists():
            return ImageFont.truetype(str(location), size)
    return ImageFont.load_default()


def main() -> None:
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    edges = {edge["id"]: edge for edge in graph["edges"]}
    nodes = {node["id"]: node for node in graph["nodes"]}
    problem_node_id = "node_0014"
    dead_end_node_id = "node_0084"
    incident_edges = [
        edge for edge in graph["edges"]
        if edge.get("enabled", True)
        and problem_node_id in {edge.get("from"), edge.get("to")}
    ]
    plan = Image.open(PLAN).convert("RGB").crop(CROP)
    plan = plan.resize(
        (plan.width * SCALE, plan.height * SCALE),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (plan.width, plan.height + HEADER), "white")
    canvas.paste(plan, (0, HEADER))
    draw = ImageDraw.Draw(canvas)

    def point(value: list[float] | tuple[float, float]) -> tuple[float, float]:
        return (
            (float(value[0]) - CROP[0]) * SCALE,
            (float(value[1]) - CROP[1]) * SCALE + HEADER,
        )

    def other_node(edge: dict) -> str:
        return edge["to"] if edge["from"] == problem_node_id else edge["from"]

    for edge in incident_edges:
        edge_id = edge["id"]
        neighbour = other_node(edge)
        if neighbour == dead_end_node_id:
            colour, width = "#ef4444", 18
        elif nodes.get(neighbour, {}).get("kind") == "endpoint":
            colour, width = "#f59e0b", 13
        else:
            colour, width = "#2563eb", 13
        draw.line(
            [point(item) for item in edge["points"]],
            fill=colour,
            width=width,
            joint="curve",
        )
        midpoint = edge["points"][len(edge["points"]) // 2]
        draw.text(
            point(midpoint),
            edge_id,
            fill=colour,
            font=font(19, bold=True),
            stroke_width=4,
            stroke_fill="white",
        )

    node_ids = {problem_node_id, dead_end_node_id}
    for edge in incident_edges:
        node_ids.add(edge["from"])
        node_ids.add(edge["to"])

    for node_id in sorted(node_ids):
        node = nodes[node_id]
        x, y = point((node["x"], node["y"]))
        radius = 18 if node_id == problem_node_id else 12
        fill = "#dc2626" if node_id == problem_node_id else "#0f172a"
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
            outline="white",
            width=5,
        )
        draw.text(
            (x + 14, y - 33),
            node_id,
            fill="#0f172a",
            font=font(21, bold=True),
            stroke_width=4,
            stroke_fill="white",
        )

    draw.text(
        (28, 18),
        f"Problem junction: {problem_node_id}",
        fill="#0f172a",
        font=font(36, bold=True),
    )
    draw.text(
        (28, 70),
        "Blue: connected alternatives   Red: current sign-compatible dead-end",
        fill="#334155",
        font=font(23),
    )
    draw.text(
        (28, 108),
        "Current graph around node_0014: "
        + ", ".join(edge["id"] for edge in incident_edges),
        fill="#166534",
        font=font(23, bold=True),
    )
    draw.text(
        (28, 146),
        "Required decision: extend/connect node_0084 branch or correct node_0014 links",
        fill="#7c2d12",
        font=font(23, bold=True),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, quality=95)
    print(OUTPUT)


if __name__ == "__main__":
    main()
