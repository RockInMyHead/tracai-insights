#!/usr/bin/env python3
"""Replay map matching on an existing analysis without rerunning GPU R3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.floorplan_constraints import apply_floorplan_constraints  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--floorplan-id", default="kerama_marazzi_2025")
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    # Saved API artifacts wrap the actual algorithm payload, while older
    # replay fixtures store it directly.
    if isinstance(source.get("analysis_result"), dict):
        source = source["analysis_result"]
    replayed = apply_floorplan_constraints(
        source,
        {"floorplan_id": args.floorplan_id},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(replayed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    diagnostics = replayed.get("floorplan_constraint") or {}
    summary = {
        "accepted": bool(diagnostics.get("accepted")),
        "reason": diagnostics.get("reason"),
        "rejection_reasons": diagnostics.get("rejection_reasons"),
        "turn_topology": diagnostics.get("turn_topology"),
        "graph_matching_used": diagnostics.get("graph_matching_used"),
        "map_matching_mode": diagnostics.get("map_matching_mode"),
        "constraint_revision": diagnostics.get("constraint_revision"),
        "map_trajectory_published": "map_trajectory" in replayed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
