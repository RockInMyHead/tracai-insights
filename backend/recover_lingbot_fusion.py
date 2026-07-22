#!/usr/bin/env python3
"""Attach a completed LingBot session to an existing R3 analysis.

This recovery path avoids re-running expensive R3 inference when only the
independent LingBot observer failed. It uses the same fusion and immutable
floor-plan gates as the normal backend pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from floorplan_constraints import apply_floorplan_constraints  # noqa: E402
from lingbot_fusion import attach_lingbot_fusion_candidate  # noqa: E402


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def convert_lingbot(
    session_id: str,
    metadata: dict[str, Any],
    trajectory_payload: dict[str, Any],
) -> dict[str, Any]:
    trajectory: list[list[float]] = []
    raw_trajectory: list[list[float]] = []
    timestamps: list[float | None] = []
    for pose in trajectory_payload.get("poses") or []:
        if not isinstance(pose, dict):
            continue
        position = pose.get("position")
        if not isinstance(position, list) or len(position) < 3:
            continue
        try:
            x, y, z = (float(position[0]), float(position[1]), float(position[2]))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        trajectory.append([x, z, y])
        raw_trajectory.append([x, y, z])
        timestamp = pose.get("timestamp_seconds")
        try:
            timestamp_value = float(timestamp) if timestamp is not None else None
        except (TypeError, ValueError):
            timestamp_value = None
        timestamps.append(
            timestamp_value
            if timestamp_value is not None and math.isfinite(timestamp_value)
            else None
        )
    return {
        "method": "lingbot_map",
        "trajectory": trajectory,
        "plan_trajectory": trajectory,
        "raw_trajectory_3d": raw_trajectory,
        "lingbot_session_id": session_id,
        "lingbot_metadata": metadata,
        "lingbot_source_timestamps_seconds": timestamps,
        "source_timestamps_seconds": timestamps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--worker-url", default="http://79.137.227.106:8004")
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()

    base_url = args.worker_url.rstrip("/")
    status = fetch_json(f"{base_url}/sessions/{args.session_id}/status")
    if status.get("status") != "completed":
        raise RuntimeError(f"LingBot session is not completed: {status}")
    metadata = fetch_json(f"{base_url}/sessions/{args.session_id}/metadata")
    trajectory_payload = fetch_json(
        f"{base_url}/sessions/{args.session_id}/trajectory"
    )
    lingbot = convert_lingbot(args.session_id, metadata, trajectory_payload)
    if len(lingbot["trajectory"]) < 2:
        raise RuntimeError("LingBot returned fewer than two valid poses")

    document = json.loads(args.analysis.read_text(encoding="utf-8"))
    r3_result = document.get("analysis_result")
    if not isinstance(r3_result, dict):
        raise RuntimeError("Analysis file has no analysis_result")
    recovered = attach_lingbot_fusion_candidate(r3_result, lingbot)
    recovered = apply_floorplan_constraints(
        recovered,
        {"floorplan_id": "kerama_marazzi_2025"},
    )
    document["analysis_result"] = recovered
    if args.backup:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.analysis, args.backup)
    temporary = args.analysis.with_suffix(args.analysis.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(args.analysis)
    candidate = recovered.get("lingbot_fusion_candidate") or {}
    floorplan = recovered.get("floorplan_constraint") or {}
    print(json.dumps({
        "session_id": args.session_id,
        "lingbot_poses": len(lingbot["trajectory"]),
        "lingbot_last_timestamp_seconds": timestamps_last(lingbot),
        "fusion_accepted": bool(candidate.get("accepted")),
        "fusion_reason": (candidate.get("diagnostics") or {}).get("reason"),
        "floorplan_accepted": bool(floorplan.get("accepted")),
        "floorplan_reason": floorplan.get("reason"),
        "selected_source": floorplan.get("trajectory_observation_source"),
    }, ensure_ascii=False, indent=2))


def timestamps_last(lingbot: dict[str, Any]) -> float | None:
    values = [
        value for value in lingbot.get("source_timestamps_seconds") or []
        if value is not None
    ]
    return float(values[-1]) if values else None


if __name__ == "__main__":
    main()
