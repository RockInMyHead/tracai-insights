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


def replace_r3_result(
    current: dict[str, Any],
    trajectory_payload: dict[str, Any],
) -> dict[str, Any]:
    """Promote a separately rebuilt R3 trajectory into an analysis document."""
    plan = (
        trajectory_payload.get("plan_trajectory")
        or trajectory_payload.get("trajectory")
        or []
    )
    if not isinstance(plan, list) or len(plan) < 2:
        raise RuntimeError("Rebuilt R3 payload has fewer than two poses")
    source = str(trajectory_payload.get("trajectory_source") or "raw")
    method = {
        "scale_aware_candidate": "r3_reconstruction_scale_aware",
        "robust_candidate": "r3_reconstruction_robust_candidate",
    }.get(source, "r3_reconstruction")
    timestamps = trajectory_payload.get("source_timestamps_seconds") or []
    distance = 0.0
    for left, right in zip(plan, plan[1:]):
        try:
            distance += math.dist(
                [float(left[0]), float(left[1])],
                [float(right[0]), float(right[1])],
            )
        except (TypeError, ValueError, IndexError):
            continue
    quality = trajectory_payload.get("trajectory_quality") or {}
    raw_3d = trajectory_payload.get("raw_trajectory_3d") or []
    updated = dict(current)
    updated.update({
        "method": method,
        "trajectory": plan,
        "plan_trajectory": plan,
        "raw_trajectory_3d": raw_3d,
        "turn_points": trajectory_payload.get("turn_points") or [],
        "frame_count": len(plan),
        "trajectory_points": len(plan),
        "r3_camera_points": raw_3d,
        "r3_raw_camera_points": raw_3d,
        "r3_source_frame_indices": (
            trajectory_payload.get("source_frame_indices") or []
        ),
        "r3_source_timestamps_seconds": timestamps,
        # Confidence belongs to the old inference and must not be silently
        # reused for a separately rebuilt trajectory.
        "r3_pose_confidence": [],
        "r3_pose_graph": trajectory_payload.get("pose_graph"),
        "r3_pose_graph_candidate": trajectory_payload.get("pose_graph_candidate"),
        "r3_scale_aware_candidate": trajectory_payload.get("scale_aware_candidate"),
        "pointcloud_status": None,
        "r3_projection": (quality.get("projection") or {}).get("method"),
    })
    stats = dict(current.get("processing_stats") or {})
    stats.update({
        "estimated_distance": round(distance, 3),
        "turns_detected": len(updated["turn_points"]),
        "avg_pose_confidence": None,
        "r3_trajectory_quality": quality,
        "r3_trajectory_source": source,
        "r3_trajectory_source_requested": trajectory_payload.get(
            "trajectory_source_requested", source
        ),
        "r3_trajectory_source_fallback_reason": trajectory_payload.get(
            "trajectory_source_fallback_reason"
        ),
        "r3_trajectory_source_selection": trajectory_payload.get(
            "trajectory_source_selection"
        ) or {},
        "r3_rebuild_recovered": True,
    })
    updated["processing_stats"] = stats
    video_info = dict(current.get("video_info") or {})
    video_info["frame_count"] = len(plan)
    if timestamps:
        try:
            video_info["duration"] = round(float(timestamps[-1]), 3)
        except (TypeError, ValueError):
            pass
    updated["video_info"] = video_info
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--worker-url", default="http://79.137.227.106:8004")
    parser.add_argument("--r3-worker-url")
    parser.add_argument("--r3-video-id")
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
    if args.r3_worker_url or args.r3_video_id:
        if not args.r3_worker_url or not args.r3_video_id:
            raise RuntimeError(
                "--r3-worker-url and --r3-video-id must be supplied together"
            )
        rebuilt_r3 = fetch_json(
            f"{args.r3_worker_url.rstrip('/')}/api/r3-trajectory/"
            f"{args.r3_video_id}"
        )
        r3_result = replace_r3_result(r3_result, rebuilt_r3)
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
        "r3_video_id": args.r3_video_id,
        "r3_points": len(recovered.get("plan_trajectory") or []),
        "r3_distance": (recovered.get("processing_stats") or {}).get(
            "estimated_distance"
        ),
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
