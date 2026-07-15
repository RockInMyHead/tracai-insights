"""Immutable selection between raw and accepted robust R3 camera trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from r3_pose_graph_optimizer import (
        load_pose_graph_candidate_c2w,
        load_pose_graph_candidate_summary,
    )
except ImportError:  # pragma: no cover - supports package-style startup
    from backend.r3_pose_graph_optimizer import (
        load_pose_graph_candidate_c2w,
        load_pose_graph_candidate_summary,
    )


def select_r3_trajectory_camera_poses(
    base: Path,
    camera_poses: list[dict[str, Any]],
    requested_source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a pose set while preserving raw artifacts and reporting fallback."""
    requested = str(requested_source or "raw").strip().lower()
    selection: dict[str, Any] = {
        "requested": requested,
        "selected": "raw",
        "fallback_reason": None,
    }
    if requested not in {"raw", "robust_candidate"}:
        selection["fallback_reason"] = "unsupported_source"
        return camera_poses, selection
    if requested == "raw":
        return camera_poses, selection

    summary = load_pose_graph_candidate_summary(base)
    if not summary.get("available", False):
        selection["fallback_reason"] = "candidate_unavailable"
        return camera_poses, selection
    if not summary.get("accepted", False):
        selection["fallback_reason"] = "candidate_rejected"
        return camera_poses, selection

    graph_path = base / "pose_graph_edges.npz"
    expected_graph_mtime = summary.get("source_graph_mtime_ns")
    try:
        current_graph_mtime = graph_path.stat().st_mtime_ns
    except OSError:
        selection["fallback_reason"] = "source_graph_missing"
        return camera_poses, selection
    if expected_graph_mtime != current_graph_mtime:
        selection["fallback_reason"] = "candidate_stale"
        return camera_poses, selection

    candidate = load_pose_graph_candidate_c2w(
        base,
        expected_count=len(camera_poses),
        accepted_only=True,
    )
    if candidate is None:
        selection["fallback_reason"] = "candidate_artifact_invalid"
        return camera_poses, selection

    selected = [
        {**camera, "pose": candidate[index].tolist()}
        for index, camera in enumerate(camera_poses)
    ]
    selection["selected"] = "robust_candidate"
    selection["candidate_objective_improvement"] = summary.get(
        "objective_improvement"
    )
    selection["candidate_runtime_seconds"] = summary.get("runtime_seconds")
    return selected, selection
