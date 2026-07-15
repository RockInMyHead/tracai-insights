"""Immutable selection between raw, robust and scale-aware R3 trajectories."""

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

try:
    from r3_scale_aware import (
        load_scale_aware_candidate_c2w,
        load_scale_aware_candidate_summary,
    )
except ImportError:  # pragma: no cover - supports package-style startup
    from backend.r3_scale_aware import (
        load_scale_aware_candidate_c2w,
        load_scale_aware_candidate_summary,
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
    if requested not in {"raw", "robust_candidate", "scale_aware_candidate"}:
        selection["fallback_reason"] = "unsupported_source"
        return camera_poses, selection
    if requested == "raw":
        return camera_poses, selection

    if requested == "scale_aware_candidate":
        scale_summary = load_scale_aware_candidate_summary(base)
        scale_candidate = load_scale_aware_candidate_c2w(
            base,
            expected_count=len(camera_poses),
            accepted_only=True,
        )
        if scale_candidate is not None:
            selected = [
                {**camera, "pose": scale_candidate[index].tolist()}
                for index, camera in enumerate(camera_poses)
            ]
            selection.update({
                "selected": "scale_aware_candidate",
                "scale_range": scale_summary.get("scale_range"),
                "floor_observations": scale_summary.get("observation_count"),
                "base_source": scale_summary.get("base_source"),
            })
            return selected, selection
        # A rejected floor candidate must not make the UI worse. Fall back to
        # the accepted SE(3) candidate when available, otherwise immutable raw.
        fallback, fallback_selection = select_r3_trajectory_camera_poses(
            base, camera_poses, "robust_candidate"
        )
        selection["selected"] = fallback_selection["selected"]
        selection["fallback_reason"] = (
            "scale_candidate_rejected"
            if scale_summary.get("available")
            else "scale_candidate_unavailable"
        )
        selection["scale_rejection_reasons"] = scale_summary.get("rejection_reasons", [])
        return fallback, selection

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
