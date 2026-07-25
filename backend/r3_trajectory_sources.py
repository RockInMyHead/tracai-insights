"""Immutable selection between raw, robust and scale-aware R3 trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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

try:
    from trajectory_geometry import trajectory_acceptance, trajectory_metrics
except ImportError:  # pragma: no cover
    from backend.trajectory_geometry import trajectory_acceptance, trajectory_metrics


def _pose_array(camera_poses: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([camera["pose"] for camera in camera_poses], dtype=np.float64)


def _candidate_record(
    name: str,
    poses: np.ndarray | None,
    summary: dict[str, Any],
    reference: np.ndarray,
) -> dict[str, Any]:
    available = poses is not None
    internal_accepted = name == "raw" or bool(summary.get("accepted"))
    record: dict[str, Any] = {
        "available": available,
        "internal_accepted": internal_accepted,
        "internal_rejection_reasons": list(summary.get("rejection_reasons") or []),
        "eligible": bool(available and internal_accepted),
        "rejection_reasons": [],
    }
    if not available:
        record["rejection_reasons"].append("artifact_unavailable")
        record["metrics"] = {"available": False}
        record["geometry_vs_raw"] = None
        return record
    centers = poses[:, :3, 3]
    record["metrics"] = trajectory_metrics(centers)
    centered = centers - centers.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = np.square(singular)
    record["projection_quality"] = {
        "motion_plane_ratio": float(
            variance[:2].sum() / max(float(variance.sum()), 1e-12)
        ),
        "dominant_axis_ratio": float(
            variance[0] / max(float(variance.sum()), 1e-12)
        ),
    }
    if name == "raw":
        record["geometry_vs_raw"] = {
            "accepted": True,
            "rejection_reasons": [],
        }
        return record
    acceptance = trajectory_acceptance(
        reference[:, :3, 3],
        centers,
        {
            "verified_loop_closure": bool(summary.get("verified_loop_closure", False)),
        },
    )
    record["geometry_vs_raw"] = acceptance
    if not acceptance["accepted"]:
        record["eligible"] = False
        record["rejection_reasons"].extend(acceptance["rejection_reasons"])
    if not internal_accepted:
        record["rejection_reasons"].append("internal_acceptance_failed")
    return record


def _materialize(
    camera_poses: list[dict[str, Any]],
    poses: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {**camera, "pose": poses[index].tolist()}
        for index, camera in enumerate(camera_poses)
    ]


def select_r3_trajectory_camera_poses(
    base: Path,
    camera_poses: list[dict[str, Any]],
    requested_source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select raw/robust/scale-aware poses through comparative geometry gates."""
    requested = str(requested_source or "raw").strip().lower()
    raw = _pose_array(camera_poses)
    selection: dict[str, Any] = {
        "requested": requested,
        "selected": "raw",
        "fallback_reason": None,
        "reason": "raw_requested",
        "candidates": {},
        "uncertain": False,
    }
    if requested not in {"raw", "robust_candidate", "scale_aware_candidate"}:
        selection["fallback_reason"] = "unsupported_source"
        return camera_poses, selection
    if requested == "raw":
        selection["candidates"]["raw"] = _candidate_record(
            "raw", raw, {"accepted": True}, raw
        )
        return camera_poses, selection

    robust_summary = load_pose_graph_candidate_summary(base)
    graph_path = base / "pose_graph_edges.npz"
    expected_graph_mtime = robust_summary.get("source_graph_mtime_ns")
    try:
        current_graph_mtime = graph_path.stat().st_mtime_ns
    except OSError:
        current_graph_mtime = None
    robust = (
        load_pose_graph_candidate_c2w(
            base, expected_count=len(camera_poses), accepted_only=False
        )
        if (
            robust_summary.get("available")
            and current_graph_mtime is not None
            and expected_graph_mtime == current_graph_mtime
        )
        else None
    )
    scale_summary = load_scale_aware_candidate_summary(base)
    scale = load_scale_aware_candidate_c2w(
        base, expected_count=len(camera_poses), accepted_only=False
    )
    records = {
        "raw": _candidate_record("raw", raw, {"accepted": True}, raw),
        "robust_candidate": _candidate_record(
            "robust_candidate", robust, robust_summary, raw
        ),
        "scale_aware_candidate": _candidate_record(
            "scale_aware_candidate", scale, scale_summary, raw
        ),
    }
    selection["candidates"] = records
    preference = (
        ("scale_aware_candidate", "robust_candidate", "raw")
        if requested == "scale_aware_candidate"
        else ("robust_candidate", "raw")
    )
    selected_name = next(
        name for name in preference if records[name]["eligible"]
    )
    selected_poses = {
        "raw": raw,
        "robust_candidate": robust,
        "scale_aware_candidate": scale,
    }[selected_name]
    selection["selected"] = selected_name
    if selected_name == requested:
        selection["reason"] = f"{selected_name}_accepted_comparatively"
    else:
        rejected = records[requested]["rejection_reasons"]
        reason = rejected[0] if rejected else "candidate_unavailable"
        if requested == "scale_aware_candidate" and not scale_summary.get("accepted"):
            compatibility_reason = (
                "scale_candidate_rejected"
                if scale_summary.get("available")
                else "scale_candidate_unavailable"
            )
        elif requested == "robust_candidate" and (
            robust_summary.get("available")
            and current_graph_mtime is not None
            and expected_graph_mtime != current_graph_mtime
        ):
            compatibility_reason = "candidate_stale"
        elif requested == "robust_candidate" and not robust_summary.get("accepted"):
            compatibility_reason = (
                "candidate_rejected"
                if robust_summary.get("available")
                else "candidate_unavailable"
            )
        else:
            compatibility_reason = reason
        selection["fallback_reason"] = compatibility_reason
        selection["reason"] = f"{requested}_{reason}"
    eligible_non_raw = [
        name for name in ("robust_candidate", "scale_aware_candidate")
        if records[name]["eligible"]
    ]
    selection["uncertain"] = len(eligible_non_raw) > 1
    return _materialize(camera_poses, selected_poses), selection
