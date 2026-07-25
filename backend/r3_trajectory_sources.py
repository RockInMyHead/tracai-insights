"""Comparative selection between immutable R3 trajectory candidates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from r3_pose_graph_optimizer import (
        load_pose_graph_candidate_c2w,
        load_pose_graph_candidate_summary,
    )
    from r3_scale_aware import (
        load_scale_aware_candidate_c2w,
        load_scale_aware_candidate_summary,
    )
    from trajectory_geometry import trajectory_acceptance, trajectory_metrics
    from r3_trajectory import compare_floor_projection_sources
except ImportError:  # pragma: no cover - supports package-style startup
    from backend.r3_pose_graph_optimizer import (
        load_pose_graph_candidate_c2w,
        load_pose_graph_candidate_summary,
    )
    from backend.r3_scale_aware import (
        load_scale_aware_candidate_c2w,
        load_scale_aware_candidate_summary,
    )
    from backend.trajectory_geometry import trajectory_acceptance, trajectory_metrics
    from backend.r3_trajectory import compare_floor_projection_sources


SOURCE_ORDER = ("raw", "robust_candidate", "scale_aware_candidate")
MAP_EVALUATION_FILE = "trajectory_source_evaluations.json"


def load_r3_trajectory_pose_sets(
    base: Path,
    camera_poses: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Load every structurally valid immutable pose set, accepted or rejected."""
    raw = np.asarray([camera["pose"] for camera in camera_poses], dtype=np.float64)
    result = {"raw": raw}
    robust_summary = load_pose_graph_candidate_summary(base)
    graph_path = base / "pose_graph_edges.npz"
    try:
        current_mtime = graph_path.stat().st_mtime_ns
    except OSError:
        current_mtime = None
    if (
        robust_summary.get("available")
        and current_mtime is not None
        and robust_summary.get("source_graph_mtime_ns") == current_mtime
    ):
        robust = load_pose_graph_candidate_c2w(
            base, expected_count=len(raw), accepted_only=False
        )
        if robust is not None:
            result["robust_candidate"] = robust
    scale = load_scale_aware_candidate_c2w(
        base, expected_count=len(raw), accepted_only=False
    )
    if scale is not None:
        result["scale_aware_candidate"] = scale
    return result


def _pose_centers(poses: np.ndarray) -> np.ndarray:
    return np.asarray(poses, dtype=np.float64)[:, :3, 3]


def _projection_quality(points: np.ndarray) -> dict[str, Any]:
    if len(points) < 3 or not np.isfinite(points).all():
        return {"available": False, "reason": "insufficient_points"}
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = np.square(singular)
    total = float(variance.sum())
    plane_ratio = float(variance[:2].sum() / max(total, 1e-12))
    line_ratio = float(variance[0] / max(total, 1e-12))
    return {
        "available": True,
        "motion_plane_ratio": round(plane_ratio, 6),
        "dominant_axis_ratio": round(line_ratio, 6),
        "quality_score": round(plane_ratio, 6),
    }


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _map_evaluations(base: Path) -> dict[str, Any]:
    path = base / MAP_EVALUATION_FILE
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    return candidates if isinstance(candidates, dict) else {}


def _map_quality(summary: dict[str, Any], external: Any) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for source in (
        summary.get("map_alignment"),
        summary.get("floorplan_alignment"),
        external,
    ):
        if isinstance(source, dict):
            evidence.update(source)
    score = _finite(
        evidence.get("score", evidence.get("map_alignment_score", evidence.get("constrained_score")))
    )
    repair_p95 = _finite(
        evidence.get("repair_p95_meters", evidence.get("correction_p95_meters"))
    )
    repair_max = _finite(
        evidence.get("repair_max_meters", evidence.get("correction_max_meters"))
    )
    accepted = evidence.get("accepted")
    return {
        "available": bool(evidence),
        "accepted": bool(accepted) if accepted is not None else None,
        "alignment_score": score,
        "repair_p95_meters": repair_p95,
        "repair_max_meters": repair_max,
        "rejection_reasons": list(evidence.get("rejection_reasons") or []),
    }


def _reason_for_geometry(source: str, reasons: list[str]) -> str:
    label = {
        "robust_candidate": "robust",
        "scale_aware_candidate": "scale_aware",
    }.get(source, source)
    if "spatial_span_ratio_out_of_bounds" in reasons:
        return f"{label}_spatial_span_collapsed"
    if "endpoint_progress_ratio_out_of_bounds" in reasons:
        return f"{label}_net_progress_regressed"
    if "segment_length_log_rmse_out_of_bounds" in reasons:
        return f"{label}_local_steps_distorted"
    if "local_direction_agreement_out_of_bounds" in reasons:
        return f"{label}_direction_regressed"
    return f"{label}_geometry_regressed"


def _candidate_record(
    name: str,
    poses: np.ndarray | None,
    summary: dict[str, Any],
    reference: np.ndarray | None,
    map_evaluation: Any,
) -> dict[str, Any]:
    available = poses is not None
    internal_accepted = True if name == "raw" else bool(summary.get("accepted"))
    record: dict[str, Any] = {
        "available": available,
        "internal_accepted": internal_accepted,
        "internal_rejection_reasons": list(summary.get("rejection_reasons") or []),
        "eligible": available and internal_accepted,
        "rejection_reasons": [],
        "metrics": trajectory_metrics(_pose_centers(poses)) if available else {"available": False},
        "projection_quality": (
            _projection_quality(_pose_centers(poses))
            if available else {"available": False}
        ),
        "map_alignment": _map_quality(summary, map_evaluation),
        "verified_loop_closure": bool(
            summary.get(
                "verified_loop_closure",
                (summary.get("config") or {}).get("verified_loop_closure", False)
                if isinstance(summary.get("config"), dict)
                else False,
            )
        ),
    }
    if not available:
        record["rejection_reasons"].append("artifact_unavailable")
    elif not internal_accepted:
        record["rejection_reasons"].append("internal_acceptance_failed")

    if available and reference is not None:
        geometry = trajectory_acceptance(_pose_centers(reference), _pose_centers(poses))
        record["geometry_vs_previous"] = geometry
        if not geometry["accepted"]:
            record["eligible"] = False
            record["rejection_reasons"].extend(geometry["rejection_reasons"])
    else:
        record["geometry_vs_previous"] = None

    map_quality = record["map_alignment"]
    if map_quality["available"] and map_quality["accepted"] is False:
        record["eligible"] = False
        record["rejection_reasons"].append("map_alignment_rejected")
    return record


def _apply_comparative_gates(
    name: str,
    record: dict[str, Any],
    previous: dict[str, Any],
) -> None:
    current_projection = record["projection_quality"]
    previous_projection = previous["projection_quality"]
    current_map = record["map_alignment"]
    missing = []
    if not current_projection.get("available"):
        missing.append("projection_quality")
    if not current_map.get("available"):
        missing.extend(("map_alignment_score", "map_repair"))
    record["selection_uncertainty"] = {
        "level": "high" if len(missing) >= 2 else "medium" if missing else "low",
        "missing_evidence": missing,
    }
    if not record["eligible"]:
        return
    if current_projection.get("available") and previous_projection.get("available"):
        if (
            float(current_projection["quality_score"])
            < float(previous_projection["quality_score"]) - 0.08
        ):
            record["eligible"] = False
            record["rejection_reasons"].append("projection_quality_regressed")

    previous_map = previous["map_alignment"]
    current_score = current_map.get("alignment_score")
    previous_score = previous_map.get("alignment_score")
    if current_score is not None and previous_score is not None:
        if float(current_score) > float(previous_score) + 0.15:
            record["eligible"] = False
            record["rejection_reasons"].append("map_alignment_score_regressed")
    current_repair = current_map.get("repair_p95_meters")
    previous_repair = previous_map.get("repair_p95_meters")
    if current_repair is not None and previous_repair is not None:
        if float(current_repair) > float(previous_repair) * 1.5 + 0.5:
            record["eligible"] = False
            record["rejection_reasons"].append("map_repair_regressed")

def select_r3_trajectory_camera_poses(
    base: Path,
    camera_poses: list[dict[str, Any]],
    requested_source: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one leaderboard and select the best eligible requested source."""
    requested = str(requested_source or "raw").strip().lower()
    raw = np.asarray([camera["pose"] for camera in camera_poses], dtype=np.float64)
    robust_summary = load_pose_graph_candidate_summary(base)
    scale_summary = load_scale_aware_candidate_summary(base)
    map_evaluations = _map_evaluations(base)

    robust: np.ndarray | None = None
    robust_reason: str | None = None
    if robust_summary.get("available"):
        graph_path = base / "pose_graph_edges.npz"
        try:
            current_mtime = graph_path.stat().st_mtime_ns
        except OSError:
            robust_reason = "source_graph_missing"
        else:
            if robust_summary.get("source_graph_mtime_ns") != current_mtime:
                robust_reason = "candidate_stale"
            else:
                robust = load_pose_graph_candidate_c2w(
                    base, expected_count=len(raw), accepted_only=False
                )
                if robust is None:
                    robust_reason = "candidate_artifact_invalid"

    scale = load_scale_aware_candidate_c2w(
        base, expected_count=len(raw), accepted_only=False
    )
    candidates = {
        "raw": _candidate_record(
            "raw", raw, {}, None, map_evaluations.get("raw")
        ),
        "robust_candidate": _candidate_record(
            "robust_candidate",
            robust,
            robust_summary,
            raw,
            map_evaluations.get("robust_candidate"),
        ),
    }
    if robust_reason:
        candidates["robust_candidate"]["eligible"] = False
        candidates["robust_candidate"]["rejection_reasons"].append(robust_reason)

    candidates["raw"]["selection_uncertainty"] = {
        "level": "high" if not candidates["raw"]["map_alignment"]["available"] else "low",
        "missing_evidence": (
            ["map_alignment_score", "map_repair"]
            if not candidates["raw"]["map_alignment"]["available"] else []
        ),
    }
    _apply_comparative_gates(
        "robust_candidate", candidates["robust_candidate"], candidates["raw"]
    )
    previous_name = (
        "robust_candidate"
        if candidates["robust_candidate"]["eligible"]
        else "raw"
    )
    previous_poses = robust if previous_name == "robust_candidate" else raw
    candidates["scale_aware_candidate"] = _candidate_record(
        "scale_aware_candidate",
        scale,
        scale_summary,
        previous_poses,
        map_evaluations.get("scale_aware_candidate"),
    )
    _apply_comparative_gates(
        "scale_aware_candidate",
        candidates["scale_aware_candidate"],
        candidates[previous_name],
    )

    supported = requested in SOURCE_ORDER
    requested_index = SOURCE_ORDER.index(requested) if supported else 0
    selected = "raw"
    reason = "raw_requested" if requested == "raw" else "requested_source_accepted"
    for name in SOURCE_ORDER[: requested_index + 1]:
        if candidates[name]["eligible"]:
            selected = name
    if not supported:
        reason = "unsupported_source"
    elif selected != requested:
        rejected = candidates[requested]
        geometry_reasons = list(
            (rejected.get("geometry_vs_previous") or {}).get("rejection_reasons") or []
        )
        if geometry_reasons:
            reason = _reason_for_geometry(requested, geometry_reasons)
        elif requested == "robust_candidate" and robust_reason:
            reason = robust_reason
        elif not rejected["internal_accepted"]:
            reason = (
                "scale_candidate_rejected"
                if requested == "scale_aware_candidate"
                else "candidate_rejected"
            )
        else:
            reason = str(
                (rejected["rejection_reasons"] or ["candidate_rejected"])[0]
            )

    selected_poses = {
        "raw": raw,
        "robust_candidate": robust,
        "scale_aware_candidate": scale,
    }[selected]
    assert selected_poses is not None
    output = [
        {**camera, "pose": selected_poses[index].tolist()}
        for index, camera in enumerate(camera_poses)
    ]
    most_reliable = "raw"
    for name in SOURCE_ORDER:
        if candidates[name]["eligible"]:
            most_reliable = name
    selection = {
        "requested": requested,
        "selected": selected,
        "candidates": candidates,
        "reason": reason,
        # Kept for older callers while the full reason is now authoritative.
        "fallback_reason": None if selected == requested else reason,
        "selection_uncertainty": candidates[selected]["selection_uncertainty"],
        "floor_projection_comparison": compare_floor_projection_sources(
            {
                name: poses
                for name, poses in {
                    "raw": raw,
                    "robust_candidate": robust,
                    "scale_aware_candidate": scale,
                }.items()
                if poses is not None
            },
            most_reliable,
        ),
    }
    return output, selection
