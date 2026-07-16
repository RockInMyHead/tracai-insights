"""Guarded R3 + LingBot trajectory fusion.

LingBot is an independent streaming reconstruction observer.  It is never
allowed to overwrite R3 merely because it produced poses: both trajectories
are first aligned by a proper (non-reflecting) 2-D similarity and compared over
the complete video.  Only geometrically consistent observations produce a
fusion candidate; the immutable floor plan makes the final source selection.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


def _finite_points(value: Any) -> np.ndarray:
    if not isinstance(value, list):
        return np.empty((0, 2), dtype=np.float64)
    points: list[list[float]] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("position") or item.get("point")
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            point = [float(item[0]), float(item[1])]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(component) for component in point):
            points.append(point)
    return np.asarray(points, dtype=np.float64)


def _finite_points_3d(value: Any) -> np.ndarray:
    """Read LingBot camera centres without accidentally treating X/Y as floor axes."""
    if not isinstance(value, list):
        return np.empty((0, 3), dtype=np.float64)
    points: list[list[float]] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("position") or item.get("point")
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        try:
            point = [float(item[0]), float(item[1]), float(item[2])]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(component) for component in point):
            points.append(point)
    return np.asarray(points, dtype=np.float64)


def _lingbot_plan_projection(lingbot_result: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a 3-D camera path onto its dominant motion plane.

    LingBot exports XYZ camera centres.  Its first two coordinates are not a
    guaranteed floor plane, so using them directly can collapse one direction
    of a production route.  PCA is sign-ambiguous; the fusion stage evaluates
    both signs of the second axis without ever reflecting the final similarity.
    """
    explicit = _finite_points(lingbot_result.get("plan_trajectory"))
    if len(explicit) >= 2:
        return explicit, {"method": "explicit_plan_trajectory"}
    points = _finite_points_3d(lingbot_result.get("trajectory"))
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.float64), {"method": "unavailable"}
    centered = points - np.median(points, axis=0)
    try:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.empty((0, 2), dtype=np.float64), {"method": "pca_failed"}
    projected = centered @ vt[:2].T
    total = max(float(np.sum(singular_values ** 2)), 1e-12)
    return projected, {
        "method": "pca_motion_plane",
        "singular_values": [round(float(value), 8) for value in singular_values],
        "explained_motion_plane_ratio": round(
            float(np.sum(singular_values[:2] ** 2)) / total, 8
        ),
        "basis": [[round(float(value), 8) for value in row] for row in vt[:2]],
        "normal": [round(float(value), 8) for value in vt[2]] if len(vt) > 2 else None,
    }


def _resample_by_time(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) == 0 or count <= 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    source_t = np.linspace(0.0, 1.0, len(points))
    target_t = np.linspace(0.0, 1.0, count)
    return np.column_stack([
        np.interp(target_t, source_t, points[:, axis])
        for axis in range(2)
    ])


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _weighted_similarity(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> Optional[tuple[float, np.ndarray, np.ndarray]]:
    total_weight = float(weights.sum())
    if len(source) < 2 or total_weight <= 1e-12:
        return None
    normalized = weights / total_weight
    source_center = np.sum(source * normalized[:, None], axis=0)
    target_center = np.sum(target * normalized[:, None], axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (source_zero * normalized[:, None]).T @ target_zero
    try:
        u, singular_values, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    rotation = u @ vt
    if float(np.linalg.det(rotation)) < 0.0:
        vt[-1, :] *= -1.0
        rotation = u @ vt
        singular_values[-1] *= -1.0
    denominator = float(np.sum(normalized * np.sum(source_zero * source_zero, axis=1)))
    if denominator <= 1e-12:
        return None
    scale = float(np.sum(singular_values) / denominator)
    if not math.isfinite(scale) or scale <= 1e-9:
        return None
    translation = target_center - scale * (source_center @ rotation)
    return scale, rotation, translation


def _robust_similarity(
    source: np.ndarray,
    target: np.ndarray,
) -> Optional[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    weights = np.ones(len(source), dtype=np.float64)
    fit: Optional[tuple[float, np.ndarray, np.ndarray]] = None
    residuals = np.zeros(len(source), dtype=np.float64)
    for _ in range(5):
        fit = _weighted_similarity(source, target, weights)
        if fit is None:
            return None
        scale, rotation, translation = fit
        aligned = scale * (source @ rotation) + translation
        residuals = np.linalg.norm(aligned - target, axis=1)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        huber = max(median + 2.5 * 1.4826 * mad, 1e-6)
        weights = np.minimum(1.0, huber / np.maximum(residuals, 1e-12))
    assert fit is not None
    return fit[0], fit[1], fit[2], residuals


def _signed_turn_radians(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    positive = lengths[lengths > 1e-9]
    if len(positive) == 0:
        return 0.0
    threshold = max(float(np.percentile(positive, 15)) * 0.2, 1e-9)
    vectors = deltas[lengths > threshold]
    total = 0.0
    for before, after in zip(vectors[:-1], vectors[1:]):
        cross = float(before[0] * after[1] - before[1] * after[0])
        dot = float(np.dot(before, after))
        total += math.atan2(cross, dot)
    return total


def _confidence_weights(values: Any, count: int) -> np.ndarray:
    if not isinstance(values, list) or len(values) < 2:
        return np.full(count, 0.5, dtype=np.float64)
    raw = np.asarray([
        float(value) if value is not None else math.nan
        for value in values
    ], dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    if len(finite) < 2:
        return np.full(count, 0.5, dtype=np.float64)
    fill = float(np.median(finite))
    raw[~np.isfinite(raw)] = fill
    low, high = np.percentile(raw, [10, 90])
    if high - low <= 1e-9:
        normalized = np.full(len(raw), 0.5, dtype=np.float64)
    else:
        normalized = np.clip((raw - low) / (high - low), 0.0, 1.0)
    return np.interp(
        np.linspace(0.0, 1.0, count),
        np.linspace(0.0, 1.0, len(normalized)),
        normalized,
    )


def build_lingbot_fusion_candidate(
    r3_result: dict[str, Any],
    lingbot_result: dict[str, Any],
) -> dict[str, Any]:
    """Return a guarded fusion candidate in the R3 plan coordinate system."""
    r3 = _finite_points(r3_result.get("plan_trajectory") or r3_result.get("trajectory"))
    lingbot, projection = _lingbot_plan_projection(lingbot_result)
    diagnostics: dict[str, Any] = {
        "method": "robust_similarity_confidence_blend_v2",
        "accepted": False,
        "r3_points": int(len(r3)),
        "lingbot_points": int(len(lingbot)),
        "lingbot_projection": projection,
    }
    independent = [
        [round(float(point[0]), 8), round(float(point[1]), 8), 0.0]
        for point in lingbot
    ]

    def rejected(reason: str) -> dict[str, Any]:
        diagnostics["reason"] = reason
        diagnostics["independent_accepted"] = len(independent) >= 6
        return {
            "accepted": False,
            "plan_trajectory": [],
            "independent_accepted": len(independent) >= 6,
            "independent_plan_trajectory": independent if len(independent) >= 6 else [],
            "diagnostics": diagnostics,
        }
    if len(r3) < 6 or len(lingbot) < 6:
        return rejected("trajectory_too_short")

    lingbot_resampled = _resample_by_time(lingbot, len(r3))
    variants = (lingbot_resampled, lingbot_resampled * np.asarray([1.0, -1.0]))
    fitted = [(variant, _robust_similarity(variant, r3)) for variant in variants]
    fitted = [(variant, fit) for variant, fit in fitted if fit is not None]
    if not fitted:
        return rejected("similarity_fit_failed")
    lingbot_resampled, fit = min(
        fitted, key=lambda item: float(np.median(item[1][3]))  # type: ignore[index]
    )
    if fit is None:
        return rejected("similarity_fit_failed")
    scale, rotation, translation, _ = fit
    aligned = scale * (lingbot_resampled @ rotation) + translation
    residuals = np.linalg.norm(aligned - r3, axis=1)
    # Full polyline length can hide large disagreement on looping routes. Use
    # the spatial extent with only a bounded contribution from travelled
    # length so a many-lap trajectory does not make every residual look tiny.
    reference_length = max(
        _polyline_length(r3) * 0.40,
        float(np.linalg.norm(np.ptp(r3, axis=0))),
        1e-9,
    )
    median_ratio = float(np.median(residuals)) / reference_length
    p95_ratio = float(np.percentile(residuals, 95)) / reference_length
    r3_turn = _signed_turn_radians(r3)
    lingbot_turn = _signed_turn_radians(aligned)
    chirality_conflict = (
        abs(r3_turn) >= math.radians(25.0)
        and abs(lingbot_turn) >= math.radians(25.0)
        and r3_turn * lingbot_turn < 0.0
    )

    diagnostics.update({
        "similarity_scale": round(scale, 8),
        "similarity_rotation_degrees": round(
            math.degrees(math.atan2(float(rotation[0, 1]), float(rotation[0, 0]))),
            3,
        ),
        "alignment_median_ratio": round(median_ratio, 6),
        "alignment_p95_ratio": round(p95_ratio, 6),
        "r3_signed_turn_degrees": round(math.degrees(r3_turn), 3),
        "lingbot_signed_turn_degrees": round(math.degrees(lingbot_turn), 3),
        "chirality_conflict": chirality_conflict,
    })
    if chirality_conflict:
        return rejected("turn_chirality_conflict")
    if median_ratio > 0.06 or p95_ratio > 0.12:
        return rejected("trajectory_disagreement_too_large")

    global_strength = float(np.clip(0.42 * (1.0 - p95_ratio / 0.12), 0.12, 0.42))
    r3_confidence = _confidence_weights(r3_result.get("r3_pose_confidence"), len(r3))
    # Preserve high-confidence R3 observations; let LingBot contribute more in
    # visually weak intervals. Endpoints remain anchored to avoid map jumps.
    local_strength = global_strength * (1.0 - 0.60 * r3_confidence)
    local_strength[0] = 0.0
    fused = r3 + local_strength[:, None] * (aligned - r3)
    diagnostics.update({
        "accepted": True,
        "reason": None,
        "fusion_strength_median": round(float(np.median(local_strength)), 4),
        "fusion_strength_max": round(float(np.max(local_strength)), 4),
    })
    trajectory = [
        [round(float(point[0]), 8), round(float(point[1]), 8), 0.0]
        for point in fused
    ]
    aligned_trajectory = [
        [round(float(point[0]), 8), round(float(point[1]), 8), 0.0]
        for point in aligned
    ]
    return {
        "accepted": True,
        "plan_trajectory": trajectory,
        "independent_accepted": True,
        "independent_plan_trajectory": independent,
        "aligned_lingbot_trajectory": aligned_trajectory,
        "diagnostics": diagnostics,
    }


def attach_lingbot_fusion_candidate(
    r3_result: dict[str, Any],
    lingbot_result: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(r3_result)
    candidate = build_lingbot_fusion_candidate(r3_result, lingbot_result)
    updated["lingbot_fusion_candidate"] = candidate
    updated["lingbot_shadow"] = {
        "method": lingbot_result.get("method", "lingbot_map"),
        "trajectory": lingbot_result.get("trajectory") or [],
        "session_id": lingbot_result.get("lingbot_session_id"),
        "metadata": lingbot_result.get("lingbot_metadata") or {},
    }
    stats = dict(updated.get("processing_stats") or {})
    stats["lingbot_fusion"] = candidate.get("diagnostics") or {}
    stats["lingbot_shadow_available"] = True
    updated["processing_stats"] = stats
    return updated
