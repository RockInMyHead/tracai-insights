"""Guarded R3 + LingBot trajectory fusion (Geometry Integrity Stack).

LingBot is an independent streaming reconstruction observer.  It is never
allowed to overwrite R3 merely because it produced poses: both trajectories
are first aligned by a proper (non-reflecting) 2-D similarity and compared over
the complete video.  Only geometrically consistent observations produce a
fusion candidate; the immutable floor plan makes the final source selection.

Geometry Integrity Protocol (GIP) invariants:
- Explicit plan coordinates never silently reflect; PCA may choose a Y-sign.
- Chirality is evaluated on the native (unmirrored) path for explicit sources.
- Independent fallback carries the selected polarity and a real quality gate.
- Correspondence prefers timestamps, otherwise arc-length (never fake "time").
- Fused endpoints stay anchored to R3.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

METHOD_TAG = "robust_similarity_confidence_blend_v3_gip"


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


def _finite_timestamps(value: Any) -> Optional[np.ndarray]:
    if not isinstance(value, list) or len(value) < 2:
        return None
    raw = np.asarray(
        [
            float(item) if item is not None else math.nan
            for item in value
        ],
        dtype=np.float64,
    )
    finite = np.flatnonzero(np.isfinite(raw))
    if len(finite) < 2:
        return None
    if len(finite) < len(raw):
        raw = np.interp(np.arange(len(raw)), finite, raw[finite])
    # np.interp below requires a monotonic parameter. Mixing frame clocks or
    # accepting a reset here silently creates false R3/LingBot correspondence.
    if float(np.ptp(raw)) <= 1e-9 or np.any(np.diff(raw) <= 0.0):
        return None
    return raw


def _lingbot_plan_projection(lingbot_result: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a 3-D camera path onto its dominant motion plane.

    LingBot exports XYZ camera centres.  Prefer ``raw_trajectory_3d`` when
    present: the TrackAI adapter also writes an XZ ``plan_trajectory`` that is
    *not* an immutable floor frame, and treating it as explicit incorrectly
    disables PCA Y-sign gauge freedom then chirality-revokes the independent
    observer.  True explicit plan trajectories (no raw 3-D) keep producer
    handedness.  PCA is sign-ambiguous; only PCA provenance may choose a
    Y-sign later.
    """
    points = _finite_points_3d(
        lingbot_result.get("raw_trajectory_3d") or lingbot_result.get("trajectory")
    )
    if len(points) >= 2:
        # Adapter often stores remapped [x, z, y] in trajectory; only trust it
        # as 3-D when a dedicated raw_trajectory_3d exists or Z span is real.
        has_raw = _finite_points_3d(lingbot_result.get("raw_trajectory_3d")).shape[0] >= 2
        z_span = float(np.ptp(points[:, 2])) if points.shape[1] >= 3 else 0.0
        if has_raw or z_span > 1e-3:
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
                "normal": (
                    [round(float(value), 8) for value in vt[2]] if len(vt) > 2 else None
                ),
            }

    explicit = _finite_points(lingbot_result.get("plan_trajectory"))
    if len(explicit) >= 2:
        return explicit, {"method": "explicit_plan_trajectory"}
    # Last resort: first two axes of trajectory (may be adapter XZ).
    planar = _finite_points(lingbot_result.get("trajectory"))
    if len(planar) >= 2:
        return planar, {"method": "explicit_plan_trajectory"}
    return np.empty((0, 2), dtype=np.float64), {"method": "unavailable"}


def _resample_by_parameter(
    points: np.ndarray,
    count: int,
    parameter: np.ndarray,
    target_parameter: Optional[np.ndarray] = None,
) -> np.ndarray:
    if len(points) == 0 or count <= 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    source_t = np.asarray(parameter, dtype=np.float64)
    source_start = float(source_t[0]) if len(source_t) else 0.0
    source_end = float(source_t[-1]) if len(source_t) else 1.0
    if (
        len(source_t) != len(points)
        or float(np.ptp(source_t)) <= 1e-12
        or np.any(~np.isfinite(source_t))
        or np.any(np.diff(source_t) <= 0.0)
    ):
        source_t = np.linspace(0.0, 1.0, len(points))
    else:
        source_t = (source_t - source_t[0]) / max(float(source_t[-1] - source_t[0]), 1e-12)
    if target_parameter is not None and len(target_parameter) == count:
        target_t = np.asarray(target_parameter, dtype=np.float64)
        target_t = (target_t - source_start) / max(source_end - source_start, 1e-12)
        target_t = np.clip(target_t, 0.0, 1.0)
    else:
        target_t = np.linspace(0.0, 1.0, count)
    return np.column_stack([
        np.interp(target_t, source_t, points[:, axis])
        for axis in range(2)
    ])


def _arc_length_parameter(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    if len(points) == 1:
        return np.asarray([0.0], dtype=np.float64)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, len(points))
    return cumulative / total


def _correspond(
    lingbot: np.ndarray,
    count: int,
    lingbot_timestamps: Optional[np.ndarray],
    r3_timestamps: Optional[np.ndarray],
) -> tuple[np.ndarray, str]:
    """Register LingBot samples onto the R3 sample count.

    Prefer absolute timestamps with a constant clock offset when both tracks
    cover nearly the same duration.  Never stretch a shorter reconstruction to
    fill a longer one — that invents false temporal correspondence (e.g. LingBot
    250s vs R³ 305s).  Otherwise use arc-length.  Index linspace is never
    advertised as time.
    """
    if len(lingbot) == 0 or count <= 0:
        return np.empty((0, 2), dtype=np.float64), "unavailable"
    if (
        lingbot_timestamps is not None
        and r3_timestamps is not None
        and len(lingbot_timestamps) == len(lingbot)
        and len(r3_timestamps) == count
    ):
        lb = lingbot_timestamps.astype(np.float64)
        r3 = r3_timestamps.astype(np.float64)
        lb_span = float(lb[-1] - lb[0])
        r3_span = float(r3[-1] - r3[0])
        if lb_span > 1e-9 and r3_span > 1e-9:
            duration_ratio = r3_span / lb_span
            # Same video coverage only — allow tiny clock skew, not 20%+ stretch.
            if 0.95 <= duration_ratio <= 1.05:
                offset = float(r3[0] - lb[0])
                mapped = lb + offset
                mapped = np.clip(mapped, float(r3[0]), float(r3[-1]))
                return _resample_by_parameter(
                    lingbot, count, mapped, target_parameter=r3
                ), "timestamp_offset"
    return _resample_by_parameter(
        lingbot, count, _arc_length_parameter(lingbot)
    ), "arc_length"


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _spatial_span(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    return float(np.linalg.norm(np.ptp(points, axis=0)))


def _effective_rank(points: np.ndarray) -> tuple[float, float]:
    if len(points) < 2:
        return 0.0, 0.0
    centered = points - np.median(points, axis=0)
    try:
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    if len(singular_values) == 0:
        return 0.0, 0.0
    primary = float(singular_values[0])
    secondary = float(singular_values[1]) if len(singular_values) > 1 else 0.0
    condition = secondary / max(primary, 1e-12)
    rank = 2.0 if condition >= 0.02 else (1.0 if primary > 1e-9 else 0.0)
    return rank, condition


def _independent_quality(
    lingbot: np.ndarray,
    projection: dict[str, Any],
) -> dict[str, Any]:
    span = _spatial_span(lingbot)
    length = _polyline_length(lingbot)
    rank, condition = _effective_rank(lingbot)
    plane_ratio = float(projection.get("explained_motion_plane_ratio") or 1.0)
    method = str(projection.get("method") or "")
    reasons: list[str] = []
    if len(lingbot) < 6:
        reasons.append("too_few_points")
    if span < 1e-3:
        reasons.append("degenerate_span")
    if length < 1e-3:
        reasons.append("degenerate_path_length")
    if rank < 1.0:
        reasons.append("rank_deficient")
    if method == "pca_motion_plane" and plane_ratio < 0.92:
        reasons.append("nonplanar_motion")
    if method in {"unavailable", "pca_failed"}:
        reasons.append("projection_unavailable")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "point_count": int(len(lingbot)),
        "spatial_span": round(span, 8),
        "path_length": round(length, 8),
        "effective_rank": round(rank, 4),
        "secondary_condition": round(condition, 8),
        "plane_energy_ratio": round(plane_ratio, 8),
    }


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


def _trajectory_list(points: np.ndarray) -> list[list[float]]:
    return [
        [round(float(point[0]), 8), round(float(point[1]), 8), 0.0]
        for point in points
    ]


_R3_TRAJECTORY_SOURCE_LABELS = {
    "raw",
    "scale_aware_candidate",
    "robust_candidate",
    "pose_graph_candidate",
}


def should_restore_lingbot_fusion_candidate(
    candidate: Any,
    *,
    requested_source: str,
    saved_source: str,
    saved_source_requested: Any = None,
) -> bool:
    """Persist fused *or* independent-quality candidates across API refresh."""
    if not isinstance(candidate, dict):
        return False
    if not bool(candidate.get("accepted") or candidate.get("independent_accepted")):
        return False
    requested = str(requested_source or "").strip()
    saved = str(saved_source or "").strip()
    saved_requested = str(saved_source_requested or "").strip()
    # Corrupted saves once stored pose-confidence arrays in r3_trajectory_source.
    # Treat only known labels as a real source gate; otherwise restore anyway.
    if saved not in _R3_TRAJECTORY_SOURCE_LABELS:
        return True
    if requested not in _R3_TRAJECTORY_SOURCE_LABELS:
        return True
    if requested == saved:
        return True
    # Recomputes / merges may leave the effective label as "raw" while the
    # analysis was produced for scale_aware_candidate (UI default). Restore
    # when the saved *requested* source matches the refresh request.
    if (
        saved_requested in _R3_TRAJECTORY_SOURCE_LABELS
        and requested == saved_requested
    ):
        return True
    return False


def build_lingbot_fusion_candidate(
    r3_result: dict[str, Any],
    lingbot_result: dict[str, Any],
) -> dict[str, Any]:
    """Return a guarded fusion candidate in the R3 plan coordinate system."""
    r3 = _finite_points(r3_result.get("plan_trajectory") or r3_result.get("trajectory"))
    lingbot, projection = _lingbot_plan_projection(lingbot_result)
    quality = _independent_quality(lingbot, projection)
    r3_timestamps = _finite_timestamps(
        r3_result.get("r3_source_timestamps_seconds")
        or r3_result.get("source_timestamps_seconds")
    )
    lingbot_timestamps = _finite_timestamps(
        lingbot_result.get("lingbot_source_timestamps_seconds")
        or lingbot_result.get("source_timestamps_seconds")
    )
    diagnostics: dict[str, Any] = {
        "method": METHOD_TAG,
        "accepted": False,
        "r3_points": int(len(r3)),
        "lingbot_points": int(len(lingbot)),
        "lingbot_projection": projection,
        "independent_quality": quality,
        "selected_sign": 1.0,
        "reflection_applied": False,
        "composite_det": 1.0,
        "correspondence_mode": "unavailable",
    }
    selected_sign = 1.0
    signed_lingbot = lingbot.copy() if len(lingbot) else lingbot

    def emit_independent() -> tuple[bool, list[list[float]]]:
        accepted = bool(quality["accepted"]) and len(signed_lingbot) >= 6
        trajectory = _trajectory_list(signed_lingbot) if accepted else []
        return accepted, trajectory

    def rejected(reason: str, *, revoke_independent: bool = False) -> dict[str, Any]:
        nonlocal quality
        if revoke_independent:
            quality = dict(quality)
            quality["accepted"] = False
            reasons = list(quality.get("reasons") or [])
            if reason not in reasons:
                reasons.append(reason)
            quality["reasons"] = reasons
            diagnostics["independent_quality"] = quality
        independent_accepted, independent_traj = emit_independent()
        diagnostics["reason"] = reason
        diagnostics["independent_accepted"] = independent_accepted
        payload = {
            "accepted": False,
            "plan_trajectory": [],
            "independent_accepted": independent_accepted,
            "independent_plan_trajectory": independent_traj,
            "diagnostics": diagnostics,
        }
        if lingbot_timestamps is not None and independent_accepted:
            payload["lingbot_source_timestamps_seconds"] = [
                round(float(value), 6) for value in lingbot_timestamps
            ]
        return payload

    if len(r3) < 6 or len(lingbot) < 6:
        return rejected("trajectory_too_short")

    lingbot_resampled, correspondence_mode = _correspond(
        lingbot, len(r3), lingbot_timestamps, r3_timestamps
    )
    diagnostics["correspondence_mode"] = correspondence_mode

    provenance = str(projection.get("method") or "")
    allow_pca_sign = provenance == "pca_motion_plane"
    variants: list[tuple[float, np.ndarray, str]] = [
        (1.0, lingbot_resampled, "native"),
    ]
    if allow_pca_sign:
        variants.append((-1.0, lingbot_resampled * np.asarray([1.0, -1.0]), "pca_y_sign"))
    else:
        # Declared adapter only — never selected silently for explicit coords.
        variants.append(
            (-1.0, lingbot_resampled * np.asarray([1.0, -1.0]), "coordinate_adapter_y_flip")
        )

    fitted: list[tuple[float, np.ndarray, str, tuple[float, np.ndarray, np.ndarray, np.ndarray]]] = []
    for sign, variant, label in variants:
        fit = _robust_similarity(variant, r3)
        if fit is not None:
            fitted.append((sign, variant, label, fit))
    if not fitted:
        return rejected("similarity_fit_failed")

    # Native chirality uses the unmirrored path after a proper similarity.
    native_entries = [item for item in fitted if item[0] > 0.0]
    if not native_entries:
        return rejected("similarity_fit_failed")
    native_sign, native_variant, native_label, native_fit = native_entries[0]
    native_aligned = native_fit[0] * (native_variant @ native_fit[1]) + native_fit[2]
    r3_turn = _signed_turn_radians(r3)
    native_lingbot_turn = _signed_turn_radians(native_aligned)
    chirality_conflict = (
        abs(r3_turn) >= math.radians(25.0)
        and abs(native_lingbot_turn) >= math.radians(25.0)
        and r3_turn * native_lingbot_turn < 0.0
    )
    diagnostics.update({
        "r3_signed_turn_degrees": round(math.degrees(r3_turn), 3),
        "lingbot_signed_turn_degrees": round(math.degrees(native_lingbot_turn), 3),
        "chirality_conflict": chirality_conflict,
        "native_hypothesis": native_label,
    })
    adapter_entries = [item for item in fitted if item[2] == "coordinate_adapter_y_flip"]
    if adapter_entries:
        diagnostics["adapter_y_flip_median_residual"] = round(
            float(np.median(adapter_entries[0][3][3])), 8
        )

    # Explicit sources: chirality veto blocks fusion and independent fallback.
    # PCA Y-sign is gauge freedom, so chirality is evaluated after sign choice.
    if chirality_conflict and not allow_pca_sign:
        return rejected("turn_chirality_conflict", revoke_independent=True)

    if allow_pca_sign:
        # Choose the lower residual among native / PCA sign. Chirality on the
        # selected signed path must still agree when both turns are large.
        selected_sign, lingbot_resampled, selected_label, fit = min(
            fitted,
            key=lambda item: float(np.median(item[3][3])),
        )
        aligned = fit[0] * (lingbot_resampled @ fit[1]) + fit[2]
        selected_turn = _signed_turn_radians(aligned)
        selected_chirality_conflict = (
            abs(r3_turn) >= math.radians(25.0)
            and abs(selected_turn) >= math.radians(25.0)
            and r3_turn * selected_turn < 0.0
        )
        diagnostics["lingbot_signed_turn_degrees"] = round(math.degrees(selected_turn), 3)
        diagnostics["chirality_conflict"] = selected_chirality_conflict
        # Cumulative micro-turn chirality is unreliable on long wiggly routes.
        # For PCA, residual gates remain authoritative; keep going so a
        # residual-good blend is not discarded solely on noisy turn sums.
        if selected_chirality_conflict:
            diagnostics["chirality_soft_conflict"] = True
    else:
        # Explicit: only the native proper hypothesis may be accepted. The
        # adapter flip remains diagnostic so silent reflection cannot invent
        # chirality agreement.
        selected_sign, lingbot_resampled, selected_label, fit = native_sign, native_variant, native_label, native_fit
        aligned = native_aligned

    scale, rotation, translation, residuals = fit
    signed_lingbot = lingbot * np.asarray([1.0, selected_sign], dtype=np.float64)
    reflection_applied = selected_sign < 0.0
    composite_det = float(np.linalg.det(rotation)) * float(selected_sign)

    reference_length = max(
        _polyline_length(r3) * 0.40,
        float(np.linalg.norm(np.ptp(r3, axis=0))),
        1e-9,
    )
    median_ratio = float(np.median(residuals)) / reference_length
    p95_ratio = float(np.percentile(residuals, 95)) / reference_length
    inlier_ratio = float(np.mean(residuals <= max(np.median(residuals) * 2.5, 1e-9)))
    _, source_condition = _effective_rank(lingbot_resampled)

    diagnostics.update({
        "selected_sign": selected_sign,
        "selected_hypothesis": selected_label,
        "reflection_applied": reflection_applied,
        "composite_det": round(composite_det, 8),
        "similarity_scale": round(scale, 8),
        "similarity_rotation_degrees": round(
            math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))),
            3,
        ),
        "alignment_median_ratio": round(median_ratio, 6),
        "alignment_p95_ratio": round(p95_ratio, 6),
        "inlier_ratio": round(inlier_ratio, 6),
        "effective_rank": round(_effective_rank(lingbot_resampled)[0], 4),
        "secondary_condition": round(source_condition, 8),
        "endpoint_displacement": 0.0,
    })

    if median_ratio > 0.06 or p95_ratio > 0.12:
        return rejected("trajectory_disagreement_too_large")

    global_strength = float(np.clip(0.42 * (1.0 - p95_ratio / 0.12), 0.12, 0.42))
    r3_confidence = _confidence_weights(r3_result.get("r3_pose_confidence"), len(r3))
    # Preserve high-confidence R3 observations; let LingBot contribute more in
    # visually weak intervals. Both endpoints stay anchored to avoid map jumps.
    local_strength = global_strength * (1.0 - 0.60 * r3_confidence)
    local_strength[0] = 0.0
    local_strength[-1] = 0.0
    fused = r3 + local_strength[:, None] * (aligned - r3)
    endpoint_displacement = float(
        max(
            np.linalg.norm(fused[0] - r3[0]),
            np.linalg.norm(fused[-1] - r3[-1]),
        )
    )
    diagnostics.update({
        "accepted": True,
        "reason": None,
        "fusion_strength_median": round(float(np.median(local_strength)), 4),
        "fusion_strength_max": round(float(np.max(local_strength)), 4),
        "endpoint_displacement": round(endpoint_displacement, 8),
    })
    independent_accepted, independent_traj = emit_independent()
    diagnostics["independent_accepted"] = independent_accepted
    payload = {
        "accepted": True,
        "plan_trajectory": _trajectory_list(fused),
        "independent_accepted": independent_accepted,
        "independent_plan_trajectory": independent_traj,
        "aligned_lingbot_trajectory": _trajectory_list(aligned),
        "diagnostics": diagnostics,
    }
    if lingbot_timestamps is not None:
        payload["lingbot_source_timestamps_seconds"] = [
            round(float(value), 6) for value in lingbot_timestamps
        ]
    return payload


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
        "raw_trajectory_3d": lingbot_result.get("raw_trajectory_3d") or [],
        "session_id": lingbot_result.get("lingbot_session_id"),
        "metadata": lingbot_result.get("lingbot_metadata") or {},
    }
    stats = dict(updated.get("processing_stats") or {})
    stats["lingbot_fusion"] = candidate.get("diagnostics") or {}
    stats["lingbot_shadow_available"] = True
    updated["processing_stats"] = stats
    return updated
