"""Robust conversion of R3 camera poses into a floor-plan trajectory.

R3 writes camera-to-world (c2w) poses.  They are useful for 3D rendering as
is, but a floor plan needs a stable two-dimensional coordinate system.  This
module keeps the two representations separate and makes turn detection depend
on the same plan trajectory that the UI renders.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _finite_vector(value: Any, length: int) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if vector.size < length:
        return None
    vector = vector[:length]
    return vector if np.isfinite(vector).all() else None


def _normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9 and np.isfinite(norm):
        return vector / norm
    if fallback is not None:
        return _normalize(fallback)
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _json_points(points: np.ndarray) -> list[list[float]]:
    return [[round(float(v), 6) for v in row] for row in points]


def _angle_delta_rad(target: float, source: float) -> float:
    return (target - source + math.pi) % (2.0 * math.pi) - math.pi


def _pose_confidences(camera_poses: Sequence[Mapping[str, Any]], pose_confidence: Any) -> np.ndarray:
    result = np.full(len(camera_poses), np.nan, dtype=np.float64)
    if pose_confidence is None:
        return result
    try:
        values = np.asarray(pose_confidence, dtype=np.float64).reshape(-1)
    except Exception:
        return result
    for index, pose in enumerate(camera_poses):
        frame = pose.get("frame", index)
        try:
            frame_index = int(frame)
        except Exception:
            frame_index = index
        if 0 <= frame_index < values.size and np.isfinite(values[frame_index]):
            result[index] = values[frame_index]
        elif index < values.size and np.isfinite(values[index]):
            result[index] = values[index]
    return result


def _fill_invalid_positions(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(points).all(axis=1)
    if finite.all():
        return points.copy(), finite
    if not finite.any():
        return np.zeros_like(points), finite

    indices = np.arange(len(points), dtype=np.float64)
    filled = points.copy()
    valid_indices = indices[finite]
    for dim in range(points.shape[1]):
        filled[:, dim] = np.interp(indices, valid_indices, points[finite, dim])
    return filled, finite


def _clean_positions(points: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Reject isolated pose spikes while preserving real turns.

    A global moving average erases short 90-degree turns.  Here we only repair
    invalid/isolated outliers, then apply a very small smoother on locally
    straight pieces of the path.
    """
    if len(points) < 3:
        return points.copy(), {
            "quality": "too_short",
            "raw_points": int(len(points)),
            "outlier_points": 0,
            "smoothed_points": 0,
        }

    filled, originally_finite = _fill_invalid_positions(points)
    steps = np.linalg.norm(np.diff(filled, axis=0), axis=1)
    positive = steps[np.isfinite(steps) & (steps > 1e-8)]
    if positive.size == 0:
        return filled, {
            "quality": "static",
            "raw_points": int(len(points)),
            "outlier_points": int((~originally_finite).sum()),
            "smoothed_points": 0,
            "raw_step_median": 0.0,
        }

    median_step = float(np.median(positive))
    p75_step = float(np.percentile(positive, 75))
    p90_step = float(np.percentile(positive, 90))
    p99_step = float(np.percentile(positive, 99))
    # p90 is often contaminated by the very teleport we are trying to reject.
    jump_limit = max(median_step * 5.0, p75_step * 2.5, 1e-6)

    finite_confidence = confidence[np.isfinite(confidence)]
    low_confidence = (
        float(np.percentile(finite_confidence, 12))
        if finite_confidence.size >= 5
        else float("-inf")
    )
    outlier = ~originally_finite

    for i in range(1, len(filled) - 1):
        before = filled[i] - filled[i - 1]
        after = filled[i + 1] - filled[i]
        before_norm = float(np.linalg.norm(before))
        after_norm = float(np.linalg.norm(after))
        local_jump = max(before_norm, after_norm) > jump_limit
        if local_jump:
            outlier[i] = True
            continue

        if before_norm <= 1e-8 or after_norm <= 1e-8:
            continue

        # A real U-turn persists after the turn.  A one-frame pose flip returns
        # to the previous direction immediately, so only remove the latter.
        cosine = float(np.dot(before, after) / (before_norm * after_norm))
        if cosine < -0.75 and i + 2 < len(filled):
            recovery = filled[i + 2] - filled[i + 1]
            recovery_norm = float(np.linalg.norm(recovery))
            continues_after_flip = (
                recovery_norm > 1e-8
                and float(np.dot(after, recovery) / (after_norm * recovery_norm)) > 0.55
            )
            pose_is_weak = not np.isfinite(confidence[i]) or confidence[i] <= low_confidence
            # Do not erase a genuine U-turn: its steps stay near the normal
            # walking baseline.  A bad absolute pose makes a much larger
            # excursion and immediately returns to a regular path.
            is_large_excursion = max(before_norm, after_norm) > max(median_step * 2.5, p75_step * 1.5, 1e-6)
            if continues_after_flip and is_large_excursion and pose_is_weak:
                outlier[i] = True

    clean = filled.copy()
    good = ~outlier
    if good.any() and not good.all():
        indices = np.arange(len(clean), dtype=np.float64)
        good_indices = indices[good]
        for dim in range(clean.shape[1]):
            clean[:, dim] = np.interp(indices, good_indices, clean[good, dim])

    smoothed = 0
    # Three-point local smoothing only on near-straight motion.  Sharp turns
    # retain their corner point instead of being averaged into a curve.
    result = clean.copy()
    for i in range(1, len(clean) - 1):
        left = clean[i] - clean[i - 1]
        right = clean[i + 1] - clean[i]
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm <= 1e-8 or right_norm <= 1e-8:
            continue
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
        if cosine > math.cos(math.radians(20.0)):
            result[i] = (clean[i - 1] + 2.0 * clean[i] + clean[i + 1]) / 4.0
            smoothed += 1

    return result, {
        "quality": "ok" if int(outlier.sum()) <= max(2, len(points) // 8) else "unstable_pose",
        "raw_points": int(len(points)),
        "outlier_points": int(outlier.sum()),
        "smoothed_points": smoothed,
        "raw_step_median": round(median_step, 6),
        "raw_step_p75": round(p75_step, 6),
        "raw_step_p90": round(p90_step, 6),
        "raw_step_p99": round(p99_step, 6),
        "jump_limit": round(jump_limit, 6),
        "low_confidence_threshold": round(low_confidence, 6) if math.isfinite(low_confidence) else None,
    }


def _camera_up_normal(rotations: Sequence[np.ndarray]) -> tuple[np.ndarray | None, float]:
    """Estimate physical camera-up in world space from OpenCV c2w poses.

    R3 exports OpenCV camera axes (X right, Y down, Z forward).  Therefore
    column 1 of c2w points *down*, and physical camera-up is ``-R[:, 1]``.
    The sign is important: it defines whether positive plan Y means left or
    right after the floor projection.
    """
    vectors: list[np.ndarray] = []
    for rotation in rotations:
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            continue
        vectors.append(_normalize(-rotation[:, 1]))
    if len(vectors) < 3:
        return None, 0.0

    # A corrupt/upside-down pose must not reverse the global map handedness.
    # Keep one hemisphere for the entire run; never choose a sign per frame.
    reference = vectors[0]
    aligned = np.array([v if float(np.dot(v, reference)) >= 0 else -v for v in vectors])
    normal = _normalize(np.median(aligned, axis=0))
    coherence = float(np.median(aligned @ normal))
    return normal, coherence


def _trajectory_plane_normal(points: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Estimate the floor normal from the camera-centre trajectory itself.

    Camera-local up is biased when an operator points the camera down.  That
    bias compresses movement in the initial direction but not after a 90°
    turn, which looks exactly like a scale change.  Once the route contains
    two non-collinear directions, the least-varying PCA axis is a substantially
    better estimate of the physical walking plane.
    """
    diagnostics: dict[str, Any] = {
        "available": False,
        "eligible": False,
        "second_axis_ratio": None,
        "planarity_ratio": None,
    }
    if len(points) < 6:
        return None, diagnostics

    centered = points - np.median(points, axis=0, keepdims=True)
    try:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, diagnostics
    if len(singular_values) < 3 or not np.isfinite(singular_values).all():
        return None, diagnostics

    first = float(singular_values[0])
    second = float(singular_values[1])
    third = float(singular_values[2])
    second_axis_ratio = second / max(first, 1e-9)
    planarity_ratio = third / max(second, 1e-9)
    normal = _normalize(vh[-1]) if first > 1e-8 else None
    # Require genuine two-dimensional route coverage.  A nearly straight path
    # leaves the plane normal underdetermined and must use camera-up instead.
    eligible = bool(
        normal is not None
        and second_axis_ratio >= 0.035
        and planarity_ratio <= 0.35
    )
    diagnostics.update({
        "available": normal is not None,
        "eligible": eligible,
        "second_axis_ratio": round(second_axis_ratio, 6),
        "planarity_ratio": round(planarity_ratio, 6),
        "singular_values": [round(float(value), 6) for value in singular_values],
    })
    return normal, diagnostics


def _project_to_floor(points: np.ndarray, rotations: Sequence[np.ndarray]) -> tuple[np.ndarray, dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Project positions into canonical plan space: X forward, Y left, Z up."""
    if len(points) == 0:
        basis = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        return points.copy(), {"method": "empty"}, basis

    origin = points[0].copy()
    centered = points - origin
    camera_up, coherence = _camera_up_normal(rotations)
    pca_normal, pca_diagnostics = _trajectory_plane_normal(points)
    normal_sign_source = "camera_physical_up"

    if pca_normal is not None and pca_diagnostics["eligible"]:
        normal = pca_normal
        method = "trajectory_plane_pca"
        if camera_up is not None:
            if float(np.dot(normal, camera_up)) < 0.0:
                normal = -normal
        else:
            dominant = int(np.argmax(np.abs(normal)))
            if normal[dominant] < 0.0:
                normal = -normal
            normal_sign_source = "deterministic_dominant_axis"
    elif camera_up is not None and coherence >= 0.55:
        normal = camera_up
        method = "camera_physical_up"
    elif pca_normal is not None:
        normal = pca_normal
        method = "trajectory_plane_pca_low_support"
        if camera_up is not None:
            if float(np.dot(normal, camera_up)) < 0.0:
                normal = -normal
        else:
            dominant = int(np.argmax(np.abs(normal)))
            if normal[dominant] < 0.0:
                normal = -normal
            normal_sign_source = "deterministic_dominant_axis"
    else:
        # R3's first c2w is normally close to identity, where physical up is
        # -Y in OpenCV coordinates.
        normal = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        method = "opencv_axis_fallback"
        normal_sign_source = "opencv_axis_convention"

    # Preserve initial walking direction as X+.  PCA alone leaves this sign
    # arbitrary and is one source of apparent 180-degree flips on the map.
    heading = np.zeros(3, dtype=np.float64)
    for index in range(1, len(centered)):
        candidate = centered[index] - np.dot(centered[index], normal) * normal
        if np.linalg.norm(candidate) > 1e-6:
            heading = candidate
            break
    if np.linalg.norm(heading) <= 1e-6 and len(points) >= 2:
        heading = centered[-1] - np.dot(centered[-1], normal) * normal
    if np.linalg.norm(heading) <= 1e-6:
        alternatives = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])]
        heading = next((axis for axis in alternatives if abs(float(np.dot(axis, normal))) < 0.9), alternatives[0])

    e1 = _normalize(heading)
    # physical_up x forward = physical_left.  This is a right-handed Cartesian
    # plan basis; the SVG renderer performs the separate Y-up -> Y-down change.
    e2 = _normalize(np.cross(normal, e1))
    e1 = _normalize(np.cross(e2, normal))
    plan = np.column_stack((centered @ e1, centered @ e2, centered @ normal))

    return plan, {
        "method": method,
        "camera_up_coherence": round(coherence, 5),
        "camera_coordinate_convention": "opencv_x_right_y_down_z_forward",
        "plan_coordinate_convention": "x_forward_y_left_z_up",
        "normal_sign_source": normal_sign_source,
        "chirality_confidence": "high" if normal_sign_source == "camera_physical_up" else "low",
        "trajectory_plane": pca_diagnostics,
        "origin": [round(float(v), 6) for v in origin],
        "basis_e1": [round(float(v), 6) for v in e1],
        "basis_e2": [round(float(v), 6) for v in e2],
        "normal": [round(float(v), 6) for v in normal],
    }, (e1, e2, normal)


def _step_window_stats(
    steps: np.ndarray,
    start: int,
    end: int,
    stationary_threshold: float,
) -> tuple[float | None, float | None, int]:
    values = steps[max(0, start): min(len(steps), end)]
    values = values[np.isfinite(values) & (values > stationary_threshold)]
    if values.size < max(4, (end - start) // 3):
        return None, None, int(values.size)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad / max(median, 1e-9), int(values.size)


def _stabilize_scale_regimes(
    plan_points: np.ndarray,
    confidence: np.ndarray,
    run_params: Mapping[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair persistent per-segment scale resets without normalizing walking speed.

    R3 fallback may re-anchor a long run with a new metric factor.  The upstream
    implementation warns when that factor changes by more than 2x but still
    accepts it.  We only repair a change when stable windows exist on both
    sides and there is strong fallback evidence (known forced boundary, a
    confidence collapse, or an extreme ratio).  Ordinary speed changes remain
    untouched.
    """
    diagnostics: dict[str, Any] = {
        "method": "persistent_step_regime_guard",
        "applied": False,
        "applied_count": 0,
        "regime_changes": [],
    }
    if len(plan_points) < 40:
        diagnostics["quality"] = "too_short"
        return plan_points.copy(), diagnostics

    deltas = np.diff(plan_points[:, :2], axis=0)
    steps = np.linalg.norm(deltas, axis=1)
    positive = steps[np.isfinite(steps) & (steps > 1e-9)]
    if positive.size < 24:
        diagnostics["quality"] = "insufficient_motion"
        return plan_points.copy(), diagnostics

    global_median = float(np.median(positive))
    stationary_threshold = max(global_median * 0.06, 1e-9)
    window = max(8, min(24, len(steps) // 10))
    if len(steps) < window * 4 + 1:
        diagnostics["quality"] = "insufficient_context"
        diagnostics["window_points"] = window
        return plan_points.copy(), diagnostics

    params = run_params if isinstance(run_params, Mapping) else {}
    try:
        max_segment_frames = max(0, int(params.get("max_segment_frames") or 0))
    except (TypeError, ValueError):
        max_segment_frames = 0
    fallback_enabled = bool(params.get("online_fallback_enabled"))
    metric_enabled = bool(params.get("metric_scale_enabled"))

    finite_confidence = confidence[np.isfinite(confidence)]
    confidence_median = float(np.median(finite_confidence)) if finite_confidence.size else None
    candidates: list[dict[str, Any]] = []
    for boundary in range(window * 2, len(steps) - window * 2 + 1):
        before_far, before_far_cv, _ = _step_window_stats(
            steps, boundary - 2 * window, boundary - window, stationary_threshold
        )
        before, before_cv, _ = _step_window_stats(
            steps, boundary - window, boundary, stationary_threshold
        )
        after, after_cv, _ = _step_window_stats(
            steps, boundary, boundary + window, stationary_threshold
        )
        after_far, after_far_cv, _ = _step_window_stats(
            steps, boundary + window, boundary + 2 * window, stationary_threshold
        )
        if None in (before_far, before, after, after_far):
            continue
        assert before_far is not None and before is not None and after is not None and after_far is not None
        assert before_far_cv is not None and before_cv is not None and after_cv is not None and after_far_cv is not None
        if max(before_far_cv, before_cv, after_cv, after_far_cv) > 0.55:
            continue
        pre_stability = max(before_far, before) / max(min(before_far, before), 1e-9)
        post_stability = max(after, after_far) / max(min(after, after_far), 1e-9)
        if pre_stability > 1.35 or post_stability > 1.35:
            continue

        ratio = after / max(before, 1e-9)
        symmetric_ratio = max(ratio, 1.0 / max(ratio, 1e-9))
        if symmetric_ratio < 1.45:
            continue

        near_forced_boundary = False
        forced_distance: int | None = None
        if fallback_enabled and max_segment_frames > 0:
            nearest = int(round(boundary / max_segment_frames)) * max_segment_frames
            forced_distance = abs(boundary - nearest)
            near_forced_boundary = nearest > 0 and forced_distance <= max(window, 12)

        local_confidence_low = False
        local_conf = confidence[
            max(0, boundary - max(3, window // 3)):
            min(len(confidence), boundary + max(3, window // 3) + 1)
        ]
        local_conf = local_conf[np.isfinite(local_conf)]
        local_confidence = float(np.median(local_conf)) if local_conf.size else None
        if local_confidence is not None and confidence_median is not None and confidence_median > 0:
            local_confidence_low = local_confidence < confidence_median * 0.72

        strong_evidence = fallback_enabled and symmetric_ratio >= 2.35
        fallback_evidence = (
            (near_forced_boundary and metric_enabled and symmetric_ratio >= 1.45)
            or (near_forced_boundary and symmetric_ratio >= 1.8)
            or (fallback_enabled and local_confidence_low and symmetric_ratio >= 1.7)
            or (metric_enabled and local_confidence_low and symmetric_ratio >= 1.6)
        )
        if not strong_evidence and not fallback_evidence:
            continue

        candidates.append({
            "boundary": boundary,
            "ratio": ratio,
            "symmetric_ratio": symmetric_ratio,
            "score": abs(math.log(max(ratio, 1e-9))),
            "near_forced_boundary": near_forced_boundary,
            "forced_boundary_distance": forced_distance,
            "local_confidence_low": local_confidence_low,
            "local_confidence": local_confidence,
        })

    # One physical reset creates a plateau of nearby candidate windows.  Keep
    # the strongest point in each plateau, then apply resets chronologically.
    selected: list[dict[str, Any]] = []
    def candidate_priority(item: Mapping[str, Any]) -> tuple[float, int, int]:
        forced_distance = item.get("forced_boundary_distance")
        return (
            float(item["score"]),
            1 if item.get("near_forced_boundary") else 0,
            -int(forced_distance) if forced_distance is not None else -10_000,
        )

    for candidate in sorted(candidates, key=candidate_priority, reverse=True):
        if any(abs(int(candidate["boundary"]) - int(item["boundary"])) <= window for item in selected):
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: int(item["boundary"]))

    corrected_deltas = deltas.copy()
    for candidate in selected:
        boundary = int(candidate["boundary"])
        correction = 1.0 / float(candidate["ratio"])
        corrected_deltas[boundary:] *= correction
        diagnostics["regime_changes"].append({
            "trajectory_index": boundary,
            "step_ratio": round(float(candidate["ratio"]), 5),
            "applied_scale": round(correction, 5),
            "near_forced_boundary": bool(candidate["near_forced_boundary"]),
            "forced_boundary_distance": candidate["forced_boundary_distance"],
            "local_confidence_low": bool(candidate["local_confidence_low"]),
        })

    corrected = plan_points.copy()
    if selected:
        corrected[1:, :2] = plan_points[0, :2] + np.cumsum(corrected_deltas, axis=0)
    output_steps = np.linalg.norm(np.diff(corrected[:, :2], axis=0), axis=1)
    output_positive = output_steps[np.isfinite(output_steps) & (output_steps > 1e-9)]
    diagnostics.update({
        "quality": "corrected" if selected else "stable_or_unconfirmed",
        "applied": bool(selected),
        "applied_count": len(selected),
        "window_points": window,
        "input_step_median": round(global_median, 6),
        "output_step_median": round(float(np.median(output_positive)), 6) if output_positive.size else 0.0,
        "fallback_enabled": fallback_enabled,
        "metric_scale_enabled": metric_enabled,
        "max_segment_frames": max_segment_frames or None,
    })
    return corrected, diagnostics


def _trajectory_headings(points: np.ndarray) -> tuple[np.ndarray, int]:
    count = len(points)
    if count < 2:
        return np.zeros(count, dtype=np.float64), 1
    half_window = max(2, min(8, count // 90 if count >= 90 else 2))
    headings = np.full(count, np.nan, dtype=np.float64)
    for i in range(count):
        left = max(0, i - half_window)
        right = min(count - 1, i + half_window)
        vector = points[right, :2] - points[left, :2]
        if float(np.linalg.norm(vector)) <= 1e-7:
            continue
        headings[i] = math.atan2(float(vector[1]), float(vector[0]))
    valid = np.isfinite(headings)
    if not valid.any():
        return np.zeros(count, dtype=np.float64), half_window
    indices = np.arange(count, dtype=np.float64)
    headings = np.interp(indices, indices[valid], np.unwrap(headings[valid]))
    return headings, half_window


def _mean_heading(headings: np.ndarray, start: int, end: int) -> tuple[float | None, float]:
    """Return circular mean and concentration for an inclusive range."""
    values = headings[max(0, start): min(len(headings), end + 1)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, 0.0
    mean_sin = float(np.mean(np.sin(values)))
    mean_cos = float(np.mean(np.cos(values)))
    concentration = float(math.hypot(mean_sin, mean_cos))
    if concentration <= 1e-8:
        return None, 0.0
    return math.atan2(mean_sin, mean_cos), concentration


def _camera_heading_signal(
    rotations: Sequence[np.ndarray] | None,
    plan_points: np.ndarray,
    floor_basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project camera-forward axes into plan space and validate them.

    Camera orientation is useful only when it follows motion.  We align its
    global sign with position tangents and expose a reliability score so an
    unrelated camera pan cannot inflate a path turn.
    """
    count = len(plan_points)
    empty = np.full(count, np.nan, dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "available": False,
        "reliable": False,
        "valid_headings": 0,
        "alignment": None,
        "alignment_support": None,
        "camera_axis": None,
    }
    if not rotations or floor_basis is None or len(rotations) != count:
        return empty, diagnostics

    e1, e2, normal = floor_basis
    trajectory_headings, _ = _trajectory_headings(plan_points)
    required_valid = max(6, min(40, count // 5))
    best: tuple[float, float, float, np.ndarray, int, int] | None = None
    # R3 normally follows the OpenCV Z-forward convention.  Test X as well:
    # its yaw deltas are equivalent, and this keeps the signal usable for
    # checkpoints exported with a different camera-axis convention.
    for axis in (2, 0):
        headings = empty.copy()
        for index, rotation in enumerate(rotations):
            if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
                continue
            direction = rotation[:, axis]
            direction = direction - float(np.dot(direction, normal)) * normal
            if float(np.linalg.norm(direction)) <= 1e-7:
                continue
            headings[index] = math.atan2(float(np.dot(direction, e2)), float(np.dot(direction, e1)))

        valid = np.isfinite(headings) & np.isfinite(trajectory_headings)
        valid_count = int(valid.sum())
        if valid_count < required_valid:
            continue
        alignment_values = np.cos(headings[valid] - trajectory_headings[valid])
        # The camera axis may be saved in the opposite sign.  Choose a single
        # global sign, never a per-frame flip, to preserve turn direction.
        if float(np.median(alignment_values)) < 0.0:
            headings[np.isfinite(headings)] += math.pi
            alignment_values = np.cos(headings[valid] - trajectory_headings[valid])
        alignment = float(np.median(alignment_values))
        support = float(np.mean(alignment_values >= math.cos(math.radians(55.0))))
        score = alignment * support
        if best is None or score > best[0]:
            best = (score, alignment, support, headings, valid_count, axis)

    if best is None:
        return empty, diagnostics

    _, alignment, support, headings, valid_count, axis = best
    diagnostics.update({
        "available": True,
        "valid_headings": valid_count,
        "alignment": round(alignment, 4),
        "alignment_support": round(support, 4),
        "camera_axis": "z" if axis == 2 else "x",
        "reliable": bool(alignment >= 0.55 and support >= 0.65),
    })
    return headings, diagnostics


def _smoothed_heading_rates(headings: np.ndarray) -> np.ndarray:
    """Return robust per-pose yaw changes in degrees.

    Multi-scale maxima are useful for one isolated corner, but on a route
    with several same-direction corners a large window can cover two or more
    turns and merge them into one event.  Per-pose curvature keeps the quiet
    straight sections between those turns, so every corner gets its own
    event before its full angle is measured from stable side headings.
    """
    if len(headings) < 2:
        return np.zeros(0, dtype=np.float64)
    rates = np.array([
        math.degrees(_angle_delta_rad(float(headings[index + 1]), float(headings[index])))
        for index in range(len(headings) - 1)
    ], dtype=np.float64)
    if len(rates) >= 3:
        median_filtered = rates.copy()
        for index in range(len(rates)):
            median_filtered[index] = float(np.median(rates[max(0, index - 1): index + 2]))
        rates = np.convolve(median_filtered, np.array([0.25, 0.5, 0.25]), mode="same")
    return rates


def _curvature_turn_events(
    plan_points: np.ndarray,
    headings: np.ndarray,
    half_window: int,
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    """Split sign-consistent curvature into distinct physical turn events."""
    rates = _smoothed_heading_rates(headings)
    if rates.size == 0:
        return [], rates, 0.35

    # 0.35 degrees per selected pose still finds a 20-degree R3 arc spread
    # over many frames.  The minimum accumulated event angle below rejects
    # isolated sub-degree pose noise.
    rate_threshold = 0.35
    signs = np.where(rates >= rate_threshold, 1, np.where(rates <= -rate_threshold, -1, 0))
    if len(signs) >= 3:
        for index in range(1, len(signs) - 1):
            if signs[index - 1] == signs[index + 1] != 0 and signs[index] != signs[index - 1]:
                signs[index] = signs[index - 1]

    raw_groups: list[tuple[int, int, int]] = []
    start: int | None = None
    last_active = -1
    active_sign = 0
    allowed_gap = max(2, half_window)
    for index, sign_value in enumerate(signs):
        sign = int(sign_value)
        if sign == 0:
            if start is not None and index - last_active > allowed_gap:
                raw_groups.append((start, last_active, active_sign))
                start = None
                active_sign = 0
            continue
        if start is None:
            start = index
            last_active = index
            active_sign = sign
        elif sign == active_sign:
            last_active = index
        else:
            raw_groups.append((start, last_active, active_sign))
            start = index
            last_active = index
            active_sign = sign
    if start is not None:
        raw_groups.append((start, last_active, active_sign))

    # Reconnect fragments of one rounded corner, while keeping real corners
    # separated by the longer low-curvature straight between them.
    merged_groups: list[tuple[int, int, int]] = []
    merge_gap = max(3, half_window * 2)
    for group in raw_groups:
        if (
            merged_groups
            and merged_groups[-1][2] == group[2]
            and group[0] - merged_groups[-1][1] <= merge_gap
        ):
            previous = merged_groups[-1]
            merged_groups[-1] = (previous[0], group[1], previous[2])
        else:
            merged_groups.append(group)

    events: list[dict[str, Any]] = []
    anchor_window = max(3, half_window)
    for rate_start, rate_end, sign in merged_groups:
        start_point = max(0, rate_start - half_window)
        end_point = min(len(plan_points) - 1, rate_end + 1 + half_window)
        before, before_concentration = _mean_heading(
            headings,
            max(0, start_point - anchor_window),
            start_point,
        )
        after, after_concentration = _mean_heading(
            headings,
            end_point,
            min(len(headings) - 1, end_point + anchor_window),
        )
        if before is None or after is None:
            continue
        angle = math.degrees(_angle_delta_rad(after, before))
        if abs(angle) < 12.0 or (angle > 0) != (sign > 0):
            continue

        rate_slice = np.abs(rates[rate_start:rate_end + 1])
        if rate_slice.size and float(rate_slice.sum()) > 1e-9:
            offsets = np.arange(rate_start, rate_end + 1, dtype=np.float64)
            center = int(round(float(np.sum(offsets * rate_slice) / np.sum(rate_slice))))
            center = min(len(plan_points) - 1, center + 1)
        else:
            center = (start_point + end_point) // 2
        event_steps = np.linalg.norm(
            np.diff(plan_points[start_point:end_point + 1, :2], axis=0),
            axis=1,
        )
        events.append({
            "center": center,
            "angle": angle,
            "magnitude": abs(angle),
            "start": start_point,
            "end": end_point,
            "span": end_point - start_point,
            "rate_start": rate_start,
            "rate_end": rate_end,
            "support": float(event_steps.sum()) if event_steps.size else 0.0,
            "before_concentration": before_concentration,
            "after_concentration": after_concentration,
        })
    return events, rates, rate_threshold


def _detect_turns(
    plan_points: np.ndarray,
    source_frame_indices: Sequence[int | None],
    confidence: np.ndarray,
    rotations: Sequence[np.ndarray] | None = None,
    floor_basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(plan_points) < 6:
        return [], {"method": "curvature_events_camera_fusion", "quality": "too_short"}

    local_headings, half_window = _trajectory_headings(plan_points)
    candidates, heading_rates, rate_threshold = _curvature_turn_events(
        plan_points,
        local_headings,
        half_window,
    )
    camera_headings, camera_diagnostics = _camera_heading_signal(rotations, plan_points, floor_basis)

    # Camera yaw is evaluated per event.  Requiring a simultaneous, same-sign
    # position bend prevents an operator pan on a straight path from becoming
    # a turn, while still rescuing the exact failure we care about: R3 shows a
    # 12--25 degree arc but its camera rotations contain the physical 90.
    camera_overrides = 0
    for candidate in candidates:
        candidate["event_angle"] = candidate["angle"]
        candidate["angle_source"] = "trajectory_curvature"
        candidate["orientation_angle"] = None
        if not camera_diagnostics["available"]:
            continue
        anchor_window = max(3, half_window)
        before, before_concentration = _mean_heading(
            camera_headings,
            max(0, candidate["start"] - anchor_window),
            candidate["start"],
        )
        after, after_concentration = _mean_heading(
            camera_headings,
            candidate["end"],
            min(len(camera_headings) - 1, candidate["end"] + anchor_window),
        )
        if before is None or after is None or min(before_concentration, after_concentration) < 0.8:
            continue
        orientation_angle = math.degrees(_angle_delta_rad(after, before))
        candidate["orientation_angle"] = orientation_angle
        position_magnitude = abs(float(candidate["angle"]))
        orientation_magnitude = abs(orientation_angle)
        same_direction = (
            position_magnitude >= 12.0
            and math.copysign(1.0, float(candidate["angle"]))
            == math.copysign(1.0, orientation_angle)
        )
        if (
            same_direction
            and 32.0 <= orientation_magnitude < 150.0
            and position_magnitude < orientation_magnitude * 0.8
        ):
            candidate["event_angle"] = orientation_angle
            candidate["angle_source"] = "camera_orientation"
            camera_overrides += 1

    finite_conf = confidence[np.isfinite(confidence)]
    weak_conf = float(np.percentile(finite_conf, 12)) if finite_conf.size >= 5 else float("-inf")
    turns: list[dict[str, Any]] = []
    last_turn_index = -10_000

    for candidate in candidates:
        angle = float(candidate["event_angle"])
        magnitude = abs(angle)
        if magnitude < 28.0:
            continue
        center = int(candidate["center"])
        movement = float(candidate["support"])
        if movement <= 1e-6:
            continue
        pose_is_weak = not np.isfinite(confidence[center]) or confidence[center] <= weak_conf
        if pose_is_weak and magnitude < 45.0 and candidate["span"] < max(3, half_window * 2):
            continue
        if center - last_turn_index < max(half_window, 3):
            continue

        if magnitude >= 135.0:
            turn_type = "u_turn"
        else:
            turn_type = "left" if angle > 0 else "right"
        source_index = source_frame_indices[center] if center < len(source_frame_indices) else None
        turns.append({
            "frame_index": source_index if source_index is not None else center,
            "r3_frame_index": center,
            "source_frame_index": source_index,
            "trajectory_index": center,
            "angle_degrees": round(angle, 1),
            "position": [round(float(v), 6) for v in plan_points[center]],
            "turn_type": turn_type,
            "confidence": round(float(confidence[center]), 5) if np.isfinite(confidence[center]) else None,
            "angle_source": candidate["angle_source"],
            "trajectory_angle_degrees": round(float(candidate["angle"]), 1),
            "camera_angle_degrees": (
                round(float(candidate["orientation_angle"]), 1)
                if candidate["orientation_angle"] is not None
                else None
            ),
            "span_points": int(candidate["span"]),
            "approach_index": int(candidate["start"]),
            "exit_index": int(candidate["end"]),
        })
        last_turn_index = center
    return turns, {
        "method": "curvature_events_camera_fusion",
        "local_half_window": half_window,
        "curvature_rate_threshold_degrees": rate_threshold,
        "max_smoothed_rate_degrees": (
            round(float(np.max(np.abs(heading_rates))), 4) if heading_rates.size else 0.0
        ),
        "candidate_count": len(candidates),
        "active_groups": len(candidates),
        "camera_overrides": camera_overrides,
        "camera_orientation": camera_diagnostics,
    }


def _apply_camera_turn_corrections(
    plan_points: np.ndarray,
    turns: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rotate future plan steps by camera-confirmed missing turn angles.

    R3 translation can turn only 20 degrees while its c2w rotations correctly
    turn 90.  Rotating the remainder around the event centre preserves every
    measured step length and the raw 3D reconstruction, but restores the
    missing yaw on the 2D plan so loops do not collapse into a broad arc.
    """
    corrected = plan_points.copy()
    applied: list[dict[str, Any]] = []
    for turn in sorted(turns, key=lambda item: int(item.get("trajectory_index", 0))):
        if turn.get("angle_source") != "camera_orientation":
            continue
        try:
            target_angle = float(turn["angle_degrees"])
            trajectory_angle = float(turn["trajectory_angle_degrees"])
            pivot_index = int(turn["trajectory_index"])
        except (KeyError, TypeError, ValueError):
            continue
        missing_angle = target_angle - trajectory_angle
        if (
            abs(missing_angle) < 12.0
            or abs(missing_angle) > 100.0
            or pivot_index < 0
            or pivot_index >= len(corrected) - 1
        ):
            continue
        radians = math.radians(missing_angle)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        pivot = corrected[pivot_index, :2].copy()
        corrected[pivot_index + 1:, :2] = (
            rotation @ (corrected[pivot_index + 1:, :2] - pivot).T
        ).T + pivot
        applied.append({
            "trajectory_index": pivot_index,
            "trajectory_angle_degrees": round(trajectory_angle, 1),
            "camera_angle_degrees": round(target_angle, 1),
            "applied_delta_degrees": round(missing_angle, 1),
        })
    return corrected, {
        "method": "camera_confirmed_piecewise_rotation",
        "applied": bool(applied),
        "applied_count": len(applied),
        "corrections": applied,
    }


def _source_frame_indices(camera_poses: Sequence[Mapping[str, Any]], frame_selection: Any) -> list[int | None]:
    source_indices: list[Any] = []
    if isinstance(frame_selection, Mapping):
        raw = frame_selection.get("source_indices")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            source_indices = list(raw)
    result: list[int | None] = []
    for index, pose in enumerate(camera_poses):
        frame = pose.get("frame", index)
        try:
            frame_index = int(frame)
        except Exception:
            frame_index = index
        source: int | None = None
        if 0 <= frame_index < len(source_indices):
            try:
                source = int(source_indices[frame_index])
            except Exception:
                source = None
        result.append(source)
    return result


def build_r3_trajectory(
    camera_poses: Iterable[Mapping[str, Any]],
    pose_confidence: Any = None,
    frame_selection: Any = None,
    run_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build matching 3D and floor-plan trajectories from R3 c2w poses.

    Returned ``trajectory`` is deliberately the plan-space trajectory for
    backward compatibility.  Consumers that render a 3D scene must use
    ``raw_trajectory_3d`` instead.
    """
    valid_poses: list[Mapping[str, Any]] = []
    points: list[np.ndarray] = []
    rotations: list[np.ndarray] = []

    for item in camera_poses:
        pose = item.get("pose") if isinstance(item, Mapping) else None
        try:
            matrix = np.asarray(pose, dtype=np.float64)
        except Exception:
            continue
        if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 4:
            continue
        translation = matrix[:3, 3]
        rotation = matrix[:3, :3]
        if not np.isfinite(translation).all() or not np.isfinite(rotation).all():
            continue
        valid_poses.append(item)
        points.append(translation.astype(np.float64))
        rotations.append(rotation.astype(np.float64))

    if not points:
        return {
            "trajectory": [],
            "plan_trajectory": [],
            "raw_plan_trajectory": [],
            "raw_trajectory_3d": [],
            "raw_camera_points": [],
            "turn_points": [],
            "source_frame_indices": [],
            "trajectory_quality": {"quality": "empty", "raw_points": 0},
        }

    raw = np.vstack(points)
    confidence = _pose_confidences(valid_poses, pose_confidence)
    source_indices = _source_frame_indices(valid_poses, frame_selection)
    cleaned_3d, filter_quality = _clean_positions(raw, confidence)
    projected_plan, plane, floor_basis = _project_to_floor(cleaned_3d, rotations)
    raw_plan, scale_stability = _stabilize_scale_regimes(
        projected_plan,
        confidence,
        run_params,
    )
    turns, turn_detection = _detect_turns(
        raw_plan,
        source_indices,
        confidence,
        rotations=rotations,
        floor_basis=floor_basis,
    )
    plan, heading_correction = _apply_camera_turn_corrections(raw_plan, turns)
    for turn in turns:
        index = int(turn.get("trajectory_index", -1))
        if 0 <= index < len(plan):
            turn["position"] = [round(float(value), 6) for value in plan[index]]
    clean_steps = np.linalg.norm(np.diff(plan[:, :2], axis=0), axis=1) if len(plan) > 1 else np.array([])

    return {
        "trajectory": _json_points(plan),
        "plan_trajectory": _json_points(plan),
        "raw_plan_trajectory": _json_points(raw_plan),
        "raw_trajectory_3d": _json_points(cleaned_3d),
        "raw_camera_points": _json_points(raw),
        "turn_points": turns,
        "source_frame_indices": source_indices,
        "pose_confidence": [round(float(v), 6) if np.isfinite(v) else None for v in confidence],
        "trajectory_quality": {
            **filter_quality,
            "projection": plane,
            "scale_stability": scale_stability,
            "turn_detection": turn_detection,
            "heading_correction": heading_correction,
            "cleaned_distance": round(float(clean_steps.sum()), 6) if clean_steps.size else 0.0,
            "turns_detected": len(turns),
        },
    }
