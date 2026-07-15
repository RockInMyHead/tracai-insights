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


R3_TRAJECTORY_POSTPROCESS_VERSION = 5


def summarize_fallback_edges(
    edges: Any,
    point_count: int = 0,
    bridge_window: int = 10,
) -> dict[str, Any]:
    """Recover accepted fallback boundaries from R3's pose-edge log.

    Upstream R3 exports every accepted replay as a cluster of ``bridge``
    edges.  Those clusters are the only trustworthy evidence that a new
    reconstruction segment started.  A nominal ``max_segment_frames`` value
    is not a boundary: confidence-triggered fallbacks happen at different
    frames, and treating every multiple as one caused production scale drift.
    """
    bridge_frames: list[int] = []
    bridge_edges = 0
    for edge in edges if isinstance(edges, list) else []:
        if not isinstance(edge, Mapping) or str(edge.get("edge_type")) != "bridge":
            continue
        try:
            frame_i = int(edge.get("frame_i"))
            frame_j = int(edge.get("frame_j"))
        except (TypeError, ValueError):
            continue
        for frame in (frame_i, frame_j):
            if frame < 0 or (point_count > 0 and frame >= point_count):
                continue
            bridge_frames.append(frame)
        bridge_edges += 1

    unique_frames = sorted(set(bridge_frames))
    if not unique_frames:
        return {
            "source": "pose_edge_log",
            "bridge_edge_count": bridge_edges,
            "events": [],
            "boundaries": [],
        }

    # Replay frames need not be consecutive when low-confidence images were
    # skipped.  A gap of several bridge windows still cleanly separates the
    # next accepted fallback on normal long-video presets.
    cluster_gap = max(4, int(bridge_window) * 3)
    clusters: list[list[int]] = [[unique_frames[0]]]
    for frame in unique_frames[1:]:
        if frame - clusters[-1][-1] > cluster_gap:
            clusters.append([frame])
        else:
            clusters[-1].append(frame)

    events: list[dict[str, Any]] = []
    boundaries: list[int] = []
    for cluster in clusters:
        boundary = cluster[-1] + 1
        if point_count > 0 and boundary >= point_count:
            continue
        boundaries.append(boundary)
        events.append({
            "boundary": boundary,
            "bridge_start": cluster[0],
            "bridge_end": cluster[-1],
            "bridge_frames": len(cluster),
        })

    return {
        "source": "pose_edge_log",
        "bridge_edge_count": bridge_edges,
        "cluster_gap": cluster_gap,
        "events": events,
        "boundaries": boundaries,
    }


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


def _camera_rotation_axis_normal(
    rotations: Sequence[np.ndarray],
    camera_up: np.ndarray | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Estimate gravity from the common world-space axis of camera turns.

    Camera-local up is tilted whenever the operator points the camera toward
    the floor.  During real left/right turns, however, c2w rotations share the
    physical vertical as their world-space rotation axis.  Accumulating axis
    outer-products removes the left/right sign while retaining that vertical.
    """
    diagnostics: dict[str, Any] = {
        "available": False,
        "reliable": False,
        "samples": 0,
        "dominance": None,
        "camera_up_alignment": None,
    }
    count = len(rotations)
    if count < 12:
        return None, diagnostics

    covariance = np.zeros((3, 3), dtype=np.float64)
    samples = 0
    spans = [span for span in (4, 8, 16, 24) if span < count]
    min_angle = math.radians(1.5)
    max_angle = math.radians(75.0)
    for span in spans:
        for index in range(count - span):
            first = rotations[index]
            second = rotations[index + span]
            if (
                first.shape != (3, 3)
                or second.shape != (3, 3)
                or not np.isfinite(first).all()
                or not np.isfinite(second).all()
            ):
                continue
            relative = second @ first.T
            cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
            angle = math.acos(cosine)
            if angle < min_angle or angle > max_angle:
                continue
            skew = np.array([
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ], dtype=np.float64)
            skew_norm = float(np.linalg.norm(skew))
            if skew_norm <= 1e-8:
                continue
            axis = skew / skew_norm
            weight = min(angle, math.radians(30.0)) ** 2
            covariance += weight * np.outer(axis, axis)
            samples += 1

    diagnostics["samples"] = samples
    diagnostics["spans"] = spans
    if samples < 12:
        return None, diagnostics
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return None, diagnostics
    if not np.isfinite(eigenvalues).all() or float(eigenvalues[-1]) <= 1e-12:
        return None, diagnostics

    normal = _normalize(eigenvectors[:, -1])
    dominance = float(eigenvalues[-1] / max(float(eigenvalues[-2]), 1e-12))
    alignment = None
    if camera_up is not None:
        if float(np.dot(normal, camera_up)) < 0.0:
            normal = -normal
        alignment = float(np.dot(normal, camera_up))
    else:
        dominant = int(np.argmax(np.abs(normal)))
        if normal[dominant] < 0.0:
            normal = -normal

    reliable = bool(dominance >= 2.0 and (alignment is None or alignment >= 0.45))
    diagnostics.update({
        "available": True,
        "reliable": reliable,
        "dominance": round(dominance, 6),
        "camera_up_alignment": round(alignment, 6) if alignment is not None else None,
        "normal": [round(float(value), 6) for value in normal],
    })
    return normal, diagnostics


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
    rotation_axis, rotation_axis_diagnostics = _camera_rotation_axis_normal(rotations, camera_up)
    pca_normal, pca_diagnostics = _trajectory_plane_normal(points)
    normal_sign_source = "camera_physical_up"

    if rotation_axis is not None and rotation_axis_diagnostics["reliable"]:
        normal = rotation_axis
        method = "camera_rotation_axis"
    elif pca_normal is not None and pca_diagnostics["eligible"]:
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
        "camera_rotation_axis": rotation_axis_diagnostics,
        "trajectory_plane": pca_diagnostics,
        "origin": [round(float(v), 6) for v in origin],
        "basis_e1": [round(float(v), 6) for v in e1],
        "basis_e2": [round(float(v), 6) for v in e2],
        "normal": [round(float(v), 6) for v in normal],
    }, (e1, e2, normal)


def _stabilize_scale_regimes(
    plan_points: np.ndarray,
    confidence: np.ndarray,
    source_indices: Sequence[int | None],
    run_params: Mapping[str, Any] | None,
    source_timestamps: Sequence[float | None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize confirmed fallback scale epochs without cumulative drift.

    The previous guard searched every point for a step-size change and then
    multiplied the *entire remaining tail*.  Several detections therefore
    compounded (2.74x × 1.78x × ... in the production run).  Here boundaries
    must come from R3's actual bridge-edge log, motion is normalized by exact
    presentation time when available, adjacent fallback segments with the same
    scale are grouped, and every epoch is scaled independently against the
    first stable epoch.
    """
    diagnostics: dict[str, Any] = {
        "method": "explicit_fallback_scale_epochs",
        "applied": False,
        "applied_count": 0,
        "regime_changes": [],
        "cumulative_scaling": False,
        "source_delta_normalized": True,
    }
    if len(plan_points) < 24:
        diagnostics["quality"] = "too_short"
        return plan_points.copy(), diagnostics

    deltas = np.diff(plan_points[:, :2], axis=0)
    steps = np.linalg.norm(deltas, axis=1)
    sample_axis: np.ndarray | None = None
    if source_timestamps is not None:
        try:
            timestamps = np.asarray([
                float(value) if value is not None else np.nan for value in source_timestamps
            ], dtype=np.float64)
        except (TypeError, ValueError):
            timestamps = np.asarray([], dtype=np.float64)
        valid_timestamps = np.isfinite(timestamps)
        minimum_timestamp_support = max(2, int(math.ceil(len(plan_points) * 0.8)))
        if timestamps.size == len(plan_points) and int(valid_timestamps.sum()) >= minimum_timestamp_support:
            indices = np.arange(len(timestamps), dtype=np.float64)
            if not valid_timestamps.all():
                timestamps = np.interp(
                    indices,
                    indices[valid_timestamps],
                    timestamps[valid_timestamps],
                )
            if np.count_nonzero(np.diff(timestamps) > 0) >= max(1, len(timestamps) - 2):
                sample_axis = timestamps
                diagnostics["motion_time_base"] = "presentation_timestamp_seconds"

    if sample_axis is None:
        try:
            source = np.asarray([
                float(value) if value is not None else np.nan for value in source_indices
            ], dtype=np.float64)
        except (TypeError, ValueError):
            source = np.arange(len(plan_points), dtype=np.float64)
        if source.size != len(plan_points):
            source = np.arange(len(plan_points), dtype=np.float64)
        valid_source = np.isfinite(source)
        if not valid_source.all():
            indices = np.arange(len(source), dtype=np.float64)
            if valid_source.any():
                source = np.interp(indices, indices[valid_source], source[valid_source])
            else:
                source = indices
        sample_axis = source
        diagnostics["motion_time_base"] = "source_frame_index"

    source_deltas = np.diff(sample_axis)
    positive_source_deltas = source_deltas[np.isfinite(source_deltas) & (source_deltas > 0)]
    default_source_delta = float(np.median(positive_source_deltas)) if positive_source_deltas.size else 1.0
    source_deltas = np.where(
        np.isfinite(source_deltas) & (source_deltas > 0),
        source_deltas,
        default_source_delta,
    )
    velocities = steps / np.maximum(source_deltas, 1e-9)
    positive = velocities[np.isfinite(velocities) & (velocities > 1e-9)]
    if positive.size < 16:
        diagnostics["quality"] = "insufficient_motion"
        return plan_points.copy(), diagnostics

    params = run_params if isinstance(run_params, Mapping) else {}
    fallback_enabled = bool(params.get("online_fallback_enabled"))
    metric_enabled = bool(params.get("metric_scale_enabled"))
    raw_boundaries = params.get("fallback_boundaries")
    if not isinstance(raw_boundaries, (list, tuple)):
        raw_boundaries = []
    boundaries: list[int] = []
    for value in raw_boundaries:
        try:
            boundary = int(value)
        except (TypeError, ValueError):
            continue
        if 8 <= boundary <= len(plan_points) - 8:
            boundaries.append(boundary)
    boundaries = sorted(set(boundaries))
    diagnostics["fallback_boundaries"] = boundaries
    diagnostics["fallback_boundary_source"] = params.get("fallback_boundary_source")
    if not fallback_enabled or not boundaries:
        diagnostics.update({
            "quality": "no_explicit_fallback_boundaries",
            "input_step_median": round(float(np.median(positive)), 6),
            "output_step_median": round(float(np.median(positive)), 6),
            "fallback_enabled": fallback_enabled,
            "metric_scale_enabled": metric_enabled,
        })
        return plan_points.copy(), diagnostics

    stationary_threshold = max(float(np.median(positive)) * 0.05, 1e-9)
    segment_limits = [0, *boundaries, len(plan_points)]
    segments: list[dict[str, Any]] = []
    segment_velocity_values: list[np.ndarray] = []
    for start, end in zip(segment_limits, segment_limits[1:]):
        values = velocities[start: max(start, end - 1)]
        values = values[np.isfinite(values) & (values > stationary_threshold)]
        segment_velocity_values.append(values)
        median = float(np.median(values)) if values.size >= 8 else None
        segments.append({
            "start": start,
            "end": end,
            "motion_samples": int(values.size),
            "median_velocity": median,
        })

    valid_segments = [segment for segment in segments if segment["median_velocity"] is not None]
    if not valid_segments:
        diagnostics["quality"] = "insufficient_segment_motion"
        diagnostics["segments"] = segments
        return plan_points.copy(), diagnostics

    # Merge adjacent fallback segments whose robust motion scale agrees.  A
    # long production run can contain many bridge resets inside one unchanged
    # scale epoch; applying a new multiplier at each reset is the old bug.
    regimes: list[dict[str, Any]] = []
    merge_ratio = 1.45
    for segment_index, segment in enumerate(segments):
        median = segment["median_velocity"]
        if median is None:
            if regimes:
                regimes[-1]["segment_indices"].append(segment_index)
            else:
                regimes.append({"segment_indices": [segment_index], "valid_values": []})
            continue
        if regimes and regimes[-1]["valid_values"]:
            previous_values = np.concatenate(regimes[-1]["valid_values"])
            previous = float(np.median(previous_values))
            ratio = max(float(median) / max(previous, 1e-9), previous / max(float(median), 1e-9))
        else:
            ratio = 1.0
        if not regimes or ratio > merge_ratio:
            regimes.append({
                "segment_indices": [segment_index],
                "valid_values": [segment_velocity_values[segment_index]],
            })
        else:
            regimes[-1]["segment_indices"].append(segment_index)
            regimes[-1]["valid_values"].append(segment_velocity_values[segment_index])

    regimes = [regime for regime in regimes if regime["segment_indices"]]
    for regime in regimes:
        value_groups = regime.pop("valid_values")
        regime_values = np.concatenate(value_groups) if value_groups else np.asarray([], dtype=np.float64)
        regime["median_velocity"] = float(np.median(regime_values)) if regime_values.size else None
        regime["start"] = segments[regime["segment_indices"][0]]["start"]
        regime["end"] = segments[regime["segment_indices"][-1]]["end"]

    anchor = next((regime for regime in regimes if regime["median_velocity"] is not None), regimes[0])
    reference_velocity = float(anchor["median_velocity"] or np.median(positive))
    segment_factors = np.ones(len(segments), dtype=np.float64)
    for regime in regimes:
        median = regime["median_velocity"]
        factor = 1.0
        raw_ratio = None
        if median is not None and median > 1e-9:
            raw_ratio = float(median) / max(reference_velocity, 1e-9)
            symmetric_ratio = max(raw_ratio, 1.0 / max(raw_ratio, 1e-9))
            if symmetric_ratio >= 1.55:
                factor = float(np.clip(1.0 / raw_ratio, 1.0 / 3.0, 3.0))
        regime["applied_scale"] = factor
        regime["raw_velocity_ratio"] = raw_ratio
        for segment_index in regime["segment_indices"]:
            segment_factors[segment_index] = factor
        if abs(factor - 1.0) > 1e-6:
            diagnostics["regime_changes"].append({
                "trajectory_index": int(regime["start"]),
                "end_index": int(regime["end"]),
                "raw_velocity_ratio": round(float(raw_ratio), 5) if raw_ratio is not None else None,
                "applied_scale": round(factor, 5),
            })

    corrected_deltas = deltas.copy()
    for segment_index, (start, end) in enumerate(zip(segment_limits, segment_limits[1:])):
        corrected_deltas[start: max(start, end - 1)] *= segment_factors[segment_index]

    corrected = plan_points.copy()
    if diagnostics["regime_changes"]:
        corrected[1:, :2] = plan_points[0, :2] + np.cumsum(corrected_deltas, axis=0)
    output_steps = np.linalg.norm(np.diff(corrected[:, :2], axis=0), axis=1)
    output_velocities = output_steps / np.maximum(source_deltas, 1e-9)
    output_positive = output_velocities[np.isfinite(output_velocities) & (output_velocities > 1e-9)]
    applied_count = len(diagnostics["regime_changes"])
    diagnostics.update({
        "quality": "corrected" if applied_count else "stable_epochs",
        "applied": bool(applied_count),
        "applied_count": applied_count,
        "reference_velocity_per_source_frame": round(reference_velocity, 6),
        "input_step_median": round(float(np.median(positive)), 6),
        "output_step_median": round(float(np.median(output_positive)), 6) if output_positive.size else 0.0,
        "fallback_enabled": fallback_enabled,
        "metric_scale_enabled": metric_enabled,
        "segments": segments,
        "regimes": regimes,
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
    source_timestamps: Sequence[float | None] | None = None,
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
        timestamp = (
            source_timestamps[center]
            if source_timestamps is not None and center < len(source_timestamps)
            else None
        )
        turns.append({
            "frame_index": source_index if source_index is not None else center,
            "r3_frame_index": center,
            "source_frame_index": source_index,
            "timestamp_seconds": round(float(timestamp), 6) if timestamp is not None else None,
            "trajectory_index": center,
            "angle_degrees": round(angle, 1),
            "position": [round(float(v), 6) for v in plan_points[center]],
            "turn_type": turn_type,
            "confidence": round(float(confidence[center]), 5) if np.isfinite(confidence[center]) else None,
            "angle_source": candidate["angle_source"],
            # Camera orientation is an independent observation.  It may
            # improve the semantic turn estimate, but it must never silently
            # rewrite the R3 translation geometry.
            "geometry_angle_degrees": round(float(candidate["angle"]), 1),
            "observation_angle_degrees": round(angle, 1),
            "geometry_mutated": False,
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
        "camera_geometry_policy": "observation_only",
        "camera_orientation": camera_diagnostics,
    }


def _summarize_camera_turn_disagreements(
    plan_points: np.ndarray,
    turns: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Report camera/translation disagreement without changing geometry.

    A camera pan, a wrong floor normal, or a single orientation sign error can
    make camera yaw disagree with the reconstructed translation.  The old
    post-processor treated camera yaw as an instruction and rotated *every*
    future plan point around the detected turn.  One bad observation therefore
    corrupted the whole remainder of a production route.

    Camera yaw remains valuable evidence for turn classification, so expose
    the discrepancy to diagnostics and the future factor-graph optimizer.  The
    returned geometry is always an exact copy of the R3 position projection.
    """
    observations: list[dict[str, Any]] = []
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
            or pivot_index >= len(plan_points) - 1
        ):
            continue
        observations.append({
            "trajectory_index": pivot_index,
            "trajectory_angle_degrees": round(trajectory_angle, 1),
            "camera_angle_degrees": round(target_angle, 1),
            "disagreement_degrees": round(missing_angle, 1),
            "action": "reported_not_applied",
        })
    return plan_points.copy(), {
        "method": "camera_orientation_observation_only",
        "geometry_mutated": False,
        "applied": False,
        "applied_count": 0,
        "suppressed_count": len(observations),
        "corrections": [],
        "observations": observations,
    }


def _robust_line_fit(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit a 2D line with a small Huber IRLS loop.

    The fit is rotation invariant: a genuine diagonal corridor stays diagonal.
    It is deliberately based only on reconstructed positions; camera yaw is
    never allowed to rewrite trajectory geometry.
    """
    if len(points) < 3 or not np.isfinite(points).all():
        return None
    weights = np.ones(len(points), dtype=np.float64)
    origin = np.median(points, axis=0)
    direction = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(5):
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-9:
            return None
        origin = np.sum(points * weights[:, None], axis=0) / weight_sum
        centered = points - origin
        covariance = (centered * weights[:, None]).T @ centered / weight_sum
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            return None
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        residuals = centered @ normal
        scale = 1.4826 * float(np.median(np.abs(residuals - np.median(residuals))))
        scale = max(scale, 1e-9)
        normalized = np.abs(residuals) / (1.5 * scale)
        weights = np.where(normalized <= 1.0, 1.0, 1.0 / np.maximum(normalized, 1e-9))
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    return origin, direction, normal


def _regularize_straight_runs(
    plan_points: np.ndarray,
    turns: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove low-frequency lateral drift only on strongly straight runs.

    Turn approach/exit windows are protected.  Each remaining run is tested
    independently using path efficiency, heading concentration and robust
    lateral residual.  Corrections taper to zero at both run boundaries so
    turns and endpoints remain observations rather than invented constraints.
    """
    diagnostics: dict[str, Any] = {
        "method": "guarded_piecewise_robust_lines",
        "applied": False,
        "accepted_runs": 0,
        "rejected_runs": 0,
        "runs": [],
        "geometry_source": "position_only",
        "axis_snapping": False,
    }
    count = len(plan_points)
    if count < 18:
        diagnostics["quality"] = "too_short"
        return plan_points.copy(), diagnostics

    protected = sorted(
        (
            max(0, int(turn.get("approach_index", turn.get("trajectory_index", 0)))),
            min(count - 1, int(turn.get("exit_index", turn.get("trajectory_index", 0)))),
        )
        for turn in turns
    )
    runs: list[tuple[int, int]] = []
    cursor = 0
    for start, end in protected:
        if start - cursor >= 11:
            runs.append((cursor, start))
        cursor = max(cursor, end)
    if count - 1 - cursor >= 11:
        runs.append((cursor, count - 1))
    if not protected:
        runs = [(0, count - 1)]

    corrected = plan_points.copy()
    median_step = float(np.median(
        np.linalg.norm(np.diff(plan_points[:, :2], axis=0), axis=1)
    ))
    median_step = max(median_step, 1e-9)

    for start, end in runs:
        xy = plan_points[start:end + 1, :2]
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        path_length = float(steps.sum())
        displacement = float(np.linalg.norm(xy[-1] - xy[0]))
        efficiency = displacement / max(path_length, 1e-9)
        fit = _robust_line_fit(xy)
        record: dict[str, Any] = {
            "start": start,
            "end": end,
            "points": end - start + 1,
            "path_efficiency": round(efficiency, 6),
            "applied": False,
        }
        if fit is None or path_length <= median_step * 10.0:
            record["reason"] = "insufficient_support"
            diagnostics["rejected_runs"] += 1
            diagnostics["runs"].append(record)
            continue
        origin, _, normal = fit
        lateral = (xy - origin) @ normal
        lateral_rmse = float(math.sqrt(float(np.mean(lateral ** 2))))
        normalized_rmse = lateral_rmse / max(displacement, median_step)
        headings = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
        valid_heading = steps > median_step * 0.08
        if valid_heading.any():
            valid_values = np.unwrap(headings[valid_heading])
            concentration = float(abs(np.mean(np.exp(1j * valid_values))))
            heading_spread = math.degrees(float(
                np.percentile(valid_values, 90) - np.percentile(valid_values, 10)
            ))
        else:
            concentration = 0.0
            heading_spread = 180.0
        record.update({
            "heading_concentration": round(concentration, 6),
            "heading_spread_degrees": round(heading_spread, 6),
            "lateral_rmse": round(lateral_rmse, 6),
            "normalized_lateral_rmse": round(normalized_rmse, 6),
        })
        eligible = (
            efficiency >= 0.94
            and concentration >= 0.90
            and heading_spread <= 15.0
            and normalized_rmse <= 0.035
            and lateral_rmse >= median_step * 0.12
        )
        if not eligible:
            record["reason"] = "not_confidently_straight"
            diagnostics["rejected_runs"] += 1
            diagnostics["runs"].append(record)
            continue

        # Keep the measured endpoints and turn boundaries fixed.  The raised
        # cosine reaches full correction after 12.5% of the run.
        sample = np.linspace(0.0, 1.0, len(xy))
        edge_fraction = min(0.125, 4.0 / max(len(xy) - 1, 1))
        ramp_in = np.clip(sample / max(edge_fraction, 1e-9), 0.0, 1.0)
        ramp_out = np.clip((1.0 - sample) / max(edge_fraction, 1e-9), 0.0, 1.0)
        blend = np.sin(0.5 * math.pi * np.minimum(ramp_in, ramp_out)) ** 2
        proposed = xy - (lateral * blend)[:, None] * normal[None, :]
        maximum_shift = float(np.max(np.linalg.norm(proposed - xy, axis=1)))
        if maximum_shift > max(median_step * 4.0, displacement * 0.045):
            record["reason"] = "correction_too_large"
            record["maximum_shift"] = round(maximum_shift, 6)
            diagnostics["rejected_runs"] += 1
            diagnostics["runs"].append(record)
            continue
        corrected[start:end + 1, :2] = proposed
        record["applied"] = True
        record["maximum_shift"] = round(maximum_shift, 6)
        diagnostics["accepted_runs"] += 1
        diagnostics["runs"].append(record)

    diagnostics["applied"] = diagnostics["accepted_runs"] > 0
    diagnostics["quality"] = "regularized" if diagnostics["applied"] else "unchanged"
    return corrected, diagnostics


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


def _source_timestamps(camera_poses: Sequence[Mapping[str, Any]], frame_selection: Any) -> list[float | None]:
    timestamps: list[Any] = []
    if isinstance(frame_selection, Mapping):
        raw = frame_selection.get("source_timestamps_seconds")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            timestamps = list(raw)
    result: list[float | None] = []
    for index, pose in enumerate(camera_poses):
        frame = pose.get("frame", index)
        try:
            frame_index = int(frame)
        except Exception:
            frame_index = index
        timestamp: float | None = None
        if 0 <= frame_index < len(timestamps):
            try:
                candidate = float(timestamps[frame_index])
                timestamp = candidate if math.isfinite(candidate) else None
            except (TypeError, ValueError):
                timestamp = None
        result.append(timestamp)
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
            "source_timestamps_seconds": [],
            "trajectory_quality": {"quality": "empty", "raw_points": 0},
        }

    raw = np.vstack(points)
    confidence = _pose_confidences(valid_poses, pose_confidence)
    source_indices = _source_frame_indices(valid_poses, frame_selection)
    source_timestamps = _source_timestamps(valid_poses, frame_selection)
    cleaned_3d, filter_quality = _clean_positions(raw, confidence)
    projected_plan, plane, floor_basis = _project_to_floor(cleaned_3d, rotations)
    raw_plan, scale_stability = _stabilize_scale_regimes(
        projected_plan,
        confidence,
        source_indices,
        run_params,
        source_timestamps=source_timestamps,
    )
    turns, turn_detection = _detect_turns(
        raw_plan,
        source_indices,
        confidence,
        source_timestamps=source_timestamps,
        rotations=rotations,
        floor_basis=floor_basis,
    )
    position_plan, structural_regularization = _regularize_straight_runs(raw_plan, turns)
    regularized_turns, regularized_turn_detection = _detect_turns(
        position_plan,
        source_indices,
        confidence,
        source_timestamps=source_timestamps,
        rotations=rotations,
        floor_basis=floor_basis,
    )
    raw_signature = [turn.get("turn_type") for turn in turns]
    regularized_signature = [turn.get("turn_type") for turn in regularized_turns]
    raw_distance = float(np.linalg.norm(np.diff(raw_plan[:, :2], axis=0), axis=1).sum())
    regularized_distance = float(
        np.linalg.norm(np.diff(position_plan[:, :2], axis=0), axis=1).sum()
    )
    distance_ratio = regularized_distance / max(raw_distance, 1e-9)
    structural_regularization["distance_ratio"] = round(distance_ratio, 6)
    if structural_regularization["applied"] and (
        raw_signature != regularized_signature or not 0.97 <= distance_ratio <= 1.01
    ):
        position_plan = raw_plan.copy()
        structural_regularization.update({
            "applied": False,
            "quality": "rejected_global_gate",
            "fallback_reason": (
                "turn_signature_changed"
                if raw_signature != regularized_signature
                else "path_length_changed"
            ),
        })
    else:
        turns = regularized_turns
        turn_detection = regularized_turn_detection
    plan, heading_correction = _summarize_camera_turn_disagreements(position_plan, turns)
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
        "source_timestamps_seconds": source_timestamps,
        "pose_confidence": [round(float(v), 6) if np.isfinite(v) else None for v in confidence],
        "trajectory_quality": {
            "postprocess_version": R3_TRAJECTORY_POSTPROCESS_VERSION,
            **filter_quality,
            "projection": plane,
            "scale_stability": scale_stability,
            "structural_regularization": structural_regularization,
            "turn_detection": turn_detection,
            "geometry_contract": {
                "plan_source": "r3_translation_guarded_piecewise_lines",
                "camera_orientation_mutates_plan": False,
                "plan_matches_raw_plan": bool(np.array_equal(plan, raw_plan)),
            },
            "camera_turn_evidence": heading_correction,
            # Backward-compatible diagnostics key.  No correction is applied
            # from post-process version 4 onward.
            "heading_correction": heading_correction,
            "cleaned_distance": round(float(clean_steps.sum()), 6) if clean_steps.size else 0.0,
            "turns_detected": len(turns),
        },
    }
