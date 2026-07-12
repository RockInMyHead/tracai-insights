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
    vectors: list[np.ndarray] = []
    for rotation in rotations:
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            continue
        # R3 stores c2w.  Column 1 is camera Y in world space; its sign does
        # not matter for the floor plane normal.
        vectors.append(_normalize(rotation[:, 1]))
    if len(vectors) < 3:
        return None, 0.0

    reference = vectors[0]
    aligned = np.array([v if float(np.dot(v, reference)) >= 0 else -v for v in vectors])
    normal = _normalize(np.median(aligned, axis=0))
    coherence = float(np.median(np.abs(aligned @ normal)))
    return normal, coherence


def _project_to_floor(points: np.ndarray, rotations: Sequence[np.ndarray]) -> tuple[np.ndarray, dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Project positions onto the camera's stable horizontal movement plane."""
    if len(points) == 0:
        basis = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        return points.copy(), {"method": "empty"}, basis

    origin = points[0].copy()
    centered = points - origin
    normal, coherence = _camera_up_normal(rotations)
    method = "camera_up"

    if normal is None or coherence < 0.72:
        # Fallback for unusual camera orientations.  The least-varying PCA
        # direction is the plane normal when the path spans a 2D floor.
        if len(points) >= 3:
            try:
                _, _, vh = np.linalg.svd(centered - np.mean(centered, axis=0, keepdims=True), full_matrices=False)
                candidate = vh[-1]
                if np.isfinite(candidate).all() and np.linalg.norm(candidate) > 1e-8:
                    normal = _normalize(candidate)
                    method = "pca_plane"
            except np.linalg.LinAlgError:
                pass
        if normal is None:
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            method = "axis_fallback"

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
    e2 = _normalize(np.cross(normal, e1))
    e1 = _normalize(np.cross(e2, normal))
    plan = np.column_stack((centered @ e1, centered @ e2, centered @ normal))

    return plan, {
        "method": method,
        "camera_up_coherence": round(coherence, 5),
        "origin": [round(float(v), 6) for v in origin],
        "basis_e1": [round(float(v), 6) for v in e1],
        "basis_e2": [round(float(v), 6) for v in e2],
        "normal": [round(float(v), 6) for v in normal],
    }, (e1, e2, normal)


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


def _turn_spans(count: int, local_half_window: int) -> list[int]:
    """Return multiple temporal scales at which a corner can be measured.

    A turn can be instantaneous in the reconstructed path, but on a long
    video R3 often spreads the same physical 90-degree corner over a few
    dozen poses.  Looking only at the local tangent then reports the central
    45--70 degrees (or misses the turn entirely).  The spans below let us
    compare the straight approach with the straight exit without using a
    global smoother.
    """
    if count < 2:
        return []
    max_span = min(72, max(3, (count - 1) // 2))
    preferred = (2, 3, 4, 6, 8, 12, 18, 27, 40, 60, 72)
    spans = {max(2, int(local_half_window))}
    spans.update(span for span in preferred if span <= max_span)
    spans.add(max_span)
    return sorted(spans)


def _position_turn_candidates(
    plan_points: np.ndarray,
    local_half_window: int,
) -> list[dict[str, Any]]:
    """Measure each possible turn from approach/exit anchor segments.

    The returned candidate is intentionally based on two pieces outside the
    candidate centre.  It therefore preserves the *full* angle of a rounded
    or slowly reconstructed corner, instead of measuring only its centre.
    """
    count = len(plan_points)
    if count < 6:
        return []

    candidates: list[dict[str, Any]] = []
    for center in range(1, count - 1):
        best: dict[str, Any] | None = None
        for span in _turn_spans(count, local_half_window):
            # Leave a gap around the prospective corner.  With a rounded turn
            # this moves each heading estimate onto the stable straight side.
            guard = max(1, int(round(span * 0.4)))
            approach_start = max(0, center - span)
            approach_end = center - guard
            exit_start = center + guard
            exit_end = min(count - 1, center + span)
            if approach_end <= approach_start or exit_end <= exit_start:
                continue

            incoming = plan_points[approach_end, :2] - plan_points[approach_start, :2]
            outgoing = plan_points[exit_end, :2] - plan_points[exit_start, :2]
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            if incoming_length <= 1e-7 or outgoing_length <= 1e-7:
                continue

            incoming_heading = math.atan2(float(incoming[1]), float(incoming[0]))
            outgoing_heading = math.atan2(float(outgoing[1]), float(outgoing[0]))
            angle = math.degrees(_angle_delta_rad(outgoing_heading, incoming_heading))
            candidate = {
                "center": center,
                "angle": angle,
                "magnitude": abs(angle),
                "span": span,
                "guard": guard,
                "approach_start": approach_start,
                "approach_end": approach_end,
                "exit_start": exit_start,
                "exit_end": exit_end,
                "support": min(incoming_length, outgoing_length),
            }
            if best is None or candidate["magnitude"] > best["magnitude"]:
                best = candidate
        if best is not None:
            candidates.append(best)
    return candidates


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


def _detect_turns(
    plan_points: np.ndarray,
    source_frame_indices: Sequence[int | None],
    confidence: np.ndarray,
    rotations: Sequence[np.ndarray] | None = None,
    floor_basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(plan_points) < 6:
        return [], {"method": "multi_scale_anchors", "quality": "too_short"}

    local_headings, half_window = _trajectory_headings(plan_points)
    local_degrees = np.zeros(len(plan_points), dtype=np.float64)
    for index in range(len(plan_points)):
        left = max(0, index - half_window)
        right = min(len(plan_points) - 1, index + half_window)
        local_degrees[index] = math.degrees(
            _angle_delta_rad(float(local_headings[right]), float(local_headings[left]))
        )
    candidates = _position_turn_candidates(plan_points, half_window)
    camera_headings, camera_diagnostics = _camera_heading_signal(rotations, plan_points, floor_basis)

    # Camera orientation is an independent cue for the common R3 failure
    # mode where a real 90-degree corner is reconstructed as a 45--70-degree
    # arc.  It is only allowed to take precedence after global validation.
    for candidate in candidates:
        candidate["event_angle"] = candidate["angle"]
        candidate["angle_source"] = "trajectory_multiscale"
        candidate["orientation_angle"] = None
        if not camera_diagnostics["reliable"]:
            continue
        before, before_concentration = _mean_heading(
            camera_headings, candidate["approach_start"], candidate["approach_end"]
        )
        after, after_concentration = _mean_heading(
            camera_headings, candidate["exit_start"], candidate["exit_end"]
        )
        if before is None or after is None or min(before_concentration, after_concentration) < 0.7:
            continue
        orientation_angle = math.degrees(_angle_delta_rad(after, before))
        candidate["orientation_angle"] = orientation_angle
        position_magnitude = abs(float(candidate["angle"]))
        orientation_magnitude = abs(orientation_angle)
        same_direction = (
            position_magnitude < 18.0
            or math.copysign(1.0, float(candidate["angle"])) == math.copysign(1.0, orientation_angle)
        )
        # Do not turn a 90-degree path corner into an orientation-only
        # U-turn.  We only correct material under-estimation in the same
        # direction and stay within normal left/right turn magnitudes.
        if (
            same_direction
            and 32.0 <= orientation_magnitude < 150.0
            and position_magnitude < orientation_magnitude * 0.8
        ):
            candidate["event_angle"] = orientation_angle
            candidate["angle_source"] = "camera_orientation"

    active = [abs(float(candidate["event_angle"])) >= 28.0 for candidate in candidates]
    groups: list[tuple[int, int]] = []
    start: int | None = None
    last_active = -1
    active_sign = 0
    merge_gap = max(2, half_window * 2)
    for index, is_active in enumerate(active):
        if is_active:
            sign = 1 if float(candidates[index]["event_angle"]) > 0 else -1
            if start is None or (active_sign and sign != active_sign):
                if start is not None:
                    groups.append((start, last_active))
                start = index
                active_sign = sign
            last_active = index
        elif start is not None and index - last_active > merge_gap:
            groups.append((start, last_active))
            start = None
            active_sign = 0
    if start is not None:
        groups.append((start, last_active))

    finite_conf = confidence[np.isfinite(confidence)]
    weak_conf = float(np.percentile(finite_conf, 12)) if finite_conf.size >= 5 else float("-inf")
    turns: list[dict[str, Any]] = []
    last_turn_index = -10_000

    for group_start, group_end in groups:
        peak = max(
            candidates[group_start:group_end + 1],
            key=lambda candidate: abs(float(candidate["event_angle"])),
        )
        angle = float(peak["event_angle"])
        magnitude = abs(angle)
        # The largest multi-scale candidate can begin before the physical
        # corner.  Use the strongest local heading change for the marker and
        # source video frame, while retaining the peak's full-angle anchors.
        group_centers = [candidate["center"] for candidate in candidates[group_start:group_end + 1]]
        center = max(group_centers, key=lambda index: abs(float(local_degrees[index])))
        movement = float(peak["support"])
        if movement <= 1e-6:
            continue
        pose_is_weak = not np.isfinite(confidence[center]) or confidence[center] <= weak_conf
        # A single weak jitter should not turn into an event.  Strong turns or
        # a multi-frame arc are still retained.
        if pose_is_weak and magnitude < 45.0 and group_end - group_start < max(2, half_window):
            continue
        if center - last_turn_index < max(half_window * 2, 3):
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
            "angle_source": peak["angle_source"],
            "trajectory_angle_degrees": round(float(peak["angle"]), 1),
            "camera_angle_degrees": (
                round(float(peak["orientation_angle"]), 1)
                if peak["orientation_angle"] is not None
                else None
            ),
            "span_points": int(peak["span"]),
            "approach_index": int(peak["approach_end"]),
            "exit_index": int(peak["exit_start"]),
        })
        last_turn_index = center
    return turns, {
        "method": "multi_scale_anchors",
        "local_half_window": half_window,
        "spans": _turn_spans(len(plan_points), half_window),
        "candidate_count": len(candidates),
        "active_groups": len(groups),
        "camera_orientation": camera_diagnostics,
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
    plan, plane, floor_basis = _project_to_floor(cleaned_3d, rotations)
    turns, turn_detection = _detect_turns(
        plan,
        source_indices,
        confidence,
        rotations=rotations,
        floor_basis=floor_basis,
    )
    clean_steps = np.linalg.norm(np.diff(plan[:, :2], axis=0), axis=1) if len(plan) > 1 else np.array([])

    return {
        "trajectory": _json_points(plan),
        "plan_trajectory": _json_points(plan),
        "raw_trajectory_3d": _json_points(cleaned_3d),
        "raw_camera_points": _json_points(raw),
        "turn_points": turns,
        "source_frame_indices": source_indices,
        "pose_confidence": [round(float(v), 6) if np.isfinite(v) else None for v in confidence],
        "trajectory_quality": {
            **filter_quality,
            "projection": plane,
            "turn_detection": turn_detection,
            "cleaned_distance": round(float(clean_steps.sum()), 6) if clean_steps.size else 0.0,
            "turns_detected": len(turns),
        },
    }
