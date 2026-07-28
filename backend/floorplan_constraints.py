"""Production floor-plan constraints for the fixed Kerama Marazzi map.

The R3 reconstruction remains the visual source of relative motion.  This
module estimates one global map similarity from the user supplied start and
heading, evaluates several scale/yaw hypotheses, and solves for a route inside
the verified walkable mask.  Visual reconstruction remains the relative-motion
observation; the fixed plan is a hard spatial constraint rather than a final
quality gate which can discard an otherwise useful route.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

try:
    from confidence_calibration import calibrated_probability
except ImportError:  # pragma: no cover - package import path
    from backend.confidence_calibration import calibrated_probability

try:
    from floorplan_graph import floorplan_graph_geometry_sha256
except ImportError:  # pragma: no cover - package import path
    from backend.floorplan_graph import floorplan_graph_geometry_sha256


DEFAULT_FLOORPLAN_ID = "kerama_marazzi_2025"
FLOORPLAN_CONSTRAINT_REVISION = (
    "kerama_event_state_branch_commit_graph_first_v35"
)
ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "floorplans"

# R3/fusion is an authoritative motion observation, while an independent
# LingBot rescue has an unconstrained monocular scale.  The latter therefore
# needs an explicit metric prior before the floor plan is allowed to choose
# its scale. Short clips use walking speed; long inspection clips may contain
# substantial stops, so their average speed is allowed to be much lower.
# The verified five-minute Kerama route averages about 0.307 m/s including
# stops, or 0.256 of the nominal 1.2 m/s.  The previous 0.30 lower bound
# rejected the ground truth before the walkable mask was evaluated.
AUTHORITATIVE_SPEED_RATIO_BOUNDS = (0.20, 2.70)
INDEPENDENT_SPEED_RATIO_BOUNDS = (0.72, 1.80)
INDEPENDENT_LOOP_CLOSED_SPEED_RATIO_BOUNDS = (0.55, 1.80)
INDEPENDENT_LONG_INSPECTION_SPEED_RATIO_BOUNDS = (0.04, 1.80)
LONG_INSPECTION_MIN_SECONDS = 480.0
# Maximum search envelope for a local obstacle correction.  The route must
# never explore another aisle merely because it is connected in the mask.
LOCAL_ASTAR_INITIAL_MARGIN_METERS = 4.0
LOCAL_ASTAR_MAX_MARGIN_METERS = 6.0
MAX_LOCAL_MAP_CORRECTION_METERS = 4.0
MAX_LOCAL_MAP_CORRECTION_ROUTE_FRACTION = 0.06
MAX_LOCAL_MAP_CORRECTION_HARD_METERS = 6.0
# One metre plus one grid-quantisation allowance (4 px ~= 0.20 m).
MASK_LOCAL_REPAIR_LIMIT_METERS = 1.05
MASK_LOCAL_REPAIR_HARD_LIMIT_METERS = 1.5
# Graph map-matching is a different estimator from local obstacle repair.  It
# may select another part of the same connected aisle network, but only while
# the mapped curve still agrees with the observation's metric length and turn
# sequence.  These are hard plausibility bounds, not a search radius.
GRAPH_MAP_MATCH_P95_CORRECTION_METERS = 12.0
GRAPH_MAP_MATCH_HARD_CORRECTION_METERS = 18.0
GRAPH_MAP_MATCH_MIN_OBSERVATIONS = 8
GRAPH_MAP_MATCH_TARGET_ACTIVE_SPEED_RATIO = 0.50
GRAPH_MAP_MATCH_MAX_VITERBI_COST_PER_ANCHOR = 1.00
TURN_TOPOLOGY_MIN_DEGREES = 18.0
MAX_TURN_TOPOLOGY_MEAN_ERROR_DEGREES = 28.0
MAX_TURN_TOPOLOGY_SIGN_MISMATCH_RATIO = 0.0
MAX_TURN_EVENT_COUNT_DELTA = 2
MAX_TURN_EVENT_ARC_SHIFT = 0.12
AUTHORITATIVE_MIN_SELECTION_MARGIN = 0.18
AUTHORITATIVE_MAX_AMBIGUOUS_ENDPOINT_METERS = 3.0
MAX_AUTHORITATIVE_SHARP_REVERSE_RATIO = 0.18
INDEPENDENT_LOOP_CLOSED_MIN_LENGTH_RATIO = 0.65
STANDARD_YAW_OFFSETS_DEGREES = (
    -60.0, -45.0, -30.0, -20.0, -15.0, -10.0, -5.0, 0.0,
    5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0,
)
REVERSED_HEADING_YAW_OFFSETS_DEGREES = tuple(
    180.0 + offset for offset in STANDARD_YAW_OFFSETS_DEGREES
)
RIGHT_QUARTER_TURN_YAW_OFFSETS_DEGREES = tuple(
    -90.0 + offset for offset in STANDARD_YAW_OFFSETS_DEGREES
)
AUTHORITATIVE_SPEED_FLAT_RATIO_BOUNDS = (0.95, 1.80)
INDEPENDENT_SPEED_FLAT_RATIO_BOUNDS = (0.95, 1.60)
INDEPENDENT_MIN_NET_PROGRESS_RATIO = 0.55
INDEPENDENT_LOOP_CLOSED_MIN_SPAN_LENGTH_RATIO = 0.25
INDEPENDENT_LOOP_CLOSED_MAX_SHARP_REVERSE_RATIO = 0.18
INDEPENDENT_MIN_SELECTION_MARGIN = 0.10
INDEPENDENT_MAX_AMBIGUOUS_SCALE_RATIO = 1.18
MAX_START_SNAP_METERS = 1.50
MAX_PUBLISHED_START_ERROR_METERS = 0.05
MAX_PUBLISHED_INITIAL_DIRECTION_ERROR_DEGREES = 5.0
GRAPH_MAP_MATCH_SOFT_INITIAL_DIRECTION_DEGREES = 30.0
GRAPH_MAP_MATCH_HARD_INITIAL_DIRECTION_DEGREES = 70.0
GRAPH_MAP_MATCH_INITIAL_DIRECTION_LOOKAHEAD_METERS = 5.0
GRAPH_MAP_MATCH_PARALLEL_BRANCH_DEGREES = 15.0
MAX_PUBLISHED_SEGMENT_METERS = 0.75


def _speed_prior_penalty(
    speed_ratio: float,
    observation_policy: str,
    duration_seconds: Optional[float] = None,
) -> float:
    """Return a flat human-speed prior with hard penalties only at extremes."""
    if not math.isfinite(speed_ratio) or speed_ratio <= 1e-9:
        return 0.0
    long_independent_inspection = (
        observation_policy == "independent"
        and duration_seconds is not None
        and duration_seconds >= LONG_INSPECTION_MIN_SECONDS
    )
    flat_lower, flat_upper = (
        (0.05, INDEPENDENT_SPEED_FLAT_RATIO_BOUNDS[1])
        if long_independent_inspection
        else
        INDEPENDENT_SPEED_FLAT_RATIO_BOUNDS
        if observation_policy == "independent"
        else AUTHORITATIVE_SPEED_FLAT_RATIO_BOUNDS
    )
    if speed_ratio < flat_lower:
        penalty = math.log(flat_lower / speed_ratio)
    elif speed_ratio > flat_upper:
        penalty = math.log(speed_ratio / flat_upper)
    else:
        penalty = 0.0
    hard_lower, hard_upper = (
        INDEPENDENT_LONG_INSPECTION_SPEED_RATIO_BOUNDS
        if long_independent_inspection
        else
        INDEPENDENT_SPEED_RATIO_BOUNDS
        if observation_policy == "independent"
        else AUTHORITATIVE_SPEED_RATIO_BOUNDS
    )
    if not hard_lower <= speed_ratio <= hard_upper:
        penalty += 4.0
    return penalty


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _normalise_points(value: Any) -> np.ndarray:
    """Keep only finite samples. Never fabricate (0,0) for bad input."""
    if not isinstance(value, list):
        return np.empty((0, 2), dtype=np.float64)
    points: list[list[float]] = []
    for item in value:
        raw: Any = item
        if isinstance(item, dict):
            raw = [
                item.get("x", item.get(0)),
                item.get("y", item.get(1)),
            ]
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        try:
            point = [float(raw[0]), float(raw[1])]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(component) for component in point):
            points.append(point)
    return np.asarray(points, dtype=np.float64)


def _stabilize_authoritative_map_observation(
    points: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract plant-scale motion from a long, high-rate R3 polyline.

    R3 remains the published observation. This filtered copy is used only by
    map estimation so frame-to-frame pose jitter cannot masquerade as aisle
    length or hundreds of alternating turns. A symmetric Gaussian preserves
    temporal order; an affine endpoint correction locks both endpoints.
    """
    raw = np.asarray(points, dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "method": "endpoint_locked_gaussian_macro_motion_v1",
        "applied": False,
        "point_count": int(len(raw)),
    }
    if len(raw) < 120:
        return raw.copy(), diagnostics

    sigma = min(24.0, max(2.0, len(raw) / 250.0))
    filtered = ndimage.gaussian_filter1d(
        raw, sigma=sigma, axis=0, mode="nearest"
    )
    fractions = np.linspace(0.0, 1.0, len(raw))[:, None]
    filtered += (
        (1.0 - fractions) * (raw[0] - filtered[0])
        + fractions * (raw[-1] - filtered[-1])
    )
    raw_length = _polyline_length(raw)
    filtered_length = _polyline_length(filtered)
    span = max(float(np.linalg.norm(np.ptp(raw, axis=0))), 1e-9)
    maximum_shift = float(np.max(np.linalg.norm(filtered - raw, axis=1)))
    noise_length_ratio = raw_length / max(filtered_length, 1e-12)
    turn_preservation = _turn_topology_metrics(raw, filtered)
    turn_events_preserved = bool(
        float(turn_preservation["mean_abs_turn_error_degrees"])
        <= MAX_TURN_TOPOLOGY_MEAN_ERROR_DEGREES
        and float(turn_preservation["sign_mismatch_ratio"])
        <= MAX_TURN_TOPOLOGY_SIGN_MISMATCH_RATIO
        and int(turn_preservation["event_count_delta"])
        <= MAX_TURN_EVENT_COUNT_DELTA
    )
    safe = bool(
        filtered_length > 1e-9
        and maximum_shift <= span * 0.04
        and noise_length_ratio >= 1.08
    )
    diagnostics.update({
        "applied": safe,
        "sigma_observations": round(float(sigma), 3),
        "raw_length_units": round(raw_length, 8),
        "macro_length_units": round(
            filtered_length if safe else raw_length, 8
        ),
        "noise_length_ratio": round(noise_length_ratio, 6),
        "maximum_filter_shift_units": round(maximum_shift, 8),
        "endpoint_locked": True,
        "turn_events_preserved": turn_events_preserved,
        "turn_event_comparison": turn_preservation,
    })
    return (filtered if safe else raw.copy()), diagnostics


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _polyline_sharp_reverse_ratio(
    points: np.ndarray,
    *,
    meters_per_pixel: float,
    min_segment_meters: float = 0.25,
    reverse_degrees: float = 135.0,
) -> float:
    """Fraction of meaningful turns that are near-U-turns.

    Mask-legal A* spikes through drawing gaps often show as a chain of
    ~180° corners even when length_ratio and collision checks still pass.
    """
    if len(points) < 3 or meters_per_pixel <= 1e-12:
        return 0.0
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1) * float(meters_per_pixel)
    keep = np.where(segments >= float(min_segment_meters))[0]
    if len(keep) < 2:
        return 0.0
    compact = np.vstack([points[0], points[keep + 1]])
    vectors = np.diff(compact, axis=0)
    if len(vectors) < 2:
        return 0.0
    left = vectors[:-1]
    right = vectors[1:]
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    valid = (left_norm > 1e-9) & (right_norm > 1e-9)
    if not np.any(valid):
        return 0.0
    cosine = np.sum(left[valid] * right[valid], axis=1) / (
        left_norm[valid] * right_norm[valid]
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(np.mean(angles >= float(reverse_degrees)))


def _polyline_progress_metrics(
    points: np.ndarray,
    *,
    meters_per_pixel: float = 1.0,
) -> dict[str, float]:
    """Scale-invariant forward progress diagnostics for an open trajectory."""
    array = np.asarray(points, dtype=np.float64)
    length_pixels = _polyline_length(array)
    endpoint_pixels = (
        float(np.linalg.norm(array[-1] - array[0])) if len(array) >= 2 else 0.0
    )
    bbox_pixels = (
        float(np.linalg.norm(np.ptp(array[:, :2], axis=0))) if len(array) else 0.0
    )
    endpoint_ratio = endpoint_pixels / max(length_pixels, 1e-12)
    span_ratio = bbox_pixels / max(length_pixels, 1e-12)
    return {
        "path_length_meters": length_pixels * meters_per_pixel,
        "endpoint_displacement_meters": endpoint_pixels * meters_per_pixel,
        "bbox_diagonal_meters": bbox_pixels * meters_per_pixel,
        "net_progress_ratio": endpoint_ratio,
        "span_length_ratio": span_ratio,
        "tortuosity": length_pixels / max(endpoint_pixels, 1e-12),
    }


def _turn_events(
    points: np.ndarray,
    *,
    samples: int = 128,
    min_turn_degrees: float = TURN_TOPOLOGY_MIN_DEGREES,
) -> list[tuple[float, float]]:
    """Return ordered ``(arc_fraction, signed_angle_degrees)`` events."""
    if len(points) < 3:
        return []
    fractions = np.linspace(0.0, 1.0, max(8, int(samples)))
    sampled = _resample_polyline(np.asarray(points, dtype=np.float64), fractions)
    vectors = np.diff(sampled, axis=0)
    headings = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    if len(headings) >= 5:
        kernel = np.ones(5, dtype=np.float64) / 5.0
        headings = np.convolve(
            np.pad(headings, (2, 2), mode="edge"),
            kernel,
            mode="valid",
        )
    increments = np.degrees(np.diff(headings))
    active = np.abs(increments) >= 1.25
    events: list[tuple[float, float]] = []
    index = 0
    while index < len(increments):
        if not active[index]:
            index += 1
            continue
        sign = math.copysign(1.0, float(increments[index]))
        end = index + 1
        gap = 0
        while end < len(increments):
            same_sign = (
                active[end]
                and math.copysign(1.0, float(increments[end])) == sign
            )
            if same_sign:
                gap = 0
                end += 1
                continue
            gap += 1
            if gap > 2:
                break
            end += 1
        stop = max(index + 1, end - gap)
        angle = float(np.sum(increments[index:stop]))
        if abs(angle) >= float(min_turn_degrees):
            center = (index + stop) / 2.0 / max(len(increments), 1)
            events.append((center, angle))
        index = max(end, index + 1)
    return events


def _turn_topology_metrics(
    source: np.ndarray,
    corrected: np.ndarray,
    *,
    samples: int = 128,
    min_turn_degrees: float = TURN_TOPOLOGY_MIN_DEGREES,
) -> dict[str, float | int]:
    """Compare route turn sequence without requiring equal point counts.

    Local obstacle repair may add or remove vertices.  What it must not do is
    turn a measured right/left sequence into another branch of the map.  The
    comparison is therefore arc-normalised and only meaningful turns vote.
    """
    if len(source) < 3:
        return {
            "turn_count": 0,
            "mean_abs_turn_error_degrees": 0.0,
            "max_abs_turn_error_degrees": 0.0,
            "sign_mismatch_ratio": 0.0,
            "source_event_count": 0,
            "corrected_event_count": 0,
            "event_count_delta": 0,
            "maximum_event_arc_shift": 0.0,
            "event_sequence_preserved": True,
        }
    if len(corrected) < 2:
        return {
            "turn_count": 0,
            "mean_abs_turn_error_degrees": 180.0,
            "max_abs_turn_error_degrees": 180.0,
            "sign_mismatch_ratio": 1.0,
            "source_event_count": 0,
            "corrected_event_count": 0,
            "event_count_delta": 0,
            "maximum_event_arc_shift": 1.0,
            "event_sequence_preserved": False,
        }
    fractions = np.linspace(0.0, 1.0, int(samples))
    source_points = _resample_polyline(source, fractions)
    corrected_points = _resample_polyline(corrected, fractions)

    source_events = _turn_events(
        source_points, samples=samples, min_turn_degrees=min_turn_degrees
    )
    corrected_events = _turn_events(
        corrected_points, samples=samples, min_turn_degrees=min_turn_degrees
    )
    if not source_events:
        return {
            "turn_count": 0,
            "mean_abs_turn_error_degrees": 0.0,
            "max_abs_turn_error_degrees": 0.0,
            "sign_mismatch_ratio": 0.0,
            "source_event_count": 0,
            "corrected_event_count": len(corrected_events),
            "event_count_delta": len(corrected_events),
            "maximum_event_arc_shift": 0.0,
            # A straight observation may legitimately acquire two same-area
            # detour corners while going around one obstacle. With no measured
            # source event there is no left/right sequence to substitute.
            "event_sequence_preserved": True,
        }

    # Match events in order and tolerate a local arc-length displacement.
    # Obstacle repair changes the distance around a corner, but it may not
    # reorder corners or substitute an opposite branch.
    cursor = 0
    errors: list[float] = []
    mismatches = 0
    matched_arc_shifts: list[float] = []
    for source_fraction, source_angle in source_events:
        candidates = [
            (candidate_index, event)
            for candidate_index, event in enumerate(
                corrected_events[cursor:],
                start=cursor,
            )
            if abs(event[0] - source_fraction) <= 0.12
        ]
        if not candidates:
            errors.append(min(180.0, abs(source_angle)))
            mismatches += 1
            continue
        best_index, (_, corrected_angle) = min(
            candidates,
            key=lambda item: (
                (2.0 if np.sign(item[1][1]) != np.sign(source_angle) else 0.0)
                + abs(item[1][0] - source_fraction)
                + abs(abs(item[1][1]) - abs(source_angle)) / 180.0
            ),
        )
        cursor = best_index + 1
        matched_arc_shifts.append(abs(float(corrected_events[best_index][0] - source_fraction)))
        delta = (corrected_angle - source_angle + 180.0) % 360.0 - 180.0
        errors.append(abs(float(delta)))
        mismatches += int(np.sign(corrected_angle) != np.sign(source_angle))

    event_count_delta = abs(len(corrected_events) - len(source_events))
    mean_error = float(np.mean(errors))
    sign_mismatch_ratio = float(mismatches / len(source_events))
    maximum_arc_shift = max(matched_arc_shifts, default=1.0)
    return {
        "turn_count": len(source_events),
        "mean_abs_turn_error_degrees": mean_error,
        "max_abs_turn_error_degrees": float(np.max(errors)),
        "sign_mismatch_ratio": sign_mismatch_ratio,
        "source_event_count": len(source_events),
        "corrected_event_count": len(corrected_events),
        "event_count_delta": event_count_delta,
        "maximum_event_arc_shift": maximum_arc_shift,
        "event_sequence_preserved": bool(
            mean_error <= MAX_TURN_TOPOLOGY_MEAN_ERROR_DEGREES
            and sign_mismatch_ratio <= MAX_TURN_TOPOLOGY_SIGN_MISMATCH_RATIO
            and event_count_delta <= MAX_TURN_EVENT_COUNT_DELTA
            and maximum_arc_shift <= MAX_TURN_EVENT_ARC_SHIFT
        ),
    }


def _resample_polyline(points: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.repeat(points[:1], len(fractions), axis=0)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    total = float(cumulative[-1])
    if total <= 1e-9:
        return np.repeat(points[:1], len(fractions), axis=0)
    targets = np.clip(fractions, 0.0, 1.0) * total
    output = np.empty((len(targets), 2), dtype=np.float64)
    segment = 0
    for index, target in enumerate(targets):
        while segment + 1 < len(cumulative) - 1 and cumulative[segment + 1] < target:
            segment += 1
        length = max(float(cumulative[segment + 1] - cumulative[segment]), 1e-12)
        alpha = (target - cumulative[segment]) / length
        output[index] = points[segment] * (1.0 - alpha) + points[segment + 1] * alpha
    return output


def _trajectory_fractions(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points), dtype=np.float64)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    return cumulative / max(float(cumulative[-1]), 1e-12)


def _resample_timestamps(values: Any, count: int) -> Any:
    if not isinstance(values, list) or len(values) < 2 or count < 2:
        return values
    source = np.asarray([_finite_float(value, math.nan) for value in values], dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(source))
    if len(finite) < 2:
        return values
    source = np.interp(np.arange(len(source)), finite, source[finite])
    return np.interp(
        np.linspace(0.0, 1.0, count), np.linspace(0.0, 1.0, len(source)), source
    ).tolist()


def _densify_polyline(points: np.ndarray, max_step_pixels: float) -> np.ndarray:
    """Bound published segment length without changing route geometry."""
    if len(points) < 2 or max_step_pixels <= 1e-9:
        return points.copy()
    output: list[np.ndarray] = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(length / max_step_pixels)))
        for index in range(1, count + 1):
            output.append(start + (end - start) * (index / count))
    return np.asarray(output, dtype=np.float64)


def _mapped_timestamps(
    source: Any,
    mapped: Any,
    timestamps: Any,
) -> list[Optional[float]]:
    """Transfer source time to an arbitrary-length map polyline by arc."""
    source_points = _normalise_points(source)
    mapped_points = _normalise_points(mapped)
    if (
        len(source_points) < 2
        or len(mapped_points) < 2
        or not isinstance(timestamps, list)
        or len(timestamps) != len(source_points)
    ):
        return []
    values = np.asarray(
        [_finite_float(value, math.nan) for value in timestamps], dtype=np.float64
    )
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) < 2:
        return []
    values = np.interp(np.arange(len(values)), finite, values[finite])
    if np.any(np.diff(values) < 0.0):
        return []
    source_fractions = _trajectory_fractions(source_points)
    unique_fractions, first_indices = np.unique(source_fractions, return_index=True)
    unique_values = values[first_indices]
    if len(unique_values):
        unique_values[0] = values[0]
        unique_values[-1] = values[-1]
    result = np.interp(
        _trajectory_fractions(mapped_points),
        unique_fractions,
        unique_values,
    )
    return [round(float(value), 6) for value in result]


def _flip_polyline_y(points: Any) -> list[list[float]]:
    """Mirror plan Y — used to resolve PCA sign gauge for independent LingBot."""
    array = _normalise_points(points)
    if len(array) == 0:
        return []
    mirrored = array.copy()
    mirrored[:, 1] *= -1.0
    return [
        [round(float(point[0]), 8), round(float(point[1]), 8), 0.0]
        for point in mirrored
    ]


def _stabilize_independent_observation(
    points: Any,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Remove frame-scale LingBot jitter without changing route topology.

    Independent LingBot can contain thousands of sub-decimetre reversals.  A
    map similarity then interprets that temporal jitter as travelled distance,
    which corrupts scale, speed, candidate cost and confidence.  A short robust
    median plus symmetric mean filter removes only high-frequency motion; the
    endpoints and point count are preserved for timestamp correspondence.
    """
    raw = _normalise_points(points)
    diagnostics: dict[str, Any] = {
        "method": "robust_temporal_median_mean_v1",
        "applied": False,
        "point_count": int(len(raw)),
    }
    # Fusion already reduces a long LingBot observation to <= 50 robust
    # temporal control points. Do not blur that 3–5 minute global trend again.
    if len(raw) < 100:
        return [
            [round(float(point[0]), 8), round(float(point[1]), 8), 0.0]
            for point in raw
        ], diagnostics

    raw_length = _polyline_length(raw)
    span = float(np.linalg.norm(np.ptp(raw, axis=0)))
    steps = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    positive_steps = steps[steps > 1e-12]
    median_step = float(np.median(positive_steps)) if len(positive_steps) else 0.0
    filtered = np.column_stack([
        ndimage.uniform_filter1d(
            ndimage.median_filter(raw[:, axis], size=5, mode="nearest"),
            size=9,
            mode="nearest",
        )
        for axis in range(2)
    ])
    filtered[0] = raw[0]
    filtered[-1] = raw[-1]
    displacement = np.linalg.norm(filtered - raw, axis=1)
    max_displacement = float(np.max(displacement)) if len(displacement) else 0.0
    allowed_displacement = max(span * 0.025, median_step * 15.0, 1e-9)
    stable_length = _polyline_length(filtered)
    safe = (
        stable_length > 1e-9
        and max_displacement <= allowed_displacement
        and stable_length <= raw_length * 1.02
    )
    selected = filtered if safe else raw
    diagnostics.update({
        "applied": bool(safe),
        "raw_length_units": round(raw_length, 8),
        "stabilized_length_units": round(_polyline_length(selected), 8),
        "noise_length_ratio": round(
            raw_length / max(_polyline_length(selected), 1e-12), 6
        ),
        "max_displacement_units": round(max_displacement, 8),
        "allowed_displacement_units": round(allowed_displacement, 8),
    })
    return [
        [round(float(point[0]), 8), round(float(point[1]), 8), 0.0]
        for point in selected
    ], diagnostics


def _r3_is_severely_fragmented(result: dict[str, Any]) -> bool:
    """Recognise graph failures without depending on one worker schema."""
    containers = [
        result,
        result.get("processing_stats") or {},
        result.get("trajectory_quality") or {},
    ]
    nested: list[dict[str, Any]] = []
    for container in containers:
        if not isinstance(container, dict):
            continue
        nested.append(container)
        for key in ("pose_graph", "graph", "r3_pose_graph", "pose_graph_summary"):
            value = container.get(key)
            if isinstance(value, dict):
                nested.append(value)
    components = 1
    coverage = 1.0
    connected = None
    total = None
    for item in nested:
        components = max(components, int(_finite_float(item.get("component_count"), 1)))
        for key in ("largest_component_coverage", "largest_component_ratio"):
            if key in item:
                coverage = min(coverage, _finite_float(item.get(key), 1.0))
        connected = item.get("connected_pose_count", item.get("connected_poses", connected))
        total = item.get("point_count", item.get("pose_count", total))
    if connected is not None and total is not None and _finite_float(total) > 0:
        coverage = min(coverage, _finite_float(connected) / _finite_float(total))
    return components >= 4 or coverage < 0.45


@dataclass(frozen=True)
class FloorplanConfig:
    map_id: str
    width: int
    height: int
    meters_per_pixel: float
    grid_cell_pixels: int = 4
    person_radius_meters: float = 0.28
    walking_speed_mps: float = 1.20
    obstacle_mask_file: str = "kerama_marazzi_2025_obstacles.png"
    obstacle_mask_sha256: str = ""
    support_mask_file: str = ""
    support_mask_sha256: str = ""
    topology_graph_file: str = ""
    topology_graph_sha256: str = ""
    source_pdf: str = "kerama-marazzi-2025.pdf"
    display_image: str = "kerama-marazzi-2025.png"
    default_anchor_reference_pixels: tuple[float, float] | None = None
    default_anchor_direction_pixels: tuple[float, float] | None = None
    default_anchor_source: str = ""


class FloorplanConstraintEngine:
    """Immutable map model plus deterministic trajectory alignment."""

    def __init__(
        self,
        config: FloorplanConfig,
        obstacle_mask: np.ndarray,
        support_mask: Optional[np.ndarray] = None,
    ):
        mask = np.asarray(obstacle_mask, dtype=bool)
        if mask.shape != (config.height, config.width):
            raise ValueError(
                f"Obstacle mask shape {mask.shape} does not match "
                f"{config.height}x{config.width}"
            )
        self.config = config
        support = None if support_mask is None else np.asarray(support_mask, dtype=bool)
        if support is not None and support.shape != mask.shape:
            raise ValueError("Floorplan support mask shape does not match obstacle mask")
        self._topology_support_pixels_added = 0
        self._topology_obstacle_pixels_overridden = 0
        if support is not None and config.topology_graph_file:
            graph_path = ASSET_ROOT / config.topology_graph_file
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            topology_support_image = Image.new(
                "L", (config.width, config.height), 0
            )
            draw = ImageDraw.Draw(topology_support_image)
            corridor_radius_meters = config.person_radius_meters + 0.35
            corridor_width_pixels = max(
                3,
                int(math.ceil(
                    2.0 * corridor_radius_meters
                    / max(config.meters_per_pixel, 1e-9)
                )),
            )
            for edge in graph.get("edges", []):
                if not edge.get("enabled", True):
                    continue
                points = [
                    (float(point[0]), float(point[1]))
                    for point in edge.get("points", [])
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                ]
                if len(points) >= 2:
                    draw.line(
                        points,
                        fill=255,
                        width=corridor_width_pixels,
                        joint="curve",
                    )
            topology_support = (
                np.asarray(topology_support_image, dtype=np.uint8) >= 128
            )
            self._topology_obstacle_pixels_overridden = int(np.count_nonzero(
                topology_support & mask
            ))
            mask = mask & ~topology_support
            self._topology_support_pixels_added = int(np.count_nonzero(
                topology_support & ~support
            ))
            support = support | topology_support
        # Obstacles are authoritative even for programmatically constructed
        # engines whose input layers overlap.
        if support is not None:
            support = support & ~mask
        self._full_mask = mask | (~support if support is not None else False)
        self._support_mask = support
        self._topology_node_points = np.empty((0, 2), dtype=np.float64)
        self._topology_authored_node_count = 0
        self._topology_adjacency: dict[int, list[tuple[int, float]]] = {}
        self._topology_segment_edges: dict[tuple[int, int], str] = {}
        self._build_grid(self._full_mask, annotation_mask=mask)
        if config.topology_graph_file:
            self._load_topology_graph()

    @classmethod
    def load(cls, map_id: str = DEFAULT_FLOORPLAN_ID) -> "FloorplanConstraintEngine":
        metadata_path = ASSET_ROOT / f"{map_id}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        default_anchor = metadata.get("default_start_anchor") or {}
        anchor_reference = default_anchor.get("reference_pixels")
        anchor_direction = default_anchor.get("direction_pixels")
        config = FloorplanConfig(
            map_id=metadata["map_id"],
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            meters_per_pixel=float(metadata["meters_per_pixel"]),
            grid_cell_pixels=int(metadata.get("grid_cell_pixels", 4)),
            person_radius_meters=float(metadata.get("person_radius_meters", 0.28)),
            walking_speed_mps=float(metadata.get("walking_speed_mps", 1.20)),
            obstacle_mask_file=metadata["obstacle_mask_file"],
            obstacle_mask_sha256=metadata.get("obstacle_mask_sha256", ""),
            support_mask_file=metadata.get("support_mask_file", ""),
            support_mask_sha256=metadata.get("support_mask_sha256", ""),
            topology_graph_file=metadata.get("topology_graph_file", ""),
            topology_graph_sha256=metadata.get("topology_graph_sha256", ""),
            source_pdf=metadata.get("source_pdf", "kerama-marazzi-2025.pdf"),
            display_image=metadata.get("display_image", "kerama-marazzi-2025.png"),
            default_anchor_reference_pixels=(
                (float(anchor_reference[0]), float(anchor_reference[1]))
                if isinstance(anchor_reference, list) and len(anchor_reference) >= 2
                else None
            ),
            default_anchor_direction_pixels=(
                (float(anchor_direction[0]), float(anchor_direction[1]))
                if isinstance(anchor_direction, list) and len(anchor_direction) >= 2
                else None
            ),
            default_anchor_source=str(default_anchor.get("source") or ""),
        )
        mask_path = ASSET_ROOT / config.obstacle_mask_file
        if config.obstacle_mask_sha256:
            actual_hash = hashlib.sha256(mask_path.read_bytes()).hexdigest()
            if actual_hash != config.obstacle_mask_sha256:
                raise ValueError(
                    f"Floorplan obstacle mask hash mismatch for {config.map_id}"
                )
        mask = np.asarray(Image.open(mask_path).convert("L")) >= 128
        support = None
        if config.support_mask_file:
            support_path = ASSET_ROOT / config.support_mask_file
            if config.support_mask_sha256:
                actual_hash = hashlib.sha256(support_path.read_bytes()).hexdigest()
                if actual_hash != config.support_mask_sha256:
                    raise ValueError(f"Floorplan support mask hash mismatch for {config.map_id}")
            support = np.asarray(Image.open(support_path).convert("L")) >= 128
        return cls(config, mask, support)

    def _load_topology_graph(self) -> None:
        graph_path = ASSET_ROOT / self.config.topology_graph_file
        if self.config.topology_graph_sha256:
            actual_hash = hashlib.sha256(graph_path.read_bytes()).hexdigest()
            if actual_hash != self.config.topology_graph_sha256:
                raise ValueError(
                    f"Floorplan topology graph hash mismatch for {self.config.map_id}"
                )
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        if graph.get("schema_version") != "trackai.floorplan_graph.v1":
            raise ValueError("Unsupported floorplan topology graph schema")
        embedded_geometry_sha = str(
            (graph.get("source") or {}).get("geometry_sha256") or ""
        )
        actual_geometry_sha = floorplan_graph_geometry_sha256(graph)
        if embedded_geometry_sha and embedded_geometry_sha != actual_geometry_sha:
            raise ValueError(
                f"Floorplan topology graph geometry mismatch for "
                f"{self.config.map_id}"
            )
        enabled_nodes = [
            node for node in graph.get("nodes", [])
            if node.get("enabled", True)
        ]
        node_index = {
            str(node["id"]): index for index, node in enumerate(enabled_nodes)
        }
        authored_points = np.asarray([
            [float(node["x"]), float(node["y"])] for node in enabled_nodes
        ], dtype=np.float64)
        points = authored_points.tolist()
        adjacency: dict[int, list[tuple[int, float]]] = {
            index: [] for index in range(len(authored_points))
        }
        segment_edges: dict[tuple[int, int], str] = {}
        maximum_sample_pixels = (
            1.0 / max(self.config.meters_per_pixel, 1e-9)
        )
        for edge in graph.get("edges", []):
            if not edge.get("enabled", True):
                continue
            left = node_index.get(str(edge.get("from") or ""))
            right = node_index.get(str(edge.get("to") or ""))
            if left is None or right is None or left == right:
                continue
            start = authored_points[left]
            end = authored_points[right]
            distance = float(np.linalg.norm(end - start))
            if distance <= 1e-9:
                continue
            segment_count = max(
                1, int(math.ceil(distance / maximum_sample_pixels))
            )
            chain = [left]
            for sample_index in range(1, segment_count):
                ratio = sample_index / segment_count
                point_index = len(points)
                points.append(((1.0 - ratio) * start + ratio * end).tolist())
                adjacency[point_index] = []
                chain.append(point_index)
            chain.append(right)
            for chain_left, chain_right in zip(chain, chain[1:]):
                weight = float(np.linalg.norm(
                    np.asarray(points[chain_right])
                    - np.asarray(points[chain_left])
                ))
                adjacency[chain_left].append((chain_right, weight))
                edge_id = str(edge.get("id") or "")
                segment_edges[(chain_left, chain_right)] = edge_id
                if edge.get("bidirectional", True):
                    adjacency[chain_right].append((chain_left, weight))
                    segment_edges[(chain_right, chain_left)] = edge_id
        self._topology_node_points = np.asarray(points, dtype=np.float64)
        self._topology_authored_node_count = len(authored_points)
        self._topology_adjacency = adjacency
        self._topology_segment_edges = segment_edges
        self._corridor_nodes = np.rint(
            self._topology_node_points
            / max(self.config.grid_cell_pixels, 1)
        ).astype(np.int32)

    def _topology_route(
        self, start: np.ndarray, end: np.ndarray
    ) -> Optional[np.ndarray]:
        if len(self._topology_node_points) == 0:
            return None
        start_index = int(np.argmin(np.linalg.norm(
            self._topology_node_points - start[None, :], axis=1
        )))
        end_index = int(np.argmin(np.linalg.norm(
            self._topology_node_points - end[None, :], axis=1
        )))
        if start_index == end_index:
            return None
        distances = {start_index: 0.0}
        parents: dict[int, int] = {}
        queue = [(0.0, start_index)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances.get(node, float("inf")):
                continue
            if node == end_index:
                break
            for neighbour, weight in self._topology_adjacency.get(node, []):
                candidate = distance + weight
                if candidate < distances.get(neighbour, float("inf")):
                    distances[neighbour] = candidate
                    parents[neighbour] = node
                    heapq.heappush(queue, (candidate, neighbour))
        if end_index not in distances:
            return None
        indices = [end_index]
        while indices[-1] != start_index:
            indices.append(parents[indices[-1]])
        indices.reverse()
        return self._topology_node_points[np.asarray(indices, dtype=int)]

    def _topology_edge_sequence(self, route: np.ndarray) -> list[str]:
        if len(route) < 2 or not self._topology_segment_edges:
            return []
        indices = [
            int(np.argmin(np.linalg.norm(
                self._topology_node_points - point[None, :], axis=1
            )))
            for point in route
        ]
        sequence: list[str] = []
        for left, right in zip(indices, indices[1:]):
            edge_id = self._topology_segment_edges.get((left, right))
            if edge_id and (not sequence or sequence[-1] != edge_id):
                sequence.append(edge_id)
        return sequence

    def _directed_edge_event_match(
        self,
        observation: np.ndarray,
    ) -> tuple[Optional[np.ndarray], dict[str, Any]]:
        """Follow connected directed graph edges using R3 turn events.

        Unlike the legacy anchor matcher, this search never asks for an
        unrelated shortest path between two observations. A hypothesis can
        move only to a neighbour of its current graph node, so parallel aisle
        hopping is impossible without a real connector.
        """
        points = np.asarray(observation, dtype=np.float64)
        if len(points) < 3 or len(self._topology_node_points) == 0:
            return None, {"reason": "directed_edge_match_unavailable"}
        target_length = _polyline_length(points)
        if target_length <= 1e-6:
            return None, {"reason": "directed_edge_match_empty_observation"}
        canonical_points = _resample_polyline(
            points, np.linspace(0.0, 1.0, 128)
        )
        expected = _turn_events(
            canonical_points,
            # Use the same canonical event resolution as the production
            # topology gate; increasing samples changes one physical bend
            # into several noise events on long R3 sequences.
            samples=128,
            min_turn_degrees=TURN_TOPOLOGY_MIN_DEGREES,
        )
        radius_pixels = 6.0 / max(self.config.meters_per_pixel, 1e-9)
        start_distances = np.linalg.norm(
            self._topology_node_points - points[0][None, :], axis=1
        )
        start_nodes = np.argsort(start_distances)[:4]
        initial_direction = self._topology_route_initial_direction(points)
        initial_direction /= max(
            float(np.linalg.norm(initial_direction)), 1e-9
        )

        initial_heading = math.atan2(
            float(initial_direction[1]), float(initial_direction[0])
        )
        # cost, node, previous, path, travelled, matched events, used arcs,
        # heading committed after the last matched R3 turn
        beam: list[
            tuple[
                float, int, int, tuple[int, ...], float, int,
                frozenset[tuple[int, int]], float,
            ]
        ] = []
        for start_node in start_nodes:
            start_node = int(start_node)
            start_cost = (
                float(start_distances[start_node]) / radius_pixels
            ) ** 2
            beam.append((
                start_cost,
                start_node,
                -1,
                (start_node,),
                0.0,
                0,
                frozenset(),
                initial_heading,
            ))

        terminals: list[
            tuple[float, tuple[int, ...], float, int]
        ] = []
        best: dict[tuple[int, int, int, int, int], float] = {}
        maximum_matched_events = 0
        maximum_travel_fraction = 0.0
        maximum_steps = min(max(len(self._topology_node_points), 32), 220)
        for _ in range(maximum_steps):
            expanded: list[
                tuple[
                    float, int, int, tuple[int, ...], float, int,
                    frozenset[tuple[int, int]], float,
                ]
            ] = []
            for (
                cost, node, previous, path, travelled, event_index, used_arcs,
                committed_heading,
            ) in beam:
                maximum_matched_events = max(
                    maximum_matched_events, event_index
                )
                maximum_travel_fraction = max(
                    maximum_travel_fraction,
                    travelled / target_length,
                )
                if (
                    travelled >= target_length * 0.65
                    and event_index == len(expected)
                ):
                    endpoint_error = float(np.linalg.norm(
                        self._topology_node_points[node] - points[-1]
                    )) / radius_pixels
                    terminals.append((
                        cost
                        + 2.0 * abs(travelled / target_length - 1.0)
                        + 0.35 * endpoint_error ** 2,
                        path,
                        travelled,
                        event_index,
                    ))
                if travelled >= target_length * 1.65:
                    continue
                current = self._topology_node_points[node]
                for neighbour, edge_length in self._topology_adjacency.get(
                    node, []
                ):
                    neighbour = int(neighbour)
                    arc = (node, neighbour)
                    if (
                        neighbour == previous
                        or neighbour in path
                        or arc in used_arcs
                    ):
                        continue
                    endpoint = self._topology_node_points[neighbour]
                    direction = endpoint - current
                    direction_length = max(
                        float(np.linalg.norm(direction)), 1e-9
                    )
                    next_event_index = event_index
                    next_committed_heading = committed_heading
                    turn_penalty = 0.0
                    direction_heading = math.atan2(
                        float(direction[1]), float(direction[0])
                    )
                    if previous < 0:
                        cosine = float(np.dot(
                            initial_direction, direction / direction_length
                        ))
                        heading_penalty, opposite = (
                            self._initial_direction_cost(cosine)
                        )
                        if opposite:
                            continue
                        turn_penalty += heading_penalty
                    else:
                        angle = math.degrees(math.atan2(
                            math.sin(
                                direction_heading - committed_heading
                            ),
                            math.cos(
                                direction_heading - committed_heading
                            ),
                        ))
                        # A physical bend may be authored as several small
                        # edge turns. Measure the cumulative heading change
                        # since the last consumed R3 event.
                        if abs(angle) >= 35.0:
                            if next_event_index >= len(expected):
                                continue
                            expected_fraction, expected_angle = expected[
                                next_event_index
                            ]
                            event_position = travelled / target_length
                            if float(expected_angle) * angle < 0.0:
                                # This is not the next measured event. Keep
                                # the connected hypothesis alive with a large
                                # penalty and wait for a compatible branch;
                                # the terminal full-event gate still forbids
                                # publishing the extra turn.
                                turn_penalty += 1.8
                            else:
                                turn_penalty += (
                                2.5 * abs(
                                    event_position - float(expected_fraction)
                                )
                                + 0.7 * abs(angle - float(expected_angle))
                                / max(abs(float(expected_angle)), 35.0)
                                )
                                next_event_index += 1
                                next_committed_heading = direction_heading
                    next_length = travelled + float(edge_length)
                    fraction = float(np.clip(
                        next_length / target_length, 0.0, 1.0
                    ))
                    observed_point = _resample_polyline(
                        points, np.asarray([fraction], dtype=np.float64)
                    )[0]
                    observation_error = (
                        float(np.linalg.norm(endpoint - observed_point))
                        / radius_pixels
                    )
                    next_cost = (
                        cost + turn_penalty + 0.22 * observation_error ** 2
                    )
                    heading_bin = int(round(
                        math.degrees(next_committed_heading) / 10.0
                    ))
                    key = (
                        node,
                        neighbour,
                        next_event_index,
                        int(round(fraction * 30.0)),
                        heading_bin,
                    )
                    if next_cost >= best.get(key, float("inf")):
                        continue
                    best[key] = next_cost
                    expanded.append((
                        next_cost,
                        neighbour,
                        node,
                        path + (neighbour,),
                        next_length,
                        next_event_index,
                        used_arcs | {arc},
                        next_committed_heading,
                    ))
            if not expanded:
                break
            expanded.sort(key=lambda item: item[0])
            beam = expanded[:600]
        if not terminals:
            return None, {
                "reason": "directed_edge_event_sequence_not_found",
                "expected_turn_events": len(expected),
                "states_evaluated": len(best),
                "maximum_matched_turn_events": maximum_matched_events,
                "maximum_travel_fraction": round(
                    maximum_travel_fraction, 6
                ),
            }
        terminals.sort(key=lambda item: item[0])
        best_terminal_topology: Optional[dict[str, Any]] = None
        best_terminal_route: Optional[np.ndarray] = None
        for objective, path, travelled, matched_events in terminals:
            route = self._topology_node_points[
                np.asarray(path, dtype=int)
            ]
            topology = _turn_topology_metrics(points, route)
            if (
                best_terminal_topology is None
                or (
                    float(topology["sign_mismatch_ratio"]),
                    abs(int(topology["event_count_delta"])),
                    float(topology["mean_abs_turn_error_degrees"]),
                )
                < (
                    float(best_terminal_topology["sign_mismatch_ratio"]),
                    abs(int(best_terminal_topology["event_count_delta"])),
                    float(best_terminal_topology[
                        "mean_abs_turn_error_degrees"
                    ]),
                )
            ):
                best_terminal_topology = topology
                best_terminal_route = route
            if not bool(topology["event_sequence_preserved"]):
                continue
            return route, {
                "reason": None,
                "method": "directed_edge_event_beam_v1",
                "objective": round(float(objective), 6),
                "anchors": len(path),
                "matched_turn_events": matched_events,
                "expected_turn_events": len(expected),
                "edge_ids": self._topology_edge_sequence(route),
                "selected_reconstructed_turn_topology": topology,
                "states_evaluated": len(best),
                "branch_commitment": "connected_edges_only",
            }
        return None, {
            "reason": "directed_edge_terminal_topology_mismatch",
            "expected_turn_events": len(expected),
            "terminal_candidates": len(terminals),
            "states_evaluated": len(best),
            "best_terminal_turn_topology": best_terminal_topology,
            "best_terminal_trajectory": (
                [
                    [round(float(point[0]), 3), round(float(point[1]), 3)]
                    for point in best_terminal_route
                ]
                if best_terminal_route is not None else []
            ),
            "best_terminal_edge_ids": (
                self._topology_edge_sequence(best_terminal_route)
                if best_terminal_route is not None else []
            ),
        }

    def _topology_route_initial_direction(
        self,
        route: np.ndarray,
        *,
        lookahead_meters: float = GRAPH_MAP_MATCH_INITIAL_DIRECTION_LOOKAHEAD_METERS,
    ) -> np.ndarray:
        """Return a stable departure direction measured along a graph route.

        The first topology segment can be shorter than a metre and inherits
        small editor/snap errors.  Measuring from the snapped graph start to a
        point several metres down the same route represents the corridor
        direction instead of the first drawing segment.
        """
        points = np.asarray(route, dtype=np.float64)
        if len(points) < 2:
            return np.zeros(2, dtype=np.float64)
        target_pixels = (
            max(float(lookahead_meters), 0.0)
            / max(self.config.meters_per_pixel, 1e-9)
        )
        travelled = 0.0
        origin = points[0]
        for left, right in zip(points, points[1:]):
            segment = right - left
            length = float(np.linalg.norm(segment))
            if length <= 1e-9:
                continue
            if travelled + length >= target_pixels:
                ratio = (target_pixels - travelled) / length
                target = left + np.clip(ratio, 0.0, 1.0) * segment
                direction = target - origin
                if float(np.linalg.norm(direction)) > 1e-9:
                    return direction
            travelled += length
        return points[-1] - origin

    @staticmethod
    def _initial_direction_cost(direction_cosine: float) -> tuple[float, bool]:
        """Return a soft heading cost and whether the route is opposite.

        Plausible start branches stay in the Viterbi beam and are resolved by
        later turn events. Only a clearly contradictory departure is rejected.
        """
        angle_degrees = math.degrees(math.acos(float(np.clip(
            direction_cosine, -1.0, 1.0
        ))))
        if angle_degrees > GRAPH_MAP_MATCH_HARD_INITIAL_DIRECTION_DEGREES:
            return float("inf"), True
        normalized = angle_degrees / max(
            GRAPH_MAP_MATCH_SOFT_INITIAL_DIRECTION_DEGREES, 1e-9
        )
        return 0.45 * normalized ** 2, False

    @classmethod
    def from_mask(
        cls,
        obstacle_mask: np.ndarray,
        *,
        meters_per_pixel: float = 0.10,
        grid_cell_pixels: int = 1,
        person_radius_meters: float = 0.0,
        walking_speed_mps: float = 1.20,
        map_id: str = "test",
    ) -> "FloorplanConstraintEngine":
        height, width = np.asarray(obstacle_mask).shape[:2]
        return cls(
            FloorplanConfig(
                map_id=map_id,
                width=int(width),
                height=int(height),
                meters_per_pixel=float(meters_per_pixel),
                grid_cell_pixels=int(grid_cell_pixels),
                person_radius_meters=float(person_radius_meters),
                walking_speed_mps=float(walking_speed_mps),
                obstacle_mask_file="",
            ),
            obstacle_mask,
        )

    def _build_grid(self, mask: np.ndarray, annotation_mask: Optional[np.ndarray] = None) -> None:
        cell = self.config.grid_cell_pixels
        rows = int(math.ceil(self.config.height / cell))
        cols = int(math.ceil(self.config.width / cell))
        padded = np.pad(
            mask,
            ((0, rows * cell - self.config.height), (0, cols * cell - self.config.width)),
            # Partial grid cells outside the PDF are physical out-of-map space.
            constant_values=True,
        )
        base = padded.reshape(rows, cell, cols, cell).any(axis=(1, 3))
        distance_to_base = ndimage.distance_transform_edt(~base)
        inflation_cells = self.config.person_radius_meters / max(
            self.config.meters_per_pixel * cell, 1e-9
        )
        occupied = base | (distance_to_base <= inflation_cells)
        self.rows = rows
        self.cols = cols
        self.occupied = occupied
        self.clearance_meters = (
            ndimage.distance_transform_edt(~occupied)
            * self.config.meters_per_pixel
            * cell
        )
        # A compact medial-axis-like corridor graph. Nodes are local clearance
        # maxima; collision-free Viterbi transitions become its implicit edges.
        ridge_window = max(3, int(round(1.2 / max(self.config.meters_per_pixel * cell, 1e-9))))
        if ridge_window % 2 == 0:
            ridge_window += 1
        ridge = (
            (~occupied)
            & (self.clearance_meters >= ndimage.maximum_filter(
                self.clearance_meters, size=ridge_window, mode="constant"
            ) - 1e-9)
            & (self.clearance_meters >= max(0.20, self.config.person_radius_meters * 0.5))
        )
        corridor_nodes = np.argwhere(ridge)[:, ::-1] if np.any(ridge) else np.empty((0, 2), dtype=int)
        if len(corridor_nodes) > 5000:
            corridor_nodes = corridor_nodes[:: int(math.ceil(len(corridor_nodes) / 5000))]
        self._corridor_nodes = corridor_nodes.astype(np.int32)
        _, nearest = ndimage.distance_transform_edt(occupied, return_indices=True)
        self._nearest_free_rows = nearest[0]
        self._nearest_free_cols = nearest[1]
        free = ~occupied
        labeled, component_count = ndimage.label(free)
        self._component_ids = labeled
        self._component_count = int(component_count)
        annotation = mask if annotation_mask is None else annotation_mask
        annotation_padded = np.pad(
            annotation,
            ((0, rows * cell - self.config.height), (0, cols * cell - self.config.width)),
            constant_values=False,
        )
        annotation_base = annotation_padded.reshape(rows, cell, cols, cell).any(axis=(1, 3))
        ys, xs = np.where(annotation_base)
        self.annotation_bbox = (
            (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            if len(xs)
            else (0, 0, cols - 1, rows - 1)
        )
        # Walkable support/free extent drives the map-scale prior. Annotation
        # ink alone only covers marked machines and underestimates the plant.
        free_ys, free_xs = np.where(free)
        if len(free_xs):
            # Prefer the largest connected free component diameter.
            if component_count > 0:
                sizes = ndimage.sum(free, labeled, index=np.arange(1, component_count + 1))
                largest = int(np.argmax(sizes)) + 1
                component_mask = labeled == largest
                free_ys, free_xs = np.where(component_mask)
            self.walkable_bbox = (
                int(free_xs.min()),
                int(free_ys.min()),
                int(free_xs.max()),
                int(free_ys.max()),
            )
        else:
            self.walkable_bbox = self.annotation_bbox

    @property
    def cell_meters(self) -> float:
        return self.config.grid_cell_pixels * self.config.meters_per_pixel

    def _pixel_to_cell(self, point: Sequence[float]) -> tuple[int, int]:
        x = int(math.floor(float(point[0]) / self.config.grid_cell_pixels))
        y = int(math.floor(float(point[1]) / self.config.grid_cell_pixels))
        return x, y

    def _cell_to_pixel(self, cell: tuple[int, int]) -> np.ndarray:
        half = self.config.grid_cell_pixels / 2.0
        return np.asarray([
            cell[0] * self.config.grid_cell_pixels + half,
            cell[1] * self.config.grid_cell_pixels + half,
        ])

    def _inside_cell(self, x: int, y: int) -> bool:
        return 0 <= x < self.cols and 0 <= y < self.rows

    def _point_occupied(self, point: Sequence[float]) -> bool:
        if not (
            0.0 <= float(point[0]) < self.config.width
            and 0.0 <= float(point[1]) < self.config.height
        ):
            return True
        x, y = self._pixel_to_cell(point)
        return not self._inside_cell(x, y) or bool(self.occupied[y, x])

    def _nearest_free(self, cell: tuple[int, int]) -> Optional[tuple[int, int]]:
        # A monocular hypothesis may leave the plan. Project its endpoint back
        # to the border so the constrained solver can recover a valid route.
        x = min(max(int(cell[0]), 0), self.cols - 1)
        y = min(max(int(cell[1]), 0), self.rows - 1)
        if not self.occupied[y, x]:
            return x, y
        ny = int(self._nearest_free_rows[y, x])
        nx = int(self._nearest_free_cols[y, x])
        return (nx, ny) if self._inside_cell(nx, ny) and not self.occupied[ny, nx] else None

    def _sample_path(self, points: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return points.copy()
        # Quarter-cell sampling is deliberately shared with line-of-sight
        # validation.  Coarser, differently phased samples can miss one grid
        # cell and falsely certify a diagonal that clips a machine corner.
        # Half-pixel phasing catches a one-cell diagonal corner even when the
        # segment crosses it exactly at the midpoint. Quarter-cell-of-grid
        # sampling (1 px on the production 4 px grid) could skip that cell,
        # while later densification materialised the missed midpoint.
        spacing = max(self.config.grid_cell_pixels * 0.125, 0.5)
        samples: list[np.ndarray] = [points[0]]
        for start, end in zip(points[:-1], points[1:]):
            count = max(1, int(math.ceil(float(np.linalg.norm(end - start)) / spacing)))
            for step in range(1, count + 1):
                samples.append(start + (end - start) * (step / count))
        return np.asarray(samples, dtype=np.float64)

    def _path_metrics(self, points: np.ndarray) -> dict[str, float]:
        samples = self._sample_path(points)
        if len(samples) == 0:
            return {"collision_ratio": 1.0, "outside_ratio": 1.0, "clearance_penalty": 1.0}
        collisions = 0
        outside = 0
        clearance_values: list[float] = []
        for point in samples:
            if not (
                0.0 <= float(point[0]) < self.config.width
                and 0.0 <= float(point[1]) < self.config.height
            ):
                outside += 1
                collisions += 1
                continue
            x, y = self._pixel_to_cell(point)
            if not self._inside_cell(x, y):
                outside += 1
                collisions += 1
                continue
            if self.occupied[y, x]:
                collisions += 1
            clearance_values.append(float(self.clearance_meters[y, x]))
        clearance_penalty = float(np.mean(np.exp(-np.asarray(clearance_values) / 0.45))) \
            if clearance_values else 1.0
        return {
            "collision_ratio": collisions / len(samples),
            "outside_ratio": outside / len(samples),
            "clearance_penalty": clearance_penalty,
        }

    @staticmethod
    def _motion_duration_seconds(timestamps: Any, points: np.ndarray) -> Optional[float]:
        """Estimate moving time so pauses do not inflate the metric scale.

        Video duration is a useful monocular-scale prior only while the
        operator moves.  R3 still jitters during a stop, so the threshold is
        derived from the run's own step distribution and falls back to full
        duration for uniformly moving sequences.
        """
        point_count = len(points)
        if not isinstance(timestamps, list) or len(timestamps) != point_count:
            return None
        values = np.asarray([_finite_float(value, math.nan) for value in timestamps], dtype=np.float64)
        finite_indices = np.flatnonzero(np.isfinite(values))
        if point_count < 2 or len(finite_indices) < 2:
            return None
        # Interpolation and speed estimation require one forward-moving clock.
        # A few duplicated timestamps are pauses; negative jumps mean samples
        # came from incompatible clocks or were associated with wrong frames.
        finite_values = values[finite_indices]
        if np.any(np.diff(finite_values) < 0.0):
            return None
        first, last = int(finite_indices[0]), int(finite_indices[-1])
        total = float(values[last] - values[first])
        if not 1.0 <= total <= 8 * 60 * 60:
            return None
        dt = np.diff(values)
        valid_dt = np.isfinite(dt) & (dt > 0.0) & (dt < 60.0)
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        positive = steps[np.isfinite(steps) & (steps > 1e-10)]
        if len(positive) < 4:
            return total
        p20, p90 = np.percentile(positive, [20, 90])
        if p20 > 1e-10 and p90 / p20 < 2.5:
            moving = valid_dt
        else:
            threshold = max(float(p20) * 2.5, float(p90) * 0.04, 1e-10)
            moving = (steps > threshold) & valid_dt
            moving = ndimage.binary_dilation(moving, structure=np.ones(3, dtype=bool)) & valid_dt
        active = float(dt[moving].sum())
        return active if active >= max(1.0, total * 0.08) else total

    def _scale_candidates(self, relative: np.ndarray, duration: Optional[float]) -> list[float]:
        raw_length = max(_polyline_length(relative), 1e-9)
        raw_span = max(float(np.ptp(relative[:, 0])), float(np.ptp(relative[:, 1])), 1e-9)
        walk_width = max(self.walkable_bbox[2] - self.walkable_bbox[0], 1)
        walk_height = max(self.walkable_bbox[3] - self.walkable_bbox[1], 1)
        map_base = max(walk_width, walk_height) * self.config.grid_cell_pixels * 0.72 / raw_span
        bases = [map_base]
        if duration is not None:
            metric_distance = self.config.walking_speed_mps * duration
            bases.append(metric_distance / self.config.meters_per_pixel / raw_length)
        factors = (
            0.38, 0.45, 0.52, 0.58, 0.66, 0.72, 0.80, 0.86, 0.93,
            1.0, 1.08, 1.16, 1.25, 1.35, 1.48, 1.60, 1.78, 1.95, 2.15, 2.35,
        )
        values = {
            round(base * factor, 9)
            for base in bases
            for factor in factors
            if math.isfinite(base * factor) and base * factor > 1e-7
        }
        return sorted(values)

    @staticmethod
    def _select_diverse_beam(
        hypotheses: list[dict[str, Any]],
        *,
        per_yaw: int = 2,
        global_top: int = 18,
    ) -> list[dict[str, Any]]:
        """Keep independent yaw and metric-scale coverage in the repair beam."""
        if not hypotheses:
            return []
        selected: list[dict[str, Any]] = []
        seen: set[tuple[float, float]] = set()

        def add(item: dict[str, Any]) -> None:
            key = (round(float(item["scale"]), 9), round(float(item["yaw"]), 3))
            if key in seen:
                return
            seen.add(key)
            selected.append(item)

        by_yaw: dict[float, list[dict[str, Any]]] = {}
        for item in hypotheses:
            by_yaw.setdefault(round(float(item["yaw"]), 3), []).append(item)
        for group in by_yaw.values():
            ordered = sorted(group, key=lambda item: float(item["score"]))
            for item in ordered[:max(1, per_yaw)]:
                add(item)

        ordered_by_score = sorted(hypotheses, key=lambda item: float(item["score"]))
        add(min(hypotheses, key=lambda item: float(item["scale"])))
        add(max(hypotheses, key=lambda item: float(item["scale"])))
        log_scales = np.asarray([
            math.log(max(float(item["scale"]), 1e-12)) for item in hypotheses
        ])
        stratum_count = max(6, per_yaw * 2)
        boundaries = np.linspace(
            float(log_scales.min()), float(log_scales.max()), stratum_count + 1
        )
        for stratum in range(stratum_count):
            lower, upper = boundaries[stratum], boundaries[stratum + 1]
            members = [
                item for item, log_scale in zip(hypotheses, log_scales)
                if log_scale >= lower - 1e-12
                and (log_scale < upper or stratum == stratum_count - 1)
            ]
            if members:
                add(min(members, key=lambda item: float(item["score"])))
        for item in ordered_by_score[:global_top]:
            add(item)
        return selected

    @staticmethod
    def _initial_heading(relative: np.ndarray) -> float:
        if len(relative) < 2:
            return 0.0
        total = _polyline_length(relative)
        if total <= 1e-8:
            return 0.0
        # Use enough travelled arc to ignore marker occlusion and the first
        # couple of unstable R3 poses, while remaining inside the first aisle.
        # At 2.5% the Kerama start-marker gap biased heading by about 13°.
        early = _resample_polyline(relative, np.asarray([0.0, 0.08]))
        delta = early[1] - early[0]
        if float(np.linalg.norm(delta)) > 1e-8:
            return math.atan2(float(delta[1]), float(delta[0]))
        for index in range(1, len(relative)):
            delta = relative[index] - relative[0]
            if float(np.linalg.norm(delta)) > 1e-8:
                return math.atan2(float(delta[1]), float(delta[0]))
        return 0.0

    def _build_hypothesis(
        self,
        relative: np.ndarray,
        start: np.ndarray,
        desired_heading: float,
        scale: float,
        yaw_offset_degrees: float,
    ) -> np.ndarray:
        rotation = desired_heading - self._initial_heading(relative) + math.radians(yaw_offset_degrees)
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        matrix = np.asarray([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float64)
        return start + (relative * scale) @ matrix.T

    def _score_hypothesis(
        self,
        points: np.ndarray,
        duration: Optional[float],
        yaw_offset_degrees: float,
        *,
        observation_policy: str,
    ) -> tuple[float, dict[str, float]]:
        metrics = self._path_metrics(points)
        length_meters = _polyline_length(points) * self.config.meters_per_pixel
        speed = length_meters / duration if duration else None
        speed_ratio = (
            speed / self.config.walking_speed_mps
            if speed is not None and self.config.walking_speed_mps > 1e-9
            else None
        )
        speed_penalty = 0.0
        if speed_ratio is not None and speed_ratio > 1e-9:
            speed_penalty = _speed_prior_penalty(
                float(speed_ratio), observation_policy, duration
            )
        score = (
            34.0 * metrics["collision_ratio"]
            + 60.0 * metrics["outside_ratio"]
            + 0.35 * metrics["clearance_penalty"]
            + 1.25 * speed_penalty
            + 0.08 * abs(yaw_offset_degrees) / 5.0
        )
        return score, {
            **metrics,
            "length_meters": length_meters,
            "speed_mps": speed if speed is not None else math.nan,
            "speed_ratio": speed_ratio if speed_ratio is not None else math.nan,
            "speed_prior_penalty": speed_penalty,
        }

    def _metric_hypothesis_is_plausible(
        self,
        speed_ratio: float,
        *,
        duration: Optional[float],
        observation_policy: str,
        loop_closure_verified: bool = False,
    ) -> bool:
        if duration is None or not math.isfinite(speed_ratio):
            return observation_policy != "independent"
        # Duration takes precedence over loop topology. A 20--25 minute
        # inspection can return near its start and still contain long stops;
        # applying the short-loop lower speed bound inflates its monocular
        # scale by an order of magnitude before map matching even starts.
        lower, upper = (
            INDEPENDENT_LONG_INSPECTION_SPEED_RATIO_BOUNDS
            if (
                observation_policy == "independent"
                and duration >= LONG_INSPECTION_MIN_SECONDS
            )
            else INDEPENDENT_LOOP_CLOSED_SPEED_RATIO_BOUNDS
            if observation_policy == "independent" and loop_closure_verified
            else INDEPENDENT_SPEED_RATIO_BOUNDS
            if observation_policy == "independent"
            else AUTHORITATIVE_SPEED_RATIO_BOUNDS
        )
        return lower <= speed_ratio <= upper

    def _path_component_count(self, points: np.ndarray) -> int:
        component_ids: set[int] = set()
        for point in self._sample_path(points):
            if self._point_occupied(point):
                continue
            x, y = self._pixel_to_cell(point)
            component = int(self._component_ids[y, x])
            if component:
                component_ids.add(component)
        return len(component_ids)

    def _segment_collides(self, start: np.ndarray, end: np.ndarray) -> bool:
        return any(self._point_occupied(point) for point in self._sample_path(
            np.vstack((start, end))
        ))

    def _astar(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        raw_segment: np.ndarray,
        _search_margin_cells: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        start = self._nearest_free(self._pixel_to_cell(start_point))
        end = self._nearest_free(self._pixel_to_cell(end_point))
        if start is None or end is None:
            return None
        if start == end:
            return np.vstack((self._cell_to_pixel(start), self._cell_to_pixel(end)))
        start_component = int(self._component_ids[start[1], start[0]])
        end_component = int(self._component_ids[end[1], end[0]])
        if start_component == 0 or end_component == 0 or start_component != end_component:
            return None

        raw_cells = np.asarray([self._pixel_to_cell(point) for point in raw_segment], dtype=np.int32)
        # This is deliberately a local repair window.  A larger search can
        # find a legal but unrelated aisle and thereby replace an R3 walk with
        # a plausible-looking route through the plant.
        local_initial_margin_meters = (
            MASK_LOCAL_REPAIR_LIMIT_METERS
            if self._support_mask is not None
            else LOCAL_ASTAR_INITIAL_MARGIN_METERS
        )
        initial_margin = max(
            2 if self._support_mask is not None else 24,
            int(round(local_initial_margin_meters / max(self.cell_meters, 1e-9))),
        )
        margin = initial_margin if _search_margin_cells is None else _search_margin_cells
        min_x = max(0, min(start[0], end[0], int(raw_cells[:, 0].min())) - margin)
        max_x = min(self.cols - 1, max(start[0], end[0], int(raw_cells[:, 0].max())) + margin)
        min_y = max(0, min(start[1], end[1], int(raw_cells[:, 1].min())) - margin)
        max_y = min(self.rows - 1, max(start[1], end[1], int(raw_cells[:, 1].max())) + margin)

        local_shape = (max_y - min_y + 1, max_x - min_x + 1)
        raw_seed = np.ones(local_shape, dtype=bool)
        for point in self._sample_path(raw_segment):
            x, y = self._pixel_to_cell(point)
            if min_x <= x <= max_x and min_y <= y <= max_y:
                raw_seed[y - min_y, x - min_x] = False
        deviation = ndimage.distance_transform_edt(raw_seed)

        neighbors = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        )
        queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
        previous: dict[tuple[int, int], tuple[int, int]] = {}
        costs = {start: 0.0}
        visited: set[tuple[int, int]] = set()

        while queue:
            _, cost, current = heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            if current == end:
                cells = [current]
                while cells[-1] != start:
                    cells.append(previous[cells[-1]])
                cells.reverse()
                route = np.asarray([self._cell_to_pixel(cell) for cell in cells])
                # A cell-by-cell route is needlessly large, but uniformly
                # resampling it back to the R3 point count can cut straight
                # through an obstacle when the source trajectory is sparse.
                # Keep the smallest collision-free set of line-of-sight
                # anchors instead.  This preserves the A* homotopy and makes
                # every rendered segment safe, independently of source FPS.
                simplified = [route[0]]
                anchor = 0
                while anchor < len(route) - 1:
                    candidate = len(route) - 1
                    while candidate > anchor + 1 and self._segment_collides(
                        route[anchor], route[candidate]
                    ):
                        candidate -= 1
                    simplified.append(route[candidate])
                    anchor = candidate
                # Grid search runs between cell centres, but callers stitch
                # this route into a continuous measured polyline. Reattach
                # the exact free endpoints through verified sub-cell
                # connectors so quantisation cannot create a new collision at
                # either splice (or move the operator's start anchor).
                if (
                    not self._point_occupied(start_point)
                    and float(np.linalg.norm(
                        np.asarray(start_point) - simplified[0]
                    )) > 1e-9
                    and not self._segment_collides(
                        np.asarray(start_point), simplified[0]
                    )
                ):
                    simplified.insert(0, np.asarray(start_point).copy())
                if (
                    not self._point_occupied(end_point)
                    and float(np.linalg.norm(
                        np.asarray(end_point) - simplified[-1]
                    )) > 1e-9
                    and not self._segment_collides(
                        simplified[-1], np.asarray(end_point)
                    )
                ):
                    simplified.append(np.asarray(end_point).copy())
                return np.asarray(simplified, dtype=np.float64)
            for dx, dy, step in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                if nx < min_x or nx > max_x or ny < min_y or ny > max_y:
                    continue
                if self.occupied[ny, nx]:
                    continue
                if dx and dy and (
                    self.occupied[current[1], nx]
                    or self.occupied[ny, current[0]]
                ):
                    # A diagonal between two blocked orthogonal cells clips
                    # the physical obstacle corner even when both endpoint
                    # cells are free.
                    continue
                clearance = float(self.clearance_meters[ny, nx])
                deviation_meters = (
                    float(deviation[ny - min_y, nx - min_x]) * self.cell_meters
                )
                # Every term is now a dimensionless multiplier of a metric
                # step. This makes the same physical map substantially
                # invariant to grid_cell_pixels.
                wall_multiplier = 0.55 * math.exp(-clearance / 0.45)
                # Stronger pull toward the visual observation so A* prefers
                # local aisle snaps over long spikes through blank CAD gaps.
                deviation_multiplier = min(8.0, 0.90 * deviation_meters)
                step_meters = step * self.cell_meters
                candidate = cost + step_meters * (
                    1.0 + wall_multiplier + deviation_multiplier
                )
                key = (nx, ny)
                if candidate >= costs.get(key, float("inf")):
                    continue
                costs[key] = candidate
                previous[key] = current
                heuristic = math.hypot(end[0] - nx, end[1] - ny) * self.cell_meters
                heappush(queue, (candidate + heuristic, candidate, key))
        # One bounded expansion handles a wide local obstruction without
        # crossing into a remote corridor.
        local_max_margin_meters = (
            MASK_LOCAL_REPAIR_LIMIT_METERS
            if self._support_mask is not None
            else LOCAL_ASTAR_MAX_MARGIN_METERS
        )
        max_margin = max(
            initial_margin,
            int(round(local_max_margin_meters / max(self.cell_meters, 1e-9))),
        )
        if margin < max_margin:
            return self._astar(
                start_point,
                end_point,
                raw_segment,
                _search_margin_cells=min(max_margin, margin * 2),
            )
        return None

    def _detour_metrics(
        self,
        route: np.ndarray,
        start_point: np.ndarray,
        end_point: np.ndarray,
        raw_segment: np.ndarray,
    ) -> dict[str, float | bool]:
        """Measure whether a detour is supported by the observed trajectory arc."""
        chord_m = float(np.linalg.norm(end_point - start_point)) * self.config.meters_per_pixel
        route_m = _polyline_length(route) * self.config.meters_per_pixel
        raw_m = _polyline_length(raw_segment) * self.config.meters_per_pixel
        if len(raw_segment) < 2 or len(route) < 2:
            return {
                "spike": False,
                "chord_meters": chord_m,
                "observed_arc_meters": raw_m,
                "route_meters": route_m,
                "max_deviation_meters": 0.0,
                "observed_tortuosity": raw_m / max(chord_m, 1e-6),
                "route_observed_length_ratio": route_m / max(raw_m, 1e-6),
            }
        max_dev = 0.0
        for point in route:
            best = float("inf")
            for left, right in zip(raw_segment[:-1], raw_segment[1:]):
                delta = right - left
                length = float(np.linalg.norm(delta))
                if length <= 1e-9:
                    best = min(best, float(np.linalg.norm(point - left)))
                    continue
                t = float(np.clip(np.dot(point - left, delta) / (length * length), 0.0, 1.0))
                best = min(best, float(np.linalg.norm(point - (left + t * delta))))
            max_dev = max(max_dev, best * self.config.meters_per_pixel)
        observed_tortuosity = raw_m / max(chord_m, 1e-6)
        route_observed_ratio = route_m / max(raw_m, 1e-6)
        deviation_limit = max(4.0, 0.40 * raw_m)
        straight_observation_spike = (
            observed_tortuosity < 1.45
            and route_m > max(3.5 * max(chord_m, 1e-6), chord_m + 8.0)
            and max_dev > deviation_limit
        )
        unsupported_inflation = (
            route_m > max(1.85 * max(raw_m, 1e-6), raw_m + 12.0)
            and max_dev > deviation_limit
        )
        remote_route = (
            max_dev > max(12.0, 0.55 * raw_m)
            and route_m > 1.25 * max(raw_m, 1e-6)
        )
        return {
            "spike": bool(
                straight_observation_spike or unsupported_inflation or remote_route
            ),
            "chord_meters": chord_m,
            "observed_arc_meters": raw_m,
            "route_meters": route_m,
            "max_deviation_meters": max_dev,
            "observed_tortuosity": observed_tortuosity,
            "route_observed_length_ratio": route_observed_ratio,
            "straight_observation_spike": straight_observation_spike,
            "unsupported_inflation": unsupported_inflation,
            "remote_route": remote_route,
        }

    def _detour_is_spike(
        self,
        route: np.ndarray,
        start_point: np.ndarray,
        end_point: np.ndarray,
        raw_segment: np.ndarray,
    ) -> bool:
        return bool(self._detour_metrics(
            route, start_point, end_point, raw_segment
        )["spike"])

    def _collision_runs(self, points: np.ndarray) -> list[tuple[int, int]]:
        bad = np.asarray([
            self._segment_collides(points[index], points[index + 1])
            for index in range(len(points) - 1)
        ], dtype=bool)
        runs: list[tuple[int, int]] = []
        index = 0
        while index < len(bad):
            if not bad[index]:
                index += 1
                continue
            start = index
            while index + 1 < len(bad) and bad[index + 1]:
                index += 1
            runs.append((start, index + 1))
            index += 1
        return runs

    def _max_collision_run_meters(self, points: np.ndarray) -> float:
        return max(
            (
                _polyline_length(points[left:right + 1])
                * self.config.meters_per_pixel
                for left, right in self._collision_runs(points)
            ),
            default=0.0,
        )

    def _repair_collisions(
        self,
        points: np.ndarray,
        *,
        failure_reasons: Optional[list[str]] = None,
        allow_provisional_spikes: bool = False,
        repair_diagnostics: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[np.ndarray], int]:
        runs = self._collision_runs(points)
        if not runs:
            return points.copy(), 0
        merged: list[tuple[int, int]] = []
        for left, right in runs:
            while left > 0 and self._point_occupied(points[left]):
                left -= 1
            while right < len(points) - 1 and self._point_occupied(points[right]):
                right += 1
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))

        rebuilt: list[np.ndarray] = []
        cursor = 0
        rerouted = 0
        for left, right in merged:
            if left == 0 and self._point_occupied(points[0]):
                return None, len(merged)
            rebuilt.extend(points[cursor:left])
            raw_segment = points[left:right + 1]
            start_cell = self._nearest_free(self._pixel_to_cell(points[left]))
            end_cell = self._nearest_free(self._pixel_to_cell(points[right]))
            connectivity_reason = None
            if start_cell is None or end_cell is None:
                connectivity_reason = "no_walkable_segment_endpoint"
            elif (
                int(self._component_ids[start_cell[1], start_cell[0]]) == 0
                or int(self._component_ids[end_cell[1], end_cell[0]]) == 0
                or int(self._component_ids[start_cell[1], start_cell[0]])
                != int(self._component_ids[end_cell[1], end_cell[0]])
            ):
                connectivity_reason = "different_walkable_components"

            route = self._astar(points[left], points[right], raw_segment)
            detour_metrics = (
                self._detour_metrics(
                    route, points[left], points[right], raw_segment
                )
                if route is not None else None
            )
            spike_rejected = bool(detour_metrics and detour_metrics["spike"])
            if repair_diagnostics is not None and detour_metrics is not None:
                repair_diagnostics.setdefault("detours", []).append({
                    key: round(float(value), 5)
                    if isinstance(value, (float, np.floating)) else value
                    for key, value in detour_metrics.items()
                })
            if spike_rejected and not allow_provisional_spikes:
                # A 3 m observation nick must not become a 30 m plant loop.
                route = None
            elif spike_rejected and repair_diagnostics is not None:
                repair_diagnostics["provisional_spike_count"] = int(
                    repair_diagnostics.get("provisional_spike_count", 0)
                ) + 1
            if route is None:
                reason = (
                    connectivity_reason
                    or ("detour_spike_rejected" if spike_rejected else "local_search_exhausted")
                )
                if failure_reasons is not None:
                    failure_reasons.append(reason)
                return None, len(merged)
            rerouted += 1
            if rebuilt and np.linalg.norm(rebuilt[-1] - route[0]) < 1e-6:
                route = route[1:]
            rebuilt.extend(route)
            cursor = right + 1
        rebuilt.extend(points[cursor:])
        # Do not force the repaired line back to the original sample count.
        # Sparse uniform resampling can reconnect two safe samples with a
        # chord through the obstacle.  Consumers already accept a polyline of
        # arbitrary length, and source-frame/turn mapping is handled by arc
        # fraction below.
        return np.asarray(rebuilt, dtype=np.float64), rerouted

    def _repair_long_trajectory_in_time_segments(
        self,
        points: np.ndarray,
        timestamps: Any,
        *,
        target_seconds: float = 180.0,
    ) -> tuple[Optional[np.ndarray], dict[str, Any]]:
        """Repair a long observation in time-local windows with shared anchors.

        Scale and yaw have already been selected by the caller for the whole
        observation.  This method is therefore allowed to correct local drift,
        but cannot independently resize or rotate individual windows.
        """
        diagnostics: dict[str, Any] = {
            "attempted": True,
            "accepted": False,
            "method": "shared_scale_time_segments_v1",
            "target_segment_seconds": target_seconds,
        }
        if not isinstance(timestamps, list) or len(timestamps) != len(points):
            diagnostics["reason"] = "segment_timestamps_unavailable"
            return None, diagnostics
        clock = np.asarray(
            [_finite_float(value, math.nan) for value in timestamps],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(clock)) or np.any(np.diff(clock) < 0.0):
            diagnostics["reason"] = "segment_timestamps_not_monotonic"
            return None, diagnostics
        elapsed = float(clock[-1] - clock[0])
        if elapsed < max(480.0, target_seconds * 2.0):
            diagnostics["reason"] = "trajectory_not_long_enough_for_segmentation"
            return None, diagnostics

        segment_count = max(2, int(math.ceil(elapsed / target_seconds)))
        boundary_times = np.linspace(float(clock[0]), float(clock[-1]), segment_count + 1)
        boundary_indices = [0]
        for boundary_time in boundary_times[1:-1]:
            search_left = max(
                boundary_indices[-1] + 2,
                int(np.searchsorted(clock, boundary_time - 45.0, side="left")),
            )
            search_right = min(
                len(points) - 3,
                int(np.searchsorted(clock, boundary_time + 45.0, side="right")),
            )
            candidates: list[tuple[float, int]] = []
            for index in range(search_left, search_right + 1):
                free_cell = self._nearest_free(self._pixel_to_cell(points[index]))
                if free_cell is None:
                    continue
                projected = self._cell_to_pixel(free_cell)
                shift_meters = float(np.linalg.norm(projected - points[index])) \
                    * self.config.meters_per_pixel
                time_penalty = abs(float(clock[index]) - float(boundary_time)) \
                    / 45.0
                candidates.append((shift_meters + 0.5 * time_penalty, index))
            if not candidates:
                diagnostics.update({
                    "reason": "segment_control_point_window_not_walkable",
                    "boundary_time_seconds": round(float(boundary_time), 3),
                })
                return None, diagnostics
            boundary_indices.append(min(candidates)[1])
        boundary_indices.append(len(points) - 1)

        anchored = points.copy()
        anchor_diagnostics: list[dict[str, Any]] = []
        maximum_anchor_shift_meters = 8.0
        for boundary_number, index in enumerate(boundary_indices):
            free_cell = self._nearest_free(self._pixel_to_cell(anchored[index]))
            if free_cell is None:
                diagnostics.update({
                    "reason": "segment_control_point_not_walkable",
                    "failed_boundary": boundary_number,
                })
                return None, diagnostics
            projected = self._cell_to_pixel(free_cell)
            shift_meters = float(np.linalg.norm(projected - anchored[index])) \
                * self.config.meters_per_pixel
            anchor_diagnostics.append({
                "boundary": boundary_number,
                "trajectory_index": int(index),
                "timestamp_seconds": round(float(clock[index]), 3),
                "projection_meters": round(shift_meters, 3),
            })
            if shift_meters > maximum_anchor_shift_meters:
                diagnostics.update({
                    "reason": "segment_control_point_too_far_from_walkable_area",
                    "failed_boundary": boundary_number,
                    "maximum_anchor_shift_meters": maximum_anchor_shift_meters,
                    "control_points": anchor_diagnostics,
                })
                return None, diagnostics
            anchored[index] = projected

        stitched: list[np.ndarray] = []
        segment_diagnostics: list[dict[str, Any]] = []
        rerouted_total = 0
        for segment_number, (left, right) in enumerate(
            zip(boundary_indices[:-1], boundary_indices[1:])
        ):
            window = anchored[left:right + 1]
            failures: list[str] = []
            repair_details: dict[str, Any] = {}
            repaired, rerouted = self._repair_collisions(
                window,
                failure_reasons=failures,
                allow_provisional_spikes=False,
                repair_diagnostics=repair_details,
            )
            segment_info = {
                "segment": segment_number,
                "start_index": int(left),
                "end_index": int(right),
                "start_seconds": round(float(clock[left]), 3),
                "end_seconds": round(float(clock[right]), 3),
                "duration_seconds": round(float(clock[right] - clock[left]), 3),
                "accepted": repaired is not None,
                "rejection_reasons": failures,
                "rerouted_segments": int(rerouted),
            }
            segment_diagnostics.append(segment_info)
            if repaired is None:
                diagnostics.update({
                    "reason": "segment_repair_failed",
                    "failed_segment": segment_number,
                    "control_points": anchor_diagnostics,
                    "segments": segment_diagnostics,
                })
                return None, diagnostics
            if stitched and np.linalg.norm(stitched[-1] - repaired[0]) < 1e-6:
                repaired = repaired[1:]
            stitched.extend(repaired)
            rerouted_total += int(rerouted)

        result = np.asarray(stitched, dtype=np.float64)
        if len(result) < 2 or self._collision_runs(result):
            diagnostics.update({
                "reason": "stitched_segment_route_not_safe",
                "control_points": anchor_diagnostics,
                "segments": segment_diagnostics,
            })
            return None, diagnostics
        diagnostics.update({
            "accepted": True,
            "reason": None,
            "elapsed_seconds": round(elapsed, 3),
            "segment_count": segment_count,
            "control_points": anchor_diagnostics,
            "segments": segment_diagnostics,
            "rerouted_segments": rerouted_total,
            "shared_scale_preserved": True,
            "red_obstacles_hard": True,
        })
        return result, diagnostics

    def _adaptive_anchor_fractions(
        self, points: np.ndarray, *, maximum: int
    ) -> np.ndarray:
        """Place anchors by travelled distance and retain curvature extrema."""
        if len(points) < 3:
            return np.linspace(0.0, 1.0, max(2, len(points)))
        fractions = _trajectory_fractions(points)
        deltas = np.diff(points[:, :2], axis=0)
        headings = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
        curvature = np.zeros(len(points), dtype=np.float64)
        if len(headings) > 1:
            curvature[1:-1] = np.abs(np.diff(headings))
        base_count = max(6, min(maximum, int(math.ceil(_polyline_length(points) / max(
            2.5 / max(self.config.meters_per_pixel, 1e-9), 1.0
        ))) + 1))
        selected = set(int(index) for index in np.searchsorted(
            fractions, np.linspace(0.0, 1.0, base_count)
        ).clip(0, len(points) - 1))
        for index in np.argsort(curvature)[::-1]:
            if curvature[index] < math.radians(3.0):
                break
            if all(abs(float(fractions[index] - fractions[item])) >= 0.018 for item in selected):
                selected.add(int(index))
            if len(selected) >= maximum:
                break
        selected.update((0, len(points) - 1))
        if len(selected) > maximum:
            mandatory = {0, len(points) - 1}
            ranked = sorted(
                selected - mandatory,
                key=lambda index: curvature[index],
                reverse=True,
            )[: maximum - 2]
            selected = mandatory | set(ranked)
        return np.asarray([fractions[index] for index in sorted(selected)], dtype=np.float64)

    def _map_state_candidates(
        self,
        observation: np.ndarray,
        guide: np.ndarray,
        *,
        radius_meters: float,
        limit: int,
        fixed: bool = False,
        required_component: Optional[int] = None,
    ) -> np.ndarray:
        if len(self._topology_node_points):
            distances = np.minimum(
                np.linalg.norm(
                    self._topology_node_points - observation[None, :], axis=1
                ),
                0.85 * np.linalg.norm(
                    self._topology_node_points - guide[None, :], axis=1
                ),
            )
            if fixed:
                return self._topology_node_points[[
                    int(np.argmin(np.linalg.norm(
                        self._topology_node_points - guide[None, :], axis=1
                    )))
                ]]
            radius_pixels = (
                radius_meters / max(self.config.meters_per_pixel, 1e-9)
            )
            indices = np.where(distances <= radius_pixels)[0]
            if not len(indices):
                return np.empty((0, 2), dtype=np.float64)
            ranked = indices[np.argsort(distances[indices])]
            return self._topology_node_points[ranked[:limit]].copy()
        if fixed:
            cell = self._nearest_free(self._pixel_to_cell(guide))
            if cell is None or (
                required_component is not None
                and int(self._component_ids[cell[1], cell[0]]) != required_component
            ):
                return np.empty((0, 2))
            return np.asarray([self._cell_to_pixel(cell)])
        radius_cells = max(2, int(math.ceil(radius_meters / max(self.cell_meters, 1e-9))))
        cells: set[tuple[int, int]] = set()
        for seed in (observation, guide):
            free = self._nearest_free(self._pixel_to_cell(seed))
            if free is not None:
                cells.add(free)
        centre = np.asarray(self._pixel_to_cell(observation))
        guide_cell = np.asarray(self._pixel_to_cell(guide))
        for origin in (centre, guide_cell):
            samples = np.linspace(-radius_cells, radius_cells, 9).round().astype(int)
            for dx in samples:
                for dy in samples:
                    if dx * dx + dy * dy > radius_cells * radius_cells:
                        continue
                    x, y = int(origin[0] + dx), int(origin[1] + dy)
                    if self._inside_cell(x, y) and not self.occupied[y, x]:
                        cells.add((x, y))
        if len(self._corridor_nodes):
            delta_observation = self._corridor_nodes - centre[None, :]
            delta_guide = self._corridor_nodes - guide_cell[None, :]
            nearby = self._corridor_nodes[
                (np.sum(delta_observation ** 2, axis=1) <= radius_cells ** 2)
                | (np.sum(delta_guide ** 2, axis=1) <= radius_cells ** 2)
            ]
            cells.update((int(item[0]), int(item[1])) for item in nearby)
        if required_component is not None:
            cells = {
                cell for cell in cells
                if int(self._component_ids[cell[1], cell[0]]) == required_component
            }
        ranked = sorted(cells, key=lambda cell: (
            min(
                float(np.linalg.norm(self._cell_to_pixel(cell) - observation)),
                float(np.linalg.norm(self._cell_to_pixel(cell) - guide)) * 0.85,
            )
            - 0.10 * float(self.clearance_meters[cell[1], cell[0]])
            / max(self.config.meters_per_pixel, 1e-9)
        ))
        chosen: list[np.ndarray] = []
        separation = max(self.config.grid_cell_pixels * 2.0, radius_cells * self.config.grid_cell_pixels * 0.12)
        for cell in ranked:
            point = self._cell_to_pixel(cell)
            if chosen and min(float(np.linalg.norm(point - prior)) for prior in chosen) < separation:
                continue
            chosen.append(point)
            if len(chosen) >= limit:
                break
        return np.asarray(chosen, dtype=np.float64)

    def _viterbi_level(
        self,
        observation: np.ndarray,
        guide: np.ndarray,
        fractions: np.ndarray,
        *,
        radius_meters: float,
        candidate_limit: int,
        allow_component_edges: bool = False,
    ) -> tuple[Optional[np.ndarray], dict[str, Any]]:
        observed = _resample_polyline(observation, fractions)
        guided = _resample_polyline(guide, fractions)
        required_component: Optional[int] = None
        if allow_component_edges:
            start_cell = self._nearest_free(self._pixel_to_cell(guided[0]))
            if start_cell is not None:
                component = int(self._component_ids[start_cell[1], start_cell[0]])
                required_component = component if component != 0 else None
        states = [
            self._map_state_candidates(
                observed[index], guided[index], radius_meters=radius_meters,
                limit=candidate_limit,
                fixed=index == 0,
                required_component=required_component,
            )
            for index in range(len(fractions))
        ]
        if any(len(layer) == 0 for layer in states):
            return None, {
                "reason": "empty_state_layer",
                "required_component": required_component,
            }
        radius_pixels = radius_meters / max(self.config.meters_per_pixel, 1e-9)
        # Establish the graph anchor first, then measure the R3 departure over
        # the same physical lookahead used for graph candidates. Translation
        # itself does not rotate R3, but this ordering makes both vectors share
        # one snapped origin and one 3-8 m scale of comparison.
        snapped_observed = (
            observed - observed[0] + states[0][0]
        )
        stable_initial_direction = self._topology_route_initial_direction(
            snapped_observed
        )
        stable_initial_direction_length = float(np.linalg.norm(
            stable_initial_direction
        ))
        if stable_initial_direction_length <= 1e-9:
            stable_initial_heading = self._initial_heading(
                snapped_observed - snapped_observed[0]
            )
            stable_initial_direction = np.asarray([
                math.cos(stable_initial_heading),
                math.sin(stable_initial_heading),
            ])
        else:
            stable_initial_direction /= stable_initial_direction_length
        source_turn_events = _turn_events(
            observed,
            samples=max(96, len(observed) * 4),
            min_turn_degrees=TURN_TOPOLOGY_MIN_DEGREES,
        )
        edge_cache: dict[tuple[int, int, int], float] = {}
        edge_route_cache: dict[tuple[int, int, int], np.ndarray] = {}
        initial_branch_directions: dict[tuple[int, int, int], np.ndarray] = {}
        parallel_initial_branch_groups = 0
        edge_rejections: dict[str, int] = {}
        routed_edge_count = 0

        def reject_edge(key: tuple[int, int, int], reason: str) -> float:
            edge_rejections[reason] = edge_rejections.get(reason, 0) + 1
            edge_cache[key] = float("inf")
            return float("inf")

        def edge(layer: int, left: int, right: int) -> float:
            nonlocal routed_edge_count
            key = (layer, left, right)
            if key in edge_cache:
                return edge_cache[key]
            start, end = states[layer - 1][left], states[layer][right]
            visual = observed[layer] - observed[layer - 1]
            mapped = end - start
            visual_length = max(float(np.linalg.norm(visual)), 1e-6)
            routed_edge = False
            routed_path: Optional[np.ndarray] = None
            if len(self._topology_node_points):
                routed_path = self._topology_route(start, end)
                if routed_path is None or len(routed_path) < 2:
                    return reject_edge(key, "explicit_graph_disconnected")
                mapped = (
                    self._topology_route_initial_direction(routed_path)
                    if layer == 1
                    else routed_path[1] - routed_path[0]
                )
                routed_edge = len(routed_path) > 2
                edge_route_cache[key] = routed_path
                if layer == 1:
                    initial_branch_directions[key] = mapped.copy()
            mapped_chord_length = max(float(np.linalg.norm(mapped)), 1e-6)
            direction_visual = (
                stable_initial_direction * visual_length
                if layer == 1 else visual
            )
            direction_cosine = float(np.clip(
                np.dot(direction_visual, mapped)
                / (visual_length * mapped_chord_length),
                -1.0,
                1.0,
            ))
            candidate_length = (
                _polyline_length(routed_path)
                if routed_path is not None
                else mapped_chord_length
            )
            chord_ratio = candidate_length / visual_length
            # Cheap geometric pruning must happen before A*: candidates that
            # already disagree in chord scale/direction cannot become valid by
            # adding a more expensive detour.
            if (
                not 0.10 <= chord_ratio <= 5.00
                or direction_cosine < -0.75
            ):
                return reject_edge(key, "geometry_pruned")
            if (
                routed_path is None
                and self._segment_collides(start, end)
            ):
                start_cell = self._nearest_free(self._pixel_to_cell(start))
                end_cell = self._nearest_free(self._pixel_to_cell(end))
                same_component = bool(
                    start_cell is not None
                    and end_cell is not None
                    and int(self._component_ids[start_cell[1], start_cell[0]]) != 0
                    and int(self._component_ids[start_cell[1], start_cell[0]])
                    == int(self._component_ids[end_cell[1], end_cell[0]])
                )
                if not allow_component_edges or not same_component:
                    return reject_edge(key, "raster_component_disconnected")
                routed_edge = True
                observed_edge = np.vstack((
                    observed[layer - 1], observed[layer]
                ))
                for margin_meters in (6.0,):
                    margin_cells = max(
                        2,
                        int(math.ceil(
                            margin_meters / max(self.cell_meters, 1e-9)
                        )),
                    )
                    candidate_route = self._astar(
                        start,
                        end,
                        observed_edge,
                        _search_margin_cells=margin_cells,
                    )
                    if candidate_route is None:
                        continue
                    detour = self._detour_metrics(
                        candidate_route, start, end, observed_edge
                    )
                    if (
                        layer > 1
                        and bool(detour["spike"])
                    ):
                        continue
                    routed_path = candidate_route
                    break
                if routed_path is None:
                    return reject_edge(key, "astar_failed")
                edge_route_cache[key] = routed_path
            mapped_length = max(
                _polyline_length(routed_path)
                if routed_path is not None
                else float(np.linalg.norm(mapped)),
                1e-6,
            )
            initial_direction_penalty = 0.0
            if layer == 1:
                initial_direction_penalty, opposite = (
                    self._initial_direction_cost(direction_cosine)
                )
                if opposite:
                    return reject_edge(key, "initial_direction_opposite")
            value = (
                1.35 * abs(math.log(mapped_length / visual_length))
                + 1.7 * (1.0 - direction_cosine)
                + (0.65 if routed_edge else 0.0)
                + initial_direction_penalty
            )
            edge_cache[key] = value
            return value

        def emission(layer: int, state: int) -> float:
            point = states[layer][state]
            observation_error = (
                float(np.linalg.norm(point - observed[layer]))
                / max(radius_pixels, 1.0)
            )
            guide_error = (
                float(np.linalg.norm(point - guided[layer]))
                / max(radius_pixels, 1.0)
            )
            # The guide represents the branch selected at the previous level.
            # Keeping a separate guide term prevents a state from hopping to a
            # nearby parallel aisle merely because both are close to R3.
            value = 0.7 * observation_error ** 2 + 0.45 * guide_error ** 2
            x, y = self._pixel_to_cell(point)
            return value + 0.08 * math.exp(
                -float(self.clearance_meters[y, x]) / 0.45
            )

        def signed_turn(left: np.ndarray, right: np.ndarray) -> float:
            return math.atan2(
                float(left[0] * right[1] - left[1] * right[0]),
                float(np.dot(left, right)),
            )

        def second_order(layer: int, grand: int, left: int, right: int) -> float:
            visual_before = observed[layer - 1] - observed[layer - 2]
            visual_after = observed[layer] - observed[layer - 1]
            mapped_before = states[layer - 1][left] - states[layer - 2][grand]
            mapped_after = states[layer][right] - states[layer - 1][left]
            prior_route = edge_route_cache.get((layer - 1, grand, left))
            next_route = edge_route_cache.get((layer, left, right))
            if prior_route is not None and len(prior_route) >= 2:
                mapped_before = prior_route[-1] - prior_route[-2]
            if next_route is not None and len(next_route) >= 2:
                mapped_after = next_route[1] - next_route[0]
            visual_before_length = max(float(np.linalg.norm(visual_before)), 1e-6)
            visual_after_length = max(float(np.linalg.norm(visual_after)), 1e-6)
            mapped_before_length = max(float(np.linalg.norm(mapped_before)), 1e-6)
            mapped_after_length = max(float(np.linalg.norm(mapped_after)), 1e-6)
            scale_before = mapped_before_length / visual_before_length
            scale_after = mapped_after_length / visual_after_length
            visual_turn = signed_turn(visual_before, visual_after)
            mapped_turn = signed_turn(mapped_before, mapped_after)
            anchor_fraction = float(fractions[layer - 1])
            expected_event = min(
                source_turn_events,
                key=lambda item: abs(float(item[0]) - anchor_fraction),
                default=None,
            )
            expected_turn = (
                math.radians(float(expected_event[1]))
                if expected_event is not None
                and abs(float(expected_event[0]) - anchor_fraction) <= 0.04
                else None
            )
            turn_delta = math.atan2(
                math.sin(mapped_turn - visual_turn),
                math.cos(mapped_turn - visual_turn),
            )
            value = (
                0.80 * abs(math.log(max(scale_after / scale_before, 1e-12)))
                + 1.20 * abs(turn_delta) / math.pi
            )
            if (
                expected_turn is not None
                and abs(mapped_turn) >= math.radians(12.0)
                and expected_turn * mapped_turn < 0.0
            ):
                # The sign sequence is part of the dynamic-programming state,
                # not merely a post-hoc quality score. An opposite branch can
                # never recover the measured event sequence later.
                edge_rejections["turn_sign_sequence_mismatch"] = (
                    edge_rejections.get("turn_sign_sequence_mismatch", 0) + 1
                )
                return float("inf")
            if prior_route is not None and next_route is not None:
                # Branch commitment. Consecutive transitions may meet at a
                # real graph node, but may not immediately traverse the same
                # corridor backwards to hop onto a parallel aisle.
                prior_in = prior_route[-1] - prior_route[-2]
                next_out = next_route[1] - next_route[0]
                denominator = max(
                    float(np.linalg.norm(prior_in))
                    * float(np.linalg.norm(next_out)),
                    1e-9,
                )
                continuation_cosine = float(
                    np.dot(prior_in, next_out) / denominator
                )
                visual_cosine = float(
                    np.dot(visual_before, visual_after)
                    / max(visual_before_length * visual_after_length, 1e-9)
                )
                if continuation_cosine < -0.92 and visual_cosine > -0.50:
                    edge_rejections["branch_commitment_backtrack"] = (
                        edge_rejections.get("branch_commitment_backtrack", 0)
                        + 1
                    )
                    return float("inf")
            if (
                abs(mapped_turn) >= math.radians(150.0)
                and abs(visual_turn) <= math.radians(90.0)
            ):
                value += 2.00
            # Branch commitment: on parallel aisles the correction vector is
            # approximately lateral. A sudden reversal or large jump of that
            # vector is a branch switch, not ordinary observation noise.
            prior_correction = states[layer - 1][left] - observed[layer - 1]
            next_correction = states[layer][right] - observed[layer]
            correction_jump = float(np.linalg.norm(
                next_correction - prior_correction
            )) / max(radius_pixels, 1.0)
            value += 1.35 * correction_jump
            if (
                np.linalg.norm(prior_correction) > radius_pixels * 0.18
                and np.linalg.norm(next_correction) > radius_pixels * 0.18
                and float(np.dot(prior_correction, next_correction)) < 0.0
            ):
                value += 2.25
            return value

        def confirmed_prefix(
            costs: np.ndarray,
            parents_so_far: list[np.ndarray],
            completed_layer: int,
        ) -> dict[str, Any]:
            """Backtrack the best finite DP history without inventing a tail."""
            if completed_layer < 1 or not np.isfinite(costs).any():
                return {}
            selected_flat = int(np.nanargmin(costs))
            left, right = np.unravel_index(selected_flat, costs.shape)
            indices = [int(left), int(right)]
            for parent in reversed(parents_so_far):
                grand = int(parent[indices[0], indices[1]])
                if grand < 0:
                    return {}
                indices.insert(0, grand)
            if len(indices) != completed_layer + 1:
                return {}
            stitched: list[np.ndarray] = []
            edge_ids: list[str] = []
            previous_node_index: Optional[int] = None
            for route_layer in range(1, len(indices)):
                route = edge_route_cache.get((
                    route_layer,
                    int(indices[route_layer - 1]),
                    int(indices[route_layer]),
                ))
                if route is None or len(route) < 2:
                    break
                if stitched and np.linalg.norm(stitched[-1] - route[0]) < 1e-6:
                    route = route[1:]
                stitched.extend(route)
                for edge_id in self._topology_edge_sequence(route):
                    if not edge_ids or edge_ids[-1] != edge_id:
                        edge_ids.append(edge_id)
                if route_layer == len(indices) - 1:
                    previous_node_index = int(np.argmin(np.linalg.norm(
                        self._topology_node_points
                        - states[route_layer - 1][indices[route_layer - 1]][None, :],
                        axis=1,
                    )))
            if len(stitched) < 2 or not edge_ids:
                return {}
            marker = np.asarray(stitched[-1], dtype=np.float64)
            marker_node = int(np.argmin(np.linalg.norm(
                self._topology_node_points - marker[None, :], axis=1
            )))
            next_visual = (
                observed[min(completed_layer + 1, len(observed) - 1)]
                - observed[completed_layer]
            )
            next_length = max(float(np.linalg.norm(next_visual)), 1e-9)
            competing: list[tuple[float, dict[str, Any]]] = []
            for neighbour, _ in self._topology_adjacency.get(marker_node, []):
                if previous_node_index is not None and neighbour == previous_node_index:
                    continue
                endpoint = self._topology_node_points[neighbour]
                vector = endpoint - marker
                cosine = float(np.dot(next_visual, vector) / max(
                    next_length * float(np.linalg.norm(vector)), 1e-9
                ))
                edge_id = self._topology_segment_edges.get(
                    (marker_node, neighbour)
                )
                if not edge_id:
                    continue
                competing.append((
                    -cosine,
                    {
                        "edge_id": edge_id,
                        "points": [
                            [round(float(value), 3) for value in marker],
                            [round(float(value), 3) for value in endpoint],
                        ],
                        "direction_cosine": round(cosine, 5),
                    },
                ))
            competing.sort(key=lambda item: item[0])
            return {
                "_confirmed_prefix": [
                    [float(value[0]), float(value[1])] for value in stitched
                ],
                "_confirmed_edge_ids": edge_ids,
                "_confirmed_fraction_end": round(
                    float(fractions[completed_layer]), 8
                ),
                "_uncertainty_marker": [
                    round(float(marker[0]), 3),
                    round(float(marker[1]), 3),
                ],
                "_competing_next_edges": [
                    item[1] for item in competing[:2]
                ],
            }

        initial_costs = np.asarray([emission(0, index) for index in range(len(states[0]))])
        if len(states) == 1:
            ranking = np.argsort(initial_costs)
            indices = [int(ranking[0])]
            final_costs = initial_costs
        else:
            pair_costs = np.full(
                (len(states[0]), len(states[1])), float("inf"), dtype=np.float64
            )
            for left, accumulated in enumerate(initial_costs):
                for right in range(len(states[1])):
                    pair_costs[left, right] = (
                        float(accumulated) + edge(1, left, right) + emission(1, right)
                    )
            # Near-collinear departures form one ambiguity class at the start.
            # Keep every member and its own geometric cost alive; subsequent
            # signed turn events select the concrete edge. In particular, do
            # not reject the whole class merely because its members are close.
            finite_initial_keys = [
                (left, right)
                for left in range(len(states[0]))
                for right in range(len(states[1]))
                if math.isfinite(float(pair_costs[left, right]))
                and (1, left, right) in initial_branch_directions
            ]
            parallel_threshold = math.cos(math.radians(
                GRAPH_MAP_MATCH_PARALLEL_BRANCH_DEGREES
            ))
            grouped_initial_keys: set[tuple[int, int]] = set()
            for index, (left, right) in enumerate(finite_initial_keys):
                if (left, right) in grouped_initial_keys:
                    continue
                direction = initial_branch_directions[(1, left, right)]
                direction_norm = max(float(np.linalg.norm(direction)), 1e-9)
                group = [(left, right)]
                for other_left, other_right in finite_initial_keys[index + 1:]:
                    other = initial_branch_directions[
                        (1, other_left, other_right)
                    ]
                    cosine = float(np.dot(direction, other) / max(
                        direction_norm * float(np.linalg.norm(other)), 1e-9
                    ))
                    if cosine >= parallel_threshold:
                        group.append((other_left, other_right))
                if len(group) > 1:
                    parallel_initial_branch_groups += 1
                    grouped_initial_keys.update(group)
            if not np.isfinite(pair_costs).any():
                start_state = states[0][int(np.argmin(initial_costs))]
                start_node = int(np.argmin(np.linalg.norm(
                    self._topology_node_points - start_state[None, :], axis=1
                )))
                start_point = self._topology_node_points[start_node]
                alternatives: list[tuple[float, dict[str, Any]]] = []
                for neighbour, _ in self._topology_adjacency.get(start_node, []):
                    endpoint = self._topology_node_points[neighbour]
                    vector = endpoint - start_point
                    cosine = float(np.dot(stable_initial_direction, vector) / max(
                        float(np.linalg.norm(vector)), 1e-9
                    ))
                    edge_id = self._topology_segment_edges.get(
                        (start_node, neighbour)
                    )
                    if edge_id:
                        alternatives.append((
                            -cosine,
                            {
                                "edge_id": edge_id,
                                "points": [
                                    [round(float(value), 3) for value in start_point],
                                    [round(float(value), 3) for value in endpoint],
                                ],
                                "direction_cosine": round(cosine, 5),
                            },
                        ))
                alternatives.sort(key=lambda item: item[0])
                return None, {
                    "reason": "corridor_graph_disconnected",
                    "failed_layer": 1,
                    "edge_rejections": edge_rejections,
                    "_confirmed_prefix": [
                        [float(start_point[0]), float(start_point[1])]
                    ],
                    "_confirmed_edge_ids": [],
                    "_confirmed_fraction_end": 0.0,
                    "_uncertainty_marker": [
                        round(float(start_point[0]), 3),
                        round(float(start_point[1]), 3),
                    ],
                    "_competing_next_edges": [
                        item[1] for item in alternatives[:2]
                    ],
                }

            parents: list[np.ndarray] = []
            for layer in range(2, len(states)):
                next_costs = np.full(
                    (len(states[layer - 1]), len(states[layer])),
                    float("inf"),
                    dtype=np.float64,
                )
                parent = np.full(next_costs.shape, -1, dtype=np.int32)
                for left in range(len(states[layer - 1])):
                    for right in range(len(states[layer])):
                        transition = edge(layer, left, right)
                        if not math.isfinite(transition):
                            continue
                        right_emission = emission(layer, right)
                        for grand in range(len(states[layer - 2])):
                            accumulated = float(pair_costs[grand, left])
                            if not math.isfinite(accumulated):
                                continue
                            value = (
                                accumulated
                                + transition
                                + second_order(layer, grand, left, right)
                                + right_emission
                            )
                            if value < next_costs[left, right]:
                                next_costs[left, right] = value
                                parent[left, right] = grand
                if not np.isfinite(next_costs).any():
                    return None, {
                        "reason": "corridor_graph_disconnected",
                        "failed_layer": layer,
                        "edge_rejections": edge_rejections,
                        **confirmed_prefix(
                            pair_costs, parents, layer - 1
                        ),
                    }
                parents.append(parent)
                pair_costs = next_costs

            flat_ranking = np.argsort(pair_costs, axis=None)
            terminal_candidates: list[
                tuple[tuple[float, float, float, float], list[int], int]
            ] = []
            # Keep every finite terminal pair. The lowest local-cost terminal
            # is not necessarily the only history with the correct global
            # signed turn sequence, especially around parallel aisles.
            for ranked_flat in flat_ranking:
                ranked_flat = int(ranked_flat)
                if not math.isfinite(float(pair_costs.reshape(-1)[ranked_flat])):
                    continue
                left, right = np.unravel_index(
                    ranked_flat, pair_costs.shape
                )
                candidate_indices = [int(left), int(right)]
                valid_backtrack = True
                for parent in reversed(parents):
                    grand = int(parent[
                        candidate_indices[0], candidate_indices[1]
                    ])
                    if grand < 0:
                        valid_backtrack = False
                        break
                    candidate_indices.insert(0, grand)
                if not valid_backtrack:
                    continue
                candidate_path = np.asarray([
                    states[layer][candidate_indices[layer]]
                    for layer in range(len(states))
                ])
                topology = _turn_topology_metrics(
                    observed, candidate_path, samples=96
                )
                terminal_candidates.append((
                    (
                        0.0
                        if bool(topology["event_sequence_preserved"])
                        else 1.0,
                        float(topology["sign_mismatch_ratio"]),
                        float(topology["mean_abs_turn_error_degrees"]),
                        float(pair_costs.reshape(-1)[ranked_flat]),
                    ),
                    candidate_indices,
                    ranked_flat,
                ))
            if not terminal_candidates:
                return None, {"reason": "viterbi_backtrack_failed"}
            # The matrix DP above keeps one parent per (previous,current)
            # pair. Topologically different histories can therefore collapse
            # before their later turn events disambiguate them. Keep a small
            # diverse history beam for every terminal pair and rank those
            # complete histories with the signed event sequence.
            history_beam: dict[
                tuple[int, int], list[tuple[float, list[int]]]
            ] = {}
            for left in range(len(states[0])):
                for right in range(len(states[1])):
                    value = float(
                        initial_costs[left]
                        + edge(1, left, right)
                        + emission(1, right)
                    )
                    if math.isfinite(value):
                        history_beam[(left, right)] = [
                            (value, [left, right])
                        ]
            for layer in range(2, len(states)):
                next_beam: dict[
                    tuple[int, int], list[tuple[float, list[int]]]
                ] = {}
                for (grand, left), histories in history_beam.items():
                    for right in range(len(states[layer])):
                        transition = edge(layer, left, right)
                        if not math.isfinite(transition):
                            continue
                        turn_cost = second_order(
                            layer, grand, left, right
                        )
                        if not math.isfinite(turn_cost):
                            continue
                        suffix = transition + turn_cost + emission(layer, right)
                        key = (left, right)
                        bucket = next_beam.setdefault(key, [])
                        for accumulated, history in histories:
                            bucket.append((
                                float(accumulated + suffix),
                                history + [right],
                            ))
                expected_prefix_signs = tuple(
                    1 if float(angle) > 0.0 else -1
                    for fraction, angle in source_turn_events
                    if float(fraction) <= float(fractions[layer]) + 0.02
                )
                signature_cache: dict[tuple[int, ...], tuple[int, ...]] = {}

                def event_signature(
                    history: list[int],
                ) -> tuple[int, ...]:
                    history_key = tuple(history)
                    cached = signature_cache.get(history_key)
                    if cached is not None:
                        return cached
                    anchor_path = np.asarray([
                        states[path_layer][history[path_layer]]
                        for path_layer in range(len(history))
                    ], dtype=np.float64)
                    if len(anchor_path) < 3:
                        return ()
                    vectors = np.diff(anchor_path, axis=0)
                    signature_values: list[int] = []
                    for before, after in zip(vectors, vectors[1:]):
                        angle = math.degrees(math.atan2(
                            float(
                                before[0] * after[1]
                                - before[1] * after[0]
                            ),
                            float(np.dot(before, after)),
                        ))
                        if abs(angle) >= TURN_TOPOLOGY_MIN_DEGREES:
                            signature_values.append(
                                1 if angle > 0.0 else -1
                            )
                    signature = tuple(signature_values)
                    signature_cache[history_key] = signature
                    return signature

                for key, histories in next_beam.items():
                    histories.sort(key=lambda item: item[0])
                    histories = histories[:96]
                    best_by_events: dict[
                        tuple[int, ...], tuple[float, list[int]]
                    ] = {}
                    for item in histories:
                        signature = event_signature(item[1])
                        current = best_by_events.get(signature)
                        if current is None or item[0] < current[0]:
                            best_by_events[signature] = item

                    def event_rank(
                        item: tuple[float, list[int]],
                    ) -> tuple[int, int, float]:
                        signature = event_signature(item[1])
                        compared = min(
                            len(signature), len(expected_prefix_signs)
                        )
                        sign_errors = sum(
                            signature[index]
                            != expected_prefix_signs[index]
                            for index in range(compared)
                        )
                        return (
                            sign_errors,
                            abs(len(signature) - len(expected_prefix_signs)),
                            float(item[0]),
                        )

                    diverse = sorted(
                        best_by_events.values(), key=event_rank
                    )
                    next_beam[key] = diverse[:16]
                history_beam = next_beam
                if not history_beam:
                    break
            beam_terminal_histories: list[
                tuple[tuple[float, float, float, float], list[int], float]
            ] = []
            for histories in history_beam.values():
                for objective, history in histories:
                    candidate_path = np.asarray([
                        states[layer][history[layer]]
                        for layer in range(len(states))
                    ])
                    topology = _turn_topology_metrics(
                        observed, candidate_path, samples=96
                    )
                    beam_terminal_histories.append((
                        (
                            0.0
                            if bool(topology["event_sequence_preserved"])
                            else 1.0,
                            float(topology["sign_mismatch_ratio"]),
                            float(topology["mean_abs_turn_error_degrees"]),
                            float(objective),
                        ),
                        history,
                        float(objective),
                    ))
            matrix_best = min(
                terminal_candidates, key=lambda item: item[0]
            )
            beam_best = (
                min(beam_terminal_histories, key=lambda item: item[0])
                if beam_terminal_histories else None
            )
            if beam_best is not None and beam_best[0] < matrix_best[0]:
                _, indices, selected_objective = beam_best
                selected_rank = -1
            else:
                _, indices, selected_flat = matrix_best
                ranking = flat_ranking
                final_costs = pair_costs.reshape(-1)
                selected_objective = float(final_costs[selected_flat])
                selected_rank = int(
                    np.where(flat_ranking == selected_flat)[0][0]
                )
            ranking = flat_ranking
            final_costs = pair_costs.reshape(-1)
            terminal_candidates_evaluated = (
                len(terminal_candidates) + len(beam_terminal_histories)
            )
        if len(states) == 1:
            selected_objective = float(final_costs[ranking[0]])
            selected_rank = 0
            terminal_candidates_evaluated = 1
        path = np.asarray([states[layer][indices[layer]] for layer in range(len(states))])
        terminal_paths: list[dict[str, Any]] = []
        if len(states) > 1:
            for _, candidate_indices, candidate_flat in terminal_candidates:
                candidate_path = np.asarray([
                    states[layer][candidate_indices[layer]]
                    for layer in range(len(states))
                ])
                candidate_routes: dict[int, np.ndarray] = {}
                for layer in range(1, len(candidate_indices)):
                    route = edge_route_cache.get((
                        layer,
                        int(candidate_indices[layer - 1]),
                        int(candidate_indices[layer]),
                    ))
                    if route is not None:
                        candidate_routes[layer - 1] = route
                terminal_paths.append({
                    "path": candidate_path,
                    "edge_routes": candidate_routes,
                    "objective": float(final_costs[candidate_flat]),
                    "cost_rank": int(
                        np.where(ranking == candidate_flat)[0][0]
                    ),
                })
            existing_histories = {
                tuple(
                    int(np.argmin(np.linalg.norm(
                        np.asarray(states[layer], dtype=np.float64)
                        - point[None, :],
                        axis=1,
                    )))
                    for layer, point in enumerate(item["path"])
                )
                for item in terminal_paths
            }
            for _, candidate_indices, objective in beam_terminal_histories:
                signature = tuple(candidate_indices)
                if signature in existing_histories:
                    continue
                candidate_path = np.asarray([
                    states[layer][candidate_indices[layer]]
                    for layer in range(len(states))
                ])
                candidate_routes: dict[int, np.ndarray] = {}
                for layer in range(1, len(candidate_indices)):
                    route = edge_route_cache.get((
                        layer,
                        int(candidate_indices[layer - 1]),
                        int(candidate_indices[layer]),
                    ))
                    if route is not None:
                        candidate_routes[layer - 1] = route
                terminal_paths.append({
                    "path": candidate_path,
                    "edge_routes": candidate_routes,
                    "objective": float(objective),
                    "cost_rank": -1,
                })
        selected_edge_routes: dict[int, np.ndarray] = {}
        for layer in range(1, len(indices)):
            route = edge_route_cache.get(
                (layer, int(indices[layer - 1]), int(indices[layer]))
            )
            if route is not None:
                selected_edge_routes[layer - 1] = route
        routed_edge_count = len(selected_edge_routes)
        margin = (
            float(final_costs[ranking[1]] - final_costs[ranking[0]])
            if len(ranking) > 1 and math.isfinite(float(final_costs[ranking[1]]))
            else None
        )
        return path, {
            "reason": None,
            "objective": round(selected_objective, 6),
            "viterbi_cost_rank": selected_rank,
            "terminal_candidates_evaluated": terminal_candidates_evaluated,
            "margin": round(margin, 6) if margin is not None else None,
            "order": 2,
            "anchors": len(fractions),
            "states_min": min(len(layer) for layer in states),
            "states_max": max(len(layer) for layer in states),
            "implicit_edges_evaluated": len(edge_cache),
            "edge_rejections": edge_rejections,
            "parallel_initial_branch_groups":
                parallel_initial_branch_groups,
            "initial_direction_policy": {
                "origin": "snapped_graph_start",
                "lookahead_meters":
                    GRAPH_MAP_MATCH_INITIAL_DIRECTION_LOOKAHEAD_METERS,
                "soft_penalty_degrees":
                    GRAPH_MAP_MATCH_SOFT_INITIAL_DIRECTION_DEGREES,
                "hard_rejection_degrees":
                    GRAPH_MAP_MATCH_HARD_INITIAL_DIRECTION_DEGREES,
                "parallel_branch_degrees":
                    GRAPH_MAP_MATCH_PARALLEL_BRANCH_DEGREES,
            },
            "component_routed_edges": routed_edge_count,
            "component_edges_enabled": allow_component_edges,
            "required_component": required_component,
            "_selected_edge_routes": selected_edge_routes,
            "_terminal_paths": terminal_paths,
        }

    @staticmethod
    def _viterbi_prefix_candidate(partial: dict[str, Any]) -> dict[str, Any]:
        trajectory = partial.get("_confirmed_prefix") or []
        edge_ids = partial.get("_confirmed_edge_ids") or []
        if len(trajectory) < 2 or not edge_ids:
            marker = partial.get("_uncertainty_marker")
            alternatives = partial.get("_competing_next_edges") or []
            if marker and alternatives:
                return {
                    "available": True,
                    "reason": "viterbi_correspondence_lost_at_start",
                    "trajectory": [marker],
                    "segments": [],
                    "confirmed_segments": 0,
                    "uncertain_segments": 0,
                    "confirmed_source_fraction": 0.0,
                    "uncertainty_source_fraction_start": 0.0,
                    "uncertainty_marker": marker,
                    "competing_next_edges": alternatives[:2],
                    "policy": "stop_at_last_confirmed_viterbi_node_v2",
                }
            return {
                "available": False,
                "reason": "no_confirmed_viterbi_edge",
                "trajectory": [],
                "segments": [],
                "uncertainty_marker": partial.get("_uncertainty_marker"),
                "competing_next_edges": [],
                "policy": "stop_at_last_confirmed_viterbi_node_v2",
            }
        return {
            "available": True,
            "reason": "viterbi_correspondence_lost",
            "trajectory": trajectory,
            "segments": [{
                "start_index": 0,
                "end_index": len(trajectory) - 1,
                "source_fraction_start": 0.0,
                "source_fraction_end": partial.get(
                    "_confirmed_fraction_end", 0.0
                ),
                "status": "confirmed",
                "edge_ids": edge_ids,
            }],
            "confirmed_segments": 1,
            "uncertain_segments": 0,
            "confirmed_source_fraction": partial.get(
                "_confirmed_fraction_end", 0.0
            ),
            "uncertainty_source_fraction_start": partial.get(
                "_confirmed_fraction_end", 0.0
            ),
            "uncertainty_marker": partial.get("_uncertainty_marker"),
            "competing_next_edges": (
                partial.get("_competing_next_edges") or []
            )[:2],
            "policy": "stop_at_last_confirmed_viterbi_node_v2",
        }

    def _multilevel_viterbi_map_match(
        self,
        observation: np.ndarray,
        baseline: np.ndarray,
        *,
        allow_global_recovery: bool = False,
    ) -> tuple[Optional[np.ndarray], dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "attempted": True,
            "accepted": False,
            "method": "directed_edge_events_then_anchor_fallback_v1",
            "corridor_graph_nodes": int(len(self._corridor_nodes)),
        }
        directed_route, directed_diag = self._directed_edge_event_match(
            observation
        )
        diagnostics["directed_edge_search"] = directed_diag
        if directed_route is not None:
            if (
                not self._point_occupied(observation[0])
                and not self._segment_collides(
                    observation[0], directed_route[0]
                )
            ):
                directed_route = directed_route.copy()
                directed_route[0] = observation[0]
            diagnostics.update({
                "reason": None,
                "method": "directed_edge_event_beam_v1",
                "fine": {
                    "reason": None,
                    "objective": directed_diag["objective"],
                    "anchors": directed_diag["anchors"],
                    "selected_reconstructed_turn_topology":
                        directed_diag[
                            "selected_reconstructed_turn_topology"
                        ],
                },
                "post_repair_segments": max(
                    0, len(directed_route) - 1
                ),
                "graph_edges": max(0, len(directed_route) - 1),
                "graph_routed_edges": max(0, len(directed_route) - 1),
                "graph_reconstruction": "directed_connected_edges",
                "start_anchor_preserved": bool(
                    np.linalg.norm(directed_route[0] - observation[0]) < 1e-6
                ),
            })
            return directed_route, diagnostics
        if os.getenv("TRACKAI_DIRECTED_EDGE_ONLY", "").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            diagnostics["reason"] = directed_diag.get("reason")
            return None, diagnostics
        coarse_fractions = self._adaptive_anchor_fractions(observation, maximum=14)
        coarse = None
        coarse_diag: dict[str, Any] = {"reason": "not_attempted"}
        coarse_attempts: list[dict[str, Any]] = []
        best_partial: dict[str, Any] = {}
        # Always prove that the local corridor cannot explain the observation
        # before widening into plant-scale branch search.
        search_levels = (
            (
                (6.0, 16, False),
                (12.0, 18, False),
                (12.0, 10, True),
                (24.0, 10, True),
                (40.0, 10, True),
            )
            if allow_global_recovery
            else ((6.0, 16, False), (12.0, 18, False))
        )
        local_loss_proven = False
        topology_blockers = {
            "turn_sign_sequence_mismatch",
            "branch_commitment_backtrack",
            "unobserved_internal_turn",
        }
        for radius_meters, candidate_limit, component_edges in search_levels:
            if radius_meters > 12.0 and not local_loss_proven:
                coarse_attempts.append({
                    "radius_meters": radius_meters,
                    "candidate_limit": candidate_limit,
                    "recovery_stage": "global_radius_blocked",
                    "reason": "local_correspondence_loss_not_proven",
                })
                continue
            coarse, coarse_diag = self._viterbi_level(
                observation,
                baseline,
                coarse_fractions,
                radius_meters=radius_meters,
                candidate_limit=candidate_limit,
                allow_component_edges=component_edges,
            )
            coarse_attempts.append({
                "radius_meters": radius_meters,
                "candidate_limit": candidate_limit,
                "recovery_stage": (
                    "local_direct_edges" if not component_edges and radius_meters <= 6.0
                    else "expanded_local_direct_edges" if not component_edges
                    else "component_edges_after_direct_failure"
                    if radius_meters <= 12.0
                    else "global_after_local_failure"
                ),
                **{
                    key: value for key, value in coarse_diag.items()
                    if not key.startswith("_")
                },
            })
            if (
                coarse_diag.get("_confirmed_prefix")
                and float(coarse_diag.get("_confirmed_fraction_end", 0.0))
                > float(best_partial.get("_confirmed_fraction_end", -1.0))
            ):
                best_partial = {
                    key: value for key, value in coarse_diag.items()
                    if key.startswith("_")
                }
            if coarse is not None:
                break
            if radius_meters == 12.0:
                rejection_keys = set(
                    (coarse_diag.get("edge_rejections") or {}).keys()
                )
                local_loss_proven = not bool(
                    rejection_keys & topology_blockers
                )
        coarse_diag.pop("_selected_edge_routes", None)
        coarse_diag.pop("_terminal_paths", None)
        diagnostics["coarse"] = coarse_diag
        diagnostics["coarse_attempts"] = coarse_attempts
        diagnostics["global_recovery_enabled"] = allow_global_recovery
        diagnostics["local_correspondence_loss_proven"] = local_loss_proven
        if coarse is None:
            diagnostics["reason"] = coarse_diag.get("reason")
            diagnostics["graph_first_candidate"] = (
                self._viterbi_prefix_candidate(best_partial)
            )
            return None, diagnostics
        fine_fractions = self._adaptive_anchor_fractions(observation, maximum=36)
        fine_guide = _resample_polyline(coarse, _trajectory_fractions(baseline))
        fine, fine_diag = self._viterbi_level(
            observation, fine_guide, fine_fractions,
            radius_meters=4.0, candidate_limit=14,
            allow_component_edges=False,
        )
        fine_attempts = [{
            "radius_meters": 4.0,
            "component_edges_enabled": False,
            **{
                key: value for key, value in fine_diag.items()
                if not key.startswith("_")
            },
        }]
        if fine is None and allow_global_recovery:
            fine, fine_diag = self._viterbi_level(
                observation, fine_guide, fine_fractions,
                radius_meters=4.0, candidate_limit=14,
                allow_component_edges=True,
            )
            fine_attempts.append({
                "radius_meters": 4.0,
                "component_edges_enabled": True,
                **{
                    key: value for key, value in fine_diag.items()
                    if not key.startswith("_")
                },
            })
        fine_partial = {
            key: value for key, value in fine_diag.items()
            if key.startswith("_")
        }
        selected_fine_edge_routes = fine_diag.pop("_selected_edge_routes", {})
        terminal_paths = fine_diag.pop("_terminal_paths", [])
        diagnostics["fine"] = fine_diag
        diagnostics["fine_attempts"] = fine_attempts
        if fine is None:
            diagnostics["reason"] = fine_diag.get("reason")
            diagnostics["graph_first_candidate"] = (
                self._viterbi_prefix_candidate(fine_partial or best_partial)
            )
            return None, diagnostics

        # Rank complete reconstructed candidates. Anchor-only topology is not
        # sufficient because a routed edge may add turns that are absent from
        # its endpoint chord.
        reconstructed_candidates: list[
            tuple[tuple[float, float, float, float], np.ndarray, dict[int, np.ndarray], dict[str, Any]]
        ] = []
        for terminal in terminal_paths:
            terminal_path = np.asarray(terminal["path"], dtype=np.float64)
            terminal_routes = terminal["edge_routes"]
            stitched_candidate: list[np.ndarray] = []
            for edge_index, (left, right) in enumerate(
                zip(terminal_path[:-1], terminal_path[1:])
            ):
                route = terminal_routes.get(edge_index)
                if route is None:
                    route = np.vstack((left, right))
                if (
                    stitched_candidate
                    and np.linalg.norm(stitched_candidate[-1] - route[0]) < 1e-6
                ):
                    route = route[1:]
                stitched_candidate.extend(route)
            reconstructed = np.asarray(stitched_candidate, dtype=np.float64)
            if len(reconstructed) < 2 or self._collision_runs(reconstructed):
                continue
            topology = _turn_topology_metrics(
                observation, reconstructed
            )
            reconstructed_candidates.append((
                (
                    0.0 if bool(topology["event_sequence_preserved"]) else 1.0,
                    float(topology["sign_mismatch_ratio"]),
                    float(topology["mean_abs_turn_error_degrees"]),
                    float(terminal["objective"]),
                ),
                terminal_path,
                terminal_routes,
                topology,
            ))
        if reconstructed_candidates:
            reconstructed_candidates.sort(key=lambda item: item[0])
            (
                _,
                fine,
                selected_fine_edge_routes,
                selected_reconstructed_topology,
            ) = reconstructed_candidates[0]
            diagnostics["fine"]["reconstructed_candidates_evaluated"] = len(
                reconstructed_candidates
            )
            diagnostics["fine"]["selected_reconstructed_turn_topology"] = (
                selected_reconstructed_topology
            )
            if len(reconstructed_candidates) >= 2:
                def terminal_edge_sequence(
                    routes: dict[int, np.ndarray],
                ) -> list[str]:
                    raw = [
                        edge_id
                        for index in sorted(routes)
                        for edge_id in self._topology_edge_sequence(routes[index])
                    ]
                    return [
                        edge_id for index, edge_id in enumerate(raw)
                        if index == 0 or edge_id != raw[index - 1]
                    ]

                selected_path = np.asarray(
                    reconstructed_candidates[0][1], dtype=np.float64
                )
                selected_routes = reconstructed_candidates[0][2]
                selected_edges = terminal_edge_sequence(selected_routes)
                competing_index = next((
                    index
                    for index, candidate in enumerate(
                        reconstructed_candidates[1:], start=1
                    )
                    if terminal_edge_sequence(candidate[2]) != selected_edges
                ), None)
                competing_path = (
                    np.asarray(
                        reconstructed_candidates[competing_index][1],
                        dtype=np.float64,
                    )
                    if competing_index is not None else None
                )
                competing_edges = (
                    terminal_edge_sequence(
                        reconstructed_candidates[competing_index][2]
                    )
                    if competing_index is not None else []
                )
            else:
                competing_path = None
            if competing_path is not None:
                compared = min(len(selected_path), len(competing_path))
                divergence_index = next((
                    index for index in range(compared)
                    if float(np.linalg.norm(
                        selected_path[index] - competing_path[index]
                    )) > max(
                        1.0,
                        0.75 / max(self.config.meters_per_pixel, 1e-9),
                    )
                ), None)
                if divergence_index is not None:
                    start = max(0, divergence_index - 1)
                    stop = min(
                        compared,
                        divergence_index + 5,
                    )
                    diagnostics["competing_branches"] = {
                        "ambiguous": True,
                        "divergence_anchor": int(divergence_index),
                        "divergence_point": [
                            round(float(value), 3)
                            for value in selected_path[
                                max(0, divergence_index - 1)
                            ]
                        ],
                        "selected": [
                            [round(float(value), 3) for value in point]
                            for point in selected_path[start:stop]
                        ],
                        "alternative": [
                            [round(float(value), 3) for value in point]
                            for point in competing_path[start:stop]
                        ],
                        "selected_turn_topology": (
                            reconstructed_candidates[0][3]
                        ),
                        "alternative_turn_topology": (
                            reconstructed_candidates[competing_index][3]
                        ),
                        "selected_edge_ids": selected_edges,
                        "alternative_edge_ids": competing_edges,
                    }

            selected_cost_per_anchor = (
                float(diagnostics["fine"].get("objective", float("inf")))
                / max(int(diagnostics["fine"].get("anchors", 0)), 1)
            )
            selected_sequence_preserved = bool(
                selected_reconstructed_topology["event_sequence_preserved"]
            )
            if (
                not selected_sequence_preserved
                or selected_cost_per_anchor
                > GRAPH_MAP_MATCH_MAX_VITERBI_COST_PER_ANCHOR
            ):
                # A connected path is not a successful graph match. Find the
                # longest prefix whose complete reconstructed turn sequence is
                # still compatible with R3 and publish only that prefix.
                confirmed_route: list[np.ndarray] = []
                confirmed_edge_ids: list[str] = []
                confirmed_fraction = 0.0
                for edge_index in range(max(0, len(fine) - 1)):
                    route = selected_fine_edge_routes.get(edge_index)
                    if route is None:
                        route = np.vstack((fine[edge_index], fine[edge_index + 1]))
                    candidate_route = list(confirmed_route)
                    candidate_piece = route
                    if (
                        candidate_route
                        and np.linalg.norm(candidate_route[-1] - route[0]) < 1e-6
                    ):
                        candidate_piece = route[1:]
                    candidate_route.extend(candidate_piece)
                    if len(candidate_route) < 2:
                        continue
                    fraction_end = float(fine_fractions[edge_index + 1])
                    source_prefix = _resample_polyline(
                        observation,
                        np.linspace(
                            0.0,
                            fraction_end,
                            max(24, 6 * (edge_index + 2)),
                        ),
                    )
                    prefix_metrics = _turn_topology_metrics(
                        source_prefix,
                        np.asarray(candidate_route, dtype=np.float64),
                    )
                    if not bool(prefix_metrics["event_sequence_preserved"]):
                        break
                    confirmed_route = candidate_route
                    for edge_id in self._topology_edge_sequence(route):
                        if (
                            not confirmed_edge_ids
                            or confirmed_edge_ids[-1] != edge_id
                        ):
                            confirmed_edge_ids.append(edge_id)
                    confirmed_fraction = fraction_end
                partial: dict[str, Any] = {}
                if len(confirmed_route) >= 2 and confirmed_edge_ids:
                    marker = np.asarray(confirmed_route[-1], dtype=np.float64)
                    partial = {
                        "_confirmed_prefix": [
                            [float(point[0]), float(point[1])]
                            for point in confirmed_route
                        ],
                        "_confirmed_edge_ids": confirmed_edge_ids,
                        "_confirmed_fraction_end": round(
                            confirmed_fraction, 8
                        ),
                        "_uncertainty_marker": [
                            round(float(marker[0]), 3),
                            round(float(marker[1]), 3),
                        ],
                        "_competing_next_edges": [],
                    }
                diagnostics["reason"] = (
                    "turn_event_sequence_mismatch"
                    if not selected_sequence_preserved
                    else "viterbi_cost_exceeded"
                )
                diagnostics["graph_first_candidate"] = (
                    self._viterbi_prefix_candidate(partial)
                )
                diagnostics["strict_graph_gate"] = {
                    "event_sequence_preserved":
                        selected_sequence_preserved,
                    "viterbi_cost_per_anchor": round(
                        selected_cost_per_anchor, 6
                    ),
                    "maximum_viterbi_cost_per_anchor":
                        GRAPH_MAP_MATCH_MAX_VITERBI_COST_PER_ANCHOR,
                }
                return None, diagnostics

        # The first map state is represented by a grid-cell centre, while the
        # operator's start marker is a sub-cell plan coordinate. Preserve that
        # exact authoritative anchor whenever the short connector is safe.
        # Otherwise a perfectly valid graph route would fail the later
        # centimetre-level publication lock solely due to grid quantisation.
        start_anchor_preserved = bool(
            not self._point_occupied(observation[0])
            and not self._segment_collides(observation[0], fine[0])
        )
        if start_anchor_preserved:
            fine = fine.copy()
            fine[0] = observation[0]

        # The HMM has selected a globally coherent sequence of corridor
        # anchors.  Reconstruct the final polyline along the actual connected
        # free-space graph.  The previous implementation interpolated anchor
        # corrections back onto every R3 sample and then called the bounded
        # local A*.  That reintroduced obstacle-crossing chords and defeated
        # the purpose of global graph matching.
        stitched: list[np.ndarray] = []
        routed_edges = 0
        graph_route_margins = tuple(
            max(2, int(math.ceil(meters / max(self.cell_meters, 1e-9))))
            for meters in (6.0, 12.0)
        )
        routed_edge_diagnostics: list[dict[str, Any]] = []
        for edge_index, (start, end) in enumerate(zip(fine[:-1], fine[1:])):
            selected_route = selected_fine_edge_routes.get(edge_index)
            if selected_route is not None:
                route = selected_route
                routed_edges += 1
            elif self._segment_collides(start, end):
                observed_edge = _resample_polyline(
                    observation,
                    np.asarray(
                        [fine_fractions[edge_index], fine_fractions[edge_index + 1]],
                        dtype=np.float64,
                    ),
                )
                route = None
                selected_margin_meters = None
                for margin_meters, margin_cells in zip(
                    (6.0, 12.0), graph_route_margins
                ):
                    candidate_route = self._astar(
                        start,
                        end,
                        observed_edge,
                        _search_margin_cells=margin_cells,
                    )
                    if candidate_route is None:
                        continue
                    # Viterbi selected this edge using the observed direction
                    # and turn sequence. Reject an A* reconstruction that
                    # silently changes its homotopy or invents a remote aisle.
                    detour = self._detour_metrics(
                        candidate_route, start, end, observed_edge
                    )
                    edge_topology = _turn_topology_metrics(
                        observed_edge, candidate_route, samples=48
                    )
                    if (
                        bool(detour["spike"])
                        or not bool(edge_topology["event_sequence_preserved"])
                    ):
                        continue
                    route = candidate_route
                    selected_margin_meters = margin_meters
                    routed_edge_diagnostics.append({
                        "edge": int(edge_index),
                        "search_margin_meters": margin_meters,
                        "route_observed_length_ratio": round(
                            float(detour["route_observed_length_ratio"]), 6
                        ),
                        "turn_topology": edge_topology,
                    })
                    break
                if route is None:
                    diagnostics.update({
                        "reason": "graph_edge_topology_preserving_route_not_found",
                        "failed_edge": int(edge_index),
                        "allowed_search_margins_meters": [6.0, 12.0],
                    })
                    return None, diagnostics
                routed_edges += 1
            else:
                route = np.vstack((start, end))
            if stitched and np.linalg.norm(stitched[-1] - route[0]) < 1e-6:
                route = route[1:]
            stitched.extend(route)

        repaired = np.asarray(stitched, dtype=np.float64)
        if len(repaired) < 2 or self._collision_runs(repaired):
            diagnostics["reason"] = "graph_reconstruction_not_safe"
            return None, diagnostics
        diagnostics.update({
            "reason": None,
            "post_repair_segments": int(routed_edges),
            "graph_edges": int(max(0, len(fine) - 1)),
            "graph_routed_edges": int(routed_edges),
            "graph_reconstruction": "bounded_observation_guided_astar",
            "routed_edge_diagnostics": routed_edge_diagnostics,
            "start_anchor_preserved": start_anchor_preserved,
        })
        return repaired, diagnostics

    def align(
        self,
        trajectory: Any,
        reference_point: Optional[dict[str, Any]],
        direction_point: Optional[dict[str, Any]],
        *,
        timestamps: Any = None,
        coordinate_convention: str = "x_forward_y_left_z_up",
        scale_candidates: Optional[Iterable[float]] = None,
        yaw_offsets_degrees: Sequence[float] = STANDARD_YAW_OFFSETS_DEGREES,
        allow_safe_shape_fallback: bool = False,
        allow_low_net_progress: bool = False,
        observation_policy: str = "authoritative",
        loop_closure_verified: bool = False,
        allow_independent_corridor_recovery: bool = False,
    ) -> dict[str, Any]:
        observation_policy = str(observation_policy or "authoritative").lower()
        if observation_policy not in {"authoritative", "independent"}:
            observation_policy = "authoritative"
        raw = _normalise_points(trajectory)
        # Authoritative R3 observations use the corridor graph in production.
        # It remains possible to disable the feature operationally without a
        # deploy. Independent/monocular observations still need an explicit
        # opt-in because their scale and topology are less constrained.
        graph_matching_setting = os.getenv(
            "TRACKAI_ENABLE_GRAPH_MAP_MATCHING", "1"
        ).strip().lower()
        graph_matching_configured = graph_matching_setting not in {
            "0", "false", "no", "off",
        }
        topology_recovery_enabled = bool(
            graph_matching_configured
            and (
                observation_policy == "authoritative"
                or allow_independent_corridor_recovery
            )
        )
        base_diagnostics: dict[str, Any] = {
            "engine": "floorplan_constraint_engine_v10_graph_map_matching",
            "map_id": self.config.map_id,
            "plan_width": self.config.width,
            "plan_height": self.config.height,
            "meters_per_pixel": self.config.meters_per_pixel,
            "person_radius_meters": self.config.person_radius_meters,
            "point_count": int(len(raw)),
            "accepted": False,
            "observation_policy": observation_policy,
            "loop_closure_verified": bool(loop_closure_verified),
            "independent_corridor_recovery_enabled": bool(
                allow_independent_corridor_recovery and topology_recovery_enabled
            ),
            "topology_recovery_enabled": topology_recovery_enabled,
            "graph_map_matching_configured": graph_matching_configured,
            "support_mask_enabled": self._support_mask is not None,
            "support_coverage_ratio": (
                round(float(np.mean(self._support_mask)), 8)
                if self._support_mask is not None else None
            ),
            "walkable_bbox_cells": list(self.walkable_bbox),
            "annotation_bbox_cells": list(self.annotation_bbox),
        }
        if len(raw) < 2:
            return {"accepted": False, "trajectory": [], "diagnostics": {**base_diagnostics, "reason": "trajectory_too_short"}}
        if not reference_point or not direction_point:
            return {"accepted": False, "trajectory": [], "diagnostics": {**base_diagnostics, "reason": "missing_start_or_direction"}}

        requested_start = np.asarray([
            _finite_float(reference_point.get("x")) / 100.0 * self.config.width,
            _finite_float(reference_point.get("y")) / 100.0 * self.config.height,
        ])
        direction = np.asarray([
            _finite_float(direction_point.get("x")) / 100.0 * self.config.width,
            _finite_float(direction_point.get("y")) / 100.0 * self.config.height,
        ])
        if float(np.linalg.norm(direction - requested_start)) < max(self.config.width, self.config.height) * 0.004:
            return {"accepted": False, "trajectory": [], "diagnostics": {**base_diagnostics, "reason": "direction_too_short"}}
        start_cell = self._nearest_free(self._pixel_to_cell(requested_start))
        if start_cell is None:
            return {"accepted": False, "trajectory": [], "diagnostics": {**base_diagnostics, "reason": "no_walkable_start"}}
        nearest_walkable_start = self._cell_to_pixel(start_cell)
        start_snap_meters = (
            float(np.linalg.norm(nearest_walkable_start - requested_start))
            * self.config.meters_per_pixel
        )
        # The operator's start is ground truth.  Never move the entire route
        # to make that point fit an imperfect support mask.
        start = requested_start.copy()
        base_diagnostics.update({
            "requested_start_pixels": requested_start.tolist(),
            "nearest_walkable_start_pixels": nearest_walkable_start.tolist(),
            "start_snap_meters": round(start_snap_meters, 3),
            "start_anchor_locked": True,
        })

        relative = raw - raw[0]
        if coordinate_convention == "x_forward_y_left_z_up":
            relative[:, 1] *= -1.0
        if observation_policy == "authoritative":
            estimation_relative, macro_motion_diagnostics = (
                _stabilize_authoritative_map_observation(relative)
            )
        else:
            estimation_relative = relative
            macro_motion_diagnostics = {
                "method": None,
                "applied": False,
                "point_count": int(len(relative)),
            }
        base_diagnostics["macro_motion_stabilization"] = (
            macro_motion_diagnostics
        )
        observation_progress = _polyline_progress_metrics(relative)
        estimation_progress = _polyline_progress_metrics(estimation_relative)
        base_diagnostics["observation_progress"] = {
            key: round(float(observation_progress[key]), 6)
            for key in ("net_progress_ratio", "span_length_ratio", "tortuosity")
        }
        base_diagnostics["map_estimation_progress"] = {
            key: round(float(estimation_progress[key]), 6)
            for key in ("net_progress_ratio", "span_length_ratio", "tortuosity")
        }
        if (
            observation_policy == "independent"
            and not allow_low_net_progress
            and observation_progress["net_progress_ratio"]
            < INDEPENDENT_MIN_NET_PROGRESS_RATIO
        ):
            return {
                "accepted": False,
                "trajectory": [],
                "diagnostics": {
                    **base_diagnostics,
                    "reason": "insufficient_independent_net_progress",
                    "rejection_reasons": ["compressed_or_looping_independent_observation"],
                    "minimum_net_progress_ratio": INDEPENDENT_MIN_NET_PROGRESS_RATIO,
                },
            }
        desired_heading = math.atan2(
            float(direction[1] - requested_start[1]),
            float(direction[0] - requested_start[0]),
        )
        duration = self._motion_duration_seconds(
            timestamps, estimation_relative
        )
        candidates = (
            list(scale_candidates)
            if scale_candidates is not None
            else self._scale_candidates(estimation_relative, duration)
        )
        hypotheses: list[dict[str, Any]] = []
        # The supplied direction fixes rotation.  Yaw search used to rotate
        # the whole route away from the operator's anchor.
        effective_yaw_offsets = (0.0,)
        for scale in candidates:
            for yaw in effective_yaw_offsets:
                points = self._build_hypothesis(
                    estimation_relative,
                    start,
                    desired_heading,
                    float(scale),
                    float(yaw),
                )
                score, metrics = self._score_hypothesis(
                    points,
                    duration,
                    float(yaw),
                    observation_policy=observation_policy,
                )
                metric_plausible = self._metric_hypothesis_is_plausible(
                    float(metrics["speed_ratio"]),
                    duration=duration,
                    observation_policy=observation_policy,
                    loop_closure_verified=loop_closure_verified,
                )
                hypotheses.append({
                    "score": score,
                    "scale": float(scale),
                    "yaw": float(yaw),
                    "points": points,
                    "metric_plausible": metric_plausible,
                    **metrics,
                })
        if not hypotheses:
            return {"accepted": False, "trajectory": [], "diagnostics": {**base_diagnostics, "reason": "no_hypotheses"}}
        hypotheses.sort(key=lambda item: item["score"])
        metric_hypotheses = [item for item in hypotheses if item["metric_plausible"]]
        if not metric_hypotheses:
            finite_speeds = [
                float(item["speed_mps"])
                for item in hypotheses
                if math.isfinite(float(item["speed_mps"]))
            ]
            return {
                "accepted": False,
                "trajectory": [],
                "diagnostics": {
                    **base_diagnostics,
                    "reason": (
                        "metric_prior_unavailable"
                        if duration is None else "metric_prior_inconsistent"
                    ),
                    "rejection_reasons": ["implausible_metric_scale"],
                    "motion_duration_seconds": (
                        round(float(duration), 3) if duration is not None else None
                    ),
                    "candidate_speed_min_mps": (
                        round(min(finite_speeds), 3) if finite_speeds else None
                    ),
                    "candidate_speed_max_mps": (
                        round(max(finite_speeds), 3) if finite_speeds else None
                    ),
                },
            }

        # The plan is part of estimation, not a post-hoc accept/reject gate.
        # Repair a yaw-diverse beam and rank only after the walkable mask is
        # enforced so raw collision scores cannot freeze the wrong homotopy.
        feasible: list[dict[str, Any]] = []
        beam = self._select_diverse_beam(metric_hypotheses)
        if (
            topology_recovery_enabled
            and observation_policy == "authoritative"
            and duration is not None
        ):
            # A long R3 clip often contains stops. Prioritise graph hypotheses
            # near a conservative active walking speed before the generic
            # collision-ranked beam; raw collision is precisely what the
            # graph is meant to resolve.
            graph_priority = sorted(
                metric_hypotheses,
                key=lambda item: (
                    abs(math.log(
                        max(float(item["speed_ratio"]), 1e-9)
                        / GRAPH_MAP_MATCH_TARGET_ACTIVE_SPEED_RATIO
                    )),
                    float(item["score"]),
                ),
            )[:6]
            reordered: list[dict[str, Any]] = []
            seen_hypotheses: set[tuple[float, float]] = set()
            for item in graph_priority + beam:
                key = (
                    round(float(item["scale"]), 9),
                    round(float(item["yaw"]), 3),
                )
                if key in seen_hypotheses:
                    continue
                seen_hypotheses.add(key)
                reordered.append(item)
            beam = reordered
        attempted = 0
        route_failure_counts: dict[str, int] = {}
        topology_recovery_attempted = 0
        topology_recovery_accepted = 0
        topology_recovery_diagnostics: list[dict[str, Any]] = []
        segmented_recovery_attempted = 0
        segmented_recovery_accepted = 0
        segmented_recovery_diagnostics: list[dict[str, Any]] = []

        def record_route_failures(reasons: list[str]) -> None:
            for reason in reasons:
                route_failure_counts[reason] = route_failure_counts.get(reason, 0) + 1

        for hypothesis in beam:
            attempted += 1
            route_failures: list[str] = []
            repair_diagnostics: dict[str, Any] = {}
            repaired, rerouted_segments = self._repair_collisions(
                hypothesis["points"],
                failure_reasons=route_failures,
                allow_provisional_spikes=topology_recovery_enabled,
                repair_diagnostics=repair_diagnostics,
            )
            record_route_failures(route_failures)
            segmented_recovery: Optional[dict[str, Any]] = None
            if repaired is None:
                can_repair_in_segments = (
                    topology_recovery_enabled
                    and
                    (
                        observation_policy == "authoritative"
                        or allow_independent_corridor_recovery
                    )
                    and duration is not None
                    and duration >= 480.0
                    and segmented_recovery_attempted < 6
                )
                if can_repair_in_segments:
                    segmented_recovery_attempted += 1
                    repaired, segmented_recovery = (
                        self._repair_long_trajectory_in_time_segments(
                            hypothesis["points"], timestamps
                        )
                    )
                    segmented_recovery_diagnostics.append({
                        "scale": round(float(hypothesis["scale"]), 9),
                        "yaw_degrees": round(float(hypothesis["yaw"]), 3),
                        **segmented_recovery,
                    })
                    if repaired is not None:
                        segmented_recovery_accepted += 1
                        rerouted_segments = int(
                            segmented_recovery.get("rerouted_segments", 0)
                        )
                        route_failures = []
                if repaired is not None:
                    topology_recovery = None
                else:
                # Local A* is intentionally bounded so it cannot invent a
                # remote plant-wide detour.  When that bounded search is the
                # only failure, let the existing corridor-graph matcher try a
                # small number of authoritative hypotheses.  Previously the
                # matcher was reachable only *after* local repair succeeded,
                # leaving genuine ``local_search_exhausted`` cases with zero
                # topology-recovery attempts.
                    can_recover_exhausted_search = (
                        topology_recovery_enabled
                        and
                        (
                            observation_policy == "authoritative"
                            or allow_independent_corridor_recovery
                        )
                        and bool({
                            "local_search_exhausted",
                            "different_walkable_components",
                        }.intersection(route_failures))
                        and len(hypothesis["points"])
                        >= GRAPH_MAP_MATCH_MIN_OBSERVATIONS
                        and topology_recovery_attempted < 3
                    )
                    if not can_recover_exhausted_search:
                        continue
                    projected_baseline = hypothesis["points"].copy()
                    for point_index, point in enumerate(projected_baseline):
                        if not self._point_occupied(point):
                            continue
                        free_cell = self._nearest_free(self._pixel_to_cell(point))
                        if free_cell is None:
                            projected_baseline = None
                            break
                        projected_baseline[point_index] = self._cell_to_pixel(free_cell)
                    if projected_baseline is None:
                        continue
                    topology_recovery_attempted += 1
                    repaired, topology_recovery = self._multilevel_viterbi_map_match(
                        hypothesis["points"],
                        projected_baseline,
                        allow_global_recovery=(
                            observation_policy == "authoritative"
                            or allow_independent_corridor_recovery
                        ),
                    )
                    topology_recovery.update({
                        "phase": "local_search_exhaustion_recovery",
                        "production_enabled": True,
                        "scale": round(float(hypothesis["scale"]), 9),
                        "yaw_degrees": round(float(hypothesis["yaw"]), 3),
                    })
                    topology_recovery_diagnostics.append(topology_recovery)
                    if repaired is None:
                        record_route_failures(["corridor_graph_recovery_failed"])
                        continue
                    topology_recovery_accepted += 1
                    topology_recovery["accepted"] = True
                    topology_recovery["reason"] = None
                    rerouted_segments = int(
                        topology_recovery.get("post_repair_segments", 0)
                    )
            else:
                topology_recovery = None
            operator_graph_match_required = bool(
                topology_recovery_enabled
                and observation_policy == "authoritative"
                and self._topology_authored_node_count > 0
                and len(hypothesis["points"])
                >= GRAPH_MAP_MATCH_MIN_OBSERVATIONS
            )
            if (
                topology_recovery_enabled
                and (
                    int(repair_diagnostics.get("provisional_spike_count", 0)) > 0
                    or operator_graph_match_required
                )
                and len(hypothesis["points"]) >= GRAPH_MAP_MATCH_MIN_OBSERVATIONS
            ):
                topology_recovery_attempted += 1
                nonlinear, topology_recovery = self._multilevel_viterbi_map_match(
                    hypothesis["points"],
                    repaired,
                    allow_global_recovery=observation_policy == "authoritative",
                )
                topology_recovery.update({
                    "phase": (
                        "operator_graph_edge_sequence"
                        if operator_graph_match_required
                        else "preselection_spike_recovery"
                    ),
                    "operator_graph_mandatory": operator_graph_match_required,
                    "provisional_detours": repair_diagnostics.get("detours", []),
                    "production_enabled": True,
                    "scale": round(float(hypothesis["scale"]), 9),
                    "yaw_degrees": round(float(hypothesis["yaw"]), 3),
                })
                topology_recovery_diagnostics.append(topology_recovery)
                if nonlinear is None:
                    record_route_failures([
                        "detour_spike_rejected",
                        "corridor_graph_recovery_failed",
                    ])
                    continue
                topology_recovery_accepted += 1
                topology_recovery["accepted"] = True
                topology_recovery["reason"] = None
                repaired = nonlinear
                rerouted_segments += int(
                    topology_recovery.get("post_repair_segments", 0)
                )
            graph_matching_used = bool(
                isinstance(topology_recovery, dict)
                and topology_recovery.get("accepted")
            )
            corrected_metrics = self._path_metrics(repaired)
            # Obstacles are hard constraints. A second pass may repair a
            # raster-edge nick, but no residual collision is publishable.
            residual_collision_budget = 0.0
            residual_segment_budget_meters = 0.0
            if not graph_matching_used and (
                corrected_metrics["outside_ratio"] > 0.0 or (
                self._collision_runs(repaired)
                and (
                    corrected_metrics["collision_ratio"] > residual_collision_budget
                    or self._max_collision_run_meters(repaired)
                    > residual_segment_budget_meters
                )
                )
            ):
                route_failures = []
                repaired_again, extra_segments = self._repair_collisions(
                    repaired, failure_reasons=route_failures
                )
                record_route_failures(route_failures)
                if repaired_again is None:
                    continue
                repaired = repaired_again
                rerouted_segments += extra_segments
                corrected_metrics = self._path_metrics(repaired)
                if corrected_metrics["outside_ratio"] > 0.0 or (
                    self._collision_runs(repaired)
                    and (
                        corrected_metrics["collision_ratio"] > residual_collision_budget
                        or self._max_collision_run_meters(repaired)
                        > residual_segment_budget_meters
                    )
                ):
                    continue
            if (
                graph_matching_used
                and float(np.linalg.norm(repaired[0] - requested_start)) > 1e-9
                and not self._segment_collides(requested_start, repaired[0])
            ):
                # A raster repair may quantise the first graph vertex back to
                # its cell centre. Reattach the exact operator anchor through
                # a verified collision-free sub-cell connector.
                repaired = np.vstack((requested_start, repaired))
                corrected_metrics = self._path_metrics(repaired)
            matched = _resample_polyline(
                repaired,
                _trajectory_fractions(hypothesis["points"]),
            )
            displacement_m = (
                np.linalg.norm(matched - hypothesis["points"], axis=1)
                * self.config.meters_per_pixel
            )
            source_length = max(_polyline_length(hypothesis["points"]), 1e-9)
            corrected_length = _polyline_length(repaired)
            length_ratio = corrected_length / source_length
            corrected_length_meters = corrected_length * self.config.meters_per_pixel
            corrected_speed = (
                corrected_length_meters / duration if duration is not None else math.nan
            )
            corrected_speed_ratio = (
                corrected_speed / self.config.walking_speed_mps
                if math.isfinite(corrected_speed) and self.config.walking_speed_mps > 1e-9
                else math.nan
            )
            metric_preserved = self._metric_hypothesis_is_plausible(
                corrected_speed_ratio,
                duration=duration,
                observation_policy=observation_policy,
                loop_closure_verified=loop_closure_verified,
            )
            corrected_progress = _polyline_progress_metrics(
                repaired,
                meters_per_pixel=self.config.meters_per_pixel,
            )
            progress_preserved = (
                observation_policy != "independent"
                or (
                    allow_low_net_progress
                    and corrected_progress["span_length_ratio"]
                    >= INDEPENDENT_LOOP_CLOSED_MIN_SPAN_LENGTH_RATIO
                )
                or corrected_progress["net_progress_ratio"]
                >= INDEPENDENT_MIN_NET_PROGRESS_RATIO
            )
            median_correction = (
                float(np.median(displacement_m)) if len(displacement_m) else 0.0
            )
            p95_correction = (
                float(np.percentile(displacement_m, 95)) if len(displacement_m) else 0.0
            )
            maximum_correction = (
                float(np.max(displacement_m)) if len(displacement_m) else 0.0
            )
            start_error_meters = (
                float(np.linalg.norm(repaired[0] - requested_start))
                * self.config.meters_per_pixel
                if len(repaired) else float("inf")
            )
            # Do not apply the 1.05 m local-A* budget to a graph estimate.
            # Local repair is judged by displacement; graph matching is judged
            # primarily by connectivity, zero collisions, metric length and
            # the ordered turn signature, with a separate global plausibility
            # envelope to prevent arbitrary remote-aisle selection.
            if graph_matching_used:
                correction_budget = GRAPH_MAP_MATCH_P95_CORRECTION_METERS
                maximum_correction_budget = (
                    GRAPH_MAP_MATCH_HARD_CORRECTION_METERS
                )
            else:
                correction_budget = (
                    MASK_LOCAL_REPAIR_LIMIT_METERS
                    if self._support_mask is not None
                    else max(
                        MAX_LOCAL_MAP_CORRECTION_METERS,
                        min(
                            MAX_LOCAL_MAP_CORRECTION_HARD_METERS,
                            float(hypothesis["length_meters"])
                            * MAX_LOCAL_MAP_CORRECTION_ROUTE_FRACTION,
                        ),
                    )
                )
                maximum_correction_budget = (
                    MASK_LOCAL_REPAIR_HARD_LIMIT_METERS
                    if self._support_mask is not None
                    else MAX_LOCAL_MAP_CORRECTION_HARD_METERS
                )
            sharp_reverse_ratio = _polyline_sharp_reverse_ratio(
                repaired,
                meters_per_pixel=self.config.meters_per_pixel,
            )
            turn_topology = _turn_topology_metrics(hypothesis["points"], repaired)
            arc_turn_topology_preserved = (
                bool(turn_topology["event_sequence_preserved"])
            )
            fine_graph_diagnostics = (
                topology_recovery.get("fine")
                if graph_matching_used
                and isinstance(topology_recovery, dict)
                and isinstance(topology_recovery.get("fine"), dict)
                else {}
            )
            graph_sequence_cost_per_anchor = (
                float(fine_graph_diagnostics["objective"])
                / max(int(fine_graph_diagnostics.get("anchors", 0)), 1)
                if fine_graph_diagnostics.get("objective") is not None
                else float("inf")
            )
            graph_sequence_preserved = bool(
                graph_matching_used
                and macro_motion_diagnostics.get("applied")
                and macro_motion_diagnostics.get("turn_events_preserved")
                and graph_sequence_cost_per_anchor
                <= GRAPH_MAP_MATCH_MAX_VITERBI_COST_PER_ANCHOR
            )
            # A low graph objective is supporting evidence only. It must never
            # override either the failure or success of the measured complete
            # left/right event sequence.
            turn_topology_preserved = bool(
                arc_turn_topology_preserved
            )
            turn_topology.update({
                "arc_event_gate_preserved": arc_turn_topology_preserved,
                "graph_sequence_gate_preserved": graph_sequence_preserved,
                "graph_viterbi_cost_per_anchor": (
                    graph_sequence_cost_per_anchor
                    if math.isfinite(graph_sequence_cost_per_anchor)
                    else None
                ),
            })
            maximum_sharp_reverse_ratio = (
                INDEPENDENT_LOOP_CLOSED_MAX_SHARP_REVERSE_RATIO
                if observation_policy == "independent" and loop_closure_verified
                else MAX_AUTHORITATIVE_SHARP_REVERSE_RATIO
            )
            max_residual_collision_meters = self._max_collision_run_meters(repaired)
            # Mild aisle repairs are OK; long invented detours and zig-zag
            # spikes through drawing gaps are not the real walk.
            minimum_length_ratio = (
                INDEPENDENT_LOOP_CLOSED_MIN_LENGTH_RATIO
                if observation_policy == "independent" and loop_closure_verified
                else 0.60
                if self._support_mask is None
                else 0.70
            )
            shape_preserved = (
                start_error_meters <= MAX_PUBLISHED_START_ERROR_METERS
                and
                p95_correction <= correction_budget
                and maximum_correction <= maximum_correction_budget
                and minimum_length_ratio <= length_ratio <= 1.50
                and sharp_reverse_ratio <= maximum_sharp_reverse_ratio
                and (
                    graph_matching_used
                    or corrected_metrics["collision_ratio"]
                    <= residual_collision_budget
                )
                and (
                    graph_matching_used
                    or max_residual_collision_meters
                    <= residual_segment_budget_meters
                )
                and turn_topology_preserved
                and metric_preserved
                and progress_preserved
            )
            graph_estimation_score = (
                graph_sequence_cost_per_anchor
                + 1.25 * _speed_prior_penalty(
                    corrected_speed_ratio,
                    observation_policy,
                    duration,
                )
                if graph_matching_used
                and math.isfinite(graph_sequence_cost_per_anchor)
                and math.isfinite(corrected_speed_ratio)
                else float(hypothesis["score"])
            )
            constrained_score = (
                graph_estimation_score
                + 0.15 * median_correction
                + 0.22 * p95_correction
                + 0.75 * abs(math.log(max(length_ratio, 1e-9)))
                + 0.08 * rerouted_segments
                + 4.0 * sharp_reverse_ratio
                + (
                    0.03 * float(turn_topology["mean_abs_turn_error_degrees"])
                    + 3.0 * float(turn_topology["sign_mismatch_ratio"])
                    + 1.5 * min(
                        int(turn_topology.get("event_count_delta", 0)), 4
                    )
                )
                + (
                    1.25 * _speed_prior_penalty(
                        corrected_speed_ratio,
                        observation_policy,
                        duration,
                    )
                    if math.isfinite(corrected_speed_ratio) else 0.0
                )
            )
            feasible.append({
                **hypothesis,
                "repaired": repaired,
                "rerouted_segments": rerouted_segments,
                "corrected_metrics": corrected_metrics,
                "displacement_m": displacement_m,
                "length_ratio": length_ratio,
                "corrected_length_meters": corrected_length_meters,
                "corrected_speed_mps": corrected_speed,
                "corrected_speed_ratio": corrected_speed_ratio,
                "metric_preserved": metric_preserved,
                "corrected_progress": corrected_progress,
                "progress_preserved": progress_preserved,
                "repair_diagnostics": repair_diagnostics,
                "topology_recovery": topology_recovery,
                "graph_matching_used": graph_matching_used,
                "graph_estimation_score": graph_estimation_score,
                "segmented_recovery": segmented_recovery,
                "correction_budget_meters": correction_budget,
                "maximum_correction_meters": maximum_correction,
                "maximum_correction_budget_meters": maximum_correction_budget,
                "start_error_meters": start_error_meters,
                "sharp_reverse_ratio": sharp_reverse_ratio,
                "maximum_sharp_reverse_ratio": maximum_sharp_reverse_ratio,
                "turn_topology": turn_topology,
                "turn_topology_preserved": turn_topology_preserved,
                "max_residual_collision_meters": max_residual_collision_meters,
                "shape_preserved": shape_preserved,
                "minimum_length_ratio": minimum_length_ratio,
                "constrained_score": constrained_score,
            })

        if not feasible:
            partial_candidates = [
                item.get("graph_first_candidate")
                for item in topology_recovery_diagnostics
                if isinstance(item.get("graph_first_candidate"), dict)
                and item["graph_first_candidate"].get("available")
            ]
            graph_first_candidate = max(
                partial_candidates,
                key=lambda item: float(
                    item.get("confirmed_source_fraction", 0.0)
                ),
                default={
                    "available": False,
                    "reason": "no_confirmed_viterbi_edge",
                    "trajectory": [],
                    "segments": [],
                    "policy": "stop_at_last_confirmed_viterbi_node_v2",
                },
            )
            return {
                "accepted": False,
                "trajectory": [],
                "diagnostics": {
                    **base_diagnostics,
                    "reason": "constraint_solution_not_found",
                    "rejection_reasons": (
                        sorted(
                            route_failure_counts,
                            key=lambda reason: (-route_failure_counts[reason], reason),
                        )
                        or ["no_collision_free_route"]
                    ),
                    "route_failure_counts": route_failure_counts,
                    "hypothesis_count": len(hypotheses),
                    "metric_hypothesis_count": len(metric_hypotheses),
                    "beam_size": len(beam),
                    "constrained_hypotheses_attempted": attempted,
                    "topology_recovery_attempted": topology_recovery_attempted,
                    "topology_recovery_accepted": topology_recovery_accepted,
                    "topology_recovery_diagnostics": topology_recovery_diagnostics,
                    "segmented_recovery_attempted": segmented_recovery_attempted,
                    "segmented_recovery_accepted": segmented_recovery_accepted,
                    "segmented_recovery_diagnostics": segmented_recovery_diagnostics,
                    "graph_first_candidate": graph_first_candidate,
                    "raw_collision_ratio": round(float(hypotheses[0]["collision_ratio"]), 6),
                },
            }

        feasible.sort(key=lambda item: item["constrained_score"])
        # Accepting an arbitrary collision-free path is worse than reporting
        # that the visual observation and map do not agree.  Only
        # shape-preserving candidates are authoritative; otherwise A*/Viterbi
        # can turn R3 into a different green branch of the same mask.
        production_feasible = [
            item for item in feasible
            if bool(item["metric_preserved"])
            and bool(item["progress_preserved"])
            and bool(item["shape_preserved"])
        ]
        shape_fallback_used = False
        shape_fallback_budget = None
        if (
            not production_feasible
            and self._support_mask is not None
            and allow_safe_shape_fallback
        ):
            # An authoritative R3/fusion observation may disagree with an
            # immutable CAD mask.  The fallback can only accept routes that
            # still preserve R3 topology; it is not allowed to publish a
            # different green branch just because the branch is collision-free.
            safe_fallbacks: list[dict[str, Any]] = []
            for item in feasible:
                metrics = item["corrected_metrics"]
                if (
                    float(metrics["outside_ratio"]) == 0.0
                    and float(metrics["collision_ratio"])
                    <= residual_collision_budget
                    and self._max_collision_run_meters(item["repaired"])
                    <= residual_segment_budget_meters
                    and bool(item["metric_preserved"])
                    and bool(item["progress_preserved"])
                    and bool(item["shape_preserved"])
                ):
                    safe_fallbacks.append(item)
            if safe_fallbacks:
                safe_fallbacks.sort(key=lambda item: (
                    float(item["constrained_score"]),
                    float(np.percentile(item["displacement_m"], 95)),
                ))
                production_feasible = safe_fallbacks
                shape_fallback_used = True
                shape_fallback_budget = float("inf")
        if not production_feasible:
            closest = feasible[0]
            topology_candidate = (
                (closest.get("topology_recovery") or {}).get(
                    "graph_first_candidate"
                )
            )
            partial_candidates = [
                candidate
                for recovery in topology_recovery_diagnostics
                for candidate in [recovery.get("graph_first_candidate")]
                if isinstance(candidate, dict)
                and candidate.get("available")
            ]
            if (
                isinstance(topology_candidate, dict)
                and topology_candidate.get("available")
            ):
                partial_candidates.append(topology_candidate)
            graph_first_candidate = max(
                partial_candidates,
                key=lambda item: float(
                    item.get("confirmed_source_fraction", 0.0)
                ),
                default={
                    "available": False,
                    "reason": "strict_gate_rejected_without_partial_prefix",
                    "trajectory": [],
                    "segments": [],
                    "policy": "stop_at_last_confirmed_viterbi_node_v2",
                },
            )
            closest_displacement = closest["displacement_m"]
            closest_p95 = (
                float(np.percentile(closest_displacement, 95))
                if len(closest_displacement) else 0.0
            )
            return {
                "accepted": False,
                "trajectory": [],
                "diagnostics": {
                    **base_diagnostics,
                    "reason": (
                        "metric_prior_inconsistent"
                        if not bool(closest.get("metric_preserved", True))
                        else "insufficient_independent_net_progress"
                        if not bool(closest.get("progress_preserved", True))
                        else "turn_event_sequence_mismatch"
                        if not bool(
                            closest.get("turn_topology_preserved", False)
                        )
                        else "map_correction_exceeds_observation_budget"
                    ),
                    "rejection_reasons": [
                        "implausible_corrected_metric_scale"
                        if not bool(closest.get("metric_preserved", True))
                        else "compressed_or_looping_independent_observation"
                        if not bool(closest.get("progress_preserved", True))
                        else "turn_event_sequence_mismatch"
                        if not bool(
                            closest.get("turn_topology_preserved", False)
                        )
                        else "topology_destroying_map_correction"
                    ],
                    "hypothesis_count": len(hypotheses),
                    "beam_size": len(beam),
                    "constrained_hypotheses_attempted": attempted,
                    "feasible_hypotheses": len(feasible),
                    "topology_recovery_attempted": topology_recovery_attempted,
                    "topology_recovery_accepted": topology_recovery_accepted,
                    "topology_recovery_diagnostics": topology_recovery_diagnostics,
                    "segmented_recovery_attempted": segmented_recovery_attempted,
                    "segmented_recovery_accepted": segmented_recovery_accepted,
                    "segmented_recovery_diagnostics": segmented_recovery_diagnostics,
                    "graph_first_candidate": graph_first_candidate,
                    "correction_p95_meters": round(closest_p95, 3),
                    "correction_budget_meters": round(
                        float(closest["correction_budget_meters"]), 3
                    ),
                    "maximum_correction_meters": round(
                        float(closest["maximum_correction_meters"]), 3
                    ),
                    "sharp_reverse_ratio": round(
                        float(closest.get("sharp_reverse_ratio", 0.0)), 4
                    ),
                    "length_ratio": round(float(closest["length_ratio"]), 5),
                    "minimum_length_ratio": round(
                        float(closest["minimum_length_ratio"]), 5
                    ),
                    "corrected_length_meters": round(
                        float(closest["corrected_length_meters"]), 3
                    ),
                    "corrected_speed_mps": round(
                        float(closest["corrected_speed_mps"]), 4
                    ) if math.isfinite(float(closest["corrected_speed_mps"])) else None,
                    "corrected_speed_ratio": round(
                        float(closest["corrected_speed_ratio"]), 4
                    ) if math.isfinite(float(closest["corrected_speed_ratio"])) else None,
                    "shape_gate_details": {
                        "p95_within_budget": closest_p95
                        <= float(closest["correction_budget_meters"]),
                        "maximum_within_hard_budget": float(
                            closest["maximum_correction_meters"]
                        ) <= float(closest["maximum_correction_budget_meters"]),
                        "length_within_budget": float(closest["minimum_length_ratio"])
                        <= float(closest["length_ratio"]) <= 1.50,
                        "sharp_reverse_within_budget": float(
                            closest.get("sharp_reverse_ratio", 0.0)
                        ) <= float(closest["maximum_sharp_reverse_ratio"]),
                        "turn_topology_preserved": bool(
                            closest.get("turn_topology_preserved", True)
                        ),
                        "metric_preserved": bool(closest["metric_preserved"]),
                        "progress_preserved": bool(closest["progress_preserved"]),
                    },
                    "turn_topology": {
                        key: (
                            round(float(value), 6)
                            if isinstance(value, (float, np.floating)) else value
                        )
                        for key, value in (
                            closest.get("turn_topology") or {}
                        ).items()
                    },
                    "corrected_progress": {
                        key: round(float(value), 6)
                        for key, value in closest["corrected_progress"].items()
                    },
                    "minimum_independent_net_progress_ratio": (
                        INDEPENDENT_MIN_NET_PROGRESS_RATIO
                        if observation_policy == "independent" else None
                    ),
                    "minimum_loop_closed_span_length_ratio": (
                        INDEPENDENT_LOOP_CLOSED_MIN_SPAN_LENGTH_RATIO
                        if observation_policy == "independent" and allow_low_net_progress
                        else None
                    ),
                },
            }
        if len(production_feasible) > 1:
            runner = next((
                item for item in production_feasible[1:]
                if (
                    max(float(item["scale"]), float(production_feasible[0]["scale"]))
                    / max(min(float(item["scale"]), float(production_feasible[0]["scale"])), 1e-12)
                    > 1.02
                    or float(np.linalg.norm(
                        np.asarray(item["repaired"])[-1]
                        - np.asarray(production_feasible[0]["repaired"])[-1]
                    )) * self.config.meters_per_pixel > 1.0
                )
            ), None)
            if runner is not None:
                best_candidate = production_feasible[0]
                selection_margin = (
                    float(runner["constrained_score"])
                    - float(best_candidate["constrained_score"])
                )
                scale_ratio = (
                    max(float(runner["scale"]), float(best_candidate["scale"]))
                    / max(min(float(runner["scale"]), float(best_candidate["scale"])), 1e-12)
                )
                endpoint_separation_meters = float(np.linalg.norm(
                    np.asarray(runner["repaired"])[-1]
                    - np.asarray(best_candidate["repaired"])[-1]
                )) * self.config.meters_per_pixel
                minimum_margin = (
                    INDEPENDENT_MIN_SELECTION_MARGIN
                    if observation_policy == "independent"
                    else AUTHORITATIVE_MIN_SELECTION_MARGIN
                )
                maximum_endpoint_separation = (
                    6.0
                    if observation_policy == "independent"
                    else AUTHORITATIVE_MAX_AMBIGUOUS_ENDPOINT_METERS
                )
                if (
                    selection_margin < minimum_margin
                    and (
                        scale_ratio > INDEPENDENT_MAX_AMBIGUOUS_SCALE_RATIO
                        or endpoint_separation_meters > maximum_endpoint_separation
                    )
                ):
                    source_name = (
                        "independent"
                        if observation_policy == "independent"
                        else "authoritative"
                    )
                    return {
                        "accepted": False,
                        "trajectory": [],
                        "diagnostics": {
                            **base_diagnostics,
                            "reason": f"ambiguous_{source_name}_map_alignment",
                            "rejection_reasons": [
                                f"{source_name}_scale_or_topology_ambiguous"
                            ],
                            "runner_up_margin": round(selection_margin, 6),
                            "ambiguous_scale_ratio": round(scale_ratio, 5),
                            "ambiguous_endpoint_separation_meters": round(
                                endpoint_separation_meters, 3
                            ),
                        },
                    }
        best = production_feasible[0]
        repaired = best["repaired"]
        corrected_metrics = best["corrected_metrics"]
        displacement_m = best["displacement_m"]
        length_ratio = float(best["length_ratio"])
        rerouted_segments = int(best["rerouted_segments"])
        p95_correction = float(np.percentile(displacement_m, 95)) if len(displacement_m) else 0.0
        speed = float(best["corrected_speed_mps"])
        nonlinear_map_matching_enabled = topology_recovery_enabled
        preselection_recovery = best.get("topology_recovery")
        if isinstance(preselection_recovery, dict):
            # This route entered the feasible set only after the graph matcher
            # replaced an A* detour which looked like an unsupported loop.
            # Preserve that evidence instead of running the same recovery a
            # second time and accidentally comparing it against itself.
            nonlinear_diagnostics = {
                **preselection_recovery,
                "attempted": True,
                "accepted": True,
                "production_enabled": True,
            }
        else:
            nonlinear_diagnostics: dict[str, Any] = {
                "attempted": False,
                "accepted": False,
                "method": "corridor_graph_multilevel_viterbi_v5_signed_events_branch_lock",
                "reason": (
                    "global_solution_stable"
                    if nonlinear_map_matching_enabled
                    else "disabled_by_operator"
                ),
                "production_enabled": nonlinear_map_matching_enabled,
            }
        should_attempt_nonlinear = (
            preselection_recovery is None
            and nonlinear_map_matching_enabled
            and (
                # An operator-authored production graph is authoritative:
                # always select/verify its edge sequence before publishing,
                # even when local A* happened to make only a small correction.
                # Raster-only maps retain the conservative legacy threshold.
                (
                    observation_policy == "authoritative"
                    and self._topology_authored_node_count > 0
                    and len(best["points"]) >= GRAPH_MAP_MATCH_MIN_OBSERVATIONS
                )
                or p95_correction > 2.0
            )
        )
        if should_attempt_nonlinear:
            topology_recovery_attempted += 1
            nonlinear, nonlinear_diagnostics = self._multilevel_viterbi_map_match(
                best["points"],
                repaired,
                allow_global_recovery=observation_policy == "authoritative",
            )
            nonlinear_diagnostics["production_enabled"] = True
            nonlinear_diagnostics["phase"] = "postselection_refinement"
            nonlinear_diagnostics["operator_graph_mandatory"] = bool(
                observation_policy == "authoritative"
                and self._topology_authored_node_count > 0
            )
            topology_recovery_diagnostics.append(nonlinear_diagnostics)
            if nonlinear is not None:
                nonlinear_matched = _resample_polyline(
                    nonlinear, _trajectory_fractions(best["points"])
                )
                nonlinear_displacement = (
                    np.linalg.norm(nonlinear_matched - best["points"], axis=1)
                    * self.config.meters_per_pixel
                )
                nonlinear_p95 = float(np.percentile(nonlinear_displacement, 95))
                nonlinear_length_ratio = _polyline_length(nonlinear) / max(
                    _polyline_length(best["points"]), 1e-9
                )
                nonlinear_length_meters = (
                    _polyline_length(nonlinear) * self.config.meters_per_pixel
                )
                nonlinear_speed = (
                    nonlinear_length_meters / duration
                    if duration is not None else math.nan
                )
                nonlinear_speed_ratio = (
                    nonlinear_speed / self.config.walking_speed_mps
                    if math.isfinite(nonlinear_speed)
                    and self.config.walking_speed_mps > 1e-9
                    else math.nan
                )
                nonlinear_metric_preserved = self._metric_hypothesis_is_plausible(
                    nonlinear_speed_ratio,
                    duration=duration,
                    observation_policy=observation_policy,
                    loop_closure_verified=loop_closure_verified,
                )
                observed_resampled = _resample_polyline(
                    best["points"], np.linspace(0.0, 1.0, 64)
                )
                candidate_resampled = _resample_polyline(
                    nonlinear, np.linspace(0.0, 1.0, 64)
                )

                def signed_turn(points: np.ndarray) -> float:
                    deltas = np.diff(points[:, :2], axis=0)
                    lengths = np.linalg.norm(deltas, axis=1)
                    deltas = deltas[lengths > 1e-6]
                    if len(deltas) < 2:
                        return 0.0
                    headings = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
                    return math.degrees(float(headings[-1] - headings[0]))

                observed_turn = signed_turn(observed_resampled)
                candidate_turn = signed_turn(candidate_resampled)
                chirality_preserved = (
                    abs(observed_turn) < 20.0 and abs(candidate_turn) < 25.0
                ) or (
                    observed_turn * candidate_turn > 0.0
                    and 0.60 <= abs(candidate_turn) / max(abs(observed_turn), 1e-9) <= 1.40
                )
                improves = (
                    nonlinear_p95 <= p95_correction * 0.97
                    or abs(math.log(max(nonlinear_length_ratio, 1e-9)))
                    < abs(math.log(max(length_ratio, 1e-9))) * 0.92
                )
                bounded = (
                    nonlinear_p95 <= min(p95_correction + 0.25, 4.0)
                    and 0.75 <= nonlinear_length_ratio <= 1.35
                    and chirality_preserved
                    and not self._collision_runs(nonlinear)
                    and self._path_metrics(nonlinear)["outside_ratio"] == 0.0
                    and nonlinear_metric_preserved
                )
                nonlinear_diagnostics.update({
                    "correction_p95_before_meters": round(p95_correction, 3),
                    "correction_p95_after_meters": round(nonlinear_p95, 3),
                    "length_ratio_before": round(length_ratio, 5),
                    "length_ratio_after": round(nonlinear_length_ratio, 5),
                    "observed_signed_turn_degrees": round(observed_turn, 3),
                    "candidate_signed_turn_degrees": round(candidate_turn, 3),
                    "chirality_preserved": chirality_preserved,
                    "speed_mps_after": (
                        round(nonlinear_speed, 4)
                        if math.isfinite(nonlinear_speed) else None
                    ),
                    "metric_preserved": nonlinear_metric_preserved,
                })
                if improves and bounded:
                    topology_recovery_accepted += 1
                    repaired = nonlinear
                    corrected_metrics = self._path_metrics(repaired)
                    displacement_m = nonlinear_displacement
                    p95_correction = nonlinear_p95
                    length_ratio = nonlinear_length_ratio
                    speed = nonlinear_speed
                    nonlinear_diagnostics["accepted"] = True
                    nonlinear_diagnostics["reason"] = None
                else:
                    nonlinear_diagnostics["accepted"] = False
                    nonlinear_diagnostics["reason"] = "safe_baseline_not_improved"
        allowed_correction = (
            float(best["correction_budget_meters"])
            if self._support_mask is not None
            else max(4.0, min(12.0, float(best["length_meters"]) * 0.18))
        )

        # Scale, speed and correction magnitude affect confidence, but they do
        # not invalidate a route already proven to lie inside the fixed plan.
        quality_warnings: list[str] = []
        if not 0.70 <= length_ratio <= 1.65:
            quality_warnings.append("route_length_changed_significantly")
        if p95_correction > allowed_correction:
            quality_warnings.append("large_map_correction_applied")
        if shape_fallback_used:
            quality_warnings.append("authoritative_safe_map_fallback")
        if preselection_recovery is not None:
            quality_warnings.append("corridor_graph_topology_recovery_applied")
        if best.get("segmented_recovery") is not None:
            quality_warnings.append("time_segmented_local_drift_correction_applied")
        final_speed_ratio = (
            speed / self.config.walking_speed_mps
            if math.isfinite(speed) and self.config.walking_speed_mps > 1e-9
            else math.nan
        )
        if not bool(best.get("metric_preserved", True)):
            quality_warnings.append("walking_speed_prior_inconsistent")
        elif (
            observation_policy == "independent"
            and math.isfinite(final_speed_ratio)
            and final_speed_ratio < INDEPENDENT_SPEED_RATIO_BOUNDS[0]
        ):
            quality_warnings.append("slow_long_inspection_route")
        if start_snap_meters > max(0.05, self.cell_meters * 0.75):
            quality_warnings.append("start_projected_to_walkable_area")

        second_score = (
            float(production_feasible[1]["constrained_score"])
            if len(production_feasible) > 1
            else float(best["constrained_score"] + 1.0)
        )
        margin = max(0.0, second_score - float(best["constrained_score"]))
        confidence = float(np.clip(
            0.35
            + min(margin / 0.8, 0.25)
            + max(0.0, 1.0 - p95_correction / max(allowed_correction, 1e-9)) * 0.30
            - min(0.08 * len(quality_warnings), 0.24),
            0.0,
            1.0,
        ))
        if observation_policy == "independent":
            quality_warnings.append("independent_monocular_rescue")
            confidence = min(confidence, 0.55)
        probability_correct, calibration = calibrated_probability(confidence)

        # Publication is stricter than candidate scoring: no residual obstacle
        # contact, no cross-component stitching, and no multi-metre sparse
        # chords that downstream consumers could mistake for one time step.
        repaired = _densify_polyline(
            repaired,
            MAX_PUBLISHED_SEGMENT_METERS
            / max(self.config.meters_per_pixel, 1e-9),
        )
        final_metrics = self._path_metrics(repaired)
        publication_repair_applied = False
        publication_rerouted_segments = 0
        publication_unsafe = (
            final_metrics["outside_ratio"] > 0.0
            or final_metrics["collision_ratio"] > 0.0
            or bool(self._collision_runs(repaired))
            or self._path_component_count(repaired) != 1
        )
        publication_repair_passes = 0
        while publication_unsafe and publication_repair_passes < 3:
            # Sparse graph paths can have legal vertices while the straight
            # chord between them clips an obstacle. Densification exposes the
            # collision; give the dense line bounded local A* cleanup passes
            # and then enforce the exact same zero-collision publication gate.
            publication_repaired, repaired_segments = (
                self._repair_collisions(repaired)
            )
            publication_repair_passes += 1
            if publication_repaired is None:
                break
            publication_rerouted_segments += int(repaired_segments)
            repaired = _densify_polyline(
                publication_repaired,
                MAX_PUBLISHED_SEGMENT_METERS
                / max(self.config.meters_per_pixel, 1e-9),
            )
            final_metrics = self._path_metrics(repaired)
            publication_repair_applied = True
            publication_unsafe = (
                final_metrics["outside_ratio"] > 0.0
                or final_metrics["collision_ratio"] > 0.0
                or bool(self._collision_runs(repaired))
                or self._path_component_count(repaired) != 1
            )
        final_progress = _polyline_progress_metrics(
            repaired,
            meters_per_pixel=self.config.meters_per_pixel,
        )
        if (
            observation_policy == "independent"
            and not allow_low_net_progress
            and final_progress["net_progress_ratio"]
            < INDEPENDENT_MIN_NET_PROGRESS_RATIO
        ):
            return {
                "accepted": False,
                "trajectory": [],
                "diagnostics": {
                    **base_diagnostics,
                    "reason": "insufficient_independent_net_progress",
                    "rejection_reasons": [
                        "compressed_or_looping_independent_observation"
                    ],
                    "net_progress_ratio": round(
                        float(final_progress["net_progress_ratio"]), 6
                    ),
                    "minimum_net_progress_ratio": INDEPENDENT_MIN_NET_PROGRESS_RATIO,
                },
            }
        if (
            final_metrics["outside_ratio"] > 0.0
            or final_metrics["collision_ratio"] > 0.0
            or self._collision_runs(repaired)
            or self._path_component_count(repaired) != 1
        ):
            return {
                "accepted": False,
                "trajectory": [],
                "diagnostics": {
                    **base_diagnostics,
                    "reason": "final_publication_invariant_failed",
                    "rejection_reasons": ["unsafe_or_disconnected_final_polyline"],
                    "publication_repair_applied": publication_repair_applied,
                    "publication_rerouted_segments": int(
                        publication_rerouted_segments
                    ),
                    "publication_repair_passes":
                        publication_repair_passes,
                    "selected_scale_pixels_per_unit": round(
                        float(best["scale"]), 8
                    ),
                    "graph_matching_used": bool(
                        best.get("graph_matching_used")
                    ),
                    "selected_topology_recovery":
                        best.get("topology_recovery"),
                    "nonlinear_map_matching": nonlinear_diagnostics,
                    "final_collision_ratio": round(
                        float(final_metrics["collision_ratio"]), 8
                    ),
                    "final_outside_ratio": round(
                        float(final_metrics["outside_ratio"]), 8
                    ),
                    "final_component_count":
                        self._path_component_count(repaired),
                    "final_collision_runs": [
                        {
                            "left": int(left),
                            "right": int(right),
                            "start": repaired[left].tolist(),
                            "end": repaired[right].tolist(),
                            "points": repaired[left:right + 1].tolist(),
                        }
                        for left, right in self._collision_runs(repaired)[:3]
                    ],
                },
            }
        corrected_metrics = final_metrics
        published_steps = np.linalg.norm(np.diff(repaired, axis=0), axis=1)
        max_published_segment_meters = (
            float(np.max(published_steps)) * self.config.meters_per_pixel
            if len(published_steps) else 0.0
        )
        published_length_meters = (
            _polyline_length(repaired) * self.config.meters_per_pixel
        )
        published_start_error_meters = (
            float(np.linalg.norm(repaired[0] - requested_start))
            * self.config.meters_per_pixel
            if len(repaired) else float("inf")
        )
        if published_start_error_meters > MAX_PUBLISHED_START_ERROR_METERS:
            return {
                "accepted": False,
                "trajectory": [],
                "diagnostics": {
                    **base_diagnostics,
                    "reason": "start_anchor_moved",
                    "rejection_reasons": ["published_start_not_locked"],
                    "start_error_meters": round(published_start_error_meters, 6),
                    "maximum_start_error_meters": MAX_PUBLISHED_START_ERROR_METERS,
                },
            }
        published_initial_heading = self._initial_heading(repaired - repaired[0])
        direction_error_radians = math.atan2(
            math.sin(published_initial_heading - desired_heading),
            math.cos(published_initial_heading - desired_heading),
        )
        published_direction_error_degrees = abs(
            math.degrees(direction_error_radians)
        )
        maximum_published_direction_error_degrees = (
            GRAPH_MAP_MATCH_HARD_INITIAL_DIRECTION_DEGREES
            if best.get("graph_matching_used")
            else MAX_PUBLISHED_INITIAL_DIRECTION_ERROR_DEGREES
        )
        if (
            published_direction_error_degrees
            > maximum_published_direction_error_degrees
        ):
            return {
                "accepted": False,
                "trajectory": [],
                "diagnostics": {
                    **base_diagnostics,
                    "reason": "initial_direction_moved",
                    "rejection_reasons": ["published_initial_direction_not_locked"],
                    "initial_direction_error_degrees": round(
                        published_direction_error_degrees, 6
                    ),
                    "maximum_initial_direction_error_degrees":
                        maximum_published_direction_error_degrees,
                },
            }
        diagnostics = {
            **base_diagnostics,
            "accepted": True,
            "reason": None,
            "rejection_reasons": [],
            "quality_warnings": quality_warnings,
            "constraint_mode": (
                "hard_obstacles_and_cad_support"
                if self._support_mask is not None else "hard_walkable_mask"
            ),
            "hypothesis_count": len(hypotheses),
            "metric_hypothesis_count": len(metric_hypotheses),
            "beam_size": len(beam),
            "constrained_hypotheses_attempted": attempted,
            "feasible_hypotheses": len(feasible),
            "topology_recovery_attempted": topology_recovery_attempted,
            "topology_recovery_accepted": topology_recovery_accepted,
            "topology_recovery_diagnostics": topology_recovery_diagnostics,
            "segmented_recovery_attempted": segmented_recovery_attempted,
            "segmented_recovery_accepted": segmented_recovery_accepted,
            "segmented_recovery_diagnostics": segmented_recovery_diagnostics,
            "shape_preserving_hypotheses": sum(
                1 for item in feasible
                if bool(item.get("shape_preserved"))
                and bool(item.get("metric_preserved"))
                and bool(item.get("progress_preserved"))
            ),
            "production_hypotheses": len(production_feasible),
            "shape_fallback_used": shape_fallback_used,
            "shape_fallback_budget_meters": (
                round(shape_fallback_budget, 3)
                if shape_fallback_budget is not None
                and math.isfinite(shape_fallback_budget)
                else None
            ),
            "shape_fallback_policy": (
                "authoritative_plan_connectivity_v2"
                if shape_fallback_used else None
            ),
            "selected_scale_pixels_per_unit": round(float(best["scale"]), 8),
            "selected_yaw_offset_degrees": round(float(best["yaw"]), 3),
            "estimated_length_meters": round(float(best["length_meters"]), 3),
            "published_length_meters": round(published_length_meters, 3),
            "motion_duration_seconds": round(float(duration), 3) if duration is not None else None,
            "estimated_speed_mps": round(speed, 3) if math.isfinite(speed) else None,
            "estimated_speed_ratio": (
                round(final_speed_ratio, 4)
                if math.isfinite(final_speed_ratio) else None
            ),
            "raw_collision_ratio": round(float(best["collision_ratio"]), 6),
            "corrected_collision_ratio": round(float(corrected_metrics["collision_ratio"]), 6),
            "max_residual_collision_meters": round(
                self._max_collision_run_meters(repaired), 3
            ),
            "outside_ratio": round(float(corrected_metrics["outside_ratio"]), 6),
            "rerouted_segments": rerouted_segments,
            "start_snap_meters": round(start_snap_meters, 3),
            "start_error_meters": round(published_start_error_meters, 6),
            "maximum_start_error_meters": MAX_PUBLISHED_START_ERROR_METERS,
            "start_anchor_locked": True,
            "initial_direction_locked": True,
            "initial_direction_error_degrees": round(
                published_direction_error_degrees, 6
            ),
            "maximum_initial_direction_error_degrees":
                maximum_published_direction_error_degrees,
            "correction_median_meters": round(float(np.median(displacement_m)), 3),
            "correction_p95_meters": round(p95_correction, 3),
            "maximum_correction_meters": round(
                float(np.max(displacement_m)) if len(displacement_m) else 0.0, 3
            ),
            "correction_budget_meters": round(allowed_correction, 3),
            "turn_topology": {
                key: (
                    round(float(value), 6)
                    if isinstance(value, (float, np.floating)) else value
                )
                for key, value in (best.get("turn_topology") or {}).items()
            },
            "sharp_reverse_ratio": round(float(best.get("sharp_reverse_ratio", 0.0)), 4),
            "publication_repair_applied": publication_repair_applied,
            "publication_rerouted_segments": int(publication_rerouted_segments),
            "publication_repair_passes": publication_repair_passes,
            "length_ratio": round(length_ratio, 5),
            "max_published_segment_meters": round(max_published_segment_meters, 4),
            "endpoint_displacement_meters": round(
                float(final_progress["endpoint_displacement_meters"]), 3
            ),
            "net_progress_ratio": round(
                float(final_progress["net_progress_ratio"]), 6
            ),
            "trajectory_tortuosity": round(
                float(final_progress["tortuosity"]), 6
            ),
            "hypothesis_score": round(float(best["score"]), 6),
            "constrained_score": round(float(best["constrained_score"]), 6),
            "runner_up_margin": round(margin, 6),
            "nonlinear_map_matching": nonlinear_diagnostics,
            "selected_detour_diagnostics": best.get("repair_diagnostics"),
            "selected_topology_recovery": best.get("topology_recovery"),
            "graph_matching_used": bool(best.get("graph_matching_used")),
            "map_matching_mode": (
                "corridor_graph_sequence"
                if best.get("graph_matching_used")
                else "local_shape_preserving"
            ),
            "selected_segmented_recovery": best.get("segmented_recovery"),
            # Kept for API compatibility; semantically this is a quality score.
            "confidence": round(confidence, 4),
            "quality_score": round(confidence, 4),
            "probability_correct": (
                round(probability_correct, 4)
                if probability_correct is not None else None
            ),
            "confidence_calibration": calibration,
            "coordinate_convention": "plan_pixels_x_right_y_down",
        }
        output = [
            [round(float(point[0]), 3), round(float(point[1]), 3), 0.0]
            for point in repaired
        ]
        return {"accepted": True, "trajectory": output, "diagnostics": diagnostics}


_ENGINE_CACHE: dict[str, FloorplanConstraintEngine] = {}


def get_floorplan_engine(map_id: str = DEFAULT_FLOORPLAN_ID) -> FloorplanConstraintEngine:
    engine = _ENGINE_CACHE.get(map_id)
    if engine is None:
        engine = FloorplanConstraintEngine.load(map_id)
        _ENGINE_CACHE[map_id] = engine
    return engine


def invalidate_floorplan_engine(map_id: str = DEFAULT_FLOORPLAN_ID) -> None:
    """Drop a cached engine after an atomic production graph update."""
    _ENGINE_CACHE.pop(map_id, None)


def _map_turn_points(
    turns: Any,
    source: Any,
    mapped: list[list[float]],
    *,
    meters_per_pixel: float,
    timestamps: Any = None,
) -> list[dict[str, Any]]:
    if not isinstance(turns, list) or not mapped:
        return []
    source_points = _normalise_points(source)
    mapped_points = _normalise_points(mapped)
    if len(source_points) == 0 or len(mapped_points) == 0:
        return []
    source_fractions = _trajectory_fractions(source_points)
    mapped_fractions = _trajectory_fractions(mapped_points)
    result: list[dict[str, Any]] = []
    for item in turns:
        if not isinstance(item, dict):
            continue
        source_index = int(round(_finite_float(item.get("trajectory_index"), 0.0)))
        source_index = max(0, min(len(source_points) - 1, source_index))
        turn_fraction = float(source_fractions[source_index])
        mapped_index = int(np.argmin(np.abs(mapped_fractions - turn_fraction)))
        approach_source = max(0, min(
            len(source_points) - 1,
            int(round(_finite_float(item.get("approach_index"), source_index))),
        ))
        exit_source = max(approach_source, min(
            len(source_points) - 1,
            int(round(_finite_float(item.get("exit_index"), source_index))),
        ))
        approach_mapped = int(np.argmin(np.abs(
            mapped_fractions - float(source_fractions[approach_source])
        )))
        exit_mapped = int(np.argmin(np.abs(
            mapped_fractions - float(source_fractions[exit_source])
        )))
        local_length_meters = (
            _polyline_length(mapped_points[approach_mapped:exit_mapped + 1])
            * meters_per_pixel
        )
        angle = abs(_finite_float(
            item.get("geometry_angle_degrees", item.get("angle_degrees")), 0.0
        ))
        duration_seconds = None
        if isinstance(timestamps, list) and len(timestamps) == len(source_points):
            start_time = _finite_float(timestamps[approach_source], math.nan)
            end_time = _finite_float(timestamps[exit_source], math.nan)
            if math.isfinite(start_time) and math.isfinite(end_time) and end_time > start_time:
                duration_seconds = end_time - start_time
        result.append({
            **item,
            "trajectory_index": mapped_index,
            "position": mapped[mapped_index],
            "map_constrained": True,
            "map_turn_arc_meters": round(local_length_meters, 4),
            "map_curvature_degrees_per_meter": (
                round(angle / local_length_meters, 4)
                if local_length_meters > 1e-6 else None
            ),
            "map_yaw_rate_degrees_per_second": (
                round(angle / duration_seconds, 4)
                if duration_seconds is not None else None
            ),
        })
    return result


def apply_floorplan_constraints(
    result: dict[str, Any],
    map_context: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate every trustworthy observer and attach only an accepted map route.

    Pose-graph fragmentation is evidence, not a veto.  R3, guarded R3/LingBot
    fusion, and both PCA polarities of an independent LingBot observation are
    evaluated against the same immutable floor plan.  The selected source is
    published only after its map alignment passes all production gates.
    """
    if not isinstance(result, dict):
        return result
    context = map_context or {}
    map_id = str(context.get("floorplan_id") or DEFAULT_FLOORPLAN_ID)
    primary_points = result.get("plan_trajectory") or result.get("trajectory") or []
    stats = dict(result.get("processing_stats") or {})
    for stale_key in (
        "map_matching_applied",
        "map_trajectory_points",
        "map_confidence",
        "map_distance_meters",
        "map_observation_source",
        "floorplan_constraint",
    ):
        stats.pop(stale_key, None)

    method = str(result.get("method") or "").lower()
    primary_source = (
        "r3"
        if method.startswith("r3")
        else ("lingbot" if method.startswith("lingbot") else (method or "visual_odometry"))
    )
    primary_convention = (
        "x_forward_y_left_z_up"
        if method.startswith("r3")
        else "x_right_y_down"
    )
    quality = result.get("trajectory_quality") or stats.get("r3_trajectory_quality") or {}
    if isinstance(quality, dict):
        projection = quality.get("projection") or {}
        if isinstance(projection, dict):
            primary_convention = str(
                projection.get("plan_coordinate_convention") or primary_convention
            )

    source_timestamps = (
        result.get("r3_source_timestamps_seconds")
        or result.get("source_timestamps_seconds")
    )
    fragmented_r3 = method.startswith("r3") and _r3_is_severely_fragmented(result)
    kerama_raster_polarity_calibrated = bool(
        map_id == DEFAULT_FLOORPLAN_ID
        and method.startswith("r3")
        and primary_convention == "x_forward_y_left_z_up"
    )
    candidate_payload = result.get("lingbot_fusion_candidate")
    candidate_payload = candidate_payload if isinstance(candidate_payload, dict) else {}

    observations: list[dict[str, Any]] = [{
        "source": primary_source,
        "variant": "primary",
        "points": primary_points,
        "timestamps": source_timestamps,
        "coordinate_convention": primary_convention,
        "selection_tier": 0,
        # Fragmentation lowers trust, but never removes a geometrically valid R3
        # route from competition.
        "source_prior": (
            1.35 if fragmented_r3 else 0.95
        ) if kerama_raster_polarity_calibrated else (
            0.45 if fragmented_r3 else 0.05
        ),
    }]

    if method.startswith("r3") and primary_convention == "x_forward_y_left_z_up":
        # R3's recovered floor plane has a chirality, but the display plan is
        # rasterised with image Y pointing down.  The camera metadata reports
        # the former convention and therefore cannot, by itself, prove the
        # correct raster polarity.  Evaluate the other *global* polarity as
        # an equal R3 observation; the fixed start/direction and hard green
        # corridor mask decide it.  This changes only one similarity transform
        # for the whole curve -- it does not mirror individual turns or invoke
        # a route-inventing A* fallback.
        observations.append({
            "source": primary_source,
            "variant": "r3_image_y_down",
            "points": primary_points,
            "timestamps": source_timestamps,
            "coordinate_convention": "x_right_y_down",
            "selection_tier": 0,
            # The operator-confirmed Kerama raster calibration resolves the
            # R3 floor-plane gauge in image-Y-down coordinates. Keep both
            # polarities evaluated, but prefer the calibrated one unless it
            # fails a hard graph or publication gate.
            "source_prior": (
                0.05 if fragmented_r3 else 0.0
            ) if kerama_raster_polarity_calibrated else (
                0.46 if fragmented_r3 else 0.06
            ),
            "polarity_candidate": "unflipped_r3_plan_y",
        })

    # R3 is the authoritative motion observation.  LingBot can be useful for
    # diagnostics, but must not silently replace the direction/topology of a
    # measured R3 path on the production floor plan.
    fusion_map_candidate_enabled = os.getenv(
        "TRACKAI_ENABLE_FUSION_MAP_CANDIDATE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    fusion_points = (
        candidate_payload.get("plan_trajectory") or []
        if candidate_payload.get("accepted") and fusion_map_candidate_enabled
        else []
    )
    if fusion_points:
        observations.append({
            "source": "r3_lingbot_fusion",
            "variant": "guarded_fusion",
            "points": fusion_points,
            "timestamps": source_timestamps,
            "coordinate_convention": primary_convention,
            "selection_tier": 0,
            # A fusion candidate has already passed the independent geometry
            # agreement gate.  Prefer it on a map-cost tie, while still
            # allowing a materially better raw R3 alignment to win.
            "source_prior": 0.0,
        })

    diagnostics_payload = candidate_payload.get("diagnostics") or {}
    independent_quality = (
        diagnostics_payload.get("independent_quality")
        if isinstance(diagnostics_payload, dict)
        else None
    )
    independent_quality_ok = bool(candidate_payload.get("independent_accepted")) and (
        not isinstance(independent_quality, dict)
        or bool(independent_quality.get("accepted", True))
    )
    independent_loop_closure_verified = bool(
        isinstance(independent_quality, dict)
        and independent_quality.get("loop_closure_verified")
    )
    independent_points = (
        candidate_payload.get("independent_plan_trajectory") or []
        if independent_quality_ok
        else []
    )
    independent_timestamps = (
        candidate_payload.get("lingbot_source_timestamps_seconds")
        or candidate_payload.get("source_timestamps_seconds")
    )
    if independent_points:
        timestamp_provenance = "lingbot_source_timestamps"
        if not isinstance(independent_timestamps, list) or len(independent_timestamps) < 2:
            independent_timestamps = _resample_timestamps(
                source_timestamps, len(independent_points)
            )
            timestamp_provenance = "r3_duration_resampled"
        stabilized_independent, independent_stabilization = (
            _stabilize_independent_observation(independent_points)
        )
        independent_prior = 0.20 if fragmented_r3 else 0.80
        observations.append({
            "source": "lingbot_independent",
            "variant": "native",
            "points": stabilized_independent,
            "timestamps": independent_timestamps,
            "coordinate_convention": "x_right_y_down",
            "selection_tier": 1,
            "source_prior": independent_prior,
            "fusion_supported": bool(candidate_payload.get("accepted")),
            "timestamp_provenance": timestamp_provenance,
            "observation_stabilization": independent_stabilization,
            "loop_closure_verified": independent_loop_closure_verified,
        })
        if independent_loop_closure_verified:
            observations.append({
                "source": "lingbot_independent",
                "variant": "native_heading_right_90",
                "points": stabilized_independent,
                "timestamps": independent_timestamps,
                "coordinate_convention": "x_right_y_down",
                "selection_tier": 2,
                "source_prior": independent_prior + 0.04,
                "fusion_supported": bool(candidate_payload.get("accepted")),
                "timestamp_provenance": timestamp_provenance,
                "observation_stabilization": independent_stabilization,
                "loop_closure_verified": True,
                "yaw_offsets_degrees": RIGHT_QUARTER_TURN_YAW_OFFSETS_DEGREES,
            })
            observations.append({
                "source": "lingbot_independent",
                "variant": "native_heading_flip_180",
                "points": stabilized_independent,
                "timestamps": independent_timestamps,
                "coordinate_convention": "x_right_y_down",
                # A verified loop can make the first-motion polarity
                # ambiguous.  Respect the requested heading first and try
                # the opposite polarity only after the native tier rejects.
                "selection_tier": 2,
                "source_prior": independent_prior + 0.08,
                "fusion_supported": bool(candidate_payload.get("accepted")),
                "timestamp_provenance": timestamp_provenance,
                "observation_stabilization": independent_stabilization,
                "loop_closure_verified": True,
                "yaw_offsets_degrees": REVERSED_HEADING_YAW_OFFSETS_DEGREES,
            })
        flipped = _flip_polyline_y(stabilized_independent)
        if flipped:
            observations.append({
                "source": "lingbot_independent",
                "variant": "y_flip",
                "points": flipped,
                "timestamps": independent_timestamps,
                "coordinate_convention": "x_right_y_down",
                "selection_tier": 1,
                "source_prior": independent_prior + 0.02,
                "fusion_supported": bool(candidate_payload.get("accepted")),
                "timestamp_provenance": timestamp_provenance,
                "observation_stabilization": independent_stabilization,
                "loop_closure_verified": independent_loop_closure_verified,
            })
            if independent_loop_closure_verified:
                observations.append({
                    "source": "lingbot_independent",
                    "variant": "y_flip_heading_flip_180",
                    "points": flipped,
                    "timestamps": independent_timestamps,
                    "coordinate_convention": "x_right_y_down",
                    "selection_tier": 2,
                    "source_prior": independent_prior + 0.06,
                    "fusion_supported": bool(candidate_payload.get("accepted")),
                    "timestamp_provenance": timestamp_provenance,
                    "observation_stabilization": independent_stabilization,
                    "loop_closure_verified": True,
                    "yaw_offsets_degrees": REVERSED_HEADING_YAW_OFFSETS_DEGREES,
                })

    source_selection: dict[str, Any] = {
        "primary": primary_source,
        "selected": None,
        "reason": "no_candidate_satisfied_floorplan",
        "r3_severely_fragmented": bool(fragmented_r3),
        "fragmentation_policy": "soft_prior_not_veto",
        "selection_policy": "r3_corridor_graph_left_heading_v7",
        "fusion_map_candidate_enabled": fusion_map_candidate_enabled,
        "kerama_raster_polarity_calibrated":
            kerama_raster_polarity_calibrated,
        "candidate_results": [],
    }
    selected_observation: Optional[dict[str, Any]] = None
    authoritative_graph_diagnostics: list[dict[str, Any]] = []
    alignment: dict[str, Any] = {
        "accepted": False,
        "trajectory": [],
        "diagnostics": {
            "engine": "floorplan_constraint_engine_v10_graph_map_matching",
            "map_id": map_id,
            "accepted": False,
            "reason": "no_candidates_evaluated",
        },
    }

    try:
        engine = get_floorplan_engine(map_id)
        reference_point = context.get("reference_point")
        direction_point = context.get("direction_point")
        anchor_source = "request"
        if (
            (not reference_point or not direction_point)
            and engine.config.default_anchor_reference_pixels is not None
            and engine.config.default_anchor_direction_pixels is not None
        ):
            reference_pixels = engine.config.default_anchor_reference_pixels
            direction_pixels = engine.config.default_anchor_direction_pixels
            reference_point = {
                "x": reference_pixels[0] / engine.config.width * 100.0,
                "y": reference_pixels[1] / engine.config.height * 100.0,
            }
            direction_point = {
                "x": direction_pixels[0] / engine.config.width * 100.0,
                "y": direction_pixels[1] / engine.config.height * 100.0,
            }
            anchor_source = engine.config.default_anchor_source or "floorplan_default_start"
        evaluated: list[dict[str, Any]] = []
        accepted: list[dict[str, Any]] = []
        selected_tier: Optional[int] = None
        for tier in (0, 1, 2):
            tier_observations = [
                observation for observation in observations
                if int(observation.get("selection_tier", 0)) == tier
            ]
            if not tier_observations:
                continue
            tier_evaluated: list[dict[str, Any]] = []
            for observation in tier_observations:
                candidate_alignment = engine.align(
                    observation["points"],
                    reference_point,
                    direction_point,
                    timestamps=observation.get("timestamps"),
                    coordinate_convention=str(observation["coordinate_convention"]),
                    allow_safe_shape_fallback=(
                        tier == 0
                    ),
                    allow_low_net_progress=bool(
                        observation.get("loop_closure_verified", False)
                    ),
                    observation_policy=(
                        "independent"
                        if observation["source"] == "lingbot_independent"
                        else "authoritative"
                    ),
                    loop_closure_verified=bool(
                        observation.get("loop_closure_verified", False)
                    ),
                    allow_independent_corridor_recovery=bool(
                        observation["source"] == "lingbot_independent"
                        and observation.get("fusion_supported", False)
                    ),
                    yaw_offsets_degrees=observation.get(
                        "yaw_offsets_degrees", STANDARD_YAW_OFFSETS_DEGREES
                    ),
                )
                diag = dict(candidate_alignment.get("diagnostics") or {})
                diag["anchor_source"] = anchor_source
                if observation["source"] != "lingbot_independent":
                    authoritative_graph_diagnostics.append(diag)
                candidate_alignment = {**candidate_alignment, "diagnostics": diag}
                if (
                    observation["source"] == "lingbot_independent"
                    and candidate_alignment.get("accepted")
                    and _finite_float(diag.get("corrected_collision_ratio"), 0.0) > 0.0
                ):
                    diag.update({
                        "accepted": False,
                        "reason": "independent_residual_collision",
                        "rejection_reasons": ["independent_residual_collision"],
                    })
                    candidate_alignment = {
                        **candidate_alignment,
                        "accepted": False,
                        "trajectory": [],
                        "diagnostics": diag,
                    }
                constrained_score = _finite_float(
                    diag.get("constrained_score"), float("inf")
                )
                correction_p95 = _finite_float(
                    diag.get("correction_p95_meters"), float("inf")
                )
                length_ratio = max(
                    _finite_float(diag.get("length_ratio"), 1.0), 1e-9
                )
                source_prior = float(observation["source_prior"])
                selection_score = (
                    constrained_score + source_prior
                    if candidate_alignment.get("accepted")
                    else float("inf")
                )
                evaluated_item = {
                    **observation,
                    "alignment": candidate_alignment,
                    "selection_score": selection_score,
                    "correction_p95": correction_p95,
                    "length_ratio": length_ratio,
                }
                evaluated.append(evaluated_item)
                tier_evaluated.append(evaluated_item)
                source_selection["candidate_results"].append({
                    "source": observation["source"],
                    "variant": observation["variant"],
                    "selection_tier": tier,
                    "accepted": bool(candidate_alignment.get("accepted")),
                    "reason": diag.get("reason"),
                    "rejection_reasons": diag.get("rejection_reasons") or [],
                    "constrained_score": (
                        round(constrained_score, 6)
                        if math.isfinite(constrained_score)
                        else None
                    ),
                    "source_prior": round(source_prior, 4),
                    "selection_score": (
                        round(selection_score, 6)
                        if math.isfinite(selection_score)
                        else None
                    ),
                    "correction_p95_meters": (
                        round(correction_p95, 3)
                        if math.isfinite(correction_p95)
                        else None
                    ),
                    "length_ratio": (
                        round(length_ratio, 5)
                        if math.isfinite(length_ratio)
                        else None
                    ),
                    "estimated_speed_mps": diag.get("estimated_speed_mps"),
                    "estimated_speed_ratio": diag.get("estimated_speed_ratio"),
                    "observation_stabilization": observation.get(
                        "observation_stabilization"
                    ),
                    "fusion_supported": bool(
                        observation.get("fusion_supported", False)
                    ),
                    "timestamp_provenance": observation.get("timestamp_provenance"),
                    "loop_closure_verified": bool(
                        observation.get("loop_closure_verified", False)
                    ),
                    "corrected_progress": diag.get("corrected_progress"),
                    "topology_recovery_enabled": bool(
                        diag.get("topology_recovery_enabled", False)
                    ),
                    "topology_recovery_attempted": int(
                        diag.get("topology_recovery_attempted", 0) or 0
                    ),
                    "topology_recovery_accepted": int(
                        diag.get("topology_recovery_accepted", 0) or 0
                    ),
                    "graph_first_candidate": diag.get("graph_first_candidate"),
                })
            accepted = [
                item for item in tier_evaluated
                if item["alignment"].get("accepted")
            ]
            if accepted:
                selected_tier = tier
                for skipped in observations:
                    if int(skipped.get("selection_tier", 0)) <= tier:
                        continue
                    source_selection["candidate_results"].append({
                        "source": skipped["source"],
                        "variant": skipped["variant"],
                        "selection_tier": int(skipped["selection_tier"]),
                        "accepted": False,
                        "skipped": True,
                        "reason": "authoritative_candidate_accepted",
                        "rejection_reasons": [],
                        "observation_stabilization": skipped.get(
                            "observation_stabilization"
                        ),
                    })
                break

        if accepted:
            accepted.sort(key=lambda item: (
                float(item["selection_score"]),
                float(item["correction_p95"]),
                abs(math.log(max(float(item["length_ratio"]), 1e-9))),
                str(item["source"]),
                str(item["variant"]),
            ))
            selected_observation = accepted[0]
            alignment = selected_observation["alignment"]
            selected_source = str(selected_observation["source"])
            source_selection.update({
                "selected": selected_source,
                "selected_variant": selected_observation["variant"],
                "selected_tier": selected_tier,
                "reason": (
                    "authoritative_candidate_accepted"
                    if selected_tier == 0
                    else "independent_fallback_after_authoritative_rejection"
                ),
                "selection_score": round(
                    float(selected_observation["selection_score"]), 6
                ),
            })
        elif evaluated:
            # Preserve the most informative rejection diagnostics, but do not
            # claim that its observer was selected for the map.
            rejection_priority = {
                "insufficient_independent_net_progress": 0,
                "map_correction_exceeds_observation_budget": 1,
                "constraint_solution_not_found": 2,
                "no_walkable_start": 3,
                "missing_start_or_direction": 4,
            }
            evaluated.sort(key=lambda item: (
                rejection_priority.get(
                    str((item["alignment"].get("diagnostics") or {}).get("reason")),
                    10,
                ),
                float(item["correction_p95"]),
            ))
            alignment = evaluated[0]["alignment"]
            source_selection["best_rejected_candidate"] = {
                "source": evaluated[0]["source"],
                "variant": evaluated[0]["variant"],
                "reason": (
                    evaluated[0]["alignment"].get("diagnostics") or {}
                ).get("reason"),
            }
    except Exception as exc:
        alignment = {
            "accepted": False,
            "trajectory": [],
            "diagnostics": {
                "engine": "floorplan_constraint_engine_v10_graph_map_matching",
                "map_id": map_id,
                "accepted": False,
                "reason": "engine_error",
                "error": str(exc),
            },
        }
        source_selection["reason"] = "floorplan_engine_error"

    updated = dict(result)
    diagnostics = dict(alignment.get("diagnostics") or {})
    # When every observation is rejected, ``alignment`` above intentionally
    # points at the most informative rejection.  That candidate may be an
    # independent LingBot observation, for which graph recovery is disabled.
    # Never let its candidate-local flag masquerade as the state of the
    # authoritative R3 graph matcher for the whole production run.
    reported_candidate_topology_enabled = bool(
        diagnostics.get("topology_recovery_enabled", False)
    )
    authoritative_topology_enabled = any(
        bool(item.get("topology_recovery_enabled", False))
        for item in authoritative_graph_diagnostics
    )
    authoritative_topology_attempted = sum(
        int(item.get("topology_recovery_attempted", 0) or 0)
        for item in authoritative_graph_diagnostics
    )
    authoritative_topology_accepted = sum(
        int(item.get("topology_recovery_accepted", 0) or 0)
        for item in authoritative_graph_diagnostics
    )
    diagnostics["reported_candidate_topology_recovery_enabled"] = (
        reported_candidate_topology_enabled
    )
    diagnostics["topology_recovery_enabled"] = (
        authoritative_topology_enabled
        if authoritative_graph_diagnostics
        else reported_candidate_topology_enabled
    )
    diagnostics["authoritative_topology_recovery_attempted"] = (
        authoritative_topology_attempted
    )
    diagnostics["authoritative_topology_recovery_accepted"] = (
        authoritative_topology_accepted
    )
    diagnostics["authoritative_graph_matching"] = {
        "candidate_count": len(authoritative_graph_diagnostics),
        "configured": any(
            bool(item.get("graph_map_matching_configured", False))
            for item in authoritative_graph_diagnostics
        ),
        "enabled": authoritative_topology_enabled,
        "attempted": authoritative_topology_attempted,
        "accepted": authoritative_topology_accepted,
    }
    graph_first_sources = [
        item
        for item in source_selection["candidate_results"]
        if item.get("source") != "lingbot_independent"
        and isinstance(item.get("graph_first_candidate"), dict)
        and item["graph_first_candidate"].get("available")
    ]
    graph_first_source = max(
        graph_first_sources,
        key=lambda item: float(
            item["graph_first_candidate"].get(
                "confirmed_source_fraction", 0.0
            )
        ),
        default=None,
    )
    if not alignment.get("accepted") and graph_first_source is not None:
        graph_first = graph_first_source["graph_first_candidate"]
        updated["graph_first_trajectory"] = graph_first["trajectory"]
        updated["graph_first_segments"] = graph_first["segments"]
        updated["graph_first_uncertainty"] = {
            "marker": graph_first.get("uncertainty_marker"),
            "competing_next_edges": graph_first.get(
                "competing_next_edges", []
            ),
        }
        updated["graph_first_metadata"] = {
            "map_id": map_id,
            "source": graph_first_source.get("source"),
            "variant": graph_first_source.get("variant"),
            "strict_map_trajectory_accepted": False,
            "confirmed_source_fraction":
                graph_first.get("confirmed_source_fraction"),
            "uncertainty_source_fraction_start":
                graph_first.get("uncertainty_source_fraction_start"),
            "confirmed_segments": graph_first.get("confirmed_segments", 0),
            "uncertain_segments": graph_first.get("uncertain_segments", 0),
            "policy": graph_first.get("policy"),
            "uncertainty_marker": graph_first.get("uncertainty_marker"),
            "competing_next_edge_count": len(
                graph_first.get("competing_next_edges") or []
            ),
        }
        diagnostics["graph_first_output_published"] = True
        diagnostics["graph_first_output"] = updated["graph_first_metadata"]
    else:
        diagnostics["graph_first_output_published"] = False
    selected_source = (
        str(selected_observation["source"])
        if selected_observation is not None and alignment.get("accepted")
        else None
    )
    diagnostics["trajectory_observation_source"] = selected_source
    diagnostics["observation_source_selection"] = source_selection
    diagnostics["constraint_revision"] = FLOORPLAN_CONSTRAINT_REVISION
    updated["floorplan_constraint"] = diagnostics
    stats["floorplan_constraint"] = diagnostics
    stats["map_matching_applied"] = bool(alignment.get("accepted"))
    stats["floorplan_id"] = map_id

    if alignment.get("accepted") and selected_observation is not None:
        mapped = alignment["trajectory"]
        selected_points = selected_observation["points"]
        selected_timestamps = selected_observation.get("timestamps")
        source_turns = (
            []
            if selected_source == "lingbot_independent"
            else result.get("turn_points")
        )
        map_turns = _map_turn_points(
            source_turns,
            selected_points,
            mapped,
            meters_per_pixel=engine.config.meters_per_pixel,
            timestamps=selected_timestamps,
        )
        updated["map_trajectory"] = mapped
        updated["map_trajectory_timestamps_seconds"] = _mapped_timestamps(
            selected_points, mapped, selected_timestamps
        )
        updated["map_trajectory_source_fractions"] = [
            round(float(value), 8)
            for value in _trajectory_fractions(_normalise_points(mapped))
        ]
        updated["map_turn_points"] = map_turns
        updated["final_turn_points"] = (
            []
            if selected_source == "lingbot_independent"
            else (map_turns or result.get("turn_points") or [])
        )
        map_distance = _finite_float(
            diagnostics.get("published_length_meters"),
            _finite_float(diagnostics.get("estimated_length_meters"))
            * _finite_float(diagnostics.get("length_ratio"), 1.0),
        )
        stats["map_trajectory_points"] = len(mapped)
        stats["map_confidence"] = diagnostics.get("confidence")
        stats["map_observation_source"] = selected_source
        stats["map_distance_meters"] = round(map_distance, 3)
        stats["estimated_distance"] = round(map_distance, 3)
        updated["map_metadata"] = {
            "map_id": map_id,
            "plan_width": diagnostics.get("plan_width"),
            "plan_height": diagnostics.get("plan_height"),
            "meters_per_pixel": diagnostics.get("meters_per_pixel"),
            "person_radius_meters": diagnostics.get("person_radius_meters"),
            "source": "fixed_floorplan_constraint_engine",
            "trajectory_observation_source": selected_source,
            "observation_variant": selected_observation["variant"],
        }
    else:
        updated.pop("map_trajectory", None)
        updated.pop("map_trajectory_timestamps_seconds", None)
        updated.pop("map_trajectory_source_fractions", None)
        updated.pop("map_turn_points", None)
        updated.pop("map_metadata", None)
        updated["final_turn_points"] = result.get("turn_points") or []

    updated["processing_stats"] = stats
    return updated
