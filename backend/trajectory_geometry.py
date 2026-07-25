"""Shared deterministic geometry metrics for 2D and 3D trajectories.

All cross-trajectory comparisons use one global, non-reflecting similarity.
They deliberately do not use map constraints or mutate either input curve.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


DEFAULT_ACCEPTANCE_THRESHOLDS = {
    "maximum_normalized_frechet": 0.08,
    "maximum_normalized_chamfer": 0.04,
    "minimum_turn_sequence_agreement": 0.70,
    "minimum_local_direction_agreement": 0.75,
    "maximum_segment_length_log_rmse": 0.50,
    "minimum_endpoint_ratio": 0.55,
    "maximum_endpoint_ratio": 1.80,
    "minimum_span_ratio": 0.55,
    "maximum_span_ratio": 1.80,
    "maximum_curvature_distribution_distance": 0.30,
}


def normalise_points(value: Any) -> np.ndarray:
    """Return finite 2D/3D points without inventing replacements.

    Mixed-dimensional inputs are rejected because silently dropping Z from only
    part of a trajectory would make path length and shape metrics inconsistent.
    Dictionary points may use ``x/y`` and optional ``z``.
    """
    if not isinstance(value, (list, tuple, np.ndarray)):
        return np.empty((0, 2), dtype=np.float64)
    points: list[list[float]] = []
    dimensions: int | None = None
    for item in value:
        if isinstance(item, Mapping):
            item = (
                [item.get("x"), item.get("y"), item.get("z")]
                if item.get("z") is not None
                else [item.get("x"), item.get("y")]
            )
        if not isinstance(item, (list, tuple, np.ndarray)) or len(item) < 2:
            continue
        current_dimensions = 3 if len(item) >= 3 else 2
        if dimensions is None:
            dimensions = current_dimensions
        if current_dimensions != dimensions:
            return np.empty((0, dimensions), dtype=np.float64)
        try:
            point = [float(item[index]) for index in range(dimensions)]
        except (TypeError, ValueError):
            continue
        if np.isfinite(point).all():
            points.append(point)
    return np.asarray(points, dtype=np.float64).reshape(-1, dimensions or 2)


def _resample_arc(points: np.ndarray, count: int = 256) -> np.ndarray:
    dimensions = points.shape[1] if points.ndim == 2 else 2
    if len(points) == 0:
        return np.empty((0, dimensions), dtype=np.float64)
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(steps)))
    if arc[-1] <= 1e-12:
        return np.repeat(points[:1], count, axis=0)
    keep = np.concatenate(([True], np.diff(arc) > 1e-12))
    target = np.linspace(0.0, float(arc[-1]), count)
    return np.column_stack([
        np.interp(target, arc[keep], points[keep, dimension])
        for dimension in range(dimensions)
    ])


def _resample_index(points: np.ndarray, count: int = 256) -> np.ndarray:
    if len(points) == 0:
        return points.copy()
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    source = np.linspace(0.0, 1.0, len(points))
    target = np.linspace(0.0, 1.0, count)
    return np.column_stack([
        np.interp(target, source, points[:, dimension])
        for dimension in range(points.shape[1])
    ])


def _turn_angles(points: np.ndarray) -> np.ndarray:
    vectors = np.diff(points, axis=0)
    if len(vectors) < 2:
        return np.empty(0, dtype=np.float64)
    left, right = vectors[:-1], vectors[1:]
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    angles = np.arccos(np.clip(
        np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12),
        -1.0,
        1.0,
    ))
    if points.shape[1] == 2:
        cross = left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]
        angles *= np.sign(cross)
    return angles


def _turns(points: np.ndarray, minimum_degrees: float = 18.0) -> list[dict[str, float | int]]:
    sampled = _resample_arc(points, min(256, max(32, len(points))))
    if len(sampled) < 5:
        return []
    window = max(2, len(sampled) // 64)
    result: list[dict[str, float | int]] = []
    if sampled.shape[1] == 2:
        headings = []
        for index in range(len(sampled)):
            left = max(0, index - window)
            right = min(len(sampled) - 1, index + window)
            vector = sampled[right] - sampled[left]
            headings.append(math.atan2(float(vector[1]), float(vector[0])))
        unwrapped = np.unwrap(np.asarray(headings))
        delta = np.degrees(
            unwrapped[np.minimum(np.arange(len(sampled)) + window, len(sampled) - 1)]
            - unwrapped[np.maximum(np.arange(len(sampled)) - window, 0)]
        )
    else:
        delta = np.zeros(len(sampled), dtype=np.float64)
        local = np.degrees(_turn_angles(sampled))
        delta[1:-1] = local
    candidates = np.flatnonzero(np.abs(delta) >= minimum_degrees)
    if not len(candidates):
        return result
    groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1) + 1)
    for group in groups:
        index = int(group[np.argmax(np.abs(delta[group]))])
        angle = float(delta[index])
        result.append({
            "sample_index": index,
            "fraction": round(index / max(len(sampled) - 1, 1), 6),
            "angle_degrees": round(angle, 3),
            "sign": 0 if sampled.shape[1] == 3 else (1 if angle > 0 else -1),
        })
    return result


def trajectory_metrics(
    value: Any,
    *,
    units_per_coordinate: float = 1.0,
    unit_name: str = "source_unit",
) -> dict[str, Any]:
    """Return the same core metrics for a 2D or 3D polyline."""
    points = normalise_points(value)
    if len(points) == 0:
        return {"available": False, "point_count": 0, "unit": unit_name}
    scale = float(units_per_coordinate)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1) * scale
    length = float(steps.sum())
    endpoint = float(np.linalg.norm(points[-1] - points[0]) * scale)
    extent = np.ptp(points, axis=0) * scale
    span = float(np.linalg.norm(extent))
    vectors = np.diff(points, axis=0)
    meaningful = vectors[np.linalg.norm(vectors, axis=1) * scale >= 0.25]
    reverse_ratio = 0.0
    if len(meaningful) >= 2:
        left, right = meaningful[:-1], meaningful[1:]
        cosine = np.sum(left * right, axis=1) / np.maximum(
            np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1),
            1e-12,
        )
        reverse_ratio = float(np.mean(
            np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) >= 135.0
        ))
    radius = np.linalg.norm(points - points[0], axis=1) * scale
    turns = _turns(points)
    result: dict[str, Any] = {
        "available": True,
        "dimensions": int(points.shape[1]),
        "point_count": int(len(points)),
        "unit": unit_name,
        "path_length": round(length, 6),
        "endpoint_displacement": round(endpoint, 6),
        "net_progress_ratio": round(endpoint / max(length, 1e-12), 6),
        "span_ratio": round(span / max(length, 1e-12), 6),
        "bbox_extents": [round(float(value), 6) for value in extent],
        "bbox_width": round(float(extent[0]), 6),
        "bbox_height": round(float(extent[1]), 6),
        "bbox_area": round(float(extent[0] * extent[1]), 6),
        "tortuosity": round(length / max(endpoint, 1e-12), 6),
        "step_p50": round(float(np.percentile(steps, 50)), 6) if len(steps) else 0.0,
        "step_p90": round(float(np.percentile(steps, 90)), 6) if len(steps) else 0.0,
        "step_p99": round(float(np.percentile(steps, 99)), 6) if len(steps) else 0.0,
        "sharp_reverse_ratio": round(reverse_ratio, 6),
        "turn_count": len(turns),
        "turns": turns,
        "near_start_fraction": {
            str(distance): round(float(np.mean(radius <= distance)), 6)
            for distance in (1, 2, 5)
        },
    }
    if points.shape[1] == 3:
        result["bbox_depth"] = round(float(extent[2]), 6)
        result["bbox_volume"] = round(float(np.prod(extent)), 6)
    return result


def _fit_similarity(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    reference_center = reference.mean(axis=0)
    candidate_center = candidate.mean(axis=0)
    reference_centered = reference - reference_center
    candidate_centered = candidate - candidate_center
    covariance = candidate_centered.T @ reference_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    denominator = float(np.sum(candidate_centered * candidate_centered))
    scale = (
        float(np.sum((candidate_centered @ rotation) * reference_centered))
        / denominator
        if denominator > 1e-12 else 1.0
    )
    # A negative scale is an orientation reversal in odd dimensions. Similarity
    # scale is physical magnitude, so keep it non-negative in both 2D and 3D.
    scale = max(scale, 1e-12)
    aligned = candidate_centered @ rotation * scale + reference_center
    return aligned, scale, rotation


def _discrete_frechet(first: np.ndarray, second: np.ndarray) -> float:
    cache = np.full((len(first), len(second)), np.nan, dtype=np.float64)
    for i in range(len(first)):
        for j in range(len(second)):
            distance = float(np.linalg.norm(first[i] - second[j]))
            if i == 0 and j == 0:
                cache[i, j] = distance
            elif i == 0:
                cache[i, j] = max(cache[i, j - 1], distance)
            elif j == 0:
                cache[i, j] = max(cache[i - 1, j], distance)
            else:
                cache[i, j] = max(
                    min(cache[i - 1, j], cache[i - 1, j - 1], cache[i, j - 1]),
                    distance,
                )
    return float(cache[-1, -1])


def _curvature_distribution_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_angles = np.abs(_turn_angles(first))
    second_angles = np.abs(_turn_angles(second))
    bins = np.linspace(0.0, math.pi, 19)
    first_hist, _ = np.histogram(first_angles, bins=bins, density=False)
    second_hist, _ = np.histogram(second_angles, bins=bins, density=False)
    first_cdf = np.cumsum(first_hist) / max(int(first_hist.sum()), 1)
    second_cdf = np.cumsum(second_hist) / max(int(second_hist.sum()), 1)
    return float(np.mean(np.abs(first_cdf - second_cdf)))


def _straight_run_preservation(first: np.ndarray, second: np.ndarray) -> float:
    """Measure whether sustained straight reference spans remain straight.

    Direction and chirality are intentionally irrelevant here: PGO may correct
    a wrong raw turn, but it may not turn a confirmed straight aisle into a
    curved or zig-zag section.
    """
    first_curvature = np.abs(_turn_angles(first))
    second_curvature = np.abs(_turn_angles(second))
    straight = first_curvature <= math.radians(8.0)
    minimum_run = max(4, len(straight) // 20)
    confirmed = np.zeros(len(straight), dtype=bool)
    indices = np.flatnonzero(straight)
    if len(indices):
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        for group in groups:
            if len(group) >= minimum_run:
                confirmed[group] = True
    if not confirmed.any():
        return 1.0
    return float(np.mean(second_curvature[confirmed] <= math.radians(15.0)))


def compare_trajectories(
    reference: Any,
    candidate: Any,
    *,
    sample_count: int = 128,
) -> dict[str, Any]:
    """Compare trajectory shapes after one non-reflecting best-fit similarity."""
    raw = normalise_points(reference)
    other = normalise_points(candidate)
    if (
        len(raw) < 2
        or len(other) < 2
        or raw.shape[1] != other.shape[1]
        or sample_count < 8
    ):
        return {
            "available": False,
            "reason": "insufficient_or_incompatible_points",
        }
    raw_arc = _resample_arc(raw, sample_count)
    other_arc = _resample_arc(other, sample_count)
    aligned, scale, rotation = _fit_similarity(raw_arc, other_arc)
    distances = np.linalg.norm(aligned - raw_arc, axis=1)
    raw_length = float(np.linalg.norm(np.diff(raw_arc, axis=0), axis=1).sum())
    normalization = max(raw_length, 1e-12)

    pairwise = np.linalg.norm(
        raw_arc[:, None, :] - aligned[None, :, :], axis=2
    )
    chamfer = 0.5 * (
        float(np.mean(np.min(pairwise, axis=1)))
        + float(np.mean(np.min(pairwise, axis=0)))
    )
    frechet = _discrete_frechet(raw_arc, aligned)

    raw_vectors = np.diff(raw_arc, axis=0)
    aligned_vectors = np.diff(aligned, axis=0)
    direction_cosine = np.sum(raw_vectors * aligned_vectors, axis=1) / np.maximum(
        np.linalg.norm(raw_vectors, axis=1)
        * np.linalg.norm(aligned_vectors, axis=1),
        1e-12,
    )
    local_direction = float(np.mean((np.clip(direction_cosine, -1.0, 1.0) + 1.0) * 0.5))

    raw_index = _resample_index(raw, sample_count)
    other_index = _resample_index(other, sample_count)
    aligned_index, _, _ = _fit_similarity(raw_index, other_index)
    raw_segment = np.linalg.norm(np.diff(raw_index, axis=0), axis=1)
    candidate_segment = np.linalg.norm(np.diff(aligned_index, axis=0), axis=1)
    valid = (raw_segment > 1e-9) & (candidate_segment > 1e-9)
    log_ratio = (
        np.log(candidate_segment[valid] / raw_segment[valid])
        if valid.any() else np.asarray([math.inf])
    )
    segment_log_rmse = float(np.sqrt(np.mean(np.square(log_ratio))))

    raw_turn = _turn_angles(raw_arc)
    candidate_turn = _turn_angles(aligned)
    if raw.shape[1] == 2:
        turn_error = np.abs(raw_turn - candidate_turn)
        turn_sequence = float(np.mean((np.cos(turn_error) + 1.0) * 0.5))
        significant = (np.abs(raw_turn) >= math.radians(8.0)) | (
            np.abs(candidate_turn) >= math.radians(8.0)
        )
        sign_agreement = (
            float(np.mean(np.sign(raw_turn[significant]) == np.sign(candidate_turn[significant])))
            if significant.any() else 1.0
        )
        turn_sequence *= sign_agreement
    else:
        significant = (np.abs(raw_turn) >= math.radians(8.0)) | (
            np.abs(candidate_turn) >= math.radians(8.0)
        )
        turn_sequence = (
            float(np.mean(
                (
                    np.cos(np.abs(
                        raw_turn[significant] - candidate_turn[significant]
                    ))
                    + 1.0
                )
                * 0.5
            ))
            if significant.any()
            else 1.0
        )
        sign_agreement = None

    raw_metrics = trajectory_metrics(raw)
    candidate_metrics = trajectory_metrics(other)
    endpoint_ratio = (
        float(candidate_metrics["net_progress_ratio"])
        / max(float(raw_metrics["net_progress_ratio"]), 1e-12)
    )
    span_ratio = (
        float(candidate_metrics["span_ratio"])
        / max(float(raw_metrics["span_ratio"]), 1e-12)
    )
    curvature_distance = _curvature_distribution_distance(raw_arc, aligned)
    straight_preservation = _straight_run_preservation(raw_arc, aligned)
    return {
        "available": True,
        "dimensions": int(raw.shape[1]),
        "fit": "arc_resampled_non_reflecting_similarity",
        "sample_count": sample_count,
        "scale_to_reference": round(scale, 8),
        "rotation_determinant": round(float(np.linalg.det(rotation)), 8),
        "pointwise_deviation": {
            "mean": round(float(np.mean(distances)), 6),
            "p50": round(float(np.percentile(distances, 50)), 6),
            "p95": round(float(np.percentile(distances, 95)), 6),
            "max": round(float(np.max(distances)), 6),
        },
        "normalized_frechet_distance": round(frechet / normalization, 6),
        "normalized_chamfer_distance": round(chamfer / normalization, 6),
        "turn_sequence_agreement": round(turn_sequence, 6),
        "turn_sign_agreement": (
            round(sign_agreement, 6) if sign_agreement is not None else None
        ),
        "local_direction_agreement": round(local_direction, 6),
        "segment_length_log_rmse": round(segment_log_rmse, 6),
        "endpoint_progress_ratio": round(endpoint_ratio, 6),
        "spatial_span_ratio": round(span_ratio, 6),
        "curvature_distribution_distance": round(curvature_distance, 6),
        "straight_run_preservation": round(straight_preservation, 6),
        "reference_metrics": raw_metrics,
        "candidate_metrics": candidate_metrics,
    }


def trajectory_acceptance(
    reference: Any,
    candidate: Any,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply explicit, caller-overridable geometry gates to a candidate.

    ``context["thresholds"]`` may override any default. A verified closed route
    may set ``verified_loop_closure=True`` to disable the endpoint-progress gate;
    all other shape gates remain active.
    """
    context = context or {}
    thresholds = dict(DEFAULT_ACCEPTANCE_THRESHOLDS)
    supplied = context.get("thresholds")
    if isinstance(supplied, Mapping):
        for key in thresholds:
            if key in supplied:
                try:
                    thresholds[key] = float(supplied[key])
                except (TypeError, ValueError):
                    pass
    comparison = compare_trajectories(
        reference,
        candidate,
        sample_count=int(context.get("sample_count", 128)),
    )
    if not comparison.get("available"):
        return {
            "accepted": False,
            "rejection_reasons": ["comparison_unavailable"],
            "thresholds": thresholds,
            "comparison": comparison,
        }
    reasons = []
    checks = (
        ("normalized_frechet_distance", "maximum_normalized_frechet", False),
        ("normalized_chamfer_distance", "maximum_normalized_chamfer", False),
        ("turn_sequence_agreement", "minimum_turn_sequence_agreement", True),
        ("local_direction_agreement", "minimum_local_direction_agreement", True),
        ("segment_length_log_rmse", "maximum_segment_length_log_rmse", False),
        (
            "curvature_distribution_distance",
            "maximum_curvature_distribution_distance",
            False,
        ),
    )
    for metric, threshold, is_minimum in checks:
        value = float(comparison[metric])
        limit = float(thresholds[threshold])
        failed = value < limit if is_minimum else value > limit
        if failed:
            reasons.append(f"{metric}_out_of_bounds")
    if not bool(context.get("verified_loop_closure", False)):
        endpoint = float(comparison["endpoint_progress_ratio"])
        if not (
            thresholds["minimum_endpoint_ratio"]
            <= endpoint
            <= thresholds["maximum_endpoint_ratio"]
        ):
            reasons.append("endpoint_progress_ratio_out_of_bounds")
    span = float(comparison["spatial_span_ratio"])
    if not (
        thresholds["minimum_span_ratio"]
        <= span
        <= thresholds["maximum_span_ratio"]
    ):
        reasons.append("spatial_span_ratio_out_of_bounds")
    return {
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "verified_loop_closure": bool(context.get("verified_loop_closure", False)),
        "thresholds": thresholds,
        "comparison": comparison,
    }


def deviation_from_reference(reference: Any, candidate: Any) -> dict[str, Any]:
    """Backward-compatible baseline summary backed by the shared comparison."""
    comparison = compare_trajectories(reference, candidate)
    if not comparison.get("available"):
        return comparison
    deviation = comparison["pointwise_deviation"]
    raw_length = float(comparison["reference_metrics"]["path_length"])
    return {
        "available": True,
        "fit": comparison["fit"],
        "sample_count": comparison["sample_count"],
        "scale_to_raw": comparison["scale_to_reference"],
        **deviation,
        "p95_over_raw_length": round(
            float(deviation["p95"]) / max(raw_length, 1e-12), 6
        ),
    }
