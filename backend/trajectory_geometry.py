"""Shared geometry metrics and comparative acceptance for trajectories."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def _points(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        return np.empty((0, 2), dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return points
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]
    return points[keep]


def _resample(points: np.ndarray, count: int = 256) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(segment)]
    if arc[-1] <= 1e-12:
        return np.repeat(points[:1], count, axis=0)
    target = np.linspace(0.0, arc[-1], count)
    return np.column_stack([
        np.interp(target, arc, points[:, axis])
        for axis in range(points.shape[1])
    ])


def _turn_angles(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.empty(0, dtype=np.float64)
    delta = np.diff(points, axis=0)
    length = np.linalg.norm(delta, axis=1)
    valid = (length[:-1] > 1e-12) & (length[1:] > 1e-12)
    if not valid.any():
        return np.empty(0, dtype=np.float64)
    first = delta[:-1][valid] / length[:-1][valid, None]
    second = delta[1:][valid] / length[1:][valid, None]
    cosine = np.clip(np.sum(first * second, axis=1), -1.0, 1.0)
    if points.shape[1] == 2:
        signed = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        return np.arctan2(signed, cosine)
    cross = np.linalg.norm(np.cross(first, second), axis=1)
    return np.arctan2(cross, cosine)


def _sharp_reverse_ratio(points: np.ndarray) -> float:
    angles = np.abs(_turn_angles(_resample(points, min(max(len(points), 32), 256))))
    return float(np.mean(angles >= math.radians(135.0))) if len(angles) else 0.0


def trajectory_metrics(points: Any) -> dict[str, Any]:
    """Return one stable metric set for a 2-D or 3-D polyline."""
    source = _points(points)
    if len(source) < 2:
        return {"available": False, "point_count": int(len(source))}
    steps = np.linalg.norm(np.diff(source, axis=0), axis=1)
    length = float(steps.sum())
    endpoint = float(np.linalg.norm(source[-1] - source[0]))
    extent = np.ptp(source, axis=0)
    bbox_diagonal = float(np.linalg.norm(extent))
    positive = steps[steps > 1e-12]
    turns = _turn_angles(_resample(source, min(max(len(source), 32), 256)))
    return {
        "available": True,
        "point_count": int(len(source)),
        "dimensions": int(source.shape[1]),
        "path_length": length,
        "endpoint_displacement": endpoint,
        "net_progress_ratio": endpoint / max(length, 1e-12),
        "bbox_diagonal": bbox_diagonal,
        "span_ratio": bbox_diagonal / max(length, 1e-12),
        "bbox_extent": extent.tolist(),
        "bbox_area": float(np.prod(np.sort(extent)[-2:])) if len(extent) >= 2 else 0.0,
        "tortuosity": length / max(endpoint, 1e-12),
        "step_p50": float(np.percentile(positive, 50)) if len(positive) else 0.0,
        "step_p90": float(np.percentile(positive, 90)) if len(positive) else 0.0,
        "step_p99": float(np.percentile(positive, 99)) if len(positive) else 0.0,
        "sharp_reverse_ratio": _sharp_reverse_ratio(source),
        "turn_count": int(np.sum(np.abs(turns) >= math.radians(18.0))),
        "absolute_turn_p50_degrees": (
            float(np.degrees(np.percentile(np.abs(turns), 50))) if len(turns) else 0.0
        ),
        "absolute_turn_p90_degrees": (
            float(np.degrees(np.percentile(np.abs(turns), 90))) if len(turns) else 0.0
        ),
    }


def _early_heading(points: np.ndarray, fraction: float = 0.08) -> np.ndarray | None:
    if len(points) < 2:
        return None
    sampled = _resample(points, 64)
    index = min(len(sampled) - 1, max(1, int(round((len(sampled) - 1) * fraction))))
    heading = sampled[index] - sampled[0]
    norm = float(np.linalg.norm(heading))
    return heading / norm if norm > 1e-12 else None


def align_trajectory_to_anchor(
    reference: Any,
    candidate: Any,
    context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align around a fixed start; direction chooses rotation and polarity.

    Translation is never estimated from centroids.  Reflection is only applied
    when explicitly requested by the coordinate convention.
    """
    options = context or {}
    ref = _points(reference)
    cand = _points(candidate)
    dimensions = min(ref.shape[1], cand.shape[1]) if len(ref) and len(cand) else 0
    if len(ref) < 2 or len(cand) < 2 or dimensions < 2:
        return cand.copy(), {"available": False, "reason": "insufficient_points"}
    ref = ref[:, :dimensions]
    cand = cand[:, :dimensions]
    ref_start = np.asarray(options.get("reference_start", ref[0]), dtype=np.float64)
    cand_start = np.asarray(options.get("candidate_start", cand[0]), dtype=np.float64)
    ref_heading = np.asarray(
        options.get("reference_direction", _early_heading(ref)),
        dtype=np.float64,
    )
    cand_heading = np.asarray(
        options.get("candidate_direction", _early_heading(cand)),
        dtype=np.float64,
    )
    if ref_heading.size < 2 or cand_heading.size < 2:
        return cand.copy(), {"available": False, "reason": "direction_unavailable"}
    ref_heading = ref_heading[:2]
    cand_heading = cand_heading[:2]
    ref_norm = float(np.linalg.norm(ref_heading))
    cand_norm = float(np.linalg.norm(cand_heading))
    if ref_norm <= 1e-12 or cand_norm <= 1e-12:
        return cand.copy(), {"available": False, "reason": "direction_degenerate"}

    reflect = bool(options.get("reflect_y", False))
    centered = cand[:, :2] - cand_start[:2]
    candidate_direction = cand_heading / cand_norm
    reflection = np.asarray([[1.0, 0.0], [0.0, -1.0]]) if reflect else np.eye(2)
    centered = centered @ reflection.T
    candidate_direction = candidate_direction @ reflection.T
    reference_direction = ref_heading / ref_norm
    source_angle = math.atan2(float(candidate_direction[1]), float(candidate_direction[0]))
    target_angle = math.atan2(float(reference_direction[1]), float(reference_direction[0]))
    angle = target_angle - source_angle
    rotation = np.asarray([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    rotated = centered @ rotation.T

    ref_resampled = _resample(ref[:, :2] - ref_start[:2], 256)
    cand_resampled = _resample(rotated, 256)
    scale = float(
        np.sum(cand_resampled * ref_resampled)
        / max(np.sum(cand_resampled * cand_resampled), 1e-12)
    )
    if not math.isfinite(scale) or scale <= 0:
        return cand.copy(), {"available": False, "reason": "scale_degenerate"}
    aligned_2d = rotated * scale + ref_start[:2]
    aligned = cand.copy()
    aligned[:, :2] = aligned_2d
    start_error = float(np.linalg.norm(aligned[0, :2] - ref_start[:2]))
    return aligned, {
        "available": True,
        "mode": "fixed_start_and_direction",
        "scale": scale,
        "rotation_degrees": math.degrees(angle),
        "reflection_applied": reflect,
        "translation_fitted_from_centroid": False,
        "reference_start": ref_start[:2].tolist(),
        "candidate_start": cand_start[:2].tolist(),
        "aligned_start": aligned[0, :2].tolist(),
        "start_error": start_error,
        "direction_error_degrees": 0.0,
    }


def _similarity(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    ref = reference - reference.mean(axis=0)
    cand = candidate - candidate.mean(axis=0)
    ref_norm = float(np.linalg.norm(ref))
    cand_norm = float(np.linalg.norm(cand))
    if ref_norm <= 1e-12 or cand_norm <= 1e-12:
        return candidate.copy(), {"available": False}
    covariance = cand.T @ ref
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    reflection_prevented = False
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
        reflection_prevented = True
    rotated = cand @ rotation
    scale = float(np.sum(rotated * ref) / max(np.sum(rotated * rotated), 1e-12))
    aligned = rotated * scale + reference.mean(axis=0)
    return aligned, {
        "available": True,
        "scale": scale,
        "rotation_determinant": float(np.linalg.det(rotation)),
        "reflection_prevented": reflection_prevented,
    }


def _chamfer(first: np.ndarray, second: np.ndarray) -> float:
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return 0.5 * (float(np.mean(np.min(distances, axis=1))) + float(np.mean(np.min(distances, axis=0))))


def _frechet(first: np.ndarray, second: np.ndarray) -> float:
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


def compare_trajectories(
    reference: Any,
    candidate: Any,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare shapes using a fixed start when an anchor context is supplied."""
    raw_reference = _points(reference)
    raw_candidate = _points(candidate)
    if len(raw_reference) < 2 or len(raw_candidate) < 2:
        return {"available": False, "reason": "insufficient_points"}
    dimensions = min(raw_reference.shape[1], raw_candidate.shape[1])
    count = min(256, max(64, min(len(raw_reference), len(raw_candidate))))
    ref = _resample(raw_reference[:, :dimensions], count)
    cand = _resample(raw_candidate[:, :dimensions], count)
    aligned, transform = (
        align_trajectory_to_anchor(ref, cand, context)
        if context is not None else _similarity(ref, cand)
    )
    if not transform["available"]:
        return {"available": False, "reason": "degenerate_similarity"}
    span = max(float(np.linalg.norm(np.ptp(ref, axis=0))), 1e-12)
    ref_delta = np.diff(ref, axis=0)
    cand_delta = np.diff(aligned, axis=0)
    ref_steps = np.linalg.norm(ref_delta, axis=1)
    cand_steps = np.linalg.norm(cand_delta, axis=1)
    direction_valid = (ref_steps > 1e-12) & (cand_steps > 1e-12)
    direction = (
        np.sum(ref_delta[direction_valid] * cand_delta[direction_valid], axis=1)
        / (ref_steps[direction_valid] * cand_steps[direction_valid])
        if direction_valid.any() else np.asarray([])
    )
    length_log_error = np.log(
        np.maximum(cand_steps, 1e-12) / np.maximum(ref_steps, 1e-12)
    )
    ref_turns = _turn_angles(ref)
    cand_turns = _turn_angles(aligned)
    significant = (np.abs(ref_turns) >= math.radians(18.0)) | (
        np.abs(cand_turns) >= math.radians(18.0)
    )
    turn_agreement = (
        float(np.mean(
            (np.sign(ref_turns[significant]) == np.sign(cand_turns[significant]))
            & (np.abs(ref_turns[significant] - cand_turns[significant]) <= math.radians(35.0))
        ))
        if significant.any() else 1.0
    )
    # Ratios are shape ratios after the same allowed similarity.  Comparing
    # pre-alignment units incorrectly rejected otherwise identical paths whose
    # coordinate scales differed (for example metres versus plan pixels).
    ref_metrics = trajectory_metrics(ref)
    candidate_metrics = trajectory_metrics(aligned)
    straight = np.abs(ref_turns) <= math.radians(8.0)
    straight_preservation = (
        float(np.mean(np.abs(cand_turns[straight]) <= math.radians(15.0)))
        if straight.any() else 1.0
    )
    return {
        "available": True,
        "similarity": transform,
        "normalized_frechet_distance": _frechet(ref, aligned) / span,
        "normalized_chamfer_distance": _chamfer(ref, aligned) / span,
        "turn_sequence_agreement": turn_agreement,
        "local_direction_agreement": float(np.mean(direction)) if len(direction) else 0.0,
        "segment_length_log_rmse": float(np.sqrt(np.mean(length_log_error ** 2))),
        "endpoint_ratio": (
            candidate_metrics["endpoint_displacement"]
            / max(ref_metrics["endpoint_displacement"], 1e-12)
        ),
        "span_ratio": (
            candidate_metrics["bbox_diagonal"]
            / max(ref_metrics["bbox_diagonal"], 1e-12)
        ),
        "curvature_p90_ratio": (
            candidate_metrics["absolute_turn_p90_degrees"]
            / max(ref_metrics["absolute_turn_p90_degrees"], 1e-12)
        ),
        "straight_run_preservation": straight_preservation,
    }


def trajectory_acceptance(
    reference: Any,
    candidate: Any,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply relative geometry gates; closed routes relax endpoint only."""
    context = dict(context or {})
    # Supplying a context selects the fixed-start comparison.  Do not let an
    # improved candidate hide a displaced start through centroid translation.
    comparison = compare_trajectories(reference, candidate, context)
    if not comparison.get("available"):
        return {
            "accepted": False,
            "rejection_reasons": ["trajectory_geometry_comparison_unavailable"],
            "comparison": comparison,
        }
    closed = bool(context.get("verified_loop_closure", False))
    limits = {
        "normalized_frechet_distance": float(context.get("maximum_normalized_frechet", 0.35)),
        "normalized_chamfer_distance": float(context.get("maximum_normalized_chamfer", 0.22)),
        "turn_sequence_agreement": float(context.get("minimum_turn_sequence_agreement", 0.70)),
        "local_direction_agreement": float(context.get("minimum_local_direction_agreement", 0.55)),
        "segment_length_log_rmse": float(context.get("maximum_segment_length_log_rmse", 0.80)),
        "span_ratio": float(context.get("minimum_span_ratio", 0.55)),
    }
    reasons: list[str] = []
    if comparison["normalized_frechet_distance"] > limits["normalized_frechet_distance"]:
        reasons.append("normalized_frechet_out_of_bounds")
    if comparison["normalized_chamfer_distance"] > limits["normalized_chamfer_distance"]:
        reasons.append("normalized_chamfer_out_of_bounds")
    if comparison["turn_sequence_agreement"] < limits["turn_sequence_agreement"]:
        reasons.append("turn_sequence_agreement_out_of_bounds")
    if comparison["local_direction_agreement"] < limits["local_direction_agreement"]:
        reasons.append("local_direction_agreement_out_of_bounds")
    if comparison["segment_length_log_rmse"] > limits["segment_length_log_rmse"]:
        reasons.append("segment_length_log_rmse_out_of_bounds")
    if comparison["span_ratio"] < limits["span_ratio"]:
        reasons.append("spatial_span_ratio_out_of_bounds")
    if not closed and comparison["endpoint_ratio"] < float(
        context.get("minimum_endpoint_ratio", 0.55)
    ):
        reasons.append("endpoint_progress_ratio_out_of_bounds")
    return {
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "comparison": comparison,
        "limits": limits,
        "verified_loop_closure": closed,
    }
