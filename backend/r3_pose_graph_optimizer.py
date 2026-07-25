"""Robust global averaging for R3 relative-pose graphs.

The optimizer deliberately produces a shadow candidate.  It never rewrites
R3 camera artifacts and never changes production geometry by itself.  R3
stores world-to-camera relative measurements with the convention

    W_j = Z_ij @ W_i

whereas its exported camera matrices are camera-to-world.  We invert the
initial cameras, solve rotation and translation synchronization separately,
then invert the candidate back to camera-to-world.

The implementation uses sparse chordal averaging plus graduated Dynamic
Covariance Scaling (DCS).  This keeps runtime bounded on long videos, rejects
inconsistent loop edges, and avoids a fragile dense nonlinear least-squares
problem over every frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import lsmr

try:
    from trajectory_geometry import compare_trajectories, trajectory_metrics
except ImportError:  # pragma: no cover - supports package-style startup
    from backend.trajectory_geometry import compare_trajectories, trajectory_metrics

try:
    from r3_pose_graph import (
        R3_ABSOLUTE_POSE_SPACE,
        R3_CONFIDENCE_SEMANTICS,
        R3_POSE_GRAPH_LEGACY_SCHEMA_VERSION,
        R3_POSE_ENCODING,
        R3_POSE_GRAPH_SCHEMA_VERSION,
        R3_RELATIVE_TRANSFORM_CONVENTION,
    )
except ImportError:  # pragma: no cover - supports package-style startup
    from backend.r3_pose_graph import (
        R3_ABSOLUTE_POSE_SPACE,
        R3_CONFIDENCE_SEMANTICS,
        R3_POSE_GRAPH_LEGACY_SCHEMA_VERSION,
        R3_POSE_ENCODING,
        R3_POSE_GRAPH_SCHEMA_VERSION,
        R3_RELATIVE_TRANSFORM_CONVENTION,
    )


@dataclass(frozen=True)
class PoseGraphOptimizerConfig:
    rotation_schedule_degrees: tuple[float, ...] = (90.0, 45.0, 20.0, 10.0)
    translation_schedule: tuple[float, ...] = (5.0, 2.0, 1.0, 0.5)
    prior_weight: float = 1e-3
    backbone_weight_floor: float = 0.5
    confidence_weight_min: float = 0.1
    confidence_weight_max: float = 10.0
    bridge_weight_factor: float = 0.5
    anchor_weight_factor: float = 1.0
    unknown_edge_weight_factor: float = 0.5
    lsmr_tolerance: float = 1e-6
    lsmr_max_iterations: int = 250
    minimum_objective_improvement: float = 0.02
    minimum_component_coverage: float = 0.95
    minimum_path_length_ratio: float = 0.5
    maximum_path_length_ratio: float = 1.5
    maximum_step_p99_ratio: float = 3.0
    maximum_residual_p90_ratio: float = 1.5
    maximum_rotation_p90_increase_degrees: float = 2.0
    maximum_translation_p90_increase: float = 0.15
    maximum_inlier_fraction_drop: float = 0.05
    verified_loop_closure: bool = False
    minimum_endpoint_displacement_ratio: float = 0.55
    minimum_spatial_span_ratio: float = 0.55
    maximum_tortuosity_factor: float = 2.0
    maximum_tortuosity_increase: float = 2.0
    maximum_near_start_fraction_increase: float = 0.25
    maximum_near_start_fraction_factor: float = 3.0
    maximum_segment_length_log_rmse: float = 0.80
    minimum_straight_run_preservation: float = 0.75
    minimum_turn_sequence_agreement: float = 0.70
    maximum_sharp_reverse_ratio_increase: float = 0.10
    maximum_backbone_gap: int = 2
    minimum_backbone_coverage: float = 0.95
    minimum_bridge_boundary_coverage: float = 1.0
    fallback_boundaries: tuple[int, ...] = ()
    local_constraint_weight_factor: float = 0.7
    verified_loop_weight_factor: float = 0.4
    distant_edge_minimum_confidence_ratio: float = 1.25
    distant_edge_maximum_rotation_error_degrees: float = 20.0
    distant_edge_minimum_translation_direction_cosine: float = 0.5
    distant_edge_minimum_matching_support: float = 2.0
    distant_edge_neighbor_radius: int = 2
    distant_edge_minimum_neighbor_matches: int = 2
    distant_edge_maximum_inverse_rotation_error_degrees: float = 5.0
    distant_edge_maximum_inverse_translation_error_ratio: float = 0.25
    loop_closure_maximum_endpoint_span_ratio: float = 0.12
    distant_edge_minimum_temporal_gap: int = 20


def _project_rotations(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64)
    u, _, vt = np.linalg.svd(matrices)
    projected = u @ vt
    negative = np.linalg.det(projected) < 0
    if negative.any():
        u[negative, :, -1] *= -1.0
        projected[negative] = u[negative] @ vt[negative]
    return projected


def _quaternion_xyzw_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norms, 1e-12)
    x, y, z, w = q.T
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrices = np.empty((len(q), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrices[:, 0, 1] = 2.0 * (xy - wz)
    matrices[:, 0, 2] = 2.0 * (xz + wy)
    matrices[:, 1, 0] = 2.0 * (xy + wz)
    matrices[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrices[:, 1, 2] = 2.0 * (yz - wx)
    matrices[:, 2, 0] = 2.0 * (xz - wy)
    matrices[:, 2, 1] = 2.0 * (yz + wx)
    matrices[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrices


def _c2w_to_w2c(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotations_c2w = _project_rotations(c2w[:, :3, :3])
    rotations_w2c = np.swapaxes(rotations_c2w, 1, 2)
    translations_w2c = -np.einsum("nij,nj->ni", rotations_w2c, c2w[:, :3, 3])
    return rotations_w2c, translations_w2c


def _w2c_to_c2w(rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    rotations_c2w = np.swapaxes(rotations, 1, 2)
    centers = -np.einsum("nij,nj->ni", rotations_c2w, translations)
    c2w = np.broadcast_to(np.eye(4, dtype=np.float64), (len(rotations), 4, 4)).copy()
    c2w[:, :3, :3] = rotations_c2w
    c2w[:, :3, 3] = centers
    return c2w


def _normalize_confidence(values: np.ndarray, config: PoseGraphOptimizerConfig) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite_positive = values[np.isfinite(values) & (values > 0)]
    median = float(np.median(finite_positive)) if finite_positive.size else 1.0
    normalized = values / max(median, 1e-8)
    return np.clip(
        normalized,
        config.confidence_weight_min,
        config.confidence_weight_max,
    )


def _safe_edge_types(
    schema_version: int,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    edge_type: np.ndarray,
) -> np.ndarray:
    """Convert old broad edge classes into conservative schema-v2 classes."""
    types = np.asarray(edge_type, dtype=np.uint8).reshape(-1)
    if schema_version != R3_POSE_GRAPH_LEGACY_SCHEMA_VERSION:
        return types
    gap = np.abs(
        np.asarray(frame_j, dtype=np.int64)
        - np.asarray(frame_i, dtype=np.int64)
    )
    converted = np.full(len(types), 255, dtype=np.uint8)
    normal = types == 0
    converted[normal & (gap <= 2)] = 0
    converted[normal & (gap > 2) & (gap <= 20)] = 1
    converted[normal & (gap > 20)] = 3
    converted[types == 1] = 2
    converted[types == 2] = 6
    return converted


def _edge_type_counts(edge_type: np.ndarray) -> dict[str, int]:
    names = {
        0: "odometry",
        1: "local_constraint",
        2: "fallback_bridge",
        3: "loop_candidate",
        4: "verified_loop",
        5: "rejected_loop",
        6: "anchor",
        255: "unknown",
    }
    values = np.asarray(edge_type, dtype=np.uint8).reshape(-1)
    return {
        names.get(int(code), "unknown"): int(count)
        for code, count in zip(*np.unique(values, return_counts=True))
    }


def _rotation_angle_degrees(matrices: np.ndarray) -> np.ndarray:
    cosine = np.clip(
        (np.trace(matrices, axis1=1, axis2=2) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def _verify_distant_edges(
    c2w_poses: np.ndarray,
    *,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    rel_pose_enc: np.ndarray,
    confidence_t: np.ndarray,
    confidence_r: np.ndarray,
    edge_type: np.ndarray,
    matching_support: np.ndarray | None,
    config: PoseGraphOptimizerConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Conservatively promote supported distant measurements.

    A promoted measurement becomes either a verified closure (type 4) when
    its endpoints are spatially close, or a long-range overlap constraint
    (type 1) otherwise. Both can influence the solve, but only the former
    changes closed-route acceptance semantics.
    """
    i = np.asarray(frame_i, dtype=np.int64).reshape(-1)
    j = np.asarray(frame_j, dtype=np.int64).reshape(-1)
    rel = np.asarray(rel_pose_enc, dtype=np.float64)
    conf_t = np.asarray(confidence_t, dtype=np.float64).reshape(-1)
    conf_r = np.asarray(confidence_r, dtype=np.float64).reshape(-1)
    types = np.asarray(edge_type, dtype=np.uint8).reshape(-1).copy()
    support = (
        np.asarray(matching_support, dtype=np.float64).reshape(-1)
        if matching_support is not None
        else np.full(len(i), np.nan, dtype=np.float64)
    )
    if len(support) != len(i):
        support = np.full(len(i), np.nan, dtype=np.float64)

    temporal_gap = np.abs(j - i)
    # A source-side "verified" label is evidence, not authority. Every
    # genuinely distant candidate must pass the same independent checks.
    candidate_indices = np.flatnonzero(
        np.isin(types, np.asarray([3, 4], dtype=np.uint8))
        & (temporal_gap > max(2, int(config.distant_edge_minimum_temporal_gap)))
    )
    if not len(candidate_indices):
        return types, {
            "candidate_count": 0,
            "verified_loop_count": 0,
            "verified_overlap_count": 0,
            "rejected_count": 0,
            "decisions": [],
        }

    rotations = _quaternion_xyzw_to_matrix(rel[:, 3:7])
    translations = rel[:, :3]
    raw_rotations, raw_translations = _c2w_to_w2c(np.asarray(c2w_poses))
    local_mask = np.isin(types, np.asarray([0, 1, 2, 6], dtype=np.uint8))
    local_conf_t = conf_t[local_mask & np.isfinite(conf_t) & (conf_t > 0)]
    local_conf_r = conf_r[local_mask & np.isfinite(conf_r) & (conf_r > 0)]
    median_t = float(np.median(local_conf_t)) if len(local_conf_t) else 1.0
    median_r = float(np.median(local_conf_r)) if len(local_conf_r) else 1.0
    centers = np.asarray(c2w_poses, dtype=np.float64)[:, :3, 3]
    span = float(np.linalg.norm(np.ptp(centers, axis=0)))
    decisions: list[dict[str, Any]] = []

    directed_pair = {(int(left), int(right)): index for index, (left, right) in enumerate(zip(i, j))}
    radius = max(1, int(config.distant_edge_neighbor_radius))
    for index in candidate_indices:
        left, right = int(i[index]), int(j[index])
        reasons: list[str] = []
        high_confidence = (
            np.isfinite(conf_t[index])
            and np.isfinite(conf_r[index])
            and conf_t[index] >= median_t * config.distant_edge_minimum_confidence_ratio
            and conf_r[index] >= median_r * config.distant_edge_minimum_confidence_ratio
        )
        if not high_confidence:
            reasons.append("low_confidence")

        predicted_rotation = raw_rotations[right] @ raw_rotations[left].T
        rotation_delta = rotations[index].T @ predicted_rotation
        rotation_error = float(_rotation_angle_degrees(rotation_delta[None])[0])
        if rotation_error > config.distant_edge_maximum_rotation_error_degrees:
            reasons.append("rotation_cycle_inconsistent")

        predicted_translation = (
            raw_translations[right] - predicted_rotation @ raw_translations[left]
        )
        measured_norm = float(np.linalg.norm(translations[index]))
        predicted_norm = float(np.linalg.norm(predicted_translation))
        if measured_norm <= 1e-9 or predicted_norm <= 1e-9:
            direction_cosine = -1.0
        else:
            direction_cosine = float(
                np.dot(translations[index], predicted_translation)
                / (measured_norm * predicted_norm)
            )
        if direction_cosine < config.distant_edge_minimum_translation_direction_cosine:
            reasons.append("translation_direction_inconsistent")

        neighbor_count = 0
        for other in candidate_indices:
            if other == index:
                continue
            if (
                abs(int(i[other]) - left) <= radius
                and abs(int(j[other]) - right) <= radius
            ):
                neighbor_count += 1
        support_value = float(support[index]) if np.isfinite(support[index]) else 0.0
        supported = (
            support_value >= config.distant_edge_minimum_matching_support
            or neighbor_count >= config.distant_edge_minimum_neighbor_matches
        )
        if not supported:
            reasons.append("isolated_match")

        inverse_index = directed_pair.get((right, left))
        inverse_available = inverse_index is not None
        inverse_consistent = True
        if inverse_available:
            reverse_rotation = rotations[inverse_index]
            inverse_rotation_delta = reverse_rotation @ rotations[index]
            inverse_rotation_error = float(
                _rotation_angle_degrees(inverse_rotation_delta[None])[0]
            )
            expected_reverse_translation = -reverse_rotation @ translations[index]
            inverse_translation_error = float(
                np.linalg.norm(translations[inverse_index] - expected_reverse_translation)
                / max(measured_norm, 1e-9)
            )
            inverse_consistent = (
                inverse_rotation_error
                <= config.distant_edge_maximum_inverse_rotation_error_degrees
                and inverse_translation_error
                <= config.distant_edge_maximum_inverse_translation_error_ratio
            )
            if not inverse_consistent:
                reasons.append("inverse_transform_inconsistent")

        endpoint_distance = float(np.linalg.norm(centers[right] - centers[left]))
        closes_route = (
            span > 1e-9
            and endpoint_distance / span
            <= config.loop_closure_maximum_endpoint_span_ratio
        )
        accepted = not reasons
        if accepted:
            types[index] = 4 if closes_route else 1
        else:
            types[index] = 5
        decisions.append({
            "edge_index": int(index),
            "frame_i": left,
            "frame_j": right,
            "accepted": accepted,
            "classification": (
                "verified_loop"
                if accepted and closes_route
                else "verified_overlap"
                if accepted
                else "rejected_loop"
            ),
            "reasons": reasons,
            "confidence_ratio_t": float(conf_t[index] / max(median_t, 1e-9)),
            "confidence_ratio_r": float(conf_r[index] / max(median_r, 1e-9)),
            "rotation_error_degrees": rotation_error,
            "translation_direction_cosine": direction_cosine,
            "matching_support": support_value,
            "neighbor_match_count": neighbor_count,
            "inverse_available": inverse_available,
            "inverse_consistent": inverse_consistent,
            "endpoint_span_ratio": endpoint_distance / max(span, 1e-9),
        })

    return types, {
        "candidate_count": len(decisions),
        "verified_loop_count": sum(
            item["classification"] == "verified_loop" for item in decisions
        ),
        "verified_overlap_count": sum(
            item["classification"] == "verified_overlap" for item in decisions
        ),
        "rejected_count": sum(not item["accepted"] for item in decisions),
        "decisions": decisions,
    }


def _canonicalize_and_deduplicate_edges(
    point_count: int,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    rel_pose_enc: np.ndarray,
    confidence: np.ndarray,
    confidence_t: np.ndarray,
    confidence_r: np.ndarray,
    edge_type: np.ndarray,
) -> dict[str, np.ndarray]:
    i = np.asarray(frame_i, dtype=np.int64).reshape(-1)
    j = np.asarray(frame_j, dtype=np.int64).reshape(-1)
    rel = np.asarray(rel_pose_enc, dtype=np.float64)
    aggregate = np.asarray(confidence, dtype=np.float64).reshape(-1)
    conf_t = np.asarray(confidence_t, dtype=np.float64).reshape(-1)
    conf_r = np.asarray(confidence_r, dtype=np.float64).reshape(-1)
    types = np.asarray(edge_type, dtype=np.uint8).reshape(-1)
    count = len(i)
    if rel.shape != (count, 9) or any(
        len(array) != count for array in (j, aggregate, conf_t, conf_r, types)
    ):
        raise ValueError("pose graph arrays have inconsistent shapes")

    conf_t = np.where(np.isfinite(conf_t) & (conf_t > 0), conf_t, aggregate)
    conf_r = np.where(np.isfinite(conf_r) & (conf_r > 0), conf_r, aggregate)
    quaternion_norm = np.linalg.norm(rel[:, 3:7], axis=1)
    valid = (
        (i >= 0)
        & (j >= 0)
        & (i < point_count)
        & (j < point_count)
        & (i != j)
        & np.isfinite(rel).all(axis=1)
        & (quaternion_norm > 0.95)
        & (quaternion_norm < 1.05)
        & np.isfinite(conf_t)
        & (conf_t > 0)
        & np.isfinite(conf_r)
        & (conf_r > 0)
        # Unverified/rejected distant matches never participate in production
        # optimization. They remain in the artifact for diagnostics.
        & np.isin(types, np.asarray([0, 1, 2, 4, 6], dtype=np.uint8))
    )
    i, j, rel = i[valid], j[valid], rel[valid]
    aggregate, conf_t, conf_r, types = (
        aggregate[valid],
        conf_t[valid],
        conf_r[valid],
        types[valid],
    )
    rotations = _quaternion_xyzw_to_matrix(rel[:, 3:7])
    translations = rel[:, :3].copy()

    reverse = i > j
    if reverse.any():
        reverse_rotations = np.swapaxes(rotations[reverse], 1, 2)
        translations[reverse] = -np.einsum(
            "nij,nj->ni", reverse_rotations, translations[reverse]
        )
        rotations[reverse] = reverse_rotations
        old_i = i[reverse].copy()
        i[reverse] = j[reverse]
        j[reverse] = old_i

    # Historical fallback logs can repeat the same pair many times.  Counting
    # every copy would turn append frequency into a fake information matrix.
    score = np.sqrt(conf_t * conf_r)
    pair_key = i * np.int64(max(point_count, 1)) + j
    order = np.lexsort((-score, pair_key))
    ordered_keys = pair_key[order]
    keep_ordered = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        keep_ordered[1:] = ordered_keys[1:] != ordered_keys[:-1]
    keep = order[keep_ordered]
    keep.sort()
    return {
        "frame_i": i[keep],
        "frame_j": j[keep],
        "rotation": rotations[keep],
        "translation": translations[keep],
        "confidence": aggregate[keep],
        "confidence_t": conf_t[keep],
        "confidence_r": conf_r[keep],
        "edge_type": types[keep],
        "input_edge_count": np.asarray([count], dtype=np.int64),
        "valid_edge_count": np.asarray([int(valid.sum())], dtype=np.int64),
        "excluded_edge_count": np.asarray([int((~valid).sum())], dtype=np.int64),
    }


def _temporal_backbone_mask(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    confidence: np.ndarray,
    point_count: int,
    edge_type: np.ndarray,
    config: PoseGraphOptimizerConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a reachable local scaffold without promoting distant edges.

    Every selected edge must originate in the component already reachable from
    frame zero. Normal edges may not cross a known fallback boundary; only a
    measured bridge edge can carry the backbone across it.
    """
    i = np.asarray(frame_i, dtype=np.int64)
    j = np.asarray(frame_j, dtype=np.int64)
    weight = np.asarray(confidence, dtype=np.float64)
    types = np.asarray(edge_type, dtype=np.uint8)
    gap = j - i
    maximum_gap = max(1, int(config.maximum_backbone_gap))
    boundaries = sorted({
        int(value)
        for value in config.fallback_boundaries
        if 0 < int(value) < point_count
    })
    mask = np.zeros(len(i), dtype=bool)
    reachable = np.zeros(point_count, dtype=bool)
    if point_count:
        reachable[0] = True
    missing: list[int] = []
    covered_boundaries: set[int] = set()

    def crossed_boundaries(left: int, right: int) -> list[int]:
        return [boundary for boundary in boundaries if left < boundary <= right]

    for target in range(1, point_count):
        candidates: list[tuple[int, float, int]] = []
        for edge_index in np.flatnonzero(j == target):
            left = int(i[edge_index])
            edge_gap = int(gap[edge_index])
            if (
                edge_gap < 1
                or edge_gap > maximum_gap
                or left < 0
                or left >= point_count
                or not reachable[left]
            ):
                continue
            crossings = crossed_boundaries(left, target)
            kind = int(types[edge_index])
            if crossings:
                if kind != 2:
                    continue
            elif kind == 2:
                # Bridge measurements are evidence for a reset boundary, not
                # ordinary odometry substitutes inside a stable epoch.
                continue
            elif kind not in {0, 1, 6}:
                continue
            candidates.append((edge_gap, -float(weight[edge_index]), int(edge_index)))
        if not candidates:
            missing.append(target)
            continue
        _, _, selected = min(candidates)
        mask[selected] = True
        reachable[target] = True
        covered_boundaries.update(crossed_boundaries(int(i[selected]), target))

    selected_gaps = gap[mask]
    coverage = float(np.mean(reachable)) if point_count else 0.0
    bridge_coverage = (
        len(covered_boundaries) / len(boundaries) if boundaries else 1.0
    )
    diagnostics = {
        "backbone_coverage": coverage,
        "reachable_frames": int(reachable.sum()),
        "maximum_backbone_gap": int(selected_gaps.max()) if len(selected_gaps) else 0,
        "configured_maximum_gap": maximum_gap,
        "missing_temporal_links": missing,
        "missing_temporal_link_count": len(missing),
        "fallback_boundaries": boundaries,
        "bridge_boundaries_covered": sorted(covered_boundaries),
        "bridge_boundary_coverage": float(bridge_coverage),
        "long_range_backbone_edges": int((selected_gaps > maximum_gap).sum()),
        "input_long_range_edges": int((gap > maximum_gap).sum()),
    }
    return mask, diagnostics


def _build_block_matrix(
    point_count: int,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    edge_rotations: np.ndarray,
    weights: np.ndarray,
    prior_weight: float,
) -> sparse.csr_matrix:
    edge_count = len(frame_i)
    sqrt_weight = np.sqrt(np.maximum(weights, 1e-12))
    row_base = np.arange(edge_count, dtype=np.int64) * 3
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    variable_j = frame_j != 0
    if variable_j.any():
        base = row_base[variable_j, None]
        axes = np.arange(3, dtype=np.int64)[None, :]
        rows.append((base + axes).reshape(-1))
        cols.append(((frame_j[variable_j, None] - 1) * 3 + axes).reshape(-1))
        data.append(np.repeat(sqrt_weight[variable_j], 3))

    variable_i = frame_i != 0
    if variable_i.any():
        base = row_base[variable_i, None, None]
        row_axes = np.arange(3, dtype=np.int64)[None, :, None]
        col_axes = np.arange(3, dtype=np.int64)[None, None, :]
        rows.append(np.broadcast_to(base + row_axes, (int(variable_i.sum()), 3, 3)).reshape(-1))
        cols.append(
            np.broadcast_to(
                (frame_i[variable_i, None, None] - 1) * 3 + col_axes,
                (int(variable_i.sum()), 3, 3),
            ).reshape(-1)
        )
        data.append(
            (-sqrt_weight[variable_i, None, None] * edge_rotations[variable_i]).reshape(-1)
        )

    if point_count > 1 and prior_weight > 0:
        prior_base = edge_count * 3
        axes = np.arange((point_count - 1) * 3, dtype=np.int64)
        rows.append(prior_base + axes)
        cols.append(axes)
        data.append(np.full(len(axes), math.sqrt(prior_weight), dtype=np.float64))

    row_count = edge_count * 3 + ((point_count - 1) * 3 if prior_weight > 0 else 0)
    return sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(row_count, max(0, (point_count - 1) * 3)),
    ).tocsr()


def _build_block_rhs(
    point_count: int,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    edge_rotations: np.ndarray,
    weights: np.ndarray,
    edge_rhs: np.ndarray,
    anchor: np.ndarray,
    prior: np.ndarray,
    prior_weight: float,
) -> np.ndarray:
    sqrt_weight = np.sqrt(np.maximum(weights, 1e-12))
    rhs = sqrt_weight[:, None] * np.asarray(edge_rhs, dtype=np.float64)
    fixed_j = frame_j == 0
    if fixed_j.any():
        rhs[fixed_j] -= sqrt_weight[fixed_j, None] * anchor
    fixed_i = frame_i == 0
    if fixed_i.any():
        rhs[fixed_i] += sqrt_weight[fixed_i, None] * np.einsum(
            "nij,j->ni", edge_rotations[fixed_i], anchor
        )
    parts = [rhs.reshape(-1)]
    if point_count > 1 and prior_weight > 0:
        parts.append(math.sqrt(prior_weight) * prior[1:].reshape(-1))
    return np.concatenate(parts)


def _solve_block_system(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    initial: np.ndarray,
    config: PoseGraphOptimizerConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = lsmr(
        matrix,
        rhs,
        atol=config.lsmr_tolerance,
        btol=config.lsmr_tolerance,
        maxiter=config.lsmr_max_iterations,
        x0=initial[1:].reshape(-1),
    )
    solved = initial.copy()
    solved[1:] = result[0].reshape(-1, 3)
    return solved, {
        "stop_code": int(result[1]),
        "iterations": int(result[2]),
        "residual_norm": float(result[3]),
        "condition_estimate": float(result[6]),
    }


def _rotation_residual_radians(
    rotations: np.ndarray,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    measurements: np.ndarray,
) -> np.ndarray:
    predicted = rotations[frame_j] @ np.swapaxes(rotations[frame_i], 1, 2)
    delta = np.swapaxes(measurements, 1, 2) @ predicted
    cosine = np.clip((np.trace(delta, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def _translation_residual(
    translations: np.ndarray,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    rotations: np.ndarray,
    measurements: np.ndarray,
) -> np.ndarray:
    predicted = translations[frame_j] - np.einsum(
        "nij,nj->ni", rotations, translations[frame_i]
    )
    return predicted - measurements


def _dcs_weights(residual: np.ndarray, scale: float) -> np.ndarray:
    phi = max(float(scale) ** 2, 1e-12)
    return np.minimum(1.0, (2.0 * phi) / (phi + np.square(residual) + 1e-12))


def _solve_rotations(
    initial: np.ndarray,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    measurements: np.ndarray,
    base_weights: np.ndarray,
    backbone: np.ndarray,
    config: PoseGraphOptimizerConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rotations = initial.copy()
    diagnostics: list[dict[str, Any]] = []

    def solve_step(weights: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
        matrix = _build_block_matrix(
            len(initial), frame_i, frame_j, measurements, weights, config.prior_weight
        )
        candidate = np.empty_like(rotations)
        solver_runs = []
        zero_rhs = np.zeros((len(frame_i), 3), dtype=np.float64)
        for column in range(3):
            prior_column = initial[:, :, column]
            current_column = rotations[:, :, column]
            rhs = _build_block_rhs(
                len(initial),
                frame_i,
                frame_j,
                measurements,
                weights,
                zero_rhs,
                initial[0, :, column],
                prior_column,
                config.prior_weight,
            )
            solved, solver = _solve_block_system(matrix, rhs, current_column, config)
            candidate[:, :, column] = solved
            solver_runs.append(solver)
        return _project_rotations(candidate), solver_runs

    # The shortest/highest-confidence incoming edge for every frame forms an
    # odometry-like scaffold.  Bootstrap from it before exposing the solution
    # to loop edges; this prevents a handful of arbitrary long-range outliers
    # from defining the first global linearization.
    bootstrap_weights = base_weights * np.where(backbone, 1.0, 1e-4)
    rotations, bootstrap_solver = solve_step(bootstrap_weights)
    diagnostics.append({
        "stage": "backbone_bootstrap",
        "backbone_edges": int(backbone.sum()),
        "solver": bootstrap_solver,
    })

    for scale_degrees in config.rotation_schedule_degrees:
        residual = _rotation_residual_radians(
            rotations, frame_i, frame_j, measurements
        )
        robust = _dcs_weights(residual, math.radians(scale_degrees))
        robust[backbone] = np.maximum(
            robust[backbone], config.backbone_weight_floor
        )
        weights = base_weights * robust
        rotations, solver_runs = solve_step(weights)
        diagnostics.append({
            "stage": "dcs",
            "scale_degrees": float(scale_degrees),
            "median_residual_degrees": float(np.degrees(np.median(residual))),
            "downweighted_edges": int((robust < 0.999).sum()),
            "solver": solver_runs,
        })
    return rotations, diagnostics


def _translation_denominator(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    measurements: np.ndarray,
) -> tuple[np.ndarray, float]:
    gap = np.maximum(frame_j - frame_i, 1).astype(np.float64)
    norms = np.linalg.norm(measurements, axis=1)
    local = (gap <= 10) & np.isfinite(norms) & (norms > 1e-8)
    per_frame = norms[local] / gap[local] if local.any() else norms[norms > 1e-8]
    step_scale = float(np.median(per_frame)) if per_frame.size else 1.0
    denominator = np.maximum(norms, step_scale * np.sqrt(gap))
    return np.maximum(denominator, 1e-8), max(step_scale, 1e-8)


def _solve_translations(
    initial: np.ndarray,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    edge_rotations: np.ndarray,
    measurements: np.ndarray,
    denominator: np.ndarray,
    base_weights: np.ndarray,
    backbone: np.ndarray,
    config: PoseGraphOptimizerConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    translations = initial.copy()
    diagnostics: list[dict[str, Any]] = []

    def solve_step(weights: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        matrix = _build_block_matrix(
            len(initial), frame_i, frame_j, edge_rotations, weights, config.prior_weight
        )
        rhs = _build_block_rhs(
            len(initial),
            frame_i,
            frame_j,
            edge_rotations,
            weights,
            measurements,
            initial[0],
            initial,
            config.prior_weight,
        )
        return _solve_block_system(matrix, rhs, translations, config)

    # Whiten translation equations by their expected baseline.  Without this,
    # one wrong long-baseline edge can dominate hundreds of local constraints
    # merely because its translation is numerically large.
    whitened_base = base_weights / np.square(denominator)
    bootstrap_weights = whitened_base * np.where(backbone, 1.0, 1e-4)
    translations, bootstrap_solver = solve_step(bootstrap_weights)
    diagnostics.append({
        "stage": "backbone_bootstrap",
        "backbone_edges": int(backbone.sum()),
        "solver": bootstrap_solver,
    })

    for scale in config.translation_schedule:
        residual = _translation_residual(
            translations, frame_i, frame_j, edge_rotations, measurements
        )
        normalized = np.linalg.norm(residual, axis=1) / denominator
        robust = _dcs_weights(normalized, scale)
        robust[backbone] = np.maximum(
            robust[backbone], config.backbone_weight_floor
        )
        weights = whitened_base * robust
        translations, solver = solve_step(weights)
        diagnostics.append({
            "stage": "dcs",
            "scale": float(scale),
            "median_normalized_residual": float(np.median(normalized)),
            "downweighted_edges": int((robust < 0.999).sum()),
            "solver": solver,
        })
    return translations, diagnostics


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    order = np.argsort(values[valid])
    ordered_values = values[valid][order]
    ordered_weights = weights[valid][order]
    threshold = float(percentile) / 100.0 * float(ordered_weights.sum())
    index = int(np.searchsorted(np.cumsum(ordered_weights), threshold, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _residual_metrics(
    rotations: np.ndarray,
    translations: np.ndarray,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    edge_rotations: np.ndarray,
    edge_translations: np.ndarray,
    denominator: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    rotation = _rotation_residual_radians(
        rotations, frame_i, frame_j, edge_rotations
    )
    translation = np.linalg.norm(
        _translation_residual(
            translations, frame_i, frame_j, edge_rotations, edge_translations
        ),
        axis=1,
    ) / denominator
    combined_sq = np.square(rotation / math.radians(15.0)) + np.square(translation)
    clipped_objective = float(
        np.sum(weights * np.minimum(combined_sq, 16.0)) / max(float(weights.sum()), 1e-12)
    )
    return {
        "objective": clipped_objective,
        "rotation_median_degrees": math.degrees(_weighted_percentile(rotation, weights, 50)),
        "rotation_p90_degrees": math.degrees(_weighted_percentile(rotation, weights, 90)),
        "translation_median_normalized": _weighted_percentile(translation, weights, 50),
        "translation_p90_normalized": _weighted_percentile(translation, weights, 90),
        "combined_inlier_fraction": float(np.average(combined_sq <= 4.0, weights=weights)),
    }


def _path_metrics(
    c2w: np.ndarray,
    *,
    near_start_radius: float | None = None,
) -> dict[str, float]:
    centers = c2w[:, :3, 3]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    shared = trajectory_metrics(centers)
    bbox_diagonal = float(np.linalg.norm(np.ptp(centers, axis=0)))
    if near_start_radius is None:
        near_start_radius = max(bbox_diagonal * 0.10, float(np.median(steps)) * 5.0)
    distance_from_start = np.linalg.norm(centers - centers[0], axis=1)
    return {
        "path_length": float(steps.sum()),
        "endpoint_displacement": float(shared.get("endpoint_displacement", 0.0)),
        "net_progress_ratio": float(shared.get("net_progress_ratio", 0.0)),
        "bbox_diagonal": bbox_diagonal,
        "span_length_ratio": float(shared.get("span_ratio", 0.0)),
        "tortuosity": float(shared.get("tortuosity", 0.0)),
        "near_start_radius": float(near_start_radius),
        "near_start_fraction": float(np.mean(distance_from_start <= near_start_radius)),
        "sharp_reverse_ratio": float(shared.get("sharp_reverse_ratio", 0.0)),
        "step_median": float(np.median(steps)) if steps.size else 0.0,
        "step_p99": float(np.percentile(steps, 99)) if steps.size else 0.0,
    }


def _geometry_rejection_reasons(
    initial_c2w: np.ndarray,
    candidate_c2w: np.ndarray,
    initial_path: dict[str, float],
    candidate_path: dict[str, float],
    config: PoseGraphOptimizerConfig,
) -> tuple[list[str], dict[str, Any]]:
    comparison = compare_trajectories(
        initial_c2w[:, :3, 3],
        candidate_c2w[:, :3, 3],
    )
    reasons: list[str] = []
    raw_endpoint = initial_path["endpoint_displacement"]
    candidate_endpoint = candidate_path["endpoint_displacement"]
    endpoint_ratio = candidate_endpoint / max(raw_endpoint, 1e-12)
    # A verified return to a visited place may legitimately have near-zero
    # endpoint displacement. Unverified open trajectories may not collapse.
    if (
        not config.verified_loop_closure
        and initial_path["net_progress_ratio"] >= 0.10
        and endpoint_ratio < config.minimum_endpoint_displacement_ratio
    ):
        reasons.append("endpoint_displacement_collapsed")

    raw_span_ratio = initial_path["span_length_ratio"]
    if (
        candidate_path["span_length_ratio"]
        < raw_span_ratio * config.minimum_spatial_span_ratio
    ):
        reasons.append("spatial_span_collapsed")

    tortuosity_limit = max(
        initial_path["tortuosity"] * config.maximum_tortuosity_factor,
        initial_path["tortuosity"] + config.maximum_tortuosity_increase,
    )
    if candidate_path["tortuosity"] > tortuosity_limit:
        reasons.append("tortuosity_regression")

    near_start_limit = max(
        initial_path["near_start_fraction"]
        + config.maximum_near_start_fraction_increase,
        initial_path["near_start_fraction"]
        * config.maximum_near_start_fraction_factor,
    )
    if (
        not config.verified_loop_closure
        and candidate_path["near_start_fraction"] > min(near_start_limit, 0.95)
    ):
        reasons.append("near_start_concentration_regression")

    if comparison.get("available"):
        if (
            float(comparison["segment_length_log_rmse"])
            > config.maximum_segment_length_log_rmse
        ):
            reasons.append("local_step_distortion")
        if (
            float(comparison["straight_run_preservation"])
            < config.minimum_straight_run_preservation
        ):
            reasons.append("straight_run_destroyed")
        if (
            float(comparison["turn_sequence_agreement"])
            < config.minimum_turn_sequence_agreement
        ):
            reasons.append("turn_sequence_destroyed")
    else:
        reasons.append("trajectory_geometry_comparison_unavailable")

    sharp_reverse_limit = max(
        initial_path["sharp_reverse_ratio"]
        + config.maximum_sharp_reverse_ratio_increase,
        initial_path["sharp_reverse_ratio"] * 2.0 + 0.02,
    )
    if candidate_path["sharp_reverse_ratio"] > sharp_reverse_limit:
        reasons.append("sharp_reverse_ratio_regression")
    return reasons, {
        "verified_loop_closure": config.verified_loop_closure,
        "endpoint_displacement_ratio": float(endpoint_ratio),
        "minimum_endpoint_displacement_ratio": config.minimum_endpoint_displacement_ratio,
        "minimum_spatial_span_ratio": config.minimum_spatial_span_ratio,
        "tortuosity_limit": float(tortuosity_limit),
        "near_start_fraction_limit": float(min(near_start_limit, 0.95)),
        "maximum_segment_length_log_rmse": config.maximum_segment_length_log_rmse,
        "minimum_straight_run_preservation": config.minimum_straight_run_preservation,
        "minimum_turn_sequence_agreement": config.minimum_turn_sequence_agreement,
        "comparison": comparison,
    }


def _component_metrics(
    point_count: int, frame_i: np.ndarray, frame_j: np.ndarray
) -> dict[str, Any]:
    adjacency = sparse.coo_matrix(
        (np.ones(len(frame_i), dtype=np.uint8), (frame_i, frame_j)),
        shape=(point_count, point_count),
    )
    component_count, labels = csgraph.connected_components(
        adjacency, directed=False, return_labels=True
    )
    sizes = np.bincount(labels, minlength=component_count)
    largest = int(sizes.max()) if sizes.size else 0
    return {
        "component_count": int(component_count),
        "largest_component_frames": largest,
        "largest_component_coverage": largest / point_count if point_count else 0.0,
    }


def optimize_pose_graph_arrays(
    c2w_poses: np.ndarray,
    *,
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    rel_pose_enc: np.ndarray,
    confidence: np.ndarray,
    confidence_t: np.ndarray,
    confidence_r: np.ndarray,
    edge_type: np.ndarray,
    matching_support: np.ndarray | None = None,
    config: PoseGraphOptimizerConfig | None = None,
) -> dict[str, Any]:
    """Build a robust shadow candidate from full R3 edge measurements."""
    config = config or PoseGraphOptimizerConfig()
    started = time.perf_counter()
    c2w = np.asarray(c2w_poses, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or len(c2w) < 2:
        raise ValueError("c2w_poses must have shape [N, 4, 4] with N >= 2")
    if not np.isfinite(c2w).all():
        raise ValueError("c2w_poses contain non-finite values")
    point_count = len(c2w)
    source_edge_type_counts = _edge_type_counts(edge_type)
    edge_type, distant_edge_verification = _verify_distant_edges(
        c2w,
        frame_i=frame_i,
        frame_j=frame_j,
        rel_pose_enc=rel_pose_enc,
        confidence_t=confidence_t,
        confidence_r=confidence_r,
        edge_type=edge_type,
        matching_support=matching_support,
        config=config,
    )
    input_edge_type_counts = _edge_type_counts(edge_type)
    edges = _canonicalize_and_deduplicate_edges(
        point_count,
        frame_i,
        frame_j,
        rel_pose_enc,
        confidence,
        confidence_t,
        confidence_r,
        edge_type,
    )
    i = edges["frame_i"]
    j = edges["frame_j"]
    if len(i) == 0:
        raise ValueError("pose graph contains no usable edges")

    initial_rotations, initial_translations = _c2w_to_w2c(c2w)
    confidence_weight_t = _normalize_confidence(edges["confidence_t"], config)
    confidence_weight_r = _normalize_confidence(edges["confidence_r"], config)
    type_factor = np.ones(len(i), dtype=np.float64)
    # Bridge edges are produced exactly when online tracking has lost enough
    # confidence to trigger fallback.  Their network confidence remains useful,
    # but is not calibrated as an information matrix and must not be promoted
    # above ordinary observations merely because it crosses a segment boundary.
    type_factor[edges["edge_type"] == 1] = config.local_constraint_weight_factor
    type_factor[edges["edge_type"] == 2] = config.bridge_weight_factor
    type_factor[edges["edge_type"] == 4] = config.verified_loop_weight_factor
    type_factor[edges["edge_type"] == 6] = config.anchor_weight_factor
    type_factor[edges["edge_type"] == 255] = config.unknown_edge_weight_factor
    confidence_weight_t *= type_factor
    confidence_weight_r *= type_factor
    joint_weight = np.sqrt(confidence_weight_t * confidence_weight_r)
    backbone, backbone_diagnostics = _temporal_backbone_mask(
        i,
        j,
        joint_weight,
        point_count,
        edges["edge_type"],
        config,
    )
    denominator, step_scale = _translation_denominator(i, j, edges["translation"])

    optimized_rotations, rotation_iterations = _solve_rotations(
        initial_rotations,
        i,
        j,
        edges["rotation"],
        confidence_weight_r,
        backbone,
        config,
    )
    optimized_translations, translation_iterations = _solve_translations(
        initial_translations,
        i,
        j,
        edges["rotation"],
        edges["translation"],
        denominator,
        confidence_weight_t,
        backbone,
        config,
    )
    candidate_c2w = _w2c_to_c2w(optimized_rotations, optimized_translations)

    before = _residual_metrics(
        initial_rotations,
        initial_translations,
        i,
        j,
        edges["rotation"],
        edges["translation"],
        denominator,
        joint_weight,
    )
    after = _residual_metrics(
        optimized_rotations,
        optimized_translations,
        i,
        j,
        edges["rotation"],
        edges["translation"],
        denominator,
        joint_weight,
    )
    initial_path = _path_metrics(c2w)
    candidate_path = _path_metrics(
        candidate_c2w,
        near_start_radius=initial_path["near_start_radius"],
    )
    component = _component_metrics(point_count, i, j)
    path_ratio = candidate_path["path_length"] / max(initial_path["path_length"], 1e-12)
    step_p99_ratio = candidate_path["step_p99"] / max(initial_path["step_p99"], 1e-12)
    displacement = np.linalg.norm(
        candidate_c2w[:, :3, 3] - c2w[:, :3, 3], axis=1
    )
    improvement = (before["objective"] - after["objective"]) / max(
        before["objective"], 1e-12
    )
    rejection_reasons: list[str] = []
    if component["largest_component_coverage"] < config.minimum_component_coverage:
        rejection_reasons.append("insufficient_graph_coverage")
    if (
        backbone_diagnostics["backbone_coverage"]
        < config.minimum_backbone_coverage
    ):
        rejection_reasons.append("insufficient_backbone_coverage")
    if (
        backbone_diagnostics["bridge_boundary_coverage"]
        < config.minimum_bridge_boundary_coverage
    ):
        rejection_reasons.append("insufficient_bridge_boundary_coverage")
    if improvement < config.minimum_objective_improvement:
        rejection_reasons.append("insufficient_objective_improvement")
    if not config.minimum_path_length_ratio <= path_ratio <= config.maximum_path_length_ratio:
        rejection_reasons.append("path_length_ratio_out_of_bounds")
    if step_p99_ratio > config.maximum_step_p99_ratio:
        rejection_reasons.append("step_p99_regression")
    rotation_p90_limit = max(
        before["rotation_p90_degrees"] * config.maximum_residual_p90_ratio,
        before["rotation_p90_degrees"]
        + config.maximum_rotation_p90_increase_degrees,
    )
    if after["rotation_p90_degrees"] > rotation_p90_limit:
        rejection_reasons.append("rotation_p90_regression")
    translation_p90_limit = max(
        before["translation_p90_normalized"] * config.maximum_residual_p90_ratio,
        before["translation_p90_normalized"]
        + config.maximum_translation_p90_increase,
    )
    if after["translation_p90_normalized"] > translation_p90_limit:
        rejection_reasons.append("translation_p90_regression")
    if (
        after["combined_inlier_fraction"]
        < before["combined_inlier_fraction"] - config.maximum_inlier_fraction_drop
    ):
        rejection_reasons.append("inlier_fraction_regression")
    if not np.isfinite(candidate_c2w).all():
        rejection_reasons.append("non_finite_candidate")
    if np.any(np.linalg.det(candidate_c2w[:, :3, :3]) < 0.999):
        rejection_reasons.append("improper_candidate_rotation")
    geometry_rejections, geometry_comparison = _geometry_rejection_reasons(
        c2w,
        candidate_c2w,
        initial_path,
        candidate_path,
        config,
    )
    rejection_reasons.extend(geometry_rejections)

    diagnostics = {
        "schema_version": 1,
        "method": "sparse_chordal_dcs_gnc_shadow",
        "coordinate_convention": "R3 world_to_camera edges; candidate exported camera_to_world",
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "point_count": point_count,
        "input_edge_count": int(edges["input_edge_count"][0]),
        "valid_edge_count": int(edges["valid_edge_count"][0]),
        "excluded_edge_count": int(edges["excluded_edge_count"][0]),
        "input_edge_type_counts": input_edge_type_counts,
        "source_edge_type_counts": source_edge_type_counts,
        "optimized_edge_type_counts": _edge_type_counts(edges["edge_type"]),
        "distant_edge_verification": distant_edge_verification,
        "deduplicated_edge_count": len(i),
        "backbone_edge_count": int(backbone.sum()),
        "backbone": backbone_diagnostics,
        "step_scale": step_scale,
        "graph": component,
        "before": before,
        "after": after,
        "objective_improvement": float(improvement),
        "initial_path": initial_path,
        "candidate_path": candidate_path,
        "path_length_ratio": float(path_ratio),
        "step_p99_ratio": float(step_p99_ratio),
        "geometry_comparison": geometry_comparison,
        "displacement_median": float(np.median(displacement)),
        "displacement_p95": float(np.percentile(displacement, 95)),
        "displacement_max": float(np.max(displacement)),
        "rotation_iterations": rotation_iterations,
        "translation_iterations": translation_iterations,
        "config": asdict(config),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return {"c2w": candidate_c2w.astype(np.float32), "diagnostics": diagnostics}


def optimize_pose_graph_file(
    c2w_poses: np.ndarray,
    graph_path: str | Path,
    config: PoseGraphOptimizerConfig | None = None,
) -> dict[str, Any]:
    source = Path(graph_path)
    with np.load(source, allow_pickle=False) as payload:
        def scalar(key: str) -> Any:
            if key not in payload.files:
                return None
            array = np.asarray(payload[key])
            if array.size != 1:
                return None
            value = array.reshape(-1)[0]
            return value.item() if isinstance(value, np.generic) else value

        metadata = {
            "schema_version": scalar("schema_version"),
            "pose_encoding": scalar("pose_encoding"),
            "transform_convention": scalar("transform_convention"),
            "frame_index_space": scalar("frame_index_space"),
            "absolute_pose_space": scalar("absolute_pose_space"),
            "confidence_semantics": scalar("confidence_semantics"),
        }
        expected = {
            "pose_encoding": R3_POSE_ENCODING,
            "transform_convention": R3_RELATIVE_TRANSFORM_CONVENTION,
            "frame_index_space": "exported_camera_index",
            "absolute_pose_space": R3_ABSOLUTE_POSE_SPACE,
            "confidence_semantics": R3_CONFIDENCE_SEMANTICS,
        }
        mismatches = {
            key: {"actual": metadata[key], "expected": expected_value}
            for key, expected_value in expected.items()
            if metadata[key] != expected_value
        }
        if metadata["schema_version"] not in {
            R3_POSE_GRAPH_LEGACY_SCHEMA_VERSION,
            R3_POSE_GRAPH_SCHEMA_VERSION,
        }:
            mismatches["schema_version"] = {
                "actual": metadata["schema_version"],
                "expected": [
                    R3_POSE_GRAPH_LEGACY_SCHEMA_VERSION,
                    R3_POSE_GRAPH_SCHEMA_VERSION,
                ],
            }
        if mismatches:
            raise ValueError(f"unsupported pose graph metadata: {mismatches}")
        safe_types = _safe_edge_types(
            int(metadata["schema_version"]),
            payload["frame_i"],
            payload["frame_j"],
            payload["edge_type"],
        )
        result = optimize_pose_graph_arrays(
            c2w_poses,
            frame_i=payload["frame_i"],
            frame_j=payload["frame_j"],
            rel_pose_enc=payload["rel_pose_enc"],
            confidence=payload["confidence"],
            confidence_t=payload["confidence_t"],
            confidence_r=payload["confidence_r"],
            edge_type=safe_types,
            matching_support=(
                payload["matching_support"]
                if "matching_support" in payload.files
                else None
            ),
            config=config,
        )
    result["diagnostics"]["source_metadata"] = metadata
    result["diagnostics"]["edge_classification_mode"] = (
        "legacy_v1_safe_classification"
        if metadata["schema_version"] == R3_POSE_GRAPH_LEGACY_SCHEMA_VERSION
        else "native_v2"
    )
    result["diagnostics"]["source_graph"] = str(source)
    result["diagnostics"]["source_graph_mtime_ns"] = source.stat().st_mtime_ns
    return result


def save_pose_graph_candidate(
    output_dir: str | Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Persist a candidate and diagnostics atomically without touching raw cameras."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    candidate_path = destination / "pose_graph_candidate.npz"
    diagnostics_path = destination / "pose_graph_candidate.json"
    diagnostics = dict(result["diagnostics"])
    with tempfile.NamedTemporaryFile(
        dir=destination, prefix=".pose_graph_candidate.", suffix=".npz", delete=False
    ) as handle:
        temporary_candidate = Path(handle.name)
    try:
        np.savez_compressed(
            temporary_candidate,
            c2w=np.asarray(result["c2w"], dtype=np.float32),
            accepted=np.asarray([bool(diagnostics.get("accepted"))], dtype=bool),
            schema_version=np.asarray([1], dtype=np.int32),
        )
        os.replace(temporary_candidate, candidate_path)
    finally:
        temporary_candidate.unlink(missing_ok=True)

    temporary_diagnostics = diagnostics_path.with_name(
        f".{diagnostics_path.name}.{os.getpid()}.tmp"
    )
    try:
        temporary_diagnostics.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_diagnostics, diagnostics_path)
    finally:
        temporary_diagnostics.unlink(missing_ok=True)
    return {
        **diagnostics,
        "candidate_path": str(candidate_path),
        "diagnostics_path": str(diagnostics_path),
    }


def load_pose_graph_candidate_summary(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir) / "pose_graph_candidate.json"
    if not path.exists():
        return {"available": False, "accepted": False, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "available": False,
            "accepted": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, dict):
        return {"available": False, "accepted": False, "error": "invalid_payload"}
    return {"available": True, **payload}


def load_pose_graph_candidate_c2w(
    output_dir: str | Path,
    *,
    expected_count: int = 0,
    accepted_only: bool = True,
) -> np.ndarray | None:
    path = Path(output_dir) / "pose_graph_candidate.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            accepted = bool(np.asarray(payload["accepted"]).reshape(-1)[0])
            c2w = np.asarray(payload["c2w"], dtype=np.float64)
    except Exception:
        return None
    if accepted_only and not accepted:
        return None
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        return None
    if expected_count > 0 and len(c2w) != expected_count:
        return None
    if not np.isfinite(c2w).all():
        return None
    return c2w


def run_pose_graph_shadow(
    output_dir: str | Path,
    c2w_poses: np.ndarray,
    config: PoseGraphOptimizerConfig | None = None,
) -> dict[str, Any]:
    """Run and persist the shadow optimizer without raising into inference."""
    destination = Path(output_dir)
    graph_path = destination / "pose_graph_edges.npz"
    if not graph_path.exists():
        return {"available": False, "accepted": False, "error": "pose_graph_missing"}
    try:
        result = optimize_pose_graph_file(c2w_poses, graph_path, config=config)
        return {"available": True, **save_pose_graph_candidate(destination, result)}
    except Exception as exc:
        diagnostics = {
            "available": False,
            "accepted": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        path = destination / "pose_graph_candidate.json"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            os.replace(temporary, path)
        except Exception:
            pass
        finally:
            temporary.unlink(missing_ok=True)
        return diagnostics
