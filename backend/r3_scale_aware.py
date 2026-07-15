"""Floor-height anchored local-scale correction for monocular R3 trajectories.

The module deliberately writes a shadow candidate. Raw R3 cameras and the
robust SE(3) candidate remain immutable until the scale candidate passes all
quality gates and is explicitly selected by the API consumer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


SCALE_CANDIDATE_FILE = "scale_aware_candidate.npz"
SCALE_SUMMARY_FILE = "scale_aware_candidate.json"
SCALE_SCHEMA_VERSION = 1


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    result = u @ vh
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1.0
        result = u @ vh
    return result


def _estimate_world_up(c2w: np.ndarray) -> tuple[np.ndarray, str]:
    centers = c2w[:, :3, 3]
    camera_up = np.median(-c2w[:, :3, 1], axis=0)
    camera_up /= max(float(np.linalg.norm(camera_up)), 1e-12)
    # A downward-looking body camera has a badly tilted local-up vector. Yaw
    # changes during real corners share the physical vertical as their world
    # rotation axis, so recover that axis before falling back to camera-up.
    covariance = np.zeros((3, 3), dtype=np.float64)
    rotation_samples = 0
    rotations = c2w[:, :3, :3]
    for span in (4, 8, 16, 24):
        if span >= len(rotations):
            continue
        for index in range(len(rotations) - span):
            relative = rotations[index + span] @ rotations[index].T
            cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
            angle = math.acos(cosine)
            if not math.radians(1.5) <= angle <= math.radians(75.0):
                continue
            skew = np.array([
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ])
            norm = float(np.linalg.norm(skew))
            if norm <= 1e-8:
                continue
            axis = skew / norm
            covariance += min(angle, math.radians(30.0)) ** 2 * np.outer(axis, axis)
            rotation_samples += 1
    if rotation_samples >= 12:
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            dominance = float(eigenvalues[-1] / max(float(eigenvalues[-2]), 1e-12))
            rotation_up = eigenvectors[:, -1]
            if float(np.dot(rotation_up, camera_up)) < 0:
                rotation_up = -rotation_up
            if dominance >= 2.0 and float(np.dot(rotation_up, camera_up)) >= 0.45:
                return rotation_up / max(float(np.linalg.norm(rotation_up)), 1e-12), "camera_rotation_axis"
        except np.linalg.LinAlgError:
            pass
    if len(centers) >= 12:
        centered = centers - np.median(centers, axis=0, keepdims=True)
        try:
            _, singular, vh = np.linalg.svd(centered, full_matrices=False)
            if (
                len(singular) >= 3
                and singular[1] / max(float(singular[0]), 1e-9) >= 0.03
                and singular[2] / max(float(singular[1]), 1e-9) <= 0.35
            ):
                normal = vh[-1]
                if float(np.dot(normal, camera_up)) < 0:
                    normal = -normal
                return normal / max(float(np.linalg.norm(normal)), 1e-12), "trajectory_plane"
        except np.linalg.LinAlgError:
            pass
    return camera_up, "median_camera_up"


def _fit_floor_plane(
    points: np.ndarray,
    expected_up: np.ndarray,
    *,
    seed: int,
) -> dict[str, float] | None:
    """Robustly fit the camera-relative floor plane with deterministic RANSAC."""
    if len(points) < 80:
        return None
    expected = expected_up / max(float(np.linalg.norm(expected_up)), 1e-12)
    rng = np.random.RandomState(seed)
    scale = float(np.median(np.linalg.norm(points, axis=1)))
    threshold = max(scale * 0.012, 1e-5)
    best_mask: np.ndarray | None = None
    best_score = -1.0
    for _ in range(72):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        normal /= norm
        alignment = abs(float(np.dot(normal, expected)))
        if alignment < 0.78:
            continue
        distance = np.abs(points @ normal - float(np.dot(sample[0], normal)))
        mask = distance <= threshold
        fraction = float(mask.mean())
        score = fraction * alignment * alignment
        if score > best_score:
            best_score = score
            best_mask = mask
    if best_mask is None or int(best_mask.sum()) < 60:
        return None

    inliers = points[best_mask]
    center = np.median(inliers, axis=0)
    try:
        _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1]
    if float(np.dot(normal, expected)) < 0:
        normal = -normal
    alignment = float(np.dot(normal, expected))
    residual = np.abs((inliers - center) @ normal)
    height = abs(float(np.dot(center, normal)))
    if height <= 1e-6 or alignment < 0.82:
        return None
    residual_ratio = float(np.median(residual)) / height
    inlier_fraction = float(best_mask.mean())
    if residual_ratio > 0.045 or inlier_fraction < 0.12:
        return None
    return {
        "height": height,
        "inlier_fraction": inlier_fraction,
        "normal_alignment": alignment,
        "residual_ratio": residual_ratio,
    }


def estimate_floor_height_observations(
    base: str | Path,
    c2w_poses: np.ndarray,
    *,
    maximum_frames: int = 180,
) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    """Estimate reconstructed camera height from a bounded depth-map sample."""
    source = Path(base)
    c2w = np.asarray(c2w_poses, dtype=np.float64)
    camera_files = sorted((source / "camera").glob("*.npz"))
    if len(camera_files) != len(c2w) or not (source / "depth").exists():
        return [], {"available": False, "reason": "camera_depth_artifacts_missing"}
    world_up, up_method = _estimate_world_up(c2w)
    sample_count = min(maximum_frames, len(camera_files))
    indices = np.unique(np.linspace(0, len(camera_files) - 1, sample_count).round().astype(int))
    observations: list[dict[str, float | int]] = []

    for index in indices:
        camera_file = camera_files[int(index)]
        depth_path = source / "depth" / f"{camera_file.stem}.npy"
        if not depth_path.exists():
            continue
        try:
            camera = np.load(camera_file)
            depth = np.asarray(np.load(depth_path), dtype=np.float64)
            if depth.ndim != 2:
                continue
            height_px, width_px = depth.shape
            intrinsics = np.asarray(camera.get("intrinsics"), dtype=np.float64)
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                continue
            fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
            cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
            if min(fx, fy) <= 1e-6:
                continue
            stride = max(2, min(height_px, width_px) // 72)
            y_start = max(int(height_px * 0.50), int(cy + height_px * 0.03))
            ys, xs = np.meshgrid(
                np.arange(y_start, max(y_start + 1, int(height_px * 0.97)), stride),
                np.arange(int(width_px * 0.04), int(width_px * 0.96), stride),
                indexing="ij",
            )
            ys, xs = ys.ravel(), xs.ravel()
            z = depth[ys, xs]
            valid = np.isfinite(z) & (z > 1e-6)
            conf_path = source / "conf" / f"{camera_file.stem}.npy"
            if conf_path.exists():
                confidence = np.asarray(np.load(conf_path), dtype=np.float64)
                if confidence.shape == depth.shape:
                    values = confidence[ys, xs]
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        valid &= values >= float(np.percentile(finite, 30))
            if int(valid.sum()) < 80:
                continue
            z, xs_v, ys_v = z[valid], xs[valid], ys[valid]
            points = np.column_stack((
                (xs_v - cx) * z / fx,
                (ys_v - cy) * z / fy,
                z,
            ))
            rotation = _project_rotation(c2w[int(index), :3, :3])
            expected_up_camera = rotation.T @ world_up
            fitted = _fit_floor_plane(points, expected_up_camera, seed=int(index) + 17)
            if fitted is None:
                continue
            observations.append({"trajectory_index": int(index), **fitted})
        except (OSError, ValueError, KeyError, np.linalg.LinAlgError):
            continue

    diagnostics: dict[str, Any] = {
        "available": bool(observations),
        "world_up_method": up_method,
        "sampled_frames": int(len(indices)),
        "accepted_observations": len(observations),
    }
    if observations:
        heights = np.asarray([float(item["height"]) for item in observations])
        diagnostics.update({
            "height_median": float(np.median(heights)),
            "height_p10": float(np.percentile(heights, 10)),
            "height_p90": float(np.percentile(heights, 90)),
        })
    return observations, diagnostics


def _smooth_log_scale(
    point_count: int,
    observation_indices: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    rows = observation_indices.astype(np.int64)
    W = sparse.coo_matrix((weights, (rows, rows)), shape=(point_count, point_count)).tocsr()
    if point_count >= 3:
        row = np.repeat(np.arange(point_count - 2), 3)
        col = np.concatenate([
            np.arange(point_count - 2),
            np.arange(1, point_count - 1),
            np.arange(2, point_count),
        ]).reshape(3, -1).T.reshape(-1)
        values = np.tile(np.array([1.0, -2.0, 1.0]), point_count - 2)
        D2 = sparse.coo_matrix((values, (row, col)), shape=(point_count - 2, point_count)).tocsr()
        regularizer = 80.0 * (D2.T @ D2)
    else:
        regularizer = sparse.csr_matrix((point_count, point_count))
    identity = sparse.eye(point_count, format="csr") * 1e-6
    rhs = np.zeros(point_count, dtype=np.float64)
    np.add.at(rhs, observation_indices, weights * targets)
    solved = spsolve(W + regularizer + identity, rhs)
    return np.asarray(solved, dtype=np.float64)


def build_scale_aware_candidate(
    c2w_poses: np.ndarray,
    observations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Create a scale-aware c2w shadow candidate from floor-height factors."""
    started = time.perf_counter()
    c2w = np.asarray(c2w_poses, dtype=np.float64)
    point_count = len(c2w)
    valid = []
    for item in observations:
        try:
            index = int(item["trajectory_index"])
            height = float(item["height"])
            inliers = float(item.get("inlier_fraction", 0.0))
            alignment = float(item.get("normal_alignment", 0.0))
            residual = float(item.get("residual_ratio", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < point_count and height > 1e-6 and inliers >= 0.12 and alignment >= 0.82:
            valid.append((index, height, inliers * alignment / max(residual, 0.005)))
    rejection_reasons: list[str] = []
    required = max(10, min(24, int(math.ceil(point_count * 0.025))))
    if len(valid) < required:
        rejection_reasons.append("insufficient_floor_observations")
    if not valid:
        return {
            "c2w": c2w.astype(np.float32),
            "diagnostics": {
                "schema_version": SCALE_SCHEMA_VERSION,
                "method": "floor_height_log_scale_graph",
                "accepted": False,
                "rejection_reasons": rejection_reasons,
                "observation_count": 0,
                "runtime_seconds": float(time.perf_counter() - started),
            },
        }

    indices = np.asarray([item[0] for item in valid], dtype=np.int64)
    heights = np.asarray([item[1] for item in valid], dtype=np.float64)
    weights = np.asarray([item[2] for item in valid], dtype=np.float64)
    log_heights = np.log(heights)
    median = float(np.median(log_heights))
    mad = 1.4826 * float(np.median(np.abs(log_heights - median)))
    if mad > 1e-9:
        keep = np.abs(log_heights - median) <= max(3.5 * mad, 0.12)
        indices, heights, weights, log_heights = (
            indices[keep], heights[keep], weights[keep], log_heights[keep]
        )
    if len(indices) < required and "insufficient_floor_observations" not in rejection_reasons:
        rejection_reasons.append("insufficient_floor_observations_after_outlier_rejection")
    weights = np.clip(weights / max(float(np.median(weights)), 1e-9), 0.15, 8.0)
    target_log_height = float(np.average(log_heights, weights=weights))
    target_log_scale = target_log_height - log_heights
    log_scale = _smooth_log_scale(point_count, indices, target_log_scale, weights)
    # Keep the global gauge unchanged: this candidate corrects local scale,
    # not the arbitrary absolute monocular unit.
    log_scale -= float(np.average(log_scale[indices], weights=weights))
    log_scale = np.clip(log_scale, -math.log(1.8), math.log(1.8))
    scale = np.exp(log_scale)

    centers = c2w[:, :3, 3]
    deltas = np.diff(centers, axis=0)
    midpoint_scale = np.sqrt(scale[:-1] * scale[1:])
    corrected_centers = centers.copy()
    corrected_centers[1:] = centers[0] + np.cumsum(deltas * midpoint_scale[:, None], axis=0)
    candidate = c2w.copy()
    candidate[:, :3, 3] = corrected_centers

    coverage = (int(indices.max()) - int(indices.min())) / max(point_count - 1, 1)
    raw_steps = np.linalg.norm(deltas, axis=1)
    corrected_steps = np.linalg.norm(np.diff(corrected_centers, axis=0), axis=1)
    raw_length = float(raw_steps.sum())
    corrected_length = float(corrected_steps.sum())
    path_ratio = corrected_length / max(raw_length, 1e-12)
    correction_range = float(scale.max() / max(float(scale.min()), 1e-12))
    before_spread = float(np.std(log_heights))
    corrected_log_heights = log_heights + log_scale[indices]
    after_spread = float(np.std(corrected_log_heights))
    improvement = 1.0 - after_spread / max(before_spread, 1e-12)
    if coverage < 0.55:
        rejection_reasons.append("insufficient_temporal_coverage")
    if not 0.72 <= path_ratio <= 1.38:
        rejection_reasons.append("path_length_ratio_out_of_bounds")
    if correction_range > 2.25:
        rejection_reasons.append("scale_range_out_of_bounds")
    if before_spread >= 0.04 and improvement < 0.35:
        rejection_reasons.append("insufficient_height_consistency_improvement")
    if not np.isfinite(candidate).all():
        rejection_reasons.append("non_finite_candidate")

    diagnostics = {
        "schema_version": SCALE_SCHEMA_VERSION,
        "method": "floor_height_log_scale_graph",
        "accepted": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "observation_count": int(len(indices)),
        "required_observations": required,
        "temporal_coverage": float(coverage),
        "height_geometric_mean": float(math.exp(target_log_height)),
        "height_log_spread_before": before_spread,
        "height_log_spread_after": after_spread,
        "height_consistency_improvement": float(improvement),
        "scale_min": float(scale.min()),
        "scale_median": float(np.median(scale)),
        "scale_max": float(scale.max()),
        "scale_range": correction_range,
        "path_length_ratio": path_ratio,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return {"c2w": candidate.astype(np.float32), "scale": scale.astype(np.float32), "diagnostics": diagnostics}


def save_scale_aware_candidate(base: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    destination = Path(base)
    destination.mkdir(parents=True, exist_ok=True)
    diagnostics = dict(result.get("diagnostics") or {})
    c2w = np.asarray(result.get("c2w"), dtype=np.float32)
    scale = np.asarray(result.get("scale", np.ones(len(c2w))), dtype=np.float32)
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, c2w=c2w, scale=scale)
        temporary.replace(destination / SCALE_CANDIDATE_FILE)
    finally:
        temporary.unlink(missing_ok=True)
    diagnostics.update({"available": True, "point_count": int(len(c2w))})
    (destination / SCALE_SUMMARY_FILE).write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return diagnostics


def load_scale_aware_candidate_summary(base: str | Path) -> dict[str, Any]:
    path = Path(base) / SCALE_SUMMARY_FILE
    if not path.exists():
        return {"available": False, "accepted": False}
    try:
        result = json.loads(path.read_text())
        result["available"] = (Path(base) / SCALE_CANDIDATE_FILE).exists()
        return result
    except (OSError, json.JSONDecodeError):
        return {"available": False, "accepted": False, "error": "invalid_summary"}


def load_scale_aware_candidate_c2w(
    base: str | Path,
    *,
    expected_count: int,
    accepted_only: bool = True,
) -> np.ndarray | None:
    summary = load_scale_aware_candidate_summary(base)
    if not summary.get("available") or (accepted_only and not summary.get("accepted")):
        return None
    try:
        with np.load(Path(base) / SCALE_CANDIDATE_FILE, allow_pickle=False) as archive:
            c2w = np.asarray(archive["c2w"], dtype=np.float64)
        if c2w.shape != (expected_count, 4, 4) or not np.isfinite(c2w).all():
            return None
        return c2w
    except (OSError, KeyError, ValueError):
        return None
