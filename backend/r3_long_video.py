"""Segment planning and robust Sim(3) stitching for long R3 videos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SegmentWindow:
    index: int
    start: int
    end: int

    @property
    def frame_count(self) -> int:
        return self.end - self.start


def plan_segment_windows(
    frame_count: int,
    segment_frames: int = 1_500,
    overlap_frames: int = 90,
) -> list[SegmentWindow]:
    """Split selected frames into overlapping, complete-coverage windows."""
    frame_count = max(0, int(frame_count))
    segment_frames = max(32, int(segment_frames))
    overlap_frames = max(3, min(int(overlap_frames), segment_frames // 3))
    if frame_count == 0:
        return []
    if frame_count <= segment_frames:
        return [SegmentWindow(0, 0, frame_count)]

    windows: list[SegmentWindow] = []
    step = segment_frames - overlap_frames
    start = 0
    while start < frame_count:
        end = min(frame_count, start + segment_frames)
        windows.append(SegmentWindow(len(windows), start, end))
        if end >= frame_count:
            break
        start += step
    return windows


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrix)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = 1.0 if np.linalg.det(u @ vh) >= 0.0 else -1.0
    return u @ correction @ vh


def estimate_pose_similarity(
    global_poses: Sequence[np.ndarray],
    local_poses: Sequence[np.ndarray],
) -> tuple[np.ndarray, float, np.ndarray, dict[str, float | int]]:
    """Estimate global_t = scale * rotation @ local_t + translation.

    Camera rotations resolve the otherwise ambiguous rotation of a nearly
    straight overlap. Translation scale is estimated robustly from overlap
    baselines and the final offset uses a coordinate-wise median.
    """
    if len(global_poses) != len(local_poses) or not global_poses:
        raise ValueError("At least one matched global/local pose pair is required")

    global_matrices = [np.asarray(pose, dtype=np.float64) for pose in global_poses]
    local_matrices = [np.asarray(pose, dtype=np.float64) for pose in local_poses]
    rotation_candidates = [
        global_pose[:3, :3] @ local_pose[:3, :3].T
        for global_pose, local_pose in zip(global_matrices, local_matrices)
    ]
    rotation = _project_rotation(np.sum(rotation_candidates, axis=0))

    global_points = np.vstack([pose[:3, 3] for pose in global_matrices])
    local_points = np.vstack([pose[:3, 3] for pose in local_matrices])
    rotated_local = (rotation @ local_points.T).T

    scale_candidates: list[float] = []
    for left in range(len(global_points) - 1):
        for right in range(left + 1, len(global_points)):
            local_distance = float(np.linalg.norm(rotated_local[right] - rotated_local[left]))
            global_distance = float(np.linalg.norm(global_points[right] - global_points[left]))
            if local_distance > 1e-6 and global_distance > 1e-6:
                scale_candidates.append(global_distance / local_distance)
    scale = float(np.median(scale_candidates)) if scale_candidates else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    # Prevent one weak overlap from catastrophically resizing the remainder.
    scale = float(np.clip(scale, 0.25, 4.0))

    offsets = global_points - scale * rotated_local
    translation = np.median(offsets, axis=0)
    predicted = scale * rotated_local + translation
    residuals = np.linalg.norm(predicted - global_points, axis=1)
    diagnostics: dict[str, float | int] = {
        "overlap_pairs": len(global_points),
        "scale_candidates": len(scale_candidates),
        "scale": round(scale, 8),
        "median_residual": round(float(np.median(residuals)), 8),
        "max_residual": round(float(np.max(residuals)), 8),
    }
    return rotation, scale, translation, diagnostics


def transform_camera_pose(
    pose: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply a world-space Sim(3) transform to one c2w camera pose."""
    source = np.asarray(pose, dtype=np.float64)
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = rotation @ source[:3, :3]
    output[:3, 3] = scale * (rotation @ source[:3, 3]) + translation
    if source.shape == (3, 4):
        return output[:3, :]
    return output


def align_segment_poses(
    local_poses: Mapping[int, np.ndarray],
    global_indices: Sequence[int],
    merged_poses: Mapping[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], float, dict[str, float | int]]:
    """Align local segment poses to already merged overlap poses."""
    matched_global: list[np.ndarray] = []
    matched_local: list[np.ndarray] = []
    for local_index, global_index in enumerate(global_indices):
        local_pose = local_poses.get(local_index)
        global_pose = merged_poses.get(int(global_index))
        if local_pose is not None and global_pose is not None:
            matched_global.append(global_pose)
            matched_local.append(local_pose)

    if not merged_poses:
        rotation = np.eye(3, dtype=np.float64)
        scale = 1.0
        translation = np.zeros(3, dtype=np.float64)
        diagnostics: dict[str, float | int] = {
            "overlap_pairs": 0,
            "scale_candidates": 0,
            "scale": 1.0,
            "median_residual": 0.0,
            "max_residual": 0.0,
        }
    elif matched_global:
        rotation, scale, translation, diagnostics = estimate_pose_similarity(
            matched_global,
            matched_local,
        )
    else:
        # Defensive fallback: preserve local scale and attach the segment's
        # first camera to the latest global camera.
        first_local_index = min(local_poses)
        latest_global_index = max(merged_poses)
        local_anchor = local_poses[first_local_index]
        global_anchor = merged_poses[latest_global_index]
        rotation = _project_rotation(global_anchor[:3, :3] @ local_anchor[:3, :3].T)
        scale = 1.0
        translation = global_anchor[:3, 3] - rotation @ local_anchor[:3, 3]
        diagnostics = {
            "overlap_pairs": 0,
            "scale_candidates": 0,
            "scale": 1.0,
            "median_residual": 0.0,
            "max_residual": 0.0,
        }

    aligned: dict[int, np.ndarray] = {}
    for local_index, global_index in enumerate(global_indices):
        pose = local_poses.get(local_index)
        if pose is None:
            continue
        aligned[int(global_index)] = transform_camera_pose(pose, rotation, scale, translation)
    return aligned, scale, diagnostics
