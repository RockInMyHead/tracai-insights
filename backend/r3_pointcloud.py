"""Memory-bounded R3 depth back-projection for production point clouds."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


ProgressCallback = Callable[[dict[str, Any]], None]
CancelCallback = Callable[[], bool]


class PointCloudBuildCancelled(RuntimeError):
    """Raised when a newer reconstruction supersedes a cloud build."""


def _env_enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _emit(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is not None:
        callback(payload)


def _save_npz_atomic(path: Path, points: np.ndarray) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        np.savez_compressed(handle, points=points)
    os.replace(temp_path, path)


def _confidence_stats(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"percentiles": {}}
    levels = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    keys = [f"p{value}" for value in levels]
    percentiles = np.percentile(finite, levels)
    return {
        "percentiles": {key: float(value) for key, value in zip(keys, percentiles)},
        "counts_by_threshold": {
            str(threshold): int((finite >= threshold).sum())
            for threshold in (0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
        },
    }


def build_sampled_pointcloud(
    output_dir: Path,
    *,
    stride: int = 4,
    max_points: int = 200_000,
    min_conf: float = 1.0,
    save_full_debug: bool | None = None,
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
    return_points: bool = True,
) -> dict[str, Any]:
    """Build a per-frame-uniform cloud without retaining every depth point.

    The production path keeps at most ``max_points`` across all frames and
    stores frame ids in column 8.  A full multi-million-point cloud is only
    assembled when ``R3_SAVE_FULL_DEBUG=true`` (or ``save_full_debug=True``).
    """
    try:
        import cv2
    except ImportError:  # RGB is optional; geometry can still be exported.
        cv2 = None

    started = time.time()
    output_dir = Path(output_dir)
    camera_dir = output_dir / "camera"
    depth_dir = output_dir / "depth"
    conf_dir = output_dir / "conf"
    frames_dir = output_dir / "frames"
    color_dir = output_dir / "color"

    camera_files = sorted(camera_dir.glob("*.npz")) if camera_dir.exists() else []
    if len(camera_files) < 2 or not depth_dir.exists():
        raise FileNotFoundError("R3 camera/depth artifacts are not available")

    stride = max(1, int(stride))
    max_points = max(1_000, int(max_points))
    save_full_debug = (
        _env_enabled("R3_SAVE_FULL_DEBUG", False)
        if save_full_debug is None
        else bool(save_full_debug)
    )
    per_frame_limit = max(1, int(math.ceil(max_points / len(camera_files))))
    sampled_chunks: list[np.ndarray] = []
    debug_chunks: list[np.ndarray] = []
    grid_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    frames_used = 0
    total_valid_points = 0

    _emit(
        progress_callback,
        status="processing",
        stage="backproject",
        progress=0,
        total_frames=len(camera_files),
        message="Построение оптимизированного 3D-облака",
    )

    for file_index, camera_file in enumerate(camera_files):
        if should_cancel is not None and should_cancel():
            raise PointCloudBuildCancelled("Point cloud build cancelled")
        try:
            camera = np.load(str(camera_file), allow_pickle=True)
            pose = np.asarray(camera["pose"], dtype=np.float32)
            intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32) if "intrinsics" in camera else None

            depth_path = depth_dir / f"{camera_file.stem}.npy"
            if not depth_path.exists():
                continue
            depth = np.asarray(np.load(str(depth_path), allow_pickle=True), dtype=np.float32)
            height, width = depth.shape

            grid_key = (height, width, stride)
            if grid_key not in grid_cache:
                ys, xs = np.meshgrid(
                    np.arange(0, height, stride, dtype=np.int32),
                    np.arange(0, width, stride, dtype=np.int32),
                    indexing="ij",
                )
                grid_cache[grid_key] = (ys.ravel(), xs.ravel())
            ys, xs = grid_cache[grid_key]
            depth_values = depth[ys, xs]
            valid = np.isfinite(depth_values) & (depth_values > 0.001)

            conf_path = conf_dir / f"{camera_file.stem}.npy"
            confidence = None
            if conf_path.exists():
                confidence = np.asarray(np.load(str(conf_path), allow_pickle=True), dtype=np.float32)
                confidence_values = confidence[ys, xs]
                valid &= np.isfinite(confidence_values) & (confidence_values > float(min_conf))
            else:
                confidence_values = np.full(depth_values.shape, 2.0, dtype=np.float32)

            valid_indices = np.flatnonzero(valid)
            if valid_indices.size == 0:
                continue
            total_valid_points += int(valid_indices.size)

            debug_indices = valid_indices if save_full_debug else None
            if valid_indices.size > per_frame_limit:
                positions = np.linspace(0, valid_indices.size - 1, per_frame_limit).round().astype(np.int64)
                production_indices = valid_indices[positions]
            else:
                production_indices = valid_indices

            rgb = None
            frame_path = color_dir / f"{camera_file.stem}.png"
            if not frame_path.exists():
                frame_path = frames_dir / f"frame_{int(camera_file.stem):06d}.jpg"
            if cv2 is not None and frame_path.exists():
                bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if rgb.shape[:2] != (height, width):
                        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)

            fx = float(intrinsics[0, 0]) if intrinsics is not None else float(max(height, width)) * 1.2
            fy = float(intrinsics[1, 1]) if intrinsics is not None else fx
            cx = float(intrinsics[0, 2]) if intrinsics is not None else width / 2.0
            cy = float(intrinsics[1, 2]) if intrinsics is not None else height / 2.0
            if pose.shape == (3, 4):
                pose_matrix = np.eye(4, dtype=np.float32)
                pose_matrix[:3, :] = pose
            else:
                pose_matrix = pose

            def make_chunk(indices: np.ndarray) -> np.ndarray:
                selected_xs = xs[indices]
                selected_ys = ys[indices]
                selected_depth = depth_values[indices]
                x_camera = (selected_xs.astype(np.float32) - cx) * selected_depth / fx
                y_camera = (selected_ys.astype(np.float32) - cy) * selected_depth / fy
                camera_points = np.stack(
                    [x_camera, y_camera, selected_depth, np.ones_like(selected_depth)],
                    axis=-1,
                )
                world_points = (pose_matrix @ camera_points.T).T[:, :3].astype(np.float32, copy=False)
                if rgb is not None:
                    colors = rgb[selected_ys, selected_xs].astype(np.float32) / 255.0
                else:
                    colors = np.full((len(indices), 3), 0.75, dtype=np.float32)
                conf_values = confidence_values[indices].astype(np.float32, copy=False)
                frame_values = np.full((len(indices), 1), int(camera_file.stem), dtype=np.float32)
                return np.concatenate(
                    [world_points, colors, conf_values[:, None], frame_values],
                    axis=1,
                ).astype(np.float32, copy=False)

            sampled_chunks.append(make_chunk(production_indices))
            if debug_indices is not None:
                debug_chunks.append(make_chunk(debug_indices))
            frames_used += 1

        except PointCloudBuildCancelled:
            raise
        except Exception:
            continue

        if file_index == len(camera_files) - 1 or (file_index + 1) % 25 == 0:
            _emit(
                progress_callback,
                status="processing",
                stage="backproject",
                progress=int((file_index + 1) * 85 / len(camera_files)),
                processed_frames=file_index + 1,
                total_frames=len(camera_files),
                message=f"3D-облако: обработано {file_index + 1}/{len(camera_files)} кадров",
            )

    if not sampled_chunks:
        raise RuntimeError("No valid R3 depth points were generated")

    production = np.concatenate(sampled_chunks, axis=0)
    finite = np.isfinite(production[:, :7]).all(axis=1)
    production = production[finite]
    if len(production) > max_points:
        indices = np.linspace(0, len(production) - 1, max_points).round().astype(np.int64)
        production = production[indices]
    if len(production) < 100:
        raise RuntimeError("R3 point cloud contains fewer than 100 valid points")

    _emit(
        progress_callback,
        status="processing",
        stage="saving",
        progress=90,
        message=f"Сохранение {len(production):,} точек",
    )
    _save_npz_atomic(output_dir / "pointcloud.npz", production)

    stats_values = production[:, 6]
    if save_full_debug and debug_chunks:
        _emit(
            progress_callback,
            status="processing",
            stage="debug",
            progress=92,
            message="Сохранение полного диагностического облака",
        )
        debug_points = np.concatenate(debug_chunks, axis=0)
        debug_points = debug_points[np.isfinite(debug_points[:, :7]).all(axis=1)]
        _save_npz_atomic(output_dir / "pointcloud_full_debug.npz", debug_points)
        stats_values = debug_points[:, 6]

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    stats = _confidence_stats(stats_values)
    stats.update({
        "total_valid_points": total_valid_points,
        "production_points": int(len(production)),
        "frames_used": frames_used,
        "sampling": "per_frame_uniform",
        "stride": stride,
        "full_debug_saved": bool(save_full_debug),
    })
    (diagnostics_dir / "conf_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    elapsed = round(time.time() - started, 2)
    _emit(
        progress_callback,
        status="completed",
        stage="ready",
        progress=100,
        message=f"3D-облако готово: {len(production):,} точек",
        points=int(len(production)),
        elapsed_seconds=elapsed,
    )
    return {
        "points": production.tolist() if return_points else None,
        "num_points": int(len(production)),
        "source_points": total_valid_points,
        "frames_used": frames_used,
        "elapsed_seconds": elapsed,
        "full_debug_saved": bool(save_full_debug),
    }
