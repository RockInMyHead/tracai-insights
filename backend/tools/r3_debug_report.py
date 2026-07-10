#!/usr/bin/env python3
"""Build a compact debug report for a completed R3 reconstruction.

Expected point format:
  pointcloud_full_debug.npz: [x, y, z, r, g, b, conf, frame_idx]
  pointcloud.npz:           [x, y, z, r, g, b, conf]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BASE_DIR = Path("/home/artem/trackai/gpu_worker_data/r3_output")
CONF_THRESHOLDS = [0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def load_points(output_dir: Path) -> tuple[np.ndarray, Path]:
    for name in ("pointcloud_full_debug.npz", "pointcloud.npz"):
        path = output_dir / name
        if path.exists():
            data = np.load(str(path))
            if "points" not in data:
                raise RuntimeError(f"{path} has no 'points' array")
            return data["points"], path
    raise FileNotFoundError(f"No pointcloud_full_debug.npz or pointcloud.npz in {output_dir}")


def percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    keys = ["p0", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p100"]
    vals = np.percentile(values, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    return {k: float(v) for k, v in zip(keys, vals)}


def inspect_camera(camera_file: Path) -> dict[str, Any]:
    data = np.load(str(camera_file))
    info: dict[str, Any] = {"file": camera_file.name, "keys": list(data.files)}
    if "pose" in data:
        pose = data["pose"]
        r = pose[:3, :3]
        t = pose[:3, 3] if pose.shape[1] >= 4 else np.zeros(3)
        info.update(
            {
                "pose_shape": list(pose.shape),
                "det_R": float(np.linalg.det(r)),
                "translation": t.tolist(),
                "translation_norm": float(np.linalg.norm(t)),
                "candidate_convention": "c2w_saved_by_r3_infer",
            }
        )
    if "intrinsics" in data:
        k = data["intrinsics"]
        info["intrinsics_shape"] = list(k.shape)
        info["intrinsics_diag"] = [float(k[0, 0]), float(k[1, 1])]
        info["intrinsics_center"] = [float(k[0, 2]), float(k[1, 2])]
    return info


def render_projection(points: np.ndarray, out_path: Path, axes: tuple[int, int], max_points: int) -> None:
    try:
        import cv2
    except Exception as exc:
        print(f"skip projection {out_path.name}: cv2 unavailable: {exc}")
        return

    if len(points) == 0:
        return

    sample = points
    if len(sample) > max_points:
        if sample.shape[1] > 6:
            idx = np.argsort(sample[:, 6])[-max_points:]
        else:
            idx = np.linspace(0, len(sample) - 1, max_points).astype(np.int64)
        sample = sample[idx]

    xy = sample[:, axes]
    finite = np.isfinite(xy).all(axis=1)
    sample = sample[finite]
    xy = xy[finite]
    if len(sample) == 0:
        return

    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    norm = np.clip((xy - lo) / span, 0, 1)

    w, h, pad = 1400, 1000, 36
    px = (pad + norm[:, 0] * (w - pad * 2)).astype(np.int32)
    py = (h - pad - norm[:, 1] * (h - pad * 2)).astype(np.int32)

    img = np.full((h, w, 3), 255, dtype=np.uint8)
    colors = np.clip(sample[:, 3:6], 0, 255).astype(np.uint8) if sample.shape[1] >= 6 else None
    if colors is None:
        colors = np.zeros((len(sample), 3), dtype=np.uint8)
    img[py, px] = colors
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def build_report(output_dir: Path, max_projection_points: int) -> dict[str, Any]:
    points, pointcloud_path = load_points(output_dir)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "success": True,
        "output_dir": str(output_dir),
        "pointcloud_file": pointcloud_path.name,
        "pointcloud_shape": list(points.shape),
        "has_conf": bool(points.ndim == 2 and points.shape[1] > 6),
        "has_frame_idx": bool(points.ndim == 2 and points.shape[1] > 7),
    }

    if points.ndim != 2 or points.shape[1] < 3:
        raise RuntimeError(f"Invalid point cloud shape: {points.shape}")

    xyz = points[:, :3]
    finite_xyz = np.isfinite(xyz).all(axis=1)
    report["finite_xyz_points"] = int(finite_xyz.sum())
    report["xyz"] = {
        "min": np.nanmin(xyz, axis=0).tolist(),
        "max": np.nanmax(xyz, axis=0).tolist(),
        "mean": np.nanmean(xyz, axis=0).tolist(),
        "std": np.nanstd(xyz, axis=0).tolist(),
    }
    if points.shape[1] >= 6:
        report["rgb"] = {
            "min": np.nanmin(points[:, 3:6], axis=0).tolist(),
            "max": np.nanmax(points[:, 3:6], axis=0).tolist(),
            "mean": np.nanmean(points[:, 3:6], axis=0).tolist(),
        }
    if points.shape[1] > 6:
        conf = points[np.isfinite(points[:, 6]), 6]
        report["confidence"] = {
            "percentiles": percentiles(conf),
            "counts_by_threshold": {str(t): int((conf >= t).sum()) for t in CONF_THRESHOLDS},
        }
    if points.shape[1] > 7:
        frames = points[np.isfinite(points[:, 7]), 7]
        report["frame_idx"] = {
            "min": int(frames.min()) if frames.size else None,
            "max": int(frames.max()) if frames.size else None,
            "unique_count": int(len(np.unique(frames))) if frames.size else 0,
        }

    counts = {
        "depth": len(list((output_dir / "depth").glob("*.npy"))),
        "conf": len(list((output_dir / "conf").glob("*.npy"))),
        "color": len(list((output_dir / "color").glob("*.png"))),
        "camera": len(list((output_dir / "camera").glob("*.npz"))),
    }
    report["files"] = counts

    run_params_path = output_dir / "run_params.json"
    if run_params_path.exists():
        try:
            report["run_params"] = json.loads(run_params_path.read_text())
        except Exception as exc:
            report["run_params"] = {"error": str(exc)}

    camera_files = sorted((output_dir / "camera").glob("*.npz"))
    sample_files = camera_files[:3] + camera_files[-3:] if len(camera_files) > 3 else camera_files
    report["camera_sample"] = []
    for camera_file in sample_files:
        try:
            report["camera_sample"].append(inspect_camera(camera_file))
        except Exception as exc:
            report["camera_sample"].append({"file": camera_file.name, "error": str(exc)})

    projections = {
        "top_x_y": ((0, 1), diagnostics_dir / "projection_top_x_y.png"),
        "front_x_z": ((0, 2), diagnostics_dir / "projection_front_x_z.png"),
        "right_y_z": ((1, 2), diagnostics_dir / "projection_right_y_z.png"),
    }
    report["projections"] = {}
    for name, (axes, path) in projections.items():
        render_projection(points, path, axes, max_projection_points)
        if path.exists():
            report["projections"][name] = str(path)

    report_path = diagnostics_dir / "debug_report.json"
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    report["report_path"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate R3 point-cloud diagnostics.")
    parser.add_argument("--video-id", help="Video id under r3_output")
    parser.add_argument("--output-dir", type=Path, help="Direct R3 output directory")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--max-projection-points", type=int, default=250000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir:
        output_dir = args.output_dir
    elif args.video_id:
        output_dir = args.base_dir / args.video_id
    else:
        raise SystemExit("Provide --video-id or --output-dir")

    report = build_report(output_dir, max_projection_points=args.max_projection_points)
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
