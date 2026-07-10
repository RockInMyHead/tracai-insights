#!/usr/bin/env python3
"""Export TrackAI point clouds using the same math as R3/view.py.

This is a backend parity tool, not a viewer. It reads an existing R3 output
directory containing depth/, color/, conf/, camera/ and writes:

  pointcloud_full_debug.r3_view_parity.npz

Point format:

  [x, y, z, r, g, b, conf, frame_idx]

The world transform is intentionally copied from R3/view.py:

  pts_cam = depth_to_cam_coords_points(depth, intrinsics)
  pts_world = pts_cam.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np


DEFAULT_BASE_DIR = Path("/home/artem/trackai/gpu_worker_data/r3_output")
R3_DIR = Path("/home/artem/trackai/R3")


def import_r3():
    if str(R3_DIR) not in sys.path:
        sys.path.insert(0, str(R3_DIR))
    from R3.utils.components_geometry import depth_to_cam_coords_points

    return depth_to_cam_coords_points


def load_frames(data_dir: Path) -> list[int]:
    depth_files = sorted(glob.glob(str(data_dir / "depth" / "*.npy")))
    return [int(os.path.splitext(os.path.basename(f))[0]) for f in depth_files]


def load_frame(data_dir: Path, frame_id: int) -> dict:
    import cv2

    tag = f"{frame_id:06d}"
    depth = np.load(str(data_dir / "depth" / f"{tag}.npy"))
    conf = np.load(str(data_dir / "conf" / f"{tag}.npy"))
    bgr = cv2.imread(str(data_dir / "color" / f"{tag}.png"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(data_dir / "color" / f"{tag}.png")
    color = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cam = np.load(str(data_dir / "camera" / f"{tag}.npz"))
    pose = cam["pose"]
    if pose.shape == (3, 4):
        pose_4x4 = np.eye(4, dtype=pose.dtype)
        pose_4x4[:3, :] = pose
        pose = pose_4x4
    return {"depth": depth, "conf": conf, "color": color, "pose": pose, "intrinsics": cam["intrinsics"]}


def frame_to_world(frame: dict, depth_to_cam_coords_points) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts_cam = depth_to_cam_coords_points(frame["depth"], frame["intrinsics"])
    pose = frame["pose"]
    if pose.shape == (3, 4):
        pose_4x4 = np.eye(4, dtype=pose.dtype)
        pose_4x4[:3, :] = pose
        pose = pose_4x4
    pts_world = (pts_cam.reshape(-1, 3) @ pose[:3, :3].T) + pose[:3, 3]
    colors = frame["color"].reshape(-1, 3)
    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32) / 255.0
    else:
        colors = colors.astype(np.float32)
        if colors.max(initial=0.0) > 1.0:
            colors = colors / 255.0
    conf = frame["conf"].reshape(-1).astype(np.float32)
    return pts_world.astype(np.float32), np.clip(colors, 0.0, 1.0), conf


def export_parity_cloud(
    output_dir: Path,
    min_conf: float,
    stride: int,
    max_points: int,
    replace_main: bool,
) -> dict:
    depth_to_cam_coords_points = import_r3()
    frame_ids = load_frames(output_dir)
    if not frame_ids:
        raise RuntimeError(f"No R3 frames found in {output_dir}")

    chunks = []
    frames_used = 0
    for frame_id in frame_ids:
        frame = load_frame(output_dir, frame_id)
        pts, colors, conf = frame_to_world(frame, depth_to_cam_coords_points)
        h, w = frame["depth"].shape
        keep = np.zeros((h, w), dtype=bool)
        keep[::stride, ::stride] = True
        keep = keep.reshape(-1)
        mask = keep & np.isfinite(pts).all(axis=1) & np.isfinite(conf) & (conf > min_conf)
        if not mask.any():
            continue
        frame_idx = np.full((int(mask.sum()), 1), float(frame_id), dtype=np.float32)
        chunks.append(np.concatenate([pts[mask], colors[mask], conf[mask, None], frame_idx], axis=1))
        frames_used += 1

    if not chunks:
        raise RuntimeError("No points generated")

    combined = np.concatenate(chunks, axis=0).astype(np.float32)
    rng = np.random.default_rng(42)
    production = combined
    if max_points > 0 and len(production) > max_points:
        idx = rng.choice(len(production), size=max_points, replace=False)
        production = production[np.sort(idx)]

    out_debug = output_dir / "pointcloud_full_debug.r3_view_parity.npz"
    np.savez_compressed(str(out_debug), points=combined)

    out_prod = output_dir / "pointcloud.r3_view_parity.npz"
    np.savez_compressed(str(out_prod), points=production[:, :7])

    if replace_main:
        np.savez_compressed(str(output_dir / "pointcloud_full_debug.npz"), points=combined)
        np.savez_compressed(str(output_dir / "pointcloud.npz"), points=production[:, :7])

    report = {
        "success": True,
        "output_dir": str(output_dir),
        "frames_total": len(frame_ids),
        "frames_used": frames_used,
        "min_conf": min_conf,
        "stride": stride,
        "max_points": max_points,
        "replace_main": replace_main,
        "debug_file": str(out_debug),
        "production_file": str(out_prod),
        "points_full": int(len(combined)),
        "points_production": int(len(production)),
        "xyz_min": combined[:, :3].min(axis=0).tolist(),
        "xyz_max": combined[:, :3].max(axis=0).tolist(),
        "xyz_std": combined[:, :3].std(axis=0).tolist(),
        "conf_percentiles": np.percentile(combined[:, 6], [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]).tolist(),
    }
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    report_path = diag_dir / "r3_view_parity_export.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    report["report_path"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--min-conf", type=float, default=1.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--replace-main", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.base_dir / args.video_id
    report = export_parity_cloud(
        output_dir=output_dir,
        min_conf=args.min_conf,
        stride=max(1, args.stride),
        max_points=args.max_points,
        replace_main=args.replace_main,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
