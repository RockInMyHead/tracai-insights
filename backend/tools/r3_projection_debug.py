#!/usr/bin/env python3
"""Generate server-side R3 point-cloud projections without Three.js.

This script is intentionally independent from the web viewer. It reads the
same NPZ cloud and saves 2D PNG projections so backend/export parity can be
checked before changing camera/mapping in the frontend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_BASE_DIR = Path("/home/artem/trackai/gpu_worker_data/r3_output")


def load_pointcloud(output_dir: Path) -> tuple[np.ndarray, Path]:
    for name in ("pointcloud_full_debug.npz", "pointcloud.npz"):
        path = output_dir / name
        if path.exists():
            data = np.load(str(path))
            key = "points" if "points" in data.files else data.files[0]
            arr = data[key]
            if arr.ndim != 2 or arr.shape[1] < 7:
                raise RuntimeError(f"Bad pointcloud shape: {arr.shape}. Expected at least x,y,z,r,g,b,conf")
            return arr, path
    raise FileNotFoundError(f"No pointcloud_full_debug.npz or pointcloud.npz in {output_dir}")


def sample_points(arr: np.ndarray, max_points: int, sampling: str) -> np.ndarray:
    if len(arr) <= max_points:
        return arr

    rng = np.random.default_rng(42)
    if sampling == "confidence_top":
        idx = np.argsort(arr[:, 6])[-max_points:]
        return arr[np.sort(idx)]

    if sampling == "random":
        idx = rng.choice(len(arr), size=max_points, replace=False)
        return arr[np.sort(idx)]

    if sampling == "per_frame_uniform":
        if arr.shape[1] < 8:
            raise RuntimeError("per_frame_uniform requires frame_idx column")
        frames = arr[:, 7]
        frame_ids = np.unique(frames[np.isfinite(frames)].astype(np.int64))
        if len(frame_ids) == 0:
            return arr[:0]
        per_frame = max(1, max_points // len(frame_ids))
        chunks = []
        for frame_id in frame_ids:
            frame_points = arr[frames == frame_id]
            if len(frame_points) > per_frame:
                idx = rng.choice(len(frame_points), size=per_frame, replace=False)
                frame_points = frame_points[np.sort(idx)]
            chunks.append(frame_points)
        result = np.concatenate(chunks, axis=0) if chunks else arr[:0]
        if len(result) > max_points:
            idx = rng.choice(len(result), size=max_points, replace=False)
            result = result[np.sort(idx)]
        return result

    raise ValueError(f"Unknown sampling: {sampling}")


def filter_points(
    arr: np.ndarray,
    min_conf: float,
    max_points: int,
    sampling: str,
    frame_start: int | None,
    frame_end: int | None,
) -> np.ndarray:
    mask = np.isfinite(arr[:, :7]).all(axis=1) & (arr[:, 6] >= min_conf)
    if frame_start is not None or frame_end is not None:
        if arr.shape[1] < 8:
            raise RuntimeError("frame filtering requires frame_idx column")
        frames = arr[:, 7]
        mask &= np.isfinite(frames)
        if frame_start is not None:
            mask &= frames >= frame_start
        if frame_end is not None:
            mask &= frames <= frame_end
    return sample_points(arr[mask], max_points=max_points, sampling=sampling)


def save_projection(arr: np.ndarray, out_path: Path, axes: tuple[int, int], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for r3_projection_debug.py") from exc

    x = arr[:, axes[0]]
    y = arr[:, axes[1]]
    rgb = arr[:, 3:6]
    if np.nanmax(rgb) > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0, 1)

    plt.figure(figsize=(12, 12))
    plt.scatter(x, y, s=0.1, c=rgb)
    plt.axis("equal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def stats(arr: np.ndarray) -> dict:
    out = {
        "shape": list(arr.shape),
        "xyz_min": arr[:, :3].min(axis=0).tolist(),
        "xyz_max": arr[:, :3].max(axis=0).tolist(),
        "xyz_std": arr[:, :3].std(axis=0).tolist(),
        "conf_percentiles": np.percentile(arr[:, 6], [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]).tolist(),
    }
    if arr.shape[1] >= 8:
        frames = arr[:, 7]
        finite = frames[np.isfinite(frames)]
        out["frame_idx"] = {
            "min": int(finite.min()) if finite.size else None,
            "max": int(finite.max()) if finite.size else None,
            "unique": int(len(np.unique(finite.astype(np.int64)))) if finite.size else 0,
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--min-conf", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=300000)
    parser.add_argument("--sampling", choices=["random", "confidence_top", "per_frame_uniform"], default="random")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.base_dir / args.video_id
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    raw, source_path = load_pointcloud(output_dir)
    filtered = filter_points(
        raw,
        min_conf=args.min_conf,
        max_points=args.max_points,
        sampling=args.sampling,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    )

    suffix = f"{args.sampling}_conf{str(args.min_conf).replace('.', '_')}_{args.max_points}"
    if args.frame_start is not None or args.frame_end is not None:
        suffix += f"_f{args.frame_start if args.frame_start is not None else 'all'}_{args.frame_end if args.frame_end is not None else 'all'}"

    outputs = {
        "top_xy": str(diag_dir / f"top_xy_{suffix}.png"),
        "top_xz": str(diag_dir / f"top_xz_{suffix}.png"),
        "front_yz": str(diag_dir / f"front_yz_{suffix}.png"),
    }
    save_projection(filtered, Path(outputs["top_xy"]), (0, 1), "Top XY")
    save_projection(filtered, Path(outputs["top_xz"]), (0, 2), "Top XZ")
    save_projection(filtered, Path(outputs["front_yz"]), (1, 2), "Front YZ")

    report = {
        "source_file": str(source_path),
        "raw": stats(raw),
        "filtered": stats(filtered) if len(filtered) else {"shape": list(filtered.shape)},
        "outputs": outputs,
    }
    report_path = diag_dir / f"projection_report_{suffix}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
