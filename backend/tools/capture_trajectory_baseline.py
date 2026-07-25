#!/usr/bin/env python3
"""Create an immutable, reproducible baseline bundle for one problematic R3 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from r3_trajectory import build_r3_trajectory  # noqa: E402
from r3_trajectory_sources import select_r3_trajectory_camera_poses  # noqa: E402
from trajectory_geometry import deviation_from_reference, trajectory_metrics  # noqa: E402


REQUIRED_FILES = (
    "pose_graph_edges.npz",
    "pose_graph_candidate.json",
    "pose_graph_candidate.npz",
    "scale_aware_candidate.json",
    "scale_aware_candidate.npz",
    "frame_selection.json",
    "run_params.json",
)
OPTIONAL_FILES = ("pose_conf.npy", "pose_edge_log.json")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _analysis_result(document: dict[str, Any]) -> dict[str, Any]:
    nested = document.get("analysis_result")
    return nested if isinstance(nested, dict) else document


def _load_camera_poses(base: Path) -> list[dict[str, Any]]:
    poses = []
    for path in sorted((base / "camera").glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            poses.append({
                "frame": int(path.stem),
                "pose": np.asarray(payload["pose"]).tolist(),
                "intrinsics": (
                    np.asarray(payload["intrinsics"]).tolist()
                    if "intrinsics" in payload else None
                ),
            })
    return poses


def _load_optional_array(base: Path, name: str) -> Any:
    path = base / name
    return np.load(path, allow_pickle=False).tolist() if path.exists() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_artifacts(r3_output: Path, destination: Path) -> list[dict[str, Any]]:
    artifacts = destination / "artifacts"
    artifacts.mkdir(parents=True)
    copied: list[Path] = []
    camera_source = r3_output / "camera"
    if not camera_source.is_dir():
        raise FileNotFoundError(f"camera directory not found: {camera_source}")
    shutil.copytree(camera_source, artifacts / "camera")
    copied.extend(path for path in (artifacts / "camera").rglob("*") if path.is_file())
    for name in REQUIRED_FILES:
        source = r3_output / name
        if not source.is_file():
            raise FileNotFoundError(f"required artifact not found: {source}")
        target = artifacts / name
        shutil.copy2(source, target)
        copied.append(target)
    for name in OPTIONAL_FILES:
        source = r3_output / name
        if source.is_file():
            target = artifacts / name
            shutil.copy2(source, target)
            copied.append(target)
    return [
        {
            "path": str(path.relative_to(destination)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(copied)
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_report(
    r3_output: Path,
    analysis_document: dict[str, Any],
    map_context: dict[str, Any],
    ground_truth: Any,
) -> tuple[dict[str, Any], dict[str, list[list[float]]]]:
    camera_poses = _load_camera_poses(r3_output)
    frame_selection = _load_json(r3_output / "frame_selection.json")
    run_params = _load_json(r3_output / "run_params.json")
    confidence = _load_optional_array(r3_output, "pose_conf.npy")
    trajectories: dict[str, list[list[float]]] = {}
    source_details: dict[str, Any] = {}
    for source in ("raw", "robust_candidate", "scale_aware_candidate"):
        selected, selection = select_r3_trajectory_camera_poses(
            r3_output, camera_poses, source
        )
        bundle = build_r3_trajectory(
            selected, confidence, frame_selection, run_params
        )
        points = bundle.get("plan_trajectory") or []
        trajectories[source] = points
        source_details[source] = {
            "selection": selection,
            "metrics": trajectory_metrics(points),
            "projection": (bundle.get("trajectory_quality") or {}).get("projection"),
            "trajectory_quality": bundle.get("trajectory_quality"),
        }

    result = _analysis_result(analysis_document)
    final_map = result.get("map_trajectory") or []
    trajectories["final_map"] = final_map
    floorplan = result.get("floorplan_constraint") or {}
    map_metadata = result.get("map_metadata") or {}
    meters_per_pixel = float(
        map_metadata.get("meters_per_pixel")
        or floorplan.get("meters_per_pixel")
        or 1.0
    )
    source_details["final_map"] = {
        "metrics": trajectory_metrics(
            final_map,
            units_per_coordinate=meters_per_pixel,
            unit_name="meter",
        ),
        "map_matching_applied": bool(
            (result.get("processing_stats") or {}).get("map_matching_applied")
        ),
        "map_repair_meters": {
            "p50": floorplan.get("correction_median_meters"),
            "p95": floorplan.get("correction_p95_meters"),
            "max": floorplan.get("maximum_correction_meters"),
        },
        "floorplan_constraint": floorplan,
    }
    raw = trajectories["raw"]
    for name, points in trajectories.items():
        source_details[name]["deviation_from_raw"] = (
            {
                "available": True,
                "fit": "identity",
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "max": 0.0,
                "p95_over_raw_length": 0.0,
            }
            if name == "raw"
            else deviation_from_reference(raw, points)
        )
    return {
        "schema_version": 1,
        "sources": source_details,
        "map_context": {
            "reference_point": map_context.get("reference_point"),
            "direction_point": map_context.get("direction_point"),
            "drawn_plan": map_context.get("drawn_plan"),
            "floorplan_id": map_context.get("floorplan_id"),
        },
        "ground_truth": {
            "metrics": trajectory_metrics(_ground_truth_points(ground_truth)),
            "deviation_from_raw": deviation_from_reference(
                trajectories["raw"], _ground_truth_points(ground_truth)
            ),
        },
        "notes": {
            "r3_units": "arbitrary reconstruction units",
            "final_map_units": "meters when meters_per_pixel is available",
            "deviation": "shape-only comparison after non-reflecting similarity fit",
        },
    }, trajectories


def _ground_truth_points(value: Any) -> Any:
    if isinstance(value, list):
        if value and isinstance(value[0], dict) and "points" in value[0]:
            points = []
            for shape in value:
                if isinstance(shape, dict) and isinstance(shape.get("points"), list):
                    points.extend(shape["points"])
            return points
        return value
    if isinstance(value, dict):
        for key in ("trajectory", "points", "ground_truth_trajectory", "route"):
            if isinstance(value.get(key), list):
                return value[key]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture all artifacts and geometry metrics for one R3 run."
    )
    parser.add_argument("--r3-output", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--map-context", type=Path, required=True)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="JSON containing the operator-drawn real trajectory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"output already exists; baseline capture never overwrites: {args.output}"
        )
    analysis = _load_json(args.analysis)
    map_context = _load_json(args.map_context)
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    try:
        files = _copy_artifacts(args.r3_output, args.output)
        shutil.copy2(args.analysis, args.output / "analysis.json")
        shutil.copy2(args.map_context, args.output / "map_context.json")
        shutil.copy2(args.ground_truth, args.output / "ground_truth.json")
        report, trajectories = build_report(
            args.r3_output, analysis, map_context, ground_truth
        )
        _write_json(args.output / "trajectory_report.json", report)
        _write_json(args.output / "trajectories.json", trajectories)
        manifest_files = [
            path for path in args.output.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        ]
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_r3_output": str(args.r3_output.resolve()),
            "source_analysis": str(args.analysis.resolve()),
            "source_map_context": str(args.map_context.resolve()),
            "source_ground_truth": str(args.ground_truth.resolve()),
            "artifact_files": files,
            "files": [
                {
                    "path": str(path.relative_to(args.output)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(manifest_files)
            ],
        }
        _write_json(args.output / "manifest.json", manifest)
    except Exception:
        shutil.rmtree(args.output)
        raise
    print(f"Baseline captured: {args.output}")


if __name__ == "__main__":
    main()
