"""Validation and diagnostics for exported R3 relative-pose measurements.

R3's final camera poses are an estimate, not the measurements that produced
that estimate.  A robust SE(3)/Sim(3) backend needs the original sparse edges,
including translation/rotation confidence, so it can reject bad constraints
and re-optimize after loop closure.  This module deliberately validates and
summarizes those edges without changing trajectory geometry.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


R3_POSE_GRAPH_SCHEMA_VERSION = 1
R3_POSE_ENCODING = "txyz_qxyzw_fovxy"
R3_RELATIVE_TRANSFORM_CONVENTION = "target_hmat=relative_hmat@reference_hmat"
R3_ABSOLUTE_POSE_SPACE = "world_to_camera"
R3_CONFIDENCE_SEMANTICS = "softplus_positive_weight_not_covariance"


def _percentile_summary(values: Any) -> dict[str, float] | None:
    array = np.asarray(values)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    percentiles = np.percentile(array, [0, 10, 50, 90, 100])
    return {
        key: round(float(value), 6)
        for key, value in zip(("min", "p10", "p50", "p90", "max"), percentiles)
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _npz_scalar(payload: Any, key: str) -> Any:
    if key not in getattr(payload, "files", []):
        return None
    array = np.asarray(payload[key])
    if array.size == 0:
        return None
    value = array.reshape(-1)[0]
    return value.item() if isinstance(value, np.generic) else value


def _connectivity_metrics(
    frame_i: np.ndarray,
    frame_j: np.ndarray,
    point_count: int,
) -> dict[str, Any]:
    if frame_i.size == 0:
        return {
            "connected_frames": 0,
            "component_count": 0,
            "largest_component_frames": 0,
            "largest_component_coverage": 0.0 if point_count > 0 else None,
        }
    maximum_frame = int(max(np.max(frame_i), np.max(frame_j)))
    capacity = max(point_count, maximum_frame + 1)
    parent = np.arange(capacity, dtype=np.int64)
    component_size = np.ones(capacity, dtype=np.int64)
    active = np.zeros(capacity, dtype=bool)
    active[frame_i] = True
    active[frame_j] = True

    def find(frame: int) -> int:
        while parent[frame] != frame:
            parent[frame] = parent[parent[frame]]
            frame = int(parent[frame])
        return frame

    for first_raw, second_raw in zip(frame_i, frame_j):
        first = int(first_raw)
        second = int(second_raw)
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            if component_size[root_first] < component_size[root_second]:
                root_first, root_second = root_second, root_first
            parent[root_second] = root_first
            component_size[root_first] += component_size[root_second]
    vertices = np.flatnonzero(active)
    roots = np.fromiter(
        (find(int(frame)) for frame in vertices),
        dtype=np.int64,
        count=len(vertices),
    )
    _, component_sizes = np.unique(roots, return_counts=True)
    largest_component = int(component_sizes.max()) if component_sizes.size else 0
    coverage = largest_component / point_count if point_count > 0 else None
    return {
        "connected_frames": len(vertices),
        "component_count": int(component_sizes.size),
        "largest_component_frames": largest_component,
        "largest_component_coverage": round(coverage, 6) if coverage is not None else None,
    }


def _summarize_pose_graph_npz(payload: Any, point_count: int) -> dict[str, Any]:
    schema_version = _npz_scalar(payload, "schema_version")
    pose_encoding = _npz_scalar(payload, "pose_encoding")
    transform_convention = _npz_scalar(payload, "transform_convention")
    frame_index_space = _npz_scalar(payload, "frame_index_space")
    absolute_pose_space = _npz_scalar(payload, "absolute_pose_space")
    confidence_semantics = _npz_scalar(payload, "confidence_semantics")
    required = (
        "frame_i",
        "frame_j",
        "rel_pose_enc",
        "confidence",
        "confidence_t",
        "confidence_r",
        "edge_type",
    )
    missing_arrays = [name for name in required if name not in getattr(payload, "files", [])]
    if missing_arrays:
        return {
            "available": True,
            "optimizer_ready": False,
            "schema_version": schema_version,
            "pose_encoding": pose_encoding,
            "transform_convention": transform_convention,
            "frame_index_space": frame_index_space,
            "absolute_pose_space": absolute_pose_space,
            "confidence_semantics": confidence_semantics,
            "error": "missing_arrays",
            "missing_arrays": missing_arrays,
        }

    frame_i = np.asarray(payload["frame_i"]).reshape(-1)
    frame_j = np.asarray(payload["frame_j"]).reshape(-1)
    confidence = np.asarray(payload["confidence"]).reshape(-1)
    confidence_t = np.asarray(payload["confidence_t"]).reshape(-1)
    confidence_r = np.asarray(payload["confidence_r"]).reshape(-1)
    edge_type = np.asarray(payload["edge_type"]).reshape(-1)
    rel_pose = np.asarray(payload["rel_pose_enc"])
    edge_count = len(frame_i)
    lengths = {
        "frame_j": len(frame_j),
        "confidence": len(confidence),
        "confidence_t": len(confidence_t),
        "confidence_r": len(confidence_r),
        "edge_type": len(edge_type),
        "rel_pose_enc": int(rel_pose.shape[0]) if rel_pose.ndim >= 1 else 0,
    }
    if any(length != edge_count for length in lengths.values()) or rel_pose.shape != (edge_count, 9):
        return {
            "available": edge_count > 0,
            "optimizer_ready": False,
            "schema_version": schema_version,
            "pose_encoding": pose_encoding,
            "transform_convention": transform_convention,
            "frame_index_space": frame_index_space,
            "absolute_pose_space": absolute_pose_space,
            "confidence_semantics": confidence_semantics,
            "edge_count": edge_count,
            "error": "array_shape_mismatch",
            "array_lengths": lengths,
            "rel_pose_shape": list(rel_pose.shape),
        }
    if not np.issubdtype(frame_i.dtype, np.integer) or not np.issubdtype(frame_j.dtype, np.integer):
        return {
            "available": edge_count > 0,
            "optimizer_ready": False,
            "schema_version": schema_version,
            "edge_count": edge_count,
            "error": "non_integer_frame_indices",
            "frame_i_dtype": str(frame_i.dtype),
            "frame_j_dtype": str(frame_j.dtype),
        }
    numeric_arrays = (confidence, confidence_t, confidence_r, rel_pose)
    if any(not np.issubdtype(array.dtype, np.number) for array in numeric_arrays):
        return {
            "available": edge_count > 0,
            "optimizer_ready": False,
            "schema_version": schema_version,
            "edge_count": edge_count,
            "error": "non_numeric_measurements",
        }

    negative = (frame_i < 0) | (frame_j < 0)
    self_mask = frame_i == frame_j
    if point_count > 0:
        out_of_range = (frame_i >= point_count) | (frame_j >= point_count)
    else:
        out_of_range = np.zeros(edge_count, dtype=bool)
    valid_index = ~(negative | self_mask | out_of_range)
    valid_index_edges = int(valid_index.sum())
    finite_relative = np.isfinite(rel_pose).all(axis=1)
    valid_relative = valid_index & finite_relative
    relative_pose_edges = int(valid_relative.sum())
    valid_split = (
        np.isfinite(confidence_t)
        & (confidence_t > 0)
        & np.isfinite(confidence_r)
        & (confidence_r > 0)
    )
    split_confidence_edges = int((valid_index & valid_split).sum())
    valid_confidence = np.isfinite(confidence) & (confidence > 0)
    invalid_confidence_edges = int((valid_index & ~valid_confidence).sum())

    quaternion_norms = np.linalg.norm(rel_pose[valid_relative, 3:7], axis=1)
    translation_norms = np.linalg.norm(rel_pose[valid_relative, :3], axis=1)
    quaternion_outliers = int(((quaternion_norms < 0.95) | (quaternion_norms > 1.05)).sum())
    valid_i = frame_i[valid_index].astype(np.int64, copy=False)
    valid_j = frame_j[valid_index].astype(np.int64, copy=False)
    connectivity = _connectivity_metrics(valid_i, valid_j, point_count)
    relative_pose_coverage = relative_pose_edges / valid_index_edges if valid_index_edges else 0.0
    split_confidence_coverage = split_confidence_edges / valid_index_edges if valid_index_edges else 0.0
    schema_matches = (
        schema_version == R3_POSE_GRAPH_SCHEMA_VERSION
        and pose_encoding == R3_POSE_ENCODING
        and transform_convention == R3_RELATIVE_TRANSFORM_CONVENTION
        and frame_index_space == "exported_camera_index"
        and absolute_pose_space == R3_ABSOLUTE_POSE_SPACE
        and confidence_semantics == R3_CONFIDENCE_SEMANTICS
    )
    connected_coverage = connectivity["largest_component_coverage"]
    optimizer_ready = bool(
        schema_matches
        and valid_index_edges > 0
        and relative_pose_coverage >= 0.95
        and split_confidence_coverage >= 0.8
        and invalid_confidence_edges == 0
        and quaternion_outliers == 0
        and (connected_coverage is None or connected_coverage >= 0.8)
    )
    type_names = {0: "normal", 1: "bridge", 2: "anchor"}
    type_counts: dict[str, int] = {}
    for code, count in zip(*np.unique(edge_type, return_counts=True)):
        name = type_names.get(int(code), "unknown")
        type_counts[name] = type_counts.get(name, 0) + int(count)

    return {
        "available": edge_count > 0,
        "optimizer_ready": optimizer_ready,
        "storage": "compressed_npz",
        "schema_version": schema_version,
        "schema_matches": schema_matches,
        "pose_encoding": pose_encoding,
        "transform_convention": transform_convention,
        "frame_index_space": frame_index_space,
        "absolute_pose_space": absolute_pose_space,
        "confidence_semantics": confidence_semantics,
        "edge_count": edge_count,
        "valid_index_edges": valid_index_edges,
        "relative_pose_edges": relative_pose_edges,
        "relative_pose_coverage": round(relative_pose_coverage, 6),
        "split_confidence_edges": split_confidence_edges,
        "split_confidence_coverage": round(split_confidence_coverage, 6),
        "invalid_records": 0,
        "invalid_indices": int(negative.sum()),
        "self_edges": int((~negative & self_mask).sum()),
        "out_of_range_edges": int((~negative & ~self_mask & out_of_range).sum()),
        "invalid_relative_poses": int((valid_index & ~finite_relative).sum()),
        "invalid_confidence_edges": invalid_confidence_edges,
        "quaternion_norm_outliers": quaternion_outliers,
        "type_counts": type_counts,
        **connectivity,
        "translation_norm": _percentile_summary(translation_norms),
        "quaternion_norm": _percentile_summary(quaternion_norms),
        "frame_gap": _percentile_summary(np.abs(valid_j - valid_i).astype(float)),
    }


def summarize_pose_graph_edges(payload: Any, point_count: int = 0) -> dict[str, Any]:
    """Validate a pose-graph sidecar and return optimizer-readiness metrics."""
    if isinstance(payload, Mapping):
        raw_edges = payload.get("edges", [])
        schema_version = payload.get("schema_version")
        pose_encoding = payload.get("pose_encoding")
        transform_convention = payload.get("transform_convention")
        frame_index_space = payload.get("frame_index_space")
        absolute_pose_space = payload.get("absolute_pose_space")
        confidence_semantics = payload.get("confidence_semantics")
    elif isinstance(payload, list):
        # Legacy pose_edge_log.json contains topology/confidence only.
        raw_edges = payload
        schema_version = 0
        pose_encoding = None
        transform_convention = None
        frame_index_space = "exported_camera_index"
        absolute_pose_space = None
        confidence_semantics = None
    else:
        raw_edges = []
        schema_version = None
        pose_encoding = None
        transform_convention = None
        frame_index_space = None
        absolute_pose_space = None
        confidence_semantics = None

    edges = raw_edges if isinstance(raw_edges, list) else []
    type_counts: dict[str, int] = {}
    invalid_records = 0
    invalid_indices = 0
    self_edges = 0
    out_of_range_edges = 0
    relative_pose_edges = 0
    invalid_relative_poses = 0
    split_confidence_edges = 0
    invalid_confidence_edges = 0
    quaternion_norm_outliers = 0
    translation_norms: list[float] = []
    quaternion_norms: list[float] = []
    frame_gaps: list[float] = []
    valid_pairs: list[tuple[int, int]] = []

    for edge in edges:
        if not isinstance(edge, Mapping):
            invalid_records += 1
            continue
        edge_type = str(edge.get("edge_type") or "unknown")
        type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        try:
            frame_i = int(edge.get("frame_i"))
            frame_j = int(edge.get("frame_j"))
        except (TypeError, ValueError):
            invalid_indices += 1
            continue
        if frame_i < 0 or frame_j < 0:
            invalid_indices += 1
            continue
        if frame_i == frame_j:
            self_edges += 1
            continue
        if point_count > 0 and (frame_i >= point_count or frame_j >= point_count):
            out_of_range_edges += 1
            continue
        valid_pairs.append((frame_i, frame_j))
        frame_gaps.append(float(abs(frame_j - frame_i)))

        confidence = _finite_number(edge.get("confidence"))
        if confidence is None or confidence <= 0:
            invalid_confidence_edges += 1
        confidence_t = _finite_number(edge.get("confidence_t"))
        confidence_r = _finite_number(edge.get("confidence_r"))
        if (
            confidence_t is not None
            and confidence_t > 0
            and confidence_r is not None
            and confidence_r > 0
        ):
            split_confidence_edges += 1

        rel_pose = edge.get("rel_pose_enc")
        try:
            vector = np.asarray(rel_pose, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            vector = np.asarray([], dtype=np.float64)
        if vector.size != 9 or not np.isfinite(vector).all():
            if rel_pose is not None:
                invalid_relative_poses += 1
            continue
        relative_pose_edges += 1
        translation_norms.append(float(np.linalg.norm(vector[:3])))
        quaternion_norm = float(np.linalg.norm(vector[3:7]))
        quaternion_norms.append(quaternion_norm)
        if not 0.95 <= quaternion_norm <= 1.05:
            quaternion_norm_outliers += 1

    vertices = sorted({frame for pair in valid_pairs for frame in pair})
    parent = {frame: frame for frame in vertices}

    def find(frame: int) -> int:
        while parent[frame] != frame:
            parent[frame] = parent[parent[frame]]
            frame = parent[frame]
        return frame

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for first, second in valid_pairs:
        union(first, second)
    component_sizes: dict[int, int] = {}
    for frame in vertices:
        root = find(frame)
        component_sizes[root] = component_sizes.get(root, 0) + 1
    largest_component = max(component_sizes.values(), default=0)

    edge_count = len(edges)
    valid_index_edges = len(valid_pairs)
    relative_pose_coverage = relative_pose_edges / valid_index_edges if valid_index_edges else 0.0
    split_confidence_coverage = split_confidence_edges / valid_index_edges if valid_index_edges else 0.0
    connected_frame_coverage = largest_component / point_count if point_count > 0 else None
    schema_matches = (
        schema_version == R3_POSE_GRAPH_SCHEMA_VERSION
        and pose_encoding == R3_POSE_ENCODING
        and transform_convention == R3_RELATIVE_TRANSFORM_CONVENTION
        and frame_index_space == "exported_camera_index"
        and absolute_pose_space == R3_ABSOLUTE_POSE_SPACE
        and confidence_semantics == R3_CONFIDENCE_SEMANTICS
    )
    optimizer_ready = bool(
        schema_matches
        and valid_index_edges > 0
        and relative_pose_coverage >= 0.95
        and split_confidence_coverage >= 0.8
        and invalid_confidence_edges == 0
        and quaternion_norm_outliers == 0
        and (connected_frame_coverage is None or connected_frame_coverage >= 0.8)
    )

    return {
        "available": edge_count > 0,
        "optimizer_ready": optimizer_ready,
        "schema_version": schema_version,
        "schema_matches": schema_matches,
        "pose_encoding": pose_encoding,
        "transform_convention": transform_convention,
        "frame_index_space": frame_index_space,
        "absolute_pose_space": absolute_pose_space,
        "confidence_semantics": confidence_semantics,
        "edge_count": edge_count,
        "valid_index_edges": valid_index_edges,
        "relative_pose_edges": relative_pose_edges,
        "relative_pose_coverage": round(relative_pose_coverage, 6),
        "split_confidence_edges": split_confidence_edges,
        "split_confidence_coverage": round(split_confidence_coverage, 6),
        "invalid_records": invalid_records,
        "invalid_indices": invalid_indices,
        "self_edges": self_edges,
        "out_of_range_edges": out_of_range_edges,
        "invalid_relative_poses": invalid_relative_poses,
        "invalid_confidence_edges": invalid_confidence_edges,
        "quaternion_norm_outliers": quaternion_norm_outliers,
        "type_counts": type_counts,
        "connected_frames": len(vertices),
        "component_count": len(component_sizes),
        "largest_component_frames": largest_component,
        "largest_component_coverage": (
            round(connected_frame_coverage, 6) if connected_frame_coverage is not None else None
        ),
        "translation_norm": _percentile_summary(translation_norms),
        "quaternion_norm": _percentile_summary(quaternion_norms),
        "frame_gap": _percentile_summary(frame_gaps),
    }


def load_pose_graph_summary(path: str | Path, point_count: int = 0) -> dict[str, Any]:
    """Load and summarize a pose-graph sidecar without raising.

    Compressed NPZ is the production format.  Its derived summary is cached
    beside the archive so diagnostics endpoints do not scan a million-edge
    graph on every request.  JSON remains supported for small fixtures and
    migration tools.
    """
    source = Path(path)
    if not source.exists():
        return {
            "available": False,
            "optimizer_ready": False,
            "path": str(source),
            "error": "missing",
        }
    if source.suffix.lower() == ".npz":
        cache_path = source.with_suffix(".summary.json")
        try:
            if cache_path.exists() and cache_path.stat().st_mtime_ns >= source.stat().st_mtime_ns:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, Mapping) and cached.get("point_count") == point_count:
                    return {"path": str(source), **dict(cached)}
        except Exception:
            pass
        try:
            with np.load(source, allow_pickle=False) as payload:
                summary = _summarize_pose_graph_npz(payload, point_count=point_count)
        except Exception as exc:
            return {
                "available": False,
                "optimizer_ready": False,
                "path": str(source),
                "error": f"{type(exc).__name__}: {exc}",
            }
        summary = {"point_count": point_count, **summary}
        temporary_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        try:
            temporary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            os.replace(temporary_path, cache_path)
        except Exception:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                pass
        return {"path": str(source), **summary}

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "available": False,
            "optimizer_ready": False,
            "path": str(source),
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"path": str(source), **summarize_pose_graph_edges(payload, point_count=point_count)}
