"""Verified Kerama route data and deterministic mask overrides."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import ndimage


ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "floorplans"
REFERENCE_ROUTE_FILE = ASSET_ROOT / "kerama_marazzi_2025_reference_route.json"
FALSE_NORTH_CORRIDOR = (600, 500, 3050, 650)


def load_reference_route(path: Optional[Path] = None) -> dict[str, Any]:
    payload = json.loads((path or REFERENCE_ROUTE_FILE).read_text(encoding="utf-8"))
    points = np.asarray(payload.get("points"), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        raise ValueError("Kerama reference route must contain at least two 2-D points")
    if not np.isfinite(points[:, :2]).all():
        raise ValueError("Kerama reference route contains non-finite coordinates")
    payload["points"] = points[:, :2].tolist()
    return payload


def reference_route_mask(
    shape: tuple[int, int],
    points: np.ndarray,
    *,
    radius_pixels: float,
) -> np.ndarray:
    """Rasterize a polyline and expand it to a verified walkable corridor."""
    route = np.zeros(shape, dtype=bool)
    height, width = shape
    for start, end in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(end - start))
        sample_count = max(2, int(math.ceil(distance * 2.0)) + 1)
        samples = start + (end - start) * np.linspace(
            0.0, 1.0, sample_count
        )[:, None]
        xs = np.clip(np.rint(samples[:, 0]).astype(np.int64), 0, width - 1)
        ys = np.clip(np.rint(samples[:, 1]).astype(np.int64), 0, height - 1)
        route[ys, xs] = True
    radius = max(1, int(math.ceil(radius_pixels)))
    axis = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    disk = xx * xx + yy * yy <= radius_pixels * radius_pixels
    return ndimage.binary_dilation(route, structure=disk)


def apply_reference_route_overrides(
    obstacle_mask: np.ndarray,
    support_mask: np.ndarray,
    *,
    meters_per_pixel: float,
    payload: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the verified passage and retained exterior protection."""
    obstacles = np.asarray(obstacle_mask, dtype=bool).copy()
    support = np.asarray(support_mask, dtype=bool).copy()
    if obstacles.shape != support.shape:
        raise ValueError("Kerama obstacle/support masks must have identical shapes")
    if meters_per_pixel <= 0.0:
        raise ValueError("meters_per_pixel must be positive")

    route_payload = payload or load_reference_route()
    points = np.asarray(route_payload["points"], dtype=np.float64)[:, :2]
    if (
        np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] >= obstacles.shape[1])
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] >= obstacles.shape[0])
    ):
        raise ValueError("Kerama reference route lies outside the canonical plan")

    x1, y1, x2, y2 = FALSE_NORTH_CORRIDOR
    obstacles[y1:y2, x1:x2] = True
    half_width_meters = float(
        route_payload.get("verified_walkable_half_width_meters", 0.75)
    )
    corridor = reference_route_mask(
        obstacles.shape,
        points,
        radius_pixels=half_width_meters / meters_per_pixel,
    )
    blocked_before = int(np.count_nonzero(obstacles & corridor))
    unsupported_before = int(np.count_nonzero((~support) & corridor))
    obstacles[corridor] = False
    support[corridor] = True
    stats = {
        "method": "operator_ground_truth_corridor_v1",
        "reference_route_file": REFERENCE_ROUTE_FILE.name,
        "verified_walkable_half_width_meters": half_width_meters,
        "corridor_pixel_count": int(corridor.sum()),
        "obstacle_pixels_cleared": blocked_before,
        "support_pixels_added": unsupported_before,
        "false_north_corridor_blocked": list(FALSE_NORTH_CORRIDOR),
    }
    return obstacles, support, stats
