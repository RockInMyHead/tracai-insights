"""Offline Kerama route fixture used only to evaluate map-matching quality."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import ndimage


ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "floorplans"
REFERENCE_ROUTE_FILE = ASSET_ROOT / "kerama_marazzi_2025_reference_route.json"


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

