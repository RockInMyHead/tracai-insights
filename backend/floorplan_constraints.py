"""Production floor-plan constraints for the fixed Kerama Marazzi map.

The R3 reconstruction remains the visual source of relative motion.  This
module estimates one global map similarity from the user supplied start and
heading, evaluates several scale/yaw hypotheses, and solves for a route inside
the verified walkable mask.  Visual reconstruction remains the relative-motion
observation; the fixed plan is a hard spatial constraint rather than a final
quality gate which can discard an otherwise useful route.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    from confidence_calibration import calibrated_probability
except ImportError:  # pragma: no cover - package import path
    from backend.confidence_calibration import calibrated_probability


DEFAULT_FLOORPLAN_ID = "kerama_marazzi_2025"
ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "floorplans"


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _normalise_points(value: Any) -> np.ndarray:
    """Keep only finite samples. Never fabricate (0,0) for bad input."""
    if not isinstance(value, list):
        return np.empty((0, 2), dtype=np.float64)
    points: list[list[float]] = []
    for item in value:
        raw: Any = item
        if isinstance(item, dict):
            raw = [
                item.get("x", item.get(0)),
                item.get("y", item.get(1)),
            ]
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        try:
            point = [float(raw[0]), float(raw[1])]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(component) for component in point):
            points.append(point)
    return np.asarray(points, dtype=np.float64)


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _polyline_sharp_reverse_ratio(
    points: np.ndarray,
    *,
    meters_per_pixel: float,
    min_segment_meters: float = 0.25,
    reverse_degrees: float = 135.0,
) -> float:
    """Fraction of meaningful turns that are near-U-turns.

    Mask-legal A* spikes through drawing gaps often show as a chain of
    ~180° corners even when length_ratio and collision checks still pass.
    """
    if len(points) < 3 or meters_per_pixel <= 1e-12:
        return 0.0
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1) * float(meters_per_pixel)
    keep = np.where(segments >= float(min_segment_meters))[0]
    if len(keep) < 2:
        return 0.0
    compact = np.vstack([points[0], points[keep + 1]])
    vectors = np.diff(compact, axis=0)
    if len(vectors) < 2:
        return 0.0
    left = vectors[:-1]
    right = vectors[1:]
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    valid = (left_norm > 1e-9) & (right_norm > 1e-9)
    if not np.any(valid):
        return 0.0
    cosine = np.sum(left[valid] * right[valid], axis=1) / (
        left_norm[valid] * right_norm[valid]
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(np.mean(angles >= float(reverse_degrees)))


def _resample_polyline(points: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.repeat(points[:1], len(fractions), axis=0)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    total = float(cumulative[-1])
    if total <= 1e-9:
        return np.repeat(points[:1], len(fractions), axis=0)
    targets = np.clip(fractions, 0.0, 1.0) * total
    output = np.empty((len(targets), 2), dtype=np.float64)
    segment = 0
    for index, target in enumerate(targets):
        while segment + 1 < len(cumulative) - 1 and cumulative[segment + 1] < target:
            segment += 1
        length = max(float(cumulative[segment + 1] - cumulative[segment]), 1e-12)
        alpha = (target - cumulative[segment]) / length
        output[index] = points[segment] * (1.0 - alpha) + points[segment + 1] * alpha
    return output


def _trajectory_fractions(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points), dtype=np.float64)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    return cumulative / max(float(cumulative[-1]), 1e-12)


def _resample_timestamps(values: Any, count: int) -> Any:
    if not isinstance(values, list) or len(values) < 2 or count < 2:
        return values
    source = np.asarray([_finite_float(value, math.nan) for value in values], dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(source))
    if len(finite) < 2:
        return values
    source = np.interp(np.arange(len(source)), finite, source[finite])
    return np.interp(
        np.linspace(0.0, 1.0, count), np.linspace(0.0, 1.0, len(source)), source
    ).tolist()


def _flip_polyline_y(points: Any) -> list[list[float]]:
    """Mirror plan Y — used to resolve PCA sign gauge for independent LingBot."""
    array = _normalise_points(points)
    if len(array) == 0:
        return []
    mirrored = array.copy()
    mirrored[:, 1] *= -1.0
    return [
        [round(float(point[0]), 8), round(float(point[1]), 8), 0.0]
        for point in mirrored
    ]


def _r3_is_severely_fragmented(result: dict[str, Any]) -> bool:
    """Recognise graph failures without depending on one worker schema."""
    containers = [
        result,
        result.get("processing_stats") or {},
        result.get("trajectory_quality") or {},
    ]
    nested: list[dict[str, Any]] = []
    for container in containers:
        if not isinstance(container, dict):
            continue
        nested.append(container)
        for key in ("pose_graph", "graph", "r3_pose_graph", "pose_graph_summary"):
            value = container.get(key)
            if isinstance(value, dict):
                nested.append(value)
    components = 1
    coverage = 1.0
    connected = None
    total = None
    for item in nested:
        components = max(components, int(_finite_float(item.get("component_count"), 1)))
        for key in ("largest_component_coverage", "largest_component_ratio"):
            if key in item:
                coverage = min(coverage, _finite_float(item.get(key), 1.0))
        connected = item.get("connected_pose_count", item.get("connected_poses", connected))
        total = item.get("point_count", item.get("pose_count", total))
    if connected is not None and total is not None and _finite_float(total) > 0:
        coverage = min(coverage, _finite_float(connected) / _finite_float(total))
    return components >= 4 or coverage < 0.45


@dataclass(frozen=True)
class FloorplanConfig:
    map_id: str
    width: int
    height: int
    meters_per_pixel: float
    grid_cell_pixels: int = 4
    person_radius_meters: float = 0.28
    walking_speed_mps: float = 1.20
    obstacle_mask_file: str = "kerama_marazzi_2025_obstacles.png"
    obstacle_mask_sha256: str = ""
    support_mask_file: str = ""
    support_mask_sha256: str = ""
    source_pdf: str = "kerama-marazzi-2025.pdf"
    display_image: str = "kerama-marazzi-2025.png"


class FloorplanConstraintEngine:
    """Immutable map model plus deterministic trajectory alignment."""

    def __init__(
        self,
        config: FloorplanConfig,
        obstacle_mask: np.ndarray,
        support_mask: Optional[np.ndarray] = None,
    ):
        mask = np.asarray(obstacle_mask, dtype=bool)
        if mask.shape != (config.height, config.width):
            raise ValueError(
                f"Obstacle mask shape {mask.shape} does not match "
                f"{config.height}x{config.width}"
            )
        self.config = config
        support = None if support_mask is None else np.asarray(support_mask, dtype=bool)
        if support is not None and support.shape != mask.shape:
            raise ValueError("Floorplan support mask shape does not match obstacle mask")
        self._full_mask = mask | (~support if support is not None else False)
        self._support_mask = support
        self._build_grid(self._full_mask, annotation_mask=mask)

    @classmethod
    def load(cls, map_id: str = DEFAULT_FLOORPLAN_ID) -> "FloorplanConstraintEngine":
        metadata_path = ASSET_ROOT / f"{map_id}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = FloorplanConfig(
            map_id=metadata["map_id"],
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            meters_per_pixel=float(metadata["meters_per_pixel"]),
            grid_cell_pixels=int(metadata.get("grid_cell_pixels", 4)),
            person_radius_meters=float(metadata.get("person_radius_meters", 0.28)),
            walking_speed_mps=float(metadata.get("walking_speed_mps", 1.20)),
            obstacle_mask_file=metadata["obstacle_mask_file"],
            obstacle_mask_sha256=metadata.get("obstacle_mask_sha256", ""),
            support_mask_file=metadata.get("support_mask_file", ""),
            support_mask_sha256=metadata.get("support_mask_sha256", ""),
            source_pdf=metadata.get("source_pdf", "kerama-marazzi-2025.pdf"),
            display_image=metadata.get("display_image", "kerama-marazzi-2025.png"),
        )
        mask_path = ASSET_ROOT / config.obstacle_mask_file
        if config.obstacle_mask_sha256:
            actual_hash = hashlib.sha256(mask_path.read_bytes()).hexdigest()
            if actual_hash != config.obstacle_mask_sha256:
                raise ValueError(
                    f"Floorplan obstacle mask hash mismatch for {config.map_id}"
                )
        mask = np.asarray(Image.open(mask_path).convert("L")) >= 128
        support = None
        if config.support_mask_file:
            support_path = ASSET_ROOT / config.support_mask_file
            if config.support_mask_sha256:
                actual_hash = hashlib.sha256(support_path.read_bytes()).hexdigest()
                if actual_hash != config.support_mask_sha256:
                    raise ValueError(f"Floorplan support mask hash mismatch for {config.map_id}")
            support = np.asarray(Image.open(support_path).convert("L")) >= 128
        return cls(config, mask, support)

    @classmethod
    def from_mask(
        cls,
        obstacle_mask: np.ndarray,
        *,
        meters_per_pixel: float = 0.10,
        grid_cell_pixels: int = 1,
        person_radius_meters: float = 0.0,
        walking_speed_mps: float = 1.20,
        map_id: str = "test",
    ) -> "FloorplanConstraintEngine":
        height, width = np.asarray(obstacle_mask).shape[:2]
        return cls(
            FloorplanConfig(
                map_id=map_id,
                width=int(width),
                height=int(height),
                meters_per_pixel=float(meters_per_pixel),
                grid_cell_pixels=int(grid_cell_pixels),
                person_radius_meters=float(person_radius_meters),
                walking_speed_mps=float(walking_speed_mps),
                obstacle_mask_file="",
            ),
            obstacle_mask,
        )

    def _build_grid(self, mask: np.ndarray, annotation_mask: Optional[np.ndarray] = None) -> None:
        cell = self.config.grid_cell_pixels
        rows = int(math.ceil(self.config.height / cell))
        cols = int(math.ceil(self.config.width / cell))
        padded = np.pad(
            mask,
            ((0, rows * cell - self.config.height), (0, cols * cell - self.config.width)),
            # Partial grid cells outside the PDF are physical out-of-map space.
            constant_values=True,
        )
        base = padded.reshape(rows, cell, cols, cell).any(axis=(1, 3))
        distance_to_base = ndimage.distance_transform_edt(~base)
        inflation_cells = self.config.person_radius_meters / max(
            self.config.meters_per_pixel * cell, 1e-9
        )
        occupied = base | (distance_to_base <= inflation_cells)
        self.rows = rows
        self.cols = cols
        self.occupied = occupied
        self.clearance_meters = (
            ndimage.distance_transform_edt(~occupied)
            * self.config.meters_per_pixel
            * cell
        )
        # A compact medial-axis-like corridor graph. Nodes are local clearance
        # maxima; collision-free Viterbi transitions become its implicit edges.
        ridge_window = max(3, int(round(1.2 / max(self.config.meters_per_pixel * cell, 1e-9))))
        if ridge_window % 2 == 0:
            ridge_window += 1
        ridge = (
            (~occupied)
            & (self.clearance_meters >= ndimage.maximum_filter(
                self.clearance_meters, size=ridge_window, mode="constant"
            ) - 1e-9)
            & (self.clearance_meters >= max(0.20, self.config.person_radius_meters * 0.5))
        )
        corridor_nodes = np.argwhere(ridge)[:, ::-1] if np.any(ridge) else np.empty((0, 2), dtype=int)
        if len(corridor_nodes) > 5000:
            corridor_nodes = corridor_nodes[:: int(math.ceil(len(corridor_nodes) / 5000))]
        self._corridor_nodes = corridor_nodes.astype(np.int32)
        _, nearest = ndimage.distance_transform_edt(occupied, return_indices=True)
        self._nearest_free_rows = nearest[0]
        self._nearest_free_cols = nearest[1]
        free = ~occupied
        labeled, component_count = ndimage.label(free)
        self._component_ids = labeled
        self._component_count = int(component_count)
        annotation = mask if annotation_mask is None else annotation_mask
        annotation_padded = np.pad(
            annotation,
            ((0, rows * cell - self.config.height), (0, cols * cell - self.config.width)),
            constant_values=False,
        )
        annotation_base = annotation_padded.reshape(rows, cell, cols, cell).any(axis=(1, 3))
        ys, xs = np.where(annotation_base)
        self.annotation_bbox = (
            (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            if len(xs)
            else (0, 0, cols - 1, rows - 1)
        )
        # Walkable support/free extent drives the map-scale prior. Annotation
        # ink alone only covers marked machines and underestimates the plant.
        free_ys, free_xs = np.where(free)
        if len(free_xs):
            # Prefer the largest connected free component diameter.
            if component_count > 0:
                sizes = ndimage.sum(free, labeled, index=np.arange(1, component_count + 1))
                largest = int(np.argmax(sizes)) + 1
                component_mask = labeled == largest
                free_ys, free_xs = np.where(component_mask)
            self.walkable_bbox = (
                int(free_xs.min()),
                int(free_ys.min()),
                int(free_xs.max()),
                int(free_ys.max()),
            )
        else:
            self.walkable_bbox = self.annotation_bbox

    @property
    def cell_meters(self) -> float:
        return self.config.grid_cell_pixels * self.config.meters_per_pixel

    def _pixel_to_cell(self, point: Sequence[float]) -> tuple[int, int]:
        x = int(math.floor(float(point[0]) / self.config.grid_cell_pixels))
        y = int(math.floor(float(point[1]) / self.config.grid_cell_pixels))
        return x, y

    def _cell_to_pixel(self, cell: tuple[int, int]) -> np.ndarray:
        half = self.config.grid_cell_pixels / 2.0
        return np.asarray([
            cell[0] * self.config.grid_cell_pixels + half,
            cell[1] * self.config.grid_cell_pixels + half,
        ])

    def _inside_cell(self, x: int, y: int) -> bool:
        return 0 <= x < self.cols and 0 <= y < self.rows

    def _point_occupied(self, point: Sequence[float]) -> bool:
        if not (
            0.0 <= float(point[0]) < self.config.width
            and 0.0 <= float(point[1]) < self.config.height
        ):
            return True
        x, y = self._pixel_to_cell(point)
        return not self._inside_cell(x, y) or bool(self.occupied[y, x])

    def _nearest_free(self, cell: tuple[int, int]) -> Optional[tuple[int, int]]:
        # A monocular hypothesis may leave the plan. Project its endpoint back
        # to the border so the constrained solver can recover a valid route.
        x = min(max(int(cell[0]), 0), self.cols - 1)
        y = min(max(int(cell[1]), 0), self.rows - 1)
        if not self.occupied[y, x]:
            return x, y
        ny = int(self._nearest_free_rows[y, x])
        nx = int(self._nearest_free_cols[y, x])
        return (nx, ny) if self._inside_cell(nx, ny) and not self.occupied[ny, nx] else None

    def _sample_path(self, points: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return points.copy()
        # Quarter-cell sampling is deliberately shared with line-of-sight
        # validation.  Coarser, differently phased samples can miss one grid
        # cell and falsely certify a diagonal that clips a machine corner.
        spacing = max(self.config.grid_cell_pixels * 0.25, 0.5)
        samples: list[np.ndarray] = [points[0]]
        for start, end in zip(points[:-1], points[1:]):
            count = max(1, int(math.ceil(float(np.linalg.norm(end - start)) / spacing)))
            for step in range(1, count + 1):
                samples.append(start + (end - start) * (step / count))
        return np.asarray(samples, dtype=np.float64)

    def _path_metrics(self, points: np.ndarray) -> dict[str, float]:
        samples = self._sample_path(points)
        if len(samples) == 0:
            return {"collision_ratio": 1.0, "outside_ratio": 1.0, "clearance_penalty": 1.0}
        collisions = 0
        outside = 0
        clearance_values: list[float] = []
        for point in samples:
            if not (
                0.0 <= float(point[0]) < self.config.width
                and 0.0 <= float(point[1]) < self.config.height
            ):
                outside += 1
                collisions += 1
                continue
            x, y = self._pixel_to_cell(point)
            if not self._inside_cell(x, y):
                outside += 1
                collisions += 1
                continue
            if self.occupied[y, x]:
                collisions += 1
            clearance_values.append(float(self.clearance_meters[y, x]))
        clearance_penalty = float(np.mean(np.exp(-np.asarray(clearance_values) / 0.45))) \
            if clearance_values else 1.0
        return {
            "collision_ratio": collisions / len(samples),
            "outside_ratio": outside / len(samples),
            "clearance_penalty": clearance_penalty,
        }

    @staticmethod
    def _motion_duration_seconds(timestamps: Any, points: np.ndarray) -> Optional[float]:
        """Estimate moving time so pauses do not inflate the metric scale.

        Video duration is a useful monocular-scale prior only while the
        operator moves.  R3 still jitters during a stop, so the threshold is
        derived from the run's own step distribution and falls back to full
        duration for uniformly moving sequences.
        """
        point_count = len(points)
        if not isinstance(timestamps, list) or len(timestamps) != point_count:
            return None
        values = np.asarray([_finite_float(value, math.nan) for value in timestamps], dtype=np.float64)
        finite_indices = np.flatnonzero(np.isfinite(values))
        if point_count < 2 or len(finite_indices) < 2:
            return None
        first, last = int(finite_indices[0]), int(finite_indices[-1])
        total = float(values[last] - values[first])
        if not 1.0 <= total <= 8 * 60 * 60:
            return None
        dt = np.diff(values)
        valid_dt = np.isfinite(dt) & (dt > 0.0) & (dt < 60.0)
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        positive = steps[np.isfinite(steps) & (steps > 1e-10)]
        if len(positive) < 4:
            return total
        p20, p90 = np.percentile(positive, [20, 90])
        if p20 > 1e-10 and p90 / p20 < 2.5:
            moving = valid_dt
        else:
            threshold = max(float(p20) * 2.5, float(p90) * 0.04, 1e-10)
            moving = (steps > threshold) & valid_dt
            moving = ndimage.binary_dilation(moving, structure=np.ones(3, dtype=bool)) & valid_dt
        active = float(dt[moving].sum())
        return active if active >= max(1.0, total * 0.08) else total

    def _scale_candidates(self, relative: np.ndarray, duration: Optional[float]) -> list[float]:
        raw_length = max(_polyline_length(relative), 1e-9)
        raw_span = max(float(np.ptp(relative[:, 0])), float(np.ptp(relative[:, 1])), 1e-9)
        walk_width = max(self.walkable_bbox[2] - self.walkable_bbox[0], 1)
        walk_height = max(self.walkable_bbox[3] - self.walkable_bbox[1], 1)
        map_base = max(walk_width, walk_height) * self.config.grid_cell_pixels * 0.72 / raw_span
        bases = [map_base]
        if duration is not None:
            metric_distance = self.config.walking_speed_mps * duration
            bases.append(metric_distance / self.config.meters_per_pixel / raw_length)
        factors = (
            0.38, 0.45, 0.52, 0.58, 0.66, 0.72, 0.80, 0.86, 0.93,
            1.0, 1.08, 1.16, 1.25, 1.35, 1.48, 1.60, 1.78, 1.95, 2.15, 2.35,
        )
        values = {
            round(base * factor, 9)
            for base in bases
            for factor in factors
            if math.isfinite(base * factor) and base * factor > 1e-7
        }
        return sorted(values)

    @staticmethod
    def _select_diverse_beam(
        hypotheses: list[dict[str, Any]],
        *,
        per_yaw: int = 3,
        global_top: int = 18,
    ) -> list[dict[str, Any]]:
        """Repair a yaw-diverse beam instead of freezing on the first raw winners."""
        if not hypotheses:
            return []
        selected: list[dict[str, Any]] = []
        seen: set[tuple[float, float]] = set()

        def add(item: dict[str, Any]) -> None:
            key = (round(float(item["scale"]), 9), round(float(item["yaw"]), 3))
            if key in seen:
                return
            seen.add(key)
            selected.append(item)

        by_yaw: dict[float, list[dict[str, Any]]] = {}
        for item in hypotheses:
            by_yaw.setdefault(round(float(item["yaw"]), 3), []).append(item)
        for group in by_yaw.values():
            for item in group[:per_yaw]:
                add(item)
        for item in hypotheses[:global_top]:
            add(item)
        return selected

    @staticmethod
    def _initial_heading(relative: np.ndarray) -> float:
        if len(relative) < 2:
            return 0.0
        # A percentage of a long video can span multiple real turns.  Only the
        # genuinely early motion is allowed to define the user supplied arrow.
        upper = min(len(relative) - 1, 48, max(5, len(relative) // 12))
        for index in range(upper, 0, -1):
            delta = relative[index] - relative[0]
            if float(np.linalg.norm(delta)) > 1e-8:
                return math.atan2(float(delta[1]), float(delta[0]))
        return 0.0

    def _build_hypothesis(
        self,
        relative: np.ndarray,
        start: np.ndarray,
        desired_heading: float,
        scale: float,
        yaw_offset_degrees: float,
    ) -> np.ndarray:
        rotation = desired_heading - self._initial_heading(relative) + math.radians(yaw_offset_degrees)
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        matrix = np.asarray([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float64)
        return start + (relative * scale) @ matrix.T

    def _score_hypothesis(
        self,
        points: np.ndarray,
        duration: Optional[float],
        yaw_offset_degrees: float,
    ) -> tuple[float, dict[str, float]]:
        metrics = self._path_metrics(points)
        length_meters = _polyline_length(points) * self.config.meters_per_pixel
        speed = length_meters / duration if duration else None
        speed_penalty = 0.0
        if speed is not None and speed > 1e-9:
            speed_penalty = abs(math.log(speed / self.config.walking_speed_mps))
            if speed < 0.20 or speed > 3.20:
                speed_penalty += 4.0
        score = (
            34.0 * metrics["collision_ratio"]
            + 60.0 * metrics["outside_ratio"]
            + 0.35 * metrics["clearance_penalty"]
            + 0.75 * speed_penalty
            + 0.08 * abs(yaw_offset_degrees) / 5.0
        )
        return score, {
            **metrics,
            "length_meters": length_meters,
            "speed_mps": speed if speed is not None else math.nan,
        }

    def _segment_collides(self, start: np.ndarray, end: np.ndarray) -> bool:
        return any(self._point_occupied(point) for point in self._sample_path(
            np.vstack((start, end))
        ))

    def _astar(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        raw_segment: np.ndarray,
        _search_margin_cells: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        start = self._nearest_free(self._pixel_to_cell(start_point))
        end = self._nearest_free(self._pixel_to_cell(end_point))
        if start is None or end is None:
            return None
        if start == end:
            return np.vstack((self._cell_to_pixel(start), self._cell_to_pixel(end)))
        start_component = int(self._component_ids[start[1], start[0]])
        end_component = int(self._component_ids[end[1], end[0]])
        if start_component == 0 or end_component == 0 or start_component != end_component:
            return None

        raw_cells = np.asarray([self._pixel_to_cell(point) for point in raw_segment], dtype=np.int32)
        # Large production machines can require a detour around their entire
        # footprint.  Thirty metres covers the wider Kerama equipment islands
        # where a 12 m local window left start/end in the same global component
        # but still made A* report no path.  On failure the search still doubles.
        initial_margin = max(24, int(round(30.0 / max(self.cell_meters, 1e-9))))
        margin = initial_margin if _search_margin_cells is None else _search_margin_cells
        min_x = max(0, min(start[0], end[0], int(raw_cells[:, 0].min())) - margin)
        max_x = min(self.cols - 1, max(start[0], end[0], int(raw_cells[:, 0].max())) + margin)
        min_y = max(0, min(start[1], end[1], int(raw_cells[:, 1].min())) - margin)
        max_y = min(self.rows - 1, max(start[1], end[1], int(raw_cells[:, 1].max())) + margin)

        local_shape = (max_y - min_y + 1, max_x - min_x + 1)
        raw_seed = np.ones(local_shape, dtype=bool)
        for point in self._sample_path(raw_segment):
            x, y = self._pixel_to_cell(point)
            if min_x <= x <= max_x and min_y <= y <= max_y:
                raw_seed[y - min_y, x - min_x] = False
        deviation = ndimage.distance_transform_edt(raw_seed)

        neighbors = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        )
        queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
        previous: dict[tuple[int, int], tuple[int, int]] = {}
        costs = {start: 0.0}
        visited: set[tuple[int, int]] = set()

        while queue:
            _, cost, current = heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            if current == end:
                cells = [current]
                while cells[-1] != start:
                    cells.append(previous[cells[-1]])
                cells.reverse()
                route = np.asarray([self._cell_to_pixel(cell) for cell in cells])
                # A cell-by-cell route is needlessly large, but uniformly
                # resampling it back to the R3 point count can cut straight
                # through an obstacle when the source trajectory is sparse.
                # Keep the smallest collision-free set of line-of-sight
                # anchors instead.  This preserves the A* homotopy and makes
                # every rendered segment safe, independently of source FPS.
                simplified = [route[0]]
                anchor = 0
                while anchor < len(route) - 1:
                    candidate = len(route) - 1
                    while candidate > anchor + 1 and self._segment_collides(
                        route[anchor], route[candidate]
                    ):
                        candidate -= 1
                    simplified.append(route[candidate])
                    anchor = candidate
                return np.asarray(simplified, dtype=np.float64)
            for dx, dy, step in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                if nx < min_x or nx > max_x or ny < min_y or ny > max_y:
                    continue
                if self.occupied[ny, nx]:
                    continue
                if dx and dy and (
                    self.occupied[current[1], nx]
                    or self.occupied[ny, current[0]]
                ):
                    # A diagonal between two blocked orthogonal cells clips
                    # the physical obstacle corner even when both endpoint
                    # cells are free.
                    continue
                clearance = float(self.clearance_meters[ny, nx])
                deviation_meters = (
                    float(deviation[ny - min_y, nx - min_x]) * self.cell_meters
                )
                # Every term is now a dimensionless multiplier of a metric
                # step. This makes the same physical map substantially
                # invariant to grid_cell_pixels.
                wall_multiplier = 0.55 * math.exp(-clearance / 0.45)
                # Stronger pull toward the visual observation so A* prefers
                # local aisle snaps over long spikes through blank CAD gaps.
                deviation_multiplier = min(8.0, 0.90 * deviation_meters)
                step_meters = step * self.cell_meters
                candidate = cost + step_meters * (
                    1.0 + wall_multiplier + deviation_multiplier
                )
                key = (nx, ny)
                if candidate >= costs.get(key, float("inf")):
                    continue
                costs[key] = candidate
                previous[key] = current
                heuristic = math.hypot(end[0] - nx, end[1] - ny) * self.cell_meters
                heappush(queue, (candidate + heuristic, candidate, key))
        # Expand locally for large machines, but never to the whole plant:
        # full-map search invents mask-legal spikes through distant CAD gaps.
        max_margin = max(initial_margin, int(round(50.0 / max(self.cell_meters, 1e-9))))
        if margin < max_margin:
            return self._astar(
                start_point,
                end_point,
                raw_segment,
                _search_margin_cells=min(max_margin, margin * 2),
            )
        return None

    def _detour_is_spike(
        self,
        route: np.ndarray,
        start_point: np.ndarray,
        end_point: np.ndarray,
        raw_segment: np.ndarray,
    ) -> bool:
        """Reject A* shortcuts that invent a long loop far from the observation."""
        chord_m = float(np.linalg.norm(end_point - start_point)) * self.config.meters_per_pixel
        route_m = _polyline_length(route) * self.config.meters_per_pixel
        if route_m > max(3.5 * max(chord_m, 1e-6), chord_m + 12.0):
            return True
        if len(raw_segment) < 2 or len(route) < 2:
            return False
        max_dev = 0.0
        for point in route:
            best = float("inf")
            for left, right in zip(raw_segment[:-1], raw_segment[1:]):
                delta = right - left
                length = float(np.linalg.norm(delta))
                if length <= 1e-9:
                    best = min(best, float(np.linalg.norm(point - left)))
                    continue
                t = float(np.clip(np.dot(point - left, delta) / (length * length), 0.0, 1.0))
                best = min(best, float(np.linalg.norm(point - (left + t * delta))))
            max_dev = max(max_dev, best * self.config.meters_per_pixel)
        return max_dev > 10.0

    def _collision_runs(self, points: np.ndarray) -> list[tuple[int, int]]:
        bad = np.asarray([
            self._segment_collides(points[index], points[index + 1])
            for index in range(len(points) - 1)
        ], dtype=bool)
        runs: list[tuple[int, int]] = []
        index = 0
        while index < len(bad):
            if not bad[index]:
                index += 1
                continue
            start = index
            while index + 1 < len(bad) and bad[index + 1]:
                index += 1
            runs.append((start, index + 1))
            index += 1
        return runs

    def _repair_collisions(self, points: np.ndarray) -> tuple[Optional[np.ndarray], int]:
        runs = self._collision_runs(points)
        if not runs:
            return points.copy(), 0
        merged: list[tuple[int, int]] = []
        for left, right in runs:
            while left > 0 and self._point_occupied(points[left]):
                left -= 1
            while right < len(points) - 1 and self._point_occupied(points[right]):
                right += 1
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))

        rebuilt: list[np.ndarray] = []
        cursor = 0
        rerouted = 0
        for left, right in merged:
            if left == 0 and self._point_occupied(points[0]):
                return None, len(merged)
            rebuilt.extend(points[cursor:left])
            raw_segment = points[left:right + 1]
            segment_m = _polyline_length(raw_segment) * self.config.meters_per_pixel
            route = self._astar(points[left], points[right], raw_segment)
            if route is not None and self._detour_is_spike(
                route, points[left], points[right], raw_segment
            ):
                # A 3 m observation nick must not become a 30 m plant loop.
                route = None
            if route is None:
                # Keep short observation nicks instead of inventing topology.
                # Longer unrepairable collisions still fail the hypothesis.
                if segment_m <= 6.0:
                    rebuilt.extend(raw_segment)
                    cursor = right + 1
                    continue
                return None, len(merged)
            rerouted += 1
            if rebuilt and np.linalg.norm(rebuilt[-1] - route[0]) < 1e-6:
                route = route[1:]
            rebuilt.extend(route)
            cursor = right + 1
        rebuilt.extend(points[cursor:])
        # Do not force the repaired line back to the original sample count.
        # Sparse uniform resampling can reconnect two safe samples with a
        # chord through the obstacle.  Consumers already accept a polyline of
        # arbitrary length, and source-frame/turn mapping is handled by arc
        # fraction below.
        return np.asarray(rebuilt, dtype=np.float64), rerouted

    def _adaptive_anchor_fractions(
        self, points: np.ndarray, *, maximum: int
    ) -> np.ndarray:
        """Place anchors by travelled distance and retain curvature extrema."""
        if len(points) < 3:
            return np.linspace(0.0, 1.0, max(2, len(points)))
        fractions = _trajectory_fractions(points)
        deltas = np.diff(points[:, :2], axis=0)
        headings = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
        curvature = np.zeros(len(points), dtype=np.float64)
        if len(headings) > 1:
            curvature[1:-1] = np.abs(np.diff(headings))
        base_count = max(6, min(maximum, int(math.ceil(_polyline_length(points) / max(
            2.5 / max(self.config.meters_per_pixel, 1e-9), 1.0
        ))) + 1))
        selected = set(int(index) for index in np.searchsorted(
            fractions, np.linspace(0.0, 1.0, base_count)
        ).clip(0, len(points) - 1))
        for index in np.argsort(curvature)[::-1]:
            if curvature[index] < math.radians(3.0):
                break
            if all(abs(float(fractions[index] - fractions[item])) >= 0.018 for item in selected):
                selected.add(int(index))
            if len(selected) >= maximum:
                break
        selected.update((0, len(points) - 1))
        if len(selected) > maximum:
            mandatory = {0, len(points) - 1}
            ranked = sorted(
                selected - mandatory,
                key=lambda index: curvature[index],
                reverse=True,
            )[: maximum - 2]
            selected = mandatory | set(ranked)
        return np.asarray([fractions[index] for index in sorted(selected)], dtype=np.float64)

    def _map_state_candidates(
        self,
        observation: np.ndarray,
        guide: np.ndarray,
        *,
        radius_meters: float,
        limit: int,
        fixed: bool = False,
    ) -> np.ndarray:
        if fixed:
            cell = self._nearest_free(self._pixel_to_cell(guide))
            return np.asarray([self._cell_to_pixel(cell)]) if cell is not None else np.empty((0, 2))
        radius_cells = max(2, int(math.ceil(radius_meters / max(self.cell_meters, 1e-9))))
        cells: set[tuple[int, int]] = set()
        for seed in (observation, guide):
            free = self._nearest_free(self._pixel_to_cell(seed))
            if free is not None:
                cells.add(free)
        centre = np.asarray(self._pixel_to_cell(observation))
        guide_cell = np.asarray(self._pixel_to_cell(guide))
        for origin in (centre, guide_cell):
            samples = np.linspace(-radius_cells, radius_cells, 9).round().astype(int)
            for dx in samples:
                for dy in samples:
                    if dx * dx + dy * dy > radius_cells * radius_cells:
                        continue
                    x, y = int(origin[0] + dx), int(origin[1] + dy)
                    if self._inside_cell(x, y) and not self.occupied[y, x]:
                        cells.add((x, y))
        if len(self._corridor_nodes):
            delta_observation = self._corridor_nodes - centre[None, :]
            delta_guide = self._corridor_nodes - guide_cell[None, :]
            nearby = self._corridor_nodes[
                (np.sum(delta_observation ** 2, axis=1) <= radius_cells ** 2)
                | (np.sum(delta_guide ** 2, axis=1) <= radius_cells ** 2)
            ]
            cells.update((int(item[0]), int(item[1])) for item in nearby)
        ranked = sorted(cells, key=lambda cell: (
            min(
                float(np.linalg.norm(self._cell_to_pixel(cell) - observation)),
                float(np.linalg.norm(self._cell_to_pixel(cell) - guide)) * 0.85,
            )
            - 0.10 * float(self.clearance_meters[cell[1], cell[0]])
            / max(self.config.meters_per_pixel, 1e-9)
        ))
        chosen: list[np.ndarray] = []
        separation = max(self.config.grid_cell_pixels * 2.0, radius_cells * self.config.grid_cell_pixels * 0.12)
        for cell in ranked:
            point = self._cell_to_pixel(cell)
            if chosen and min(float(np.linalg.norm(point - prior)) for prior in chosen) < separation:
                continue
            chosen.append(point)
            if len(chosen) >= limit:
                break
        return np.asarray(chosen, dtype=np.float64)

    def _viterbi_level(
        self,
        observation: np.ndarray,
        guide: np.ndarray,
        fractions: np.ndarray,
        *,
        radius_meters: float,
        candidate_limit: int,
    ) -> tuple[Optional[np.ndarray], dict[str, Any]]:
        observed = _resample_polyline(observation, fractions)
        guided = _resample_polyline(guide, fractions)
        states = [
            self._map_state_candidates(
                observed[index], guided[index], radius_meters=radius_meters,
                limit=candidate_limit, fixed=index == 0,
            )
            for index in range(len(fractions))
        ]
        if any(len(layer) == 0 for layer in states):
            return None, {"reason": "empty_state_layer"}
        radius_pixels = radius_meters / max(self.config.meters_per_pixel, 1e-9)
        edge_cache: dict[tuple[int, int, int], float] = {}

        def edge(layer: int, left: int, right: int) -> float:
            key = (layer, left, right)
            if key in edge_cache:
                return edge_cache[key]
            start, end = states[layer - 1][left], states[layer][right]
            if self._segment_collides(start, end):
                value = float("inf")
            else:
                visual = observed[layer] - observed[layer - 1]
                mapped = end - start
                visual_length = max(float(np.linalg.norm(visual)), 1e-6)
                mapped_length = max(float(np.linalg.norm(mapped)), 1e-6)
                cosine = float(np.clip(
                    np.dot(visual, mapped) / (visual_length * mapped_length), -1.0, 1.0
                ))
                value = (
                    1.35 * abs(math.log(mapped_length / visual_length))
                    + 1.7 * (1.0 - cosine)
                )
            edge_cache[key] = value
            return value

        costs = np.asarray([
            0.7 * (float(np.linalg.norm(point - observed[0])) / max(radius_pixels, 1.0)) ** 2
            for point in states[0]
        ])
        parents: list[np.ndarray] = []
        for layer in range(1, len(states)):
            next_costs = np.full(len(states[layer]), float("inf"))
            parent = np.full(len(states[layer]), -1, dtype=np.int32)
            for right, point in enumerate(states[layer]):
                emission = 0.7 * (
                    float(np.linalg.norm(point - observed[layer])) / max(radius_pixels, 1.0)
                ) ** 2
                x, y = self._pixel_to_cell(point)
                emission += 0.08 * math.exp(-float(self.clearance_meters[y, x]) / 0.45)
                for left, accumulated in enumerate(costs):
                    transition = edge(layer, left, right)
                    value = float(accumulated) + transition + emission
                    if value < next_costs[right]:
                        next_costs[right] = value
                        parent[right] = left
            if not np.isfinite(next_costs).any():
                return None, {"reason": "corridor_graph_disconnected", "failed_layer": layer}
            parents.append(parent)
            costs = next_costs
        ranking = np.argsort(costs)
        state_index = int(ranking[0])
        indices = [state_index]
        for parent in reversed(parents):
            state_index = int(parent[state_index])
            indices.append(state_index)
        indices.reverse()
        path = np.asarray([states[layer][indices[layer]] for layer in range(len(states))])
        margin = (
            float(costs[ranking[1]] - costs[ranking[0]]) if len(ranking) > 1 else None
        )
        return path, {
            "reason": None,
            "objective": round(float(costs[ranking[0]]), 6),
            "margin": round(margin, 6) if margin is not None else None,
            "anchors": len(fractions),
            "states_min": min(len(layer) for layer in states),
            "states_max": max(len(layer) for layer in states),
            "implicit_edges_evaluated": len(edge_cache),
        }

    def _multilevel_viterbi_map_match(
        self, observation: np.ndarray, baseline: np.ndarray
    ) -> tuple[Optional[np.ndarray], dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "attempted": True,
            "accepted": False,
            "method": "corridor_graph_multilevel_viterbi_v2",
            "corridor_graph_nodes": int(len(self._corridor_nodes)),
        }
        coarse_fractions = self._adaptive_anchor_fractions(observation, maximum=14)
        coarse, coarse_diag = self._viterbi_level(
            observation, baseline, coarse_fractions,
            radius_meters=12.0, candidate_limit=18,
        )
        diagnostics["coarse"] = coarse_diag
        if coarse is None:
            diagnostics["reason"] = coarse_diag.get("reason")
            return None, diagnostics
        fine_fractions = self._adaptive_anchor_fractions(observation, maximum=36)
        fine_guide = _resample_polyline(coarse, _trajectory_fractions(baseline))
        fine, fine_diag = self._viterbi_level(
            observation, fine_guide, fine_fractions,
            radius_meters=4.0, candidate_limit=14,
        )
        diagnostics["fine"] = fine_diag
        if fine is None:
            diagnostics["reason"] = fine_diag.get("reason")
            return None, diagnostics
        source_fraction = _trajectory_fractions(observation)
        fine_observed = _resample_polyline(observation, fine_fractions)
        correction = fine - fine_observed
        warped = observation.copy()
        warped[:, 0] += np.interp(source_fraction, fine_fractions, correction[:, 0])
        warped[:, 1] += np.interp(source_fraction, fine_fractions, correction[:, 1])
        repaired, reroutes = self._repair_collisions(warped)
        if repaired is None or self._collision_runs(repaired):
            diagnostics["reason"] = "dense_reconstruction_not_safe"
            return None, diagnostics
        diagnostics.update({"reason": None, "post_repair_segments": int(reroutes)})
        return repaired, diagnostics

    def align(
        self,
        trajectory: Any,
        reference_point: Optional[dict[str, Any]],
        direction_point: Optional[dict[str, Any]],
        *,
        timestamps: Any = None,
        coordinate_convention: str = "x_forward_y_left_z_up",
        scale_candidates: Optional[Iterable[float]] = None,
        yaw_offsets_degrees: Sequence[float] = (
            -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0
        ),
    ) -> dict[str, Any]:
        raw = _normalise_points(trajectory)
        base_diagnostics: dict[str, Any] = {
            "engine": "floorplan_constraint_engine_v5_exterior_guarded",
            "map_id": self.config.map_id,
            "plan_width": self.config.width,
            "plan_height": self.config.height,
            "meters_per_pixel": self.config.meters_per_pixel,
            "person_radius_meters": self.config.person_radius_meters,
            "point_count": int(len(raw)),
            "accepted": False,
            "support_mask_enabled": self._support_mask is not None,
            "support_coverage_ratio": (
                round(float(np.mean(self._support_mask)), 8)
                if self._support_mask is not None else None
            ),
            "walkable_bbox_cells": list(self.walkable_bbox),
            "annotation_bbox_cells": list(self.annotation_bbox),
        }
        if len(raw) < 2:
            return {"accepted": False, "trajectory": [], "diagnostics": {**base_diagnostics, "reason": "trajectory_too_short"}}
        if not reference_point or not direction_point:
            return {"accepted": False, "trajectory": [], "diagnostics": {**base_diagnostics, "reason": "missing_start_or_direction"}}

        requested_start = np.asarray([
            _finite_float(reference_point.get("x")) / 100.0 * self.config.width,
            _finite_float(reference_point.get("y")) / 100.0 * self.config.height,
        ])
        direction = np.asarray([
            _finite_float(direction_point.get("x")) / 100.0 * self.config.width,
            _finite_float(direction_point.get("y")) / 100.0 * self.config.height,
        ])
        if float(np.linalg.norm(direction - requested_start)) < max(self.config.width, self.config.height) * 0.004:
            return {"accepted": False, "trajectory": [], "diagnostics": {**base_diagnostics, "reason": "direction_too_short"}}
        start_cell = self._nearest_free(self._pixel_to_cell(requested_start))
        if start_cell is None:
            return {"accepted": False, "trajectory": [], "diagnostics": {**base_diagnostics, "reason": "no_walkable_start"}}
        start = self._cell_to_pixel(start_cell)
        start_snap_meters = (
            float(np.linalg.norm(start - requested_start))
            * self.config.meters_per_pixel
        )

        relative = raw - raw[0]
        if coordinate_convention == "x_forward_y_left_z_up":
            relative[:, 1] *= -1.0
        desired_heading = math.atan2(
            float(direction[1] - requested_start[1]),
            float(direction[0] - requested_start[0]),
        )
        duration = self._motion_duration_seconds(timestamps, relative)
        candidates = list(scale_candidates) if scale_candidates is not None else self._scale_candidates(relative, duration)
        hypotheses: list[dict[str, Any]] = []
        for scale in candidates:
            for yaw in yaw_offsets_degrees:
                points = self._build_hypothesis(relative, start, desired_heading, float(scale), float(yaw))
                score, metrics = self._score_hypothesis(points, duration, float(yaw))
                hypotheses.append({"score": score, "scale": float(scale), "yaw": float(yaw), "points": points, **metrics})
        if not hypotheses:
            return {"accepted": False, "trajectory": [], "diagnostics": {**base_diagnostics, "reason": "no_hypotheses"}}
        hypotheses.sort(key=lambda item: item["score"])

        # The plan is part of estimation, not a post-hoc accept/reject gate.
        # Repair a yaw-diverse beam and rank only after the walkable mask is
        # enforced so raw collision scores cannot freeze the wrong homotopy.
        feasible: list[dict[str, Any]] = []
        beam = self._select_diverse_beam(hypotheses)
        attempted = 0
        for hypothesis in beam:
            attempted += 1
            repaired, rerouted_segments = self._repair_collisions(hypothesis["points"])
            if repaired is None:
                continue
            corrected_metrics = self._path_metrics(repaired)
            # Dense sample metrics can still report a single-pixel nick after
            # A* + line-of-sight simplify, even when every polyline segment
            # is collision-free.  Prefer another local repair, but allow a small
            # residual collision budget when the only alternative is a spike.
            residual_collision_budget = 0.035
            if corrected_metrics["outside_ratio"] > 0.0 or (
                self._collision_runs(repaired)
                and corrected_metrics["collision_ratio"] > residual_collision_budget
            ):
                repaired_again, extra_segments = self._repair_collisions(repaired)
                if repaired_again is None:
                    continue
                repaired = repaired_again
                rerouted_segments += extra_segments
                corrected_metrics = self._path_metrics(repaired)
                if corrected_metrics["outside_ratio"] > 0.0 or (
                    self._collision_runs(repaired)
                    and corrected_metrics["collision_ratio"] > residual_collision_budget
                ):
                    continue
            matched = _resample_polyline(
                repaired,
                _trajectory_fractions(hypothesis["points"]),
            )
            displacement_m = (
                np.linalg.norm(matched - hypothesis["points"], axis=1)
                * self.config.meters_per_pixel
            )
            source_length = max(_polyline_length(hypothesis["points"]), 1e-9)
            corrected_length = _polyline_length(repaired)
            length_ratio = corrected_length / source_length
            median_correction = (
                float(np.median(displacement_m)) if len(displacement_m) else 0.0
            )
            p95_correction = (
                float(np.percentile(displacement_m, 95)) if len(displacement_m) else 0.0
            )
            # A floor plan may select scale/yaw and make local obstacle
            # repairs; it may not invent a different route.  The former
            # production weights made a 12 m displacement almost free
            # (0.005 * p95), so a geometrically absurd route could outrank a
            # faithful observation merely because it had more clearance.
            # Cap stays below the ~13–15 m "mask-legal spike" regime that still
            # looked wrong on Kerama despite length_ratio ≈ 1.17.
            correction_budget = max(
                2.5,
                min(11.0, float(hypothesis["length_meters"]) * 0.09),
            )
            sharp_reverse_ratio = _polyline_sharp_reverse_ratio(
                repaired,
                meters_per_pixel=self.config.meters_per_pixel,
            )
            # Mild aisle repairs are OK; long invented detours and zig-zag
            # spikes through drawing gaps are not the real walk.
            shape_preserved = (
                p95_correction <= correction_budget
                and 0.70 <= length_ratio <= 1.50
                and sharp_reverse_ratio <= 0.08
            )
            constrained_score = (
                float(hypothesis["score"])
                + 0.15 * median_correction
                + 0.08 * p95_correction
                + 0.75 * abs(math.log(max(length_ratio, 1e-9)))
                + 0.08 * rerouted_segments
                + 4.0 * sharp_reverse_ratio
            )
            feasible.append({
                **hypothesis,
                "repaired": repaired,
                "rerouted_segments": rerouted_segments,
                "corrected_metrics": corrected_metrics,
                "displacement_m": displacement_m,
                "length_ratio": length_ratio,
                "correction_budget_meters": correction_budget,
                "sharp_reverse_ratio": sharp_reverse_ratio,
                "shape_preserved": shape_preserved,
                "constrained_score": constrained_score,
            })

        if not feasible:
            return {
                "accepted": False,
                "trajectory": [],
                "diagnostics": {
                    **base_diagnostics,
                    "reason": "constraint_solution_not_found",
                    "rejection_reasons": ["no_collision_free_route"],
                    "hypothesis_count": len(hypotheses),
                    "beam_size": len(beam),
                    "constrained_hypotheses_attempted": attempted,
                    "raw_collision_ratio": round(float(hypotheses[0]["collision_ratio"]), 6),
                },
            }

        feasible.sort(key=lambda item: item["constrained_score"])
        # With a positive support mask, accepting an arbitrary collision-free
        # path is worse than reporting that the visual observation and map do
        # not agree.  Only shape-preserving candidates are authoritative.
        production_feasible = (
            [item for item in feasible if item["shape_preserved"]]
            if self._support_mask is not None
            else feasible
        )
        if not production_feasible:
            closest = feasible[0]
            closest_displacement = closest["displacement_m"]
            closest_p95 = (
                float(np.percentile(closest_displacement, 95))
                if len(closest_displacement) else 0.0
            )
            return {
                "accepted": False,
                "trajectory": [],
                "diagnostics": {
                    **base_diagnostics,
                    "reason": "map_correction_exceeds_observation_budget",
                    "rejection_reasons": ["topology_destroying_map_correction"],
                    "hypothesis_count": len(hypotheses),
                    "beam_size": len(beam),
                    "constrained_hypotheses_attempted": attempted,
                    "feasible_hypotheses": len(feasible),
                    "correction_p95_meters": round(closest_p95, 3),
                    "correction_budget_meters": round(
                        float(closest["correction_budget_meters"]), 3
                    ),
                    "sharp_reverse_ratio": round(
                        float(closest.get("sharp_reverse_ratio", 0.0)), 4
                    ),
                    "length_ratio": round(float(closest["length_ratio"]), 5),
                },
            }
        best = production_feasible[0]
        repaired = best["repaired"]
        corrected_metrics = best["corrected_metrics"]
        displacement_m = best["displacement_m"]
        length_ratio = float(best["length_ratio"])
        rerouted_segments = int(best["rerouted_segments"])
        p95_correction = float(np.percentile(displacement_m, 95)) if len(displacement_m) else 0.0
        experimental_hmm_enabled = os.getenv(
            "TRACKAI_ENABLE_EXPERIMENTAL_MULTILEVEL_HMM", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        nonlinear_diagnostics: dict[str, Any] = {
            "attempted": False,
            "accepted": False,
            "method": "corridor_graph_multilevel_viterbi_v2",
            "reason": (
                "global_solution_stable"
                if experimental_hmm_enabled
                else "disabled_pending_production_validation"
            ),
            "production_enabled": experimental_hmm_enabled,
        }
        if experimental_hmm_enabled and (rerouted_segments > 0 or p95_correction > 2.0):
            nonlinear, nonlinear_diagnostics = self._multilevel_viterbi_map_match(
                best["points"], repaired
            )
            if nonlinear is not None:
                nonlinear_matched = _resample_polyline(
                    nonlinear, _trajectory_fractions(best["points"])
                )
                nonlinear_displacement = (
                    np.linalg.norm(nonlinear_matched - best["points"], axis=1)
                    * self.config.meters_per_pixel
                )
                nonlinear_p95 = float(np.percentile(nonlinear_displacement, 95))
                nonlinear_length_ratio = _polyline_length(nonlinear) / max(
                    _polyline_length(best["points"]), 1e-9
                )
                observed_resampled = _resample_polyline(
                    best["points"], np.linspace(0.0, 1.0, 64)
                )
                candidate_resampled = _resample_polyline(
                    nonlinear, np.linspace(0.0, 1.0, 64)
                )

                def signed_turn(points: np.ndarray) -> float:
                    deltas = np.diff(points[:, :2], axis=0)
                    lengths = np.linalg.norm(deltas, axis=1)
                    deltas = deltas[lengths > 1e-6]
                    if len(deltas) < 2:
                        return 0.0
                    headings = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
                    return math.degrees(float(headings[-1] - headings[0]))

                observed_turn = signed_turn(observed_resampled)
                candidate_turn = signed_turn(candidate_resampled)
                chirality_preserved = (
                    abs(observed_turn) < 20.0 and abs(candidate_turn) < 25.0
                ) or (
                    observed_turn * candidate_turn > 0.0
                    and 0.60 <= abs(candidate_turn) / max(abs(observed_turn), 1e-9) <= 1.40
                )
                improves = (
                    nonlinear_p95 <= p95_correction * 0.97
                    or abs(math.log(max(nonlinear_length_ratio, 1e-9)))
                    < abs(math.log(max(length_ratio, 1e-9))) * 0.92
                )
                bounded = (
                    nonlinear_p95 <= min(p95_correction + 0.25, 4.0)
                    and 0.75 <= nonlinear_length_ratio <= 1.35
                    and chirality_preserved
                    and not self._collision_runs(nonlinear)
                    and self._path_metrics(nonlinear)["outside_ratio"] == 0.0
                )
                nonlinear_diagnostics.update({
                    "correction_p95_before_meters": round(p95_correction, 3),
                    "correction_p95_after_meters": round(nonlinear_p95, 3),
                    "length_ratio_before": round(length_ratio, 5),
                    "length_ratio_after": round(nonlinear_length_ratio, 5),
                    "observed_signed_turn_degrees": round(observed_turn, 3),
                    "candidate_signed_turn_degrees": round(candidate_turn, 3),
                    "chirality_preserved": chirality_preserved,
                })
                if improves and bounded:
                    repaired = nonlinear
                    corrected_metrics = self._path_metrics(repaired)
                    displacement_m = nonlinear_displacement
                    p95_correction = nonlinear_p95
                    length_ratio = nonlinear_length_ratio
                    nonlinear_diagnostics["accepted"] = True
                    nonlinear_diagnostics["reason"] = None
                else:
                    nonlinear_diagnostics["accepted"] = False
                    nonlinear_diagnostics["reason"] = "safe_baseline_not_improved"
        allowed_correction = (
            float(best["correction_budget_meters"])
            if self._support_mask is not None
            else max(4.0, min(12.0, float(best["length_meters"]) * 0.18))
        )

        # Scale, speed and correction magnitude affect confidence, but they do
        # not invalidate a route already proven to lie inside the fixed plan.
        quality_warnings: list[str] = []
        if not 0.70 <= length_ratio <= 1.65:
            quality_warnings.append("route_length_changed_significantly")
        if p95_correction > allowed_correction:
            quality_warnings.append("large_map_correction_applied")
        if float(corrected_metrics.get("collision_ratio", 0.0)) > 0.0:
            quality_warnings.append("residual_micro_collisions_kept_to_preserve_shape")
        speed = float(best["speed_mps"])
        if math.isfinite(speed) and not 0.20 <= speed <= 3.20:
            quality_warnings.append("walking_speed_prior_inconsistent")
        if start_snap_meters > max(0.05, self.cell_meters * 0.75):
            quality_warnings.append("start_projected_to_walkable_area")

        second_score = (
            float(feasible[1]["constrained_score"])
            if len(feasible) > 1
            else float(best["constrained_score"] + 1.0)
        )
        margin = max(0.0, second_score - float(best["constrained_score"]))
        confidence = float(np.clip(
            0.35
            + min(margin / 0.8, 0.25)
            + max(0.0, 1.0 - p95_correction / max(allowed_correction, 1e-9)) * 0.30
            - min(0.08 * len(quality_warnings), 0.24),
            0.0,
            1.0,
        ))
        probability_correct, calibration = calibrated_probability(confidence)
        diagnostics = {
            **base_diagnostics,
            "accepted": True,
            "reason": None,
            "rejection_reasons": [],
            "quality_warnings": quality_warnings,
            "constraint_mode": (
                "hard_obstacles_and_cad_support"
                if self._support_mask is not None else "hard_walkable_mask"
            ),
            "hypothesis_count": len(hypotheses),
            "beam_size": len(beam),
            "constrained_hypotheses_attempted": attempted,
            "feasible_hypotheses": len(feasible),
            "shape_preserving_hypotheses": len(production_feasible),
            "selected_scale_pixels_per_unit": round(float(best["scale"]), 8),
            "selected_yaw_offset_degrees": round(float(best["yaw"]), 3),
            "estimated_length_meters": round(float(best["length_meters"]), 3),
            "motion_duration_seconds": round(float(duration), 3) if duration is not None else None,
            "estimated_speed_mps": round(speed, 3) if math.isfinite(speed) else None,
            "raw_collision_ratio": round(float(best["collision_ratio"]), 6),
            "corrected_collision_ratio": round(float(corrected_metrics["collision_ratio"]), 6),
            "outside_ratio": round(float(corrected_metrics["outside_ratio"]), 6),
            "rerouted_segments": rerouted_segments,
            "start_snap_meters": round(start_snap_meters, 3),
            "correction_median_meters": round(float(np.median(displacement_m)), 3),
            "correction_p95_meters": round(p95_correction, 3),
            "correction_budget_meters": round(allowed_correction, 3),
            "sharp_reverse_ratio": round(float(best.get("sharp_reverse_ratio", 0.0)), 4),
            "length_ratio": round(length_ratio, 5),
            "hypothesis_score": round(float(best["score"]), 6),
            "constrained_score": round(float(best["constrained_score"]), 6),
            "runner_up_margin": round(margin, 6),
            "nonlinear_map_matching": nonlinear_diagnostics,
            # Kept for API compatibility; semantically this is a quality score.
            "confidence": round(confidence, 4),
            "quality_score": round(confidence, 4),
            "probability_correct": (
                round(probability_correct, 4)
                if probability_correct is not None else None
            ),
            "confidence_calibration": calibration,
            "coordinate_convention": "plan_pixels_x_right_y_down",
        }
        output = [
            [round(float(point[0]), 3), round(float(point[1]), 3), 0.0]
            for point in repaired
        ]
        return {"accepted": True, "trajectory": output, "diagnostics": diagnostics}


_ENGINE_CACHE: dict[str, FloorplanConstraintEngine] = {}


def get_floorplan_engine(map_id: str = DEFAULT_FLOORPLAN_ID) -> FloorplanConstraintEngine:
    engine = _ENGINE_CACHE.get(map_id)
    if engine is None:
        engine = FloorplanConstraintEngine.load(map_id)
        _ENGINE_CACHE[map_id] = engine
    return engine


def _map_turn_points(
    turns: Any,
    source: Any,
    mapped: list[list[float]],
    *,
    meters_per_pixel: float,
    timestamps: Any = None,
) -> list[dict[str, Any]]:
    if not isinstance(turns, list) or not mapped:
        return []
    source_points = _normalise_points(source)
    mapped_points = _normalise_points(mapped)
    if len(source_points) == 0 or len(mapped_points) == 0:
        return []
    source_fractions = _trajectory_fractions(source_points)
    mapped_fractions = _trajectory_fractions(mapped_points)
    result: list[dict[str, Any]] = []
    for item in turns:
        if not isinstance(item, dict):
            continue
        source_index = int(round(_finite_float(item.get("trajectory_index"), 0.0)))
        source_index = max(0, min(len(source_points) - 1, source_index))
        turn_fraction = float(source_fractions[source_index])
        mapped_index = int(np.argmin(np.abs(mapped_fractions - turn_fraction)))
        approach_source = max(0, min(
            len(source_points) - 1,
            int(round(_finite_float(item.get("approach_index"), source_index))),
        ))
        exit_source = max(approach_source, min(
            len(source_points) - 1,
            int(round(_finite_float(item.get("exit_index"), source_index))),
        ))
        approach_mapped = int(np.argmin(np.abs(
            mapped_fractions - float(source_fractions[approach_source])
        )))
        exit_mapped = int(np.argmin(np.abs(
            mapped_fractions - float(source_fractions[exit_source])
        )))
        local_length_meters = (
            _polyline_length(mapped_points[approach_mapped:exit_mapped + 1])
            * meters_per_pixel
        )
        angle = abs(_finite_float(
            item.get("geometry_angle_degrees", item.get("angle_degrees")), 0.0
        ))
        duration_seconds = None
        if isinstance(timestamps, list) and len(timestamps) == len(source_points):
            start_time = _finite_float(timestamps[approach_source], math.nan)
            end_time = _finite_float(timestamps[exit_source], math.nan)
            if math.isfinite(start_time) and math.isfinite(end_time) and end_time > start_time:
                duration_seconds = end_time - start_time
        result.append({
            **item,
            "trajectory_index": mapped_index,
            "position": mapped[mapped_index],
            "map_constrained": True,
            "map_turn_arc_meters": round(local_length_meters, 4),
            "map_curvature_degrees_per_meter": (
                round(angle / local_length_meters, 4)
                if local_length_meters > 1e-6 else None
            ),
            "map_yaw_rate_degrees_per_second": (
                round(angle / duration_seconds, 4)
                if duration_seconds is not None else None
            ),
        })
    return result


def apply_floorplan_constraints(
    result: dict[str, Any],
    map_context: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Attach an accepted map trajectory without mutating visual-source data."""
    if not isinstance(result, dict):
        return result
    context = map_context or {}
    map_id = str(context.get("floorplan_id") or DEFAULT_FLOORPLAN_ID)
    points = result.get("plan_trajectory") or result.get("trajectory") or []
    stats = dict(result.get("processing_stats") or {})
    for stale_key in (
        "map_matching_applied",
        "map_trajectory_points",
        "map_confidence",
        "map_distance_meters",
        "floorplan_constraint",
    ):
        stats.pop(stale_key, None)
    method = str(result.get("method") or "").lower()
    convention = (
        "x_forward_y_left_z_up"
        if method.startswith("r3")
        else "x_right_y_down"
    )
    quality = result.get("trajectory_quality") or stats.get("r3_trajectory_quality") or {}
    if isinstance(quality, dict):
        projection = quality.get("projection") or {}
        if isinstance(projection, dict):
            convention = str(projection.get("plan_coordinate_convention") or convention)
    candidate_payload = result.get("lingbot_fusion_candidate")
    candidate_points: Any = []
    independent_points: Any = []
    independent_quality_ok = False
    if isinstance(candidate_payload, dict) and candidate_payload.get("accepted"):
        candidate_points = candidate_payload.get("plan_trajectory") or []
    if isinstance(candidate_payload, dict) and candidate_payload.get("independent_accepted"):
        independent_points = candidate_payload.get("independent_plan_trajectory") or []
        diagnostics = candidate_payload.get("diagnostics") or {}
        independent_quality = (
            diagnostics.get("independent_quality") if isinstance(diagnostics, dict) else None
        )
        if isinstance(independent_quality, dict):
            independent_quality_ok = bool(independent_quality.get("accepted", True))
        else:
            independent_quality_ok = True
    fragmented_r3 = method.startswith("r3") and _r3_is_severely_fragmented(result)
    use_independent = bool(
        fragmented_r3 and independent_points and independent_quality_ok
    )
    primary_observation_source = (
        "r3"
        if method.startswith("r3")
        else ("lingbot" if method.startswith("lingbot") else (method or "visual_odometry"))
    )
    selected_points = points
    selected_observation_source = primary_observation_source
    source_selection: dict[str, Any] = {
        "primary": primary_observation_source,
        "candidate": (
            "lingbot_independent" if use_independent
            else ("r3_lingbot_fusion" if candidate_points else None)
        ),
        "selected": primary_observation_source,
        "reason": "no_fusion_candidate" if not candidate_points else "primary_preferred",
    }
    try:
        engine = get_floorplan_engine(map_id)
        source_timestamps = (
            result.get("r3_source_timestamps_seconds")
            or result.get("source_timestamps_seconds")
        )
        if use_independent:
            selected_points = independent_points
            selected_observation_source = "lingbot_independent"
            independent_timestamps = None
            if isinstance(candidate_payload, dict):
                independent_timestamps = (
                    candidate_payload.get("lingbot_source_timestamps_seconds")
                    or candidate_payload.get("source_timestamps_seconds")
                )
            if not isinstance(independent_timestamps, list) or len(independent_timestamps) < 2:
                independent_timestamps = _resample_timestamps(
                    source_timestamps, len(independent_points)
                )
            # PCA Y-sign is gauge freedom.  When R³ is fragmented it cannot
            # adjudicate chirality, so score both polarities on the floor plan.
            independent_variants: list[tuple[str, Any]] = [
                ("native", independent_points),
            ]
            flipped = _flip_polyline_y(independent_points)
            if flipped:
                independent_variants.append(("y_flip", flipped))

            def _independent_rank(payload: dict[str, Any]) -> tuple[float, float, float, float]:
                diag = payload.get("diagnostics") or {}
                accepted_rank = 0.0 if payload.get("accepted") else 1.0
                length_ratio = max(_finite_float(diag.get("length_ratio"), 1e9), 1e-9)
                p95 = _finite_float(diag.get("correction_p95_meters"), 1e9)
                score = _finite_float(diag.get("constrained_score"), 1e9)
                return (
                    accepted_rank,
                    abs(math.log(length_ratio)),
                    p95,
                    score,
                )

            alignment = {
                "accepted": False,
                "trajectory": [],
                "diagnostics": {"reason": "independent_polarity_unavailable"},
            }
            best_label = "native"
            best_rank = (1.0, float("inf"), float("inf"), float("inf"))
            polarity_diagnostics: list[dict[str, Any]] = []
            for label, variant_points in independent_variants:
                candidate_alignment = engine.align(
                    variant_points,
                    context.get("reference_point"),
                    context.get("direction_point"),
                    timestamps=independent_timestamps,
                    coordinate_convention="x_right_y_down",
                )
                diag = candidate_alignment.get("diagnostics") or {}
                polarity_diagnostics.append({
                    "label": label,
                    "accepted": bool(candidate_alignment.get("accepted")),
                    "length_ratio": diag.get("length_ratio"),
                    "correction_p95_meters": diag.get("correction_p95_meters"),
                    "constrained_score": diag.get("constrained_score"),
                })
                rank = _independent_rank(candidate_alignment)
                if rank < best_rank:
                    best_rank = rank
                    best_label = label
                    alignment = candidate_alignment
                    selected_points = variant_points
            source_selection.update({
                "selected": selected_observation_source,
                "reason": "fragmented_r3_uses_independent_lingbot",
                "r3_severely_fragmented": True,
                "independent_polarity": best_label,
                "independent_polarity_candidates": polarity_diagnostics,
            })
        else:
            if fragmented_r3 and independent_points and not independent_quality_ok:
                source_selection["independent_rejected_reason"] = "independent_quality_failed"
            alignment = engine.align(
                points,
                context.get("reference_point"),
                context.get("direction_point"),
                timestamps=source_timestamps,
                coordinate_convention=convention,
            )
        if candidate_points and not use_independent:
            candidate_alignment = engine.align(
                candidate_points,
                context.get("reference_point"),
                context.get("direction_point"),
                timestamps=result.get("r3_source_timestamps_seconds") or result.get("source_timestamps_seconds"),
                coordinate_convention=convention,
            )
            primary_diag = alignment.get("diagnostics") or {}
            candidate_diag = candidate_alignment.get("diagnostics") or {}
            primary_score = _finite_float(primary_diag.get("constrained_score"), float("inf"))
            candidate_score = _finite_float(candidate_diag.get("constrained_score"), float("inf"))
            primary_correction = _finite_float(
                primary_diag.get("correction_p95_meters"), float("inf")
            )
            candidate_correction = _finite_float(
                candidate_diag.get("correction_p95_meters"), float("inf")
            )
            candidate_is_preferred = bool(candidate_alignment.get("accepted")) and (
                not alignment.get("accepted")
                or (
                    candidate_score <= primary_score + 0.10
                    and candidate_correction <= primary_correction * 1.10 + 0.25
                )
            )
            source_selection.update({
                "primary_constrained_score": (
                    round(primary_score, 6) if math.isfinite(primary_score) else None
                ),
                "candidate_constrained_score": (
                    round(candidate_score, 6) if math.isfinite(candidate_score) else None
                ),
                "primary_correction_p95_meters": (
                    round(primary_correction, 3) if math.isfinite(primary_correction) else None
                ),
                "candidate_correction_p95_meters": (
                    round(candidate_correction, 3) if math.isfinite(candidate_correction) else None
                ),
            })
            if candidate_is_preferred:
                alignment = candidate_alignment
                selected_points = candidate_points
                selected_observation_source = "r3_lingbot_fusion"
                source_selection.update({
                    "selected": selected_observation_source,
                    "reason": (
                        "primary_constraint_failed"
                        if not primary_diag.get("accepted")
                        else "fusion_supported_by_floorplan"
                    ),
                })
            else:
                source_selection["reason"] = "primary_has_lower_map_cost"
    except Exception as exc:
        alignment = {
            "accepted": False,
            "trajectory": [],
            "diagnostics": {
                "engine": "floorplan_constraint_engine_v5_exterior_guarded",
                "map_id": map_id,
                "accepted": False,
                "reason": "engine_error",
                "error": str(exc),
            },
        }

    updated = dict(result)
    diagnostics = dict(alignment["diagnostics"])
    diagnostics["trajectory_observation_source"] = selected_observation_source
    diagnostics["observation_source_selection"] = source_selection
    updated["floorplan_constraint"] = diagnostics
    stats["floorplan_constraint"] = diagnostics
    stats["map_matching_applied"] = bool(alignment["accepted"])
    stats["floorplan_id"] = map_id
    if alignment["accepted"]:
        mapped = alignment["trajectory"]
        # R3 turn indices have no valid correspondence to an independent
        # LingBot trajectory and would create authoritative-looking false turns.
        source_turns = [] if selected_observation_source == "lingbot_independent" else result.get("turn_points")
        selected_timestamps = (
            candidate_payload.get("lingbot_source_timestamps_seconds")
            if selected_observation_source == "lingbot_independent"
            and isinstance(candidate_payload, dict)
            else source_timestamps
        )
        map_turns = _map_turn_points(
            source_turns,
            selected_points,
            mapped,
            meters_per_pixel=engine.config.meters_per_pixel,
            timestamps=selected_timestamps,
        )
        updated["map_trajectory"] = mapped
        updated["map_turn_points"] = map_turns
        updated["final_turn_points"] = (
            []
            if selected_observation_source == "lingbot_independent"
            else (map_turns or result.get("turn_points") or [])
        )
        map_distance = (
            _finite_float(diagnostics.get("estimated_length_meters"))
            * _finite_float(diagnostics.get("length_ratio"), 1.0)
        )
        stats["map_trajectory_points"] = len(mapped)
        stats["map_confidence"] = diagnostics.get("confidence")
        stats["map_observation_source"] = selected_observation_source
        stats["map_distance_meters"] = round(map_distance, 3)
        stats["estimated_distance"] = round(map_distance, 3)
        updated["map_metadata"] = {
            "map_id": map_id,
            "plan_width": diagnostics.get("plan_width"),
            "plan_height": diagnostics.get("plan_height"),
            "meters_per_pixel": diagnostics.get("meters_per_pixel"),
            "person_radius_meters": diagnostics.get("person_radius_meters"),
            "source": "fixed_floorplan_constraint_engine",
            "trajectory_observation_source": selected_observation_source,
        }
    else:
        updated.pop("map_trajectory", None)
        updated.pop("map_turn_points", None)
        updated.pop("map_metadata", None)
        updated["final_turn_points"] = result.get("turn_points") or []
    updated["processing_stats"] = stats
    return updated
