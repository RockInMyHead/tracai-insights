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
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage


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
            constant_values=False,
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
        factors = (0.45, 0.58, 0.72, 0.86, 1.0, 1.16, 1.35, 1.60, 1.95, 2.35)
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
        per_yaw: int = 2,
        global_top: int = 12,
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
                wall_cost = 0.42 / max(clearance + 0.12, 0.12)
                raw_cost = 0.055 * float(deviation[ny - min_y, nx - min_x])
                candidate = cost + step * (1.0 + wall_cost + raw_cost)
                key = (nx, ny)
                if candidate >= costs.get(key, float("inf")):
                    continue
                costs[key] = candidate
                previous[key] = current
                heuristic = math.hypot(end[0] - nx, end[1] - ny)
                heappush(queue, (candidate + heuristic, candidate, key))
        # A route around a long machine can leave the first local window.
        # Expand deterministically instead of declaring the plan disconnected.
        full_margin = max(self.rows, self.cols)
        if margin < full_margin:
            return self._astar(
                start_point,
                end_point,
                raw_segment,
                _search_margin_cells=min(full_margin, margin * 2),
            )
        return None

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
        for left, right in merged:
            if left == 0 and self._point_occupied(points[0]):
                return None, len(merged)
            rebuilt.extend(points[cursor:left])
            route = self._astar(points[left], points[right], points[left:right + 1])
            if route is None:
                return None, len(merged)
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
        return np.asarray(rebuilt, dtype=np.float64), len(merged)

    def align(
        self,
        trajectory: Any,
        reference_point: Optional[dict[str, Any]],
        direction_point: Optional[dict[str, Any]],
        *,
        timestamps: Any = None,
        coordinate_convention: str = "x_forward_y_left_z_up",
        scale_candidates: Optional[Iterable[float]] = None,
        yaw_offsets_degrees: Sequence[float] = (-10.0, -5.0, 0.0, 5.0, 10.0),
    ) -> dict[str, Any]:
        raw = _normalise_points(trajectory)
        base_diagnostics: dict[str, Any] = {
            "engine": "floorplan_constraint_engine_v2_gip",
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
            # is collision-free.  The hard contract is segment safety.
            if self._collision_runs(repaired) or corrected_metrics["outside_ratio"] > 0.0:
                repaired_again, extra_segments = self._repair_collisions(repaired)
                if repaired_again is None:
                    continue
                repaired = repaired_again
                rerouted_segments += extra_segments
                corrected_metrics = self._path_metrics(repaired)
                if self._collision_runs(repaired) or corrected_metrics["outside_ratio"] > 0.0:
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
            constrained_score = (
                float(hypothesis["score"])
                + 0.02 * median_correction
                + 0.005 * p95_correction
                + 0.30 * abs(math.log(max(length_ratio, 1e-9)))
                + 0.02 * rerouted_segments
            )
            feasible.append({
                **hypothesis,
                "repaired": repaired,
                "rerouted_segments": rerouted_segments,
                "corrected_metrics": corrected_metrics,
                "displacement_m": displacement_m,
                "length_ratio": length_ratio,
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
        best = feasible[0]
        repaired = best["repaired"]
        corrected_metrics = best["corrected_metrics"]
        displacement_m = best["displacement_m"]
        length_ratio = float(best["length_ratio"])
        rerouted_segments = int(best["rerouted_segments"])
        p95_correction = float(np.percentile(displacement_m, 95)) if len(displacement_m) else 0.0
        allowed_correction = max(
            4.0,
            min(12.0, float(best["length_meters"]) * 0.18),
        )

        # Scale, speed and correction magnitude affect confidence, but they do
        # not invalidate a route already proven to lie inside the fixed plan.
        quality_warnings: list[str] = []
        if not 0.70 <= length_ratio <= 1.65:
            quality_warnings.append("route_length_changed_significantly")
        if p95_correction > allowed_correction:
            quality_warnings.append("large_map_correction_applied")
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
            "length_ratio": round(length_ratio, 5),
            "hypothesis_score": round(float(best["score"]), 6),
            "constrained_score": round(float(best["constrained_score"]), 6),
            "runner_up_margin": round(margin, 6),
            "confidence": round(confidence, 4),
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
        result.append({
            **item,
            "trajectory_index": mapped_index,
            "position": mapped[mapped_index],
            "map_constrained": True,
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
            alignment = engine.align(
                independent_points,
                context.get("reference_point"),
                context.get("direction_point"),
                timestamps=independent_timestamps,
                coordinate_convention="x_right_y_down",
            )
            source_selection.update({
                "selected": selected_observation_source,
                "reason": "fragmented_r3_uses_independent_lingbot",
                "r3_severely_fragmented": True,
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
                "engine": "floorplan_constraint_engine_v2_gip",
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
        map_turns = _map_turn_points(source_turns, selected_points, mapped)
        updated["map_trajectory"] = mapped
        updated["map_turn_points"] = map_turns
        updated["final_turn_points"] = map_turns or result.get("turn_points") or []
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
