"""
Map postprocessing: привязка траектории к плану и пересчёт поворотов по карте.

Источник истины по поворотам:
  1. IMU yaw — если появится.
  2. Map-corrected path turns — если есть план.
  3. Integrated visual yaw из VO (raw_turn_points / turn_points).
  4. Chord-angle validator — secondary check only.

Исправления:
- унифицирован контракт траектории: x, y, heading_deg;
- сохранена backward compatibility со старым полем z;
- финальные turn points считаются по map_path и сериализуются с heading_deg;
- map_trajectory теперь хранит heading_deg по сегментам карты, а не фиктивный z=0.
"""

import base64
import math
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


GRID_CELL = 8
LUMINANCE_THRESHOLD = 0.6
SUBSAMPLE = 5


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_heading(point: Any) -> float:
    if isinstance(point, dict):
        if "heading_deg" in point:
            return _to_float(point.get("heading_deg"), 0.0)
        return _to_float(point.get("z", point.get(2, 0.0)), 0.0)
    if isinstance(point, (list, tuple)) and len(point) > 2:
        return _to_float(point[2], 0.0)
    return 0.0


def _normalize_trajectory(traj: Any) -> List[Dict[str, float]]:
    if not isinstance(traj, list):
        return []
    points: List[Dict[str, float]] = []
    for point in traj:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append({
                "x": _to_float(point[0]),
                "y": _to_float(point[1]),
                "heading_deg": _get_heading(point),
            })
        elif isinstance(point, dict):
            points.append({
                "x": _to_float(point.get("x", point.get(0, 0.0))),
                "y": _to_float(point.get("y", point.get(1, 0.0))),
                "heading_deg": _get_heading(point),
            })
    return points


def _decode_data_url_image(data_url: str) -> Optional[np.ndarray]:
    if not isinstance(data_url, str) or "base64," not in data_url:
        return None
    try:
        raw = base64.b64decode(data_url.split("base64,", 1)[1])
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _render_drawn_plan(shapes: Any, width: int = 800, height: int = 600) -> Optional[np.ndarray]:
    if not isinstance(shapes, list):
        return None
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        points = shape.get("points") or []
        if len(points) < 2:
            continue
        p1 = points[0]
        p2 = points[1]
        x1 = int(round(_to_float(p1.get("x"))))
        y1 = int(round(_to_float(p1.get("y"))))
        x2 = int(round(_to_float(p2.get("x"))))
        y2 = int(round(_to_float(p2.get("y"))))
        if shape.get("type") == "rect":
            cv2.rectangle(canvas, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), (0, 0, 0), thickness=-1)
        else:
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), thickness=8, lineType=cv2.LINE_AA)
    return canvas


def _skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    img = binary_mask.copy()
    skeleton = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        residue = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return cv2.dilate(skeleton, np.ones((3, 3), dtype=np.uint8), iterations=1)


def _build_occupancy_grid(plan_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
    height, width = plan_image.shape[:2]
    gray = cv2.cvtColor(plan_image, cv2.COLOR_BGR2GRAY)
    cols = int(math.ceil(width / GRID_CELL))
    rows = int(math.ceil(height / GRID_CELL))
    grid = np.zeros(cols * rows, dtype=np.uint8)

    for gy in range(rows):
        for gx in range(cols):
            y1 = gy * GRID_CELL
            y2 = min((gy + 1) * GRID_CELL, height)
            x1 = gx * GRID_CELL
            x2 = min((gx + 1) * GRID_CELL, width)
            roi = gray[y1:y2, x1:x2]
            avg = float(np.mean(roi) / 255.0) if roi.size else 1.0
            grid[gy * cols + gx] = 0 if avg >= LUMINANCE_THRESHOLD else 1

    free_mask = ((grid.reshape(rows, cols) == 0).astype(np.uint8)) * 255
    clearance = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)
    skeleton = _skeletonize(free_mask)
    return grid, clearance, skeleton, cols, rows, width, height


def _to_grid(x: float, y: float, cols: int, rows: int) -> Tuple[int, int]:
    gx = max(0, min(cols - 1, int(x / GRID_CELL)))
    gy = max(0, min(rows - 1, int(y / GRID_CELL)))
    return gx, gy


def _to_world(gx: int, gy: int) -> Tuple[float, float]:
    return gx * GRID_CELL + GRID_CELL / 2.0, gy * GRID_CELL + GRID_CELL / 2.0


def _is_wall(grid: np.ndarray, cols: int, rows: int, gx: int, gy: int) -> bool:
    if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
        return True
    return bool(grid[gy * cols + gx] == 1)


def _nearest_free(grid: np.ndarray, cols: int, rows: int, gx: int, gy: int) -> Tuple[int, int]:
    if not _is_wall(grid, cols, rows, gx, gy):
        return gx, gy
    for radius in range(1, 6):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = gx + dx, gy + dy
                if not _is_wall(grid, cols, rows, nx, ny):
                    return nx, ny
    return gx, gy


def _nearest_mask_point(mask: np.ndarray, gx: int, gy: int, max_radius: int = 8) -> Optional[Tuple[int, int]]:
    rows, cols = mask.shape[:2]
    if 0 <= gx < cols and 0 <= gy < rows and mask[gy, gx] > 0:
        return gx, gy
    for radius in range(1, max_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < cols and 0 <= ny < rows and mask[ny, nx] > 0:
                    return nx, ny
    return None


def _astar(
    grid: np.ndarray,
    clearance: np.ndarray,
    cols: int,
    rows: int,
    start: Tuple[int, int],
    end: Tuple[int, int],
    allowed_mask: Optional[np.ndarray] = None,
) -> List[Tuple[float, float]]:
    if allowed_mask is not None:
        start = _nearest_mask_point(allowed_mask, *start) or _nearest_free(grid, cols, rows, *start)
        end = _nearest_mask_point(allowed_mask, *end) or _nearest_free(grid, cols, rows, *end)
    else:
        start = _nearest_free(grid, cols, rows, *start)
        end = _nearest_free(grid, cols, rows, *end)
    start_key = start[1] * cols + start[0]
    end_key = end[1] * cols + end[0]

    open_heap: List[Tuple[float, int]] = [(0.0, start_key)]
    came_from: Dict[int, int] = {}
    g_score: Dict[int, float] = {start_key: 0.0}
    visited = set()
    neighbors = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while open_heap:
        _, current = heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == end_key:
            path: List[Tuple[float, float]] = []
            cur = current
            while True:
                gx = cur % cols
                gy = cur // cols
                path.append(_to_world(gx, gy))
                if cur == start_key:
                    break
                cur = came_from[cur]
            path.reverse()
            return path

        cgx = current % cols
        cgy = current // cols
        for dx, dy in neighbors:
            ngx, ngy = cgx + dx, cgy + dy
            if _is_wall(grid, cols, rows, ngx, ngy):
                continue
            if allowed_mask is not None and allowed_mask[ngy, ngx] == 0:
                continue
            nkey = ngy * cols + ngx
            step_cost = math.sqrt(dx * dx + dy * dy)
            clearance_value = float(clearance[ngy, ngx]) if clearance.size else 1.0
            wall_penalty = 1.0 / max(clearance_value + 0.5, 0.5)
            tentative = g_score[current] + step_cost * (1.0 + 0.25 * wall_penalty)
            if tentative >= g_score.get(nkey, float("inf")):
                continue
            came_from[nkey] = current
            g_score[nkey] = tentative
            heuristic = math.hypot(end[0] - ngx, end[1] - ngy)
            heappush(open_heap, (tentative + heuristic, nkey))

    sx, sy = _to_world(*start)
    ex, ey = _to_world(*end)
    return [(sx, sy), (ex, ey)]


def _build_anchor_points(points: List[Dict[str, float]], turn_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(points) < 2:
        return [{"point": p, "source_index": i, "turn": None} for i, p in enumerate(points)]

    anchor_indices = {0, len(points) - 1}
    turn_by_index: Dict[int, Dict[str, Any]] = {}
    for turn in turn_points:
        idx = int(round(_to_float(turn.get("trajectory_index"), 0.0)))
        idx = max(0, min(len(points) - 1, idx))
        anchor_indices.add(idx)
        turn_by_index[idx] = turn

    sorted_indices = sorted(anchor_indices)
    expanded_indices: List[int] = []
    max_gap = max(24, len(points) // 10)
    for idx, current in enumerate(sorted_indices):
        expanded_indices.append(current)
        if idx == len(sorted_indices) - 1:
            continue
        nxt = sorted_indices[idx + 1]
        gap = nxt - current
        if gap > max_gap:
            for extra in range(current + max_gap, nxt, max_gap):
                expanded_indices.append(extra)

    expanded_indices = sorted(set(expanded_indices))
    return [
        {
            "point": points[idx],
            "source_index": idx,
            "turn": turn_by_index.get(idx),
        }
        for idx in expanded_indices
    ]


def _build_observations(points: List[Dict[str, float]], turn_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(points) < 2:
        return [{"point": point, "source_index": idx, "turn": None} for idx, point in enumerate(points)]

    turn_by_index: Dict[int, Dict[str, Any]] = {}
    indices = {0, len(points) - 1}
    stride = max(10, len(points) // 24)
    for idx in range(0, len(points), stride):
        indices.add(idx)

    for turn in turn_points:
        idx = int(round(_to_float(turn.get("trajectory_index"), 0.0)))
        idx = max(0, min(len(points) - 1, idx))
        indices.add(idx)
        turn_by_index[idx] = turn

    sorted_indices = sorted(indices)
    observations: List[Dict[str, Any]] = []
    for idx in sorted_indices:
        observations.append({
            "point": points[idx],
            "source_index": idx,
            "turn": turn_by_index.get(idx),
        })
    return observations


def _angle_diff(a: float, b: float) -> float:
    diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff)


def _path_length(path: List[Tuple[float, float]]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))


def _skeleton_candidates(skeleton: np.ndarray, point: Dict[str, float], cols: int, rows: int, grid: np.ndarray, limit: int = 4) -> List[Tuple[int, int]]:
    gx, gy = _to_grid(point["x"], point["y"], cols, rows)
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return [_nearest_free(grid, cols, rows, gx, gy)]
    distances = ((xs - gx) ** 2 + (ys - gy) ** 2).astype(np.float32)
    order = np.argsort(distances)[:limit]
    return [(int(xs[idx]), int(ys[idx])) for idx in order]


def _route_cache_key(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _apply_loop_closure_to_state_sequence(
    observations: List[Dict[str, Any]],
    chosen_states: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], int]:
    if len(chosen_states) < 3:
        return chosen_states, 0

    closure_radius = 3.0
    min_gap = 3
    merged = list(chosen_states)
    loop_closures = 0

    headings: List[float] = []
    for idx in range(len(observations)):
        if idx == 0:
            a = observations[idx]["point"]
            b = observations[min(1, len(observations) - 1)]["point"]
        elif idx == len(observations) - 1:
            a = observations[max(0, idx - 1)]["point"]
            b = observations[idx]["point"]
        else:
            a = observations[idx - 1]["point"]
            b = observations[idx + 1]["point"]
        headings.append(math.atan2(b["y"] - a["y"], b["x"] - a["x"]))

    for idx in range(1, len(merged)):
        curr = merged[idx]
        best_prior: Optional[Tuple[int, int]] = None
        best_dist = float("inf")
        for prev_idx in range(0, idx - min_gap):
            prev = merged[prev_idx]
            state_dist = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
            if state_dist > closure_radius:
                continue
            if _angle_diff(headings[idx], headings[prev_idx]) > (40.0 * math.pi / 180.0):
                continue
            current_turn = observations[idx].get("turn")
            prev_turn = observations[prev_idx].get("turn")
            if current_turn and prev_turn and current_turn.get("turn_type") != prev_turn.get("turn_type"):
                continue
            if state_dist < best_dist:
                best_prior = prev
                best_dist = state_dist
        if best_prior is not None and best_prior != curr:
            merged[idx] = best_prior
            loop_closures += 1

    for idx in range(1, len(merged) - 1):
        if merged[idx - 1] == merged[idx + 1] and merged[idx] != merged[idx - 1]:
            merged[idx] = merged[idx - 1]

    return merged, loop_closures


def _graph_match_path(
    observations: List[Dict[str, Any]],
    grid: np.ndarray,
    clearance: np.ndarray,
    skeleton: np.ndarray,
    cols: int,
    rows: int,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    if len(observations) < 2 or np.count_nonzero(skeleton) == 0:
        return [], {"loop_closures": 0, "state_count": 0}

    candidate_sets = [
        _skeleton_candidates(skeleton, obs["point"], cols, rows, grid)
        for obs in observations
    ]
    route_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[Tuple[float, float]]] = {}
    dp: List[List[float]] = [[float("inf")] * len(candidates) for candidates in candidate_sets]
    prev_choice: List[List[Optional[int]]] = [[None] * len(candidates) for candidates in candidate_sets]

    for cand_idx, candidate in enumerate(candidate_sets[0]):
        wx, wy = _to_world(*candidate)
        obs_point = observations[0]["point"]
        dp[0][cand_idx] = math.hypot(wx - obs_point["x"], wy - obs_point["y"]) / max(GRID_CELL * 2.0, 1.0)

    for anchor_idx in range(1, len(observations)):
        raw_start = observations[anchor_idx - 1]["point"]
        raw_end = observations[anchor_idx]["point"]
        raw_len = math.hypot(raw_end["x"] - raw_start["x"], raw_end["y"] - raw_start["y"])
        raw_heading = math.atan2(raw_end["y"] - raw_start["y"], raw_end["x"] - raw_start["x"])

        for curr_idx, curr_candidate in enumerate(candidate_sets[anchor_idx]):
            curr_world = _to_world(*curr_candidate)
            endpoint_penalty = math.hypot(curr_world[0] - raw_end["x"], curr_world[1] - raw_end["y"])
            emission_cost = endpoint_penalty / max(raw_len + 10.0, 18.0)

            for prev_idx, prev_candidate in enumerate(candidate_sets[anchor_idx - 1]):
                if math.isinf(dp[anchor_idx - 1][prev_idx]):
                    continue
                cache_key = _route_cache_key(prev_candidate, curr_candidate)
                path = route_cache.get(cache_key)
                if path is None:
                    path = _astar(grid, clearance, cols, rows, prev_candidate, curr_candidate, allowed_mask=skeleton)
                    route_cache[cache_key] = path
                if len(path) < 2:
                    continue

                graph_len = _path_length(path)
                graph_heading = math.atan2(path[-1][1] - path[0][1], path[-1][0] - path[0][0]) if len(path) >= 2 else raw_heading
                length_cost = abs(graph_len - raw_len) / max(raw_len, 1.0)
                heading_cost = _angle_diff(graph_heading, raw_heading) / math.pi
                turn_bonus = 0.0
                if observations[anchor_idx].get("turn") is not None:
                    turn_angle = _to_float(observations[anchor_idx]["turn"].get("angle_degrees"), 0.0)
                    if 35.0 <= turn_angle <= 150.0:
                        turn_bonus = -0.12

                transition_cost = dp[anchor_idx - 1][prev_idx]
                transition_cost += (0.85 * length_cost) + (0.55 * heading_cost)
                transition_cost += emission_cost
                transition_cost += turn_bonus

                if transition_cost < dp[anchor_idx][curr_idx]:
                    dp[anchor_idx][curr_idx] = transition_cost
                    prev_choice[anchor_idx][curr_idx] = prev_idx

    if not dp[-1] or all(math.isinf(score) for score in dp[-1]):
        return [], {"loop_closures": 0, "state_count": 0}

    end_choice = min(range(len(dp[-1])), key=lambda idx: dp[-1][idx])
    node_choices = [end_choice]
    for anchor_idx in range(len(observations) - 1, 0, -1):
        prev_idx = prev_choice[anchor_idx][node_choices[-1]]
        if prev_idx is None:
            return [], {"loop_closures": 0, "state_count": 0}
        node_choices.append(prev_idx)
    node_choices.reverse()

    chosen_states = [candidate_sets[idx][node_choices[idx]] for idx in range(len(node_choices))]
    chosen_states, loop_closures = _apply_loop_closure_to_state_sequence(observations, chosen_states)

    corrected: List[List[float]] = []
    for anchor_idx in range(1, len(observations)):
        prev_candidate = chosen_states[anchor_idx - 1]
        curr_candidate = chosen_states[anchor_idx]
        cache_key = _route_cache_key(prev_candidate, curr_candidate)
        path = route_cache.get(cache_key) or _astar(grid, clearance, cols, rows, prev_candidate, curr_candidate, allowed_mask=skeleton)
        if len(path) < 2:
            return [], {"loop_closures": loop_closures, "state_count": len(chosen_states)}
        limit = len(path) - 1 if anchor_idx < len(observations) - 1 else len(path)
        for px, py in path[:limit]:
            corrected.append([round(float(px), 3), round(float(py), 3), 0.0])

    return corrected, {"loop_closures": loop_closures, "state_count": len(chosen_states)}


def _snap_path_to_clearance(path: List[List[float]], grid: np.ndarray, clearance: np.ndarray, cols: int, rows: int) -> List[List[float]]:
    if not path:
        return path

    snapped: List[List[float]] = []
    for point in path:
        gx, gy = _to_grid(point[0], point[1], cols, rows)
        best = (gx, gy)
        best_score = -float("inf")
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = gx + dx, gy + dy
                if _is_wall(grid, cols, rows, nx, ny):
                    continue
                clearance_value = float(clearance[ny, nx]) if clearance.size else 0.0
                distance_penalty = math.hypot(dx, dy) * 0.35
                score = clearance_value - distance_penalty
                if score > best_score:
                    best_score = score
                    best = (nx, ny)
        wx, wy = _to_world(*best)
        snapped.append([round(float(wx), 3), round(float(wy), 3), point[2] if len(point) > 2 else 0.0])
    return snapped


def _infer_headings_for_path(path: List[List[float]]) -> List[List[float]]:
    if not path:
        return path
    if len(path) == 1:
        return [[path[0][0], path[0][1], path[0][2] if len(path) > 2 else 0.0]]

    out: List[List[float]] = []
    for idx in range(len(path)):
        if idx == len(path) - 1:
            prev = path[idx - 1]
            curr = path[idx]
            heading = math.degrees(math.atan2(curr[1] - prev[1], curr[0] - prev[0]))
        else:
            curr = path[idx]
            nxt = path[idx + 1]
            heading = math.degrees(math.atan2(nxt[1] - curr[1], nxt[0] - curr[0]))
        out.append([round(float(path[idx][0]), 3), round(float(path[idx][1]), 3), round(float(heading), 2)])
    return out


def _smooth_path(path: List[List[float]], grid: np.ndarray, clearance: np.ndarray, cols: int, rows: int) -> List[List[float]]:
    if len(path) < 3:
        return _infer_headings_for_path(path)

    arr = np.array(path, dtype=np.float32)
    smoothed = arr.copy()
    for idx in range(1, len(arr) - 1):
        smoothed[idx, 0] = (arr[idx - 1, 0] + 2.0 * arr[idx, 0] + arr[idx + 1, 0]) / 4.0
        smoothed[idx, 1] = (arr[idx - 1, 1] + 2.0 * arr[idx, 1] + arr[idx + 1, 1]) / 4.0
    snapped = _snap_path_to_clearance(smoothed.tolist(), grid, clearance, cols, rows)

    deduped: List[List[float]] = []
    for point in snapped:
        if not deduped or math.hypot(point[0] - deduped[-1][0], point[1] - deduped[-1][1]) >= 1.0:
            deduped.append(point)
    return _infer_headings_for_path(deduped)


def _correct_path(
    points: List[Dict[str, float]],
    turn_points: List[Dict[str, Any]],
    grid: np.ndarray,
    clearance: np.ndarray,
    cols: int,
    rows: int,
    skeleton: np.ndarray,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    if len(points) < 2:
        return [[round(p["x"], 3), round(p["y"], 3), round(p.get("heading_deg", 0.0), 2)] for p in points], {"loop_closures": 0, "state_count": 0}

    observations = _build_observations(points, turn_points)
    global_graph_corrected, global_info = _graph_match_path(observations, grid, clearance, skeleton, cols, rows)
    if global_graph_corrected:
        return _smooth_path(global_graph_corrected, grid, clearance, cols, rows), global_info

    anchors = _build_anchor_points(points, turn_points)
    graph_corrected, graph_info = _graph_match_path(anchors, grid, clearance, skeleton, cols, rows)
    if graph_corrected:
        return _smooth_path(graph_corrected, grid, clearance, cols, rows), graph_info

    waypoints = [anchor["point"] for anchor in anchors]
    if len(waypoints) < 2:
        waypoints = points[::SUBSAMPLE]
        if not waypoints or waypoints[-1] is not points[-1]:
            waypoints.append(points[-1])

    corrected: List[List[float]] = []
    for idx in range(len(waypoints) - 1):
        start = _to_grid(waypoints[idx]["x"], waypoints[idx]["y"], cols, rows)
        end = _to_grid(waypoints[idx + 1]["x"], waypoints[idx + 1]["y"], cols, rows)
        segment = _astar(grid, clearance, cols, rows, start, end, allowed_mask=skeleton)
        if len(segment) <= 2:
            segment = _astar(grid, clearance, cols, rows, start, end)
        limit = len(segment) - 1 if idx < len(waypoints) - 2 else len(segment)
        for sx, sy in segment[:limit]:
            corrected.append([round(float(sx), 3), round(float(sy), 3), 0.0])
    return _smooth_path(corrected, grid, clearance, cols, rows), {"loop_closures": 0, "state_count": len(anchors)}


def _build_transformed_points(
    points: List[Dict[str, float]],
    plan_width: int,
    plan_height: int,
    reference_point: Optional[Dict[str, Any]],
    direction_point: Optional[Dict[str, Any]],
    scale: float,
) -> List[Dict[str, float]]:
    if not points:
        return []

    start_x = points[0]["x"]
    start_y = points[0]["y"]
    if reference_point:
        ref_x = (_to_float(reference_point.get("x")) / 100.0) * plan_width
        ref_y = (_to_float(reference_point.get("y")) / 100.0) * plan_height
    else:
        ref_x = plan_width / 2.0
        ref_y = plan_height / 2.0

    rotation = 0.0
    if reference_point and direction_point and len(points) >= 2:
        dir_x = (_to_float(direction_point.get("x")) / 100.0) * plan_width
        dir_y = (_to_float(direction_point.get("y")) / 100.0) * plan_height
        direction_angle = math.atan2(dir_y - ref_y, dir_x - ref_x)
        seg_len = max(2, min(20, max(2, int(len(points) * 0.1))))
        p0 = points[0]
        pN = points[seg_len - 1]
        traj_angle = math.atan2((pN["y"] - p0["y"]) * scale, (pN["x"] - p0["x"]) * scale)
        rotation = direction_angle - traj_angle

    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    transformed: List[Dict[str, float]] = []
    for point in points:
        px = (point["x"] - start_x) * scale + ref_x
        py = (point["y"] - start_y) * scale + ref_y
        dx = px - ref_x
        dy = py - ref_y
        transformed.append({
            "x": dx * cos_r - dy * sin_r + ref_x,
            "y": dx * sin_r + dy * cos_r + ref_y,
            "heading_deg": point.get("heading_deg", 0.0),
        })
    return transformed


def _point_walk_score(point: Dict[str, float], grid: np.ndarray, cols: int, rows: int, width: int, height: int) -> float:
    x = point["x"]
    y = point["y"]
    if x < 0 or y < 0 or x >= width or y >= height:
        return -1.0
    gx, gy = _to_grid(x, y, cols, rows)
    return 1.0 if not _is_wall(grid, cols, rows, gx, gy) else -0.5


def _segment_walk_ratio(points: List[Dict[str, float]], grid: np.ndarray, cols: int, rows: int, width: int, height: int) -> float:
    if len(points) < 2:
        return 0.0
    free = 0
    total = 0
    for idx in range(len(points) - 1):
        a = points[idx]
        b = points[idx + 1]
        samples = max(2, int(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / GRID_CELL))
        for step in range(samples + 1):
            t = step / max(samples, 1)
            sample = {
                "x": a["x"] * (1.0 - t) + b["x"] * t,
                "y": a["y"] * (1.0 - t) + b["y"] * t,
            }
            total += 1
            if _point_walk_score(sample, grid, cols, rows, width, height) > 0:
                free += 1
    return free / max(total, 1)


def _select_plan_scale(
    points: List[Dict[str, float]],
    grid: np.ndarray,
    cols: int,
    rows: int,
    width: int,
    height: int,
    reference_point: Optional[Dict[str, Any]],
    direction_point: Optional[Dict[str, Any]],
) -> float:
    if len(points) < 2:
        return 1.0

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    raw_span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    target_span = min(width, height) * 0.45
    base_scale = max(0.05, min(20.0, target_span / raw_span))
    candidates = [base_scale * factor for factor in (0.3, 0.45, 0.65, 0.85, 1.0, 1.2, 1.5, 1.8, 2.2, 2.8)]

    best_scale = base_scale
    best_score = -float("inf")
    for candidate in candidates:
        transformed = _build_transformed_points(points, width, height, reference_point, direction_point, candidate)
        point_scores = [_point_walk_score(point, grid, cols, rows, width, height) for point in transformed]
        inside_ratio = sum(1 for score in point_scores if score > 0) / max(len(point_scores), 1)
        outside_ratio = sum(1 for score in point_scores if score < 0) / max(len(point_scores), 1)
        segment_ratio = _segment_walk_ratio(transformed, grid, cols, rows, width, height)
        score = (inside_ratio * 0.65) + (segment_ratio * 0.45) - (outside_ratio * 0.35)
        if score > best_score:
            best_score = score
            best_scale = candidate
    return float(best_scale)


def _rdp(pts: np.ndarray, epsilon: float) -> np.ndarray:
    n = len(pts)
    if n < 3:
        return pts
    start = pts[0]
    end = pts[-1]
    if n == 2:
        return pts
    seg = end - start
    seg_len = np.linalg.norm(seg) + 1e-12
    seg_unit = seg / seg_len
    d = pts - start
    proj = np.dot(d, seg_unit)
    proj = np.clip(proj, 0, seg_len)
    closest = start + proj[:, np.newaxis] * seg_unit
    dists = np.linalg.norm(pts - closest, axis=1)
    idx = int(np.argmax(dists))
    dmax = float(dists[idx])
    if dmax < epsilon:
        return np.array([start, end], dtype=np.float32)
    left = _rdp(pts[: idx + 1], epsilon)
    right = _rdp(pts[idx:], epsilon)
    return np.vstack([left[:-1], right]) if len(right) > 0 else left


def _merge_collinear(pts: np.ndarray, angle_tol_deg: float = 12.0) -> np.ndarray:
    if len(pts) < 3:
        return pts
    angle_tol = math.radians(angle_tol_deg)
    out = [pts[0]]
    i = 1
    while i < len(pts):
        a = np.array(out[-1], dtype=np.float64)
        b = np.array(pts[i], dtype=np.float64)
        j = i + 1
        while j < len(pts):
            c = np.array(pts[j], dtype=np.float64)
            v1 = b - a
            v2 = c - b
            n1 = np.linalg.norm(v1) + 1e-12
            n2 = np.linalg.norm(v2) + 1e-12
            v1 = v1 / n1
            v2 = v2 / n2
            cos_a = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = math.acos(cos_a)
            if angle > angle_tol:
                break
            b = c
            j += 1
        out.append(pts[j - 1] if j > i else pts[i])
        i = j
    return np.array(out, dtype=np.float32)


def _snap_turn_angle(angle: float) -> float:
    candidates = [45.0, 90.0, 135.0, 180.0]
    best = min(candidates, key=lambda x: abs(x - angle))
    return best if abs(best - angle) <= 12.0 else angle


def _heading_at_path_index(path: List[List[float]], idx: int) -> float:
    idx = max(0, min(len(path) - 1, idx))
    if len(path[idx]) > 2:
        return _to_float(path[idx][2], 0.0)
    if idx == len(path) - 1 and idx > 0:
        prev = path[idx - 1]
        curr = path[idx]
        return math.degrees(math.atan2(curr[1] - prev[1], curr[0] - prev[0]))
    if idx < len(path) - 1:
        curr = path[idx]
        nxt = path[idx + 1]
        return math.degrees(math.atan2(nxt[1] - curr[1], nxt[0] - curr[0]))
    return 0.0


def _detect_turns_on_map_path(path: List[List[float]]) -> List[Dict[str, Any]]:
    if len(path) < 5:
        return []

    pts = np.array([[float(p[0]), float(p[1])] for p in path], dtype=np.float32)
    simplified = _rdp(pts, epsilon=6.0)
    if len(simplified) < 3:
        return []

    simplified = _merge_collinear(simplified, angle_tol_deg=12.0)
    if len(simplified) < 3:
        return []

    result: List[Dict[str, Any]] = []
    for i in range(1, len(simplified) - 1):
        a = simplified[i - 1]
        b = simplified[i]
        c = simplified[i + 1]

        h1 = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        h2 = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]))
        delta = ((h2 - h1 + 180.0) % 360.0) - 180.0

        if abs(delta) < 20.0:
            continue

        snapped = _snap_turn_angle(abs(delta))
        j_closest = int(np.argmin([math.hypot(path[j][0] - float(b[0]), path[j][1] - float(b[1])) for j in range(len(path))]))
        heading_deg = _heading_at_path_index(path, j_closest)
        result.append({
            "frame_index": 0,
            "trajectory_index": j_closest,
            "position": [float(b[0]), float(b[1]), round(float(heading_deg), 2)],
            "angle_degrees": round(snapped, 1),
            "raw_angle_degrees": round(delta, 1),
            "turn_type": "left" if delta > 0 else "right",
            "confidence": 1.0,
            "source": "map_path",
        })
    return result


def _transform_turn_points(
    turn_points: List[Dict[str, Any]],
    source_points: List[Dict[str, float]],
    mapped_points: List[List[float]],
) -> List[Dict[str, Any]]:
    if not turn_points:
        return []
    last_idx = max(len(source_points) - 1, 1)
    mapped_last_idx = max(len(mapped_points) - 1, 0)
    result = []
    for turn in turn_points:
        idx = int(round((_to_float(turn.get("trajectory_index")) / last_idx) * mapped_last_idx)) if mapped_last_idx else 0
        idx = max(0, min(mapped_last_idx, idx))
        mapped_pos = mapped_points[idx] if mapped_points else turn.get("position", [0, 0, 0])
        result.append({
            **turn,
            "trajectory_index": idx,
            "position": mapped_pos,
        })
    return result


def apply_map_postprocessing(result: Dict[str, Any], map_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not result or not map_context:
        return result

    floor_plan_data = map_context.get("floor_plan_data")
    drawn_plan = map_context.get("drawn_plan")
    reference_point = map_context.get("reference_point")
    direction_point = map_context.get("direction_point")

    plan_image = _decode_data_url_image(floor_plan_data) if floor_plan_data else None
    if plan_image is None and drawn_plan:
        plan_image = _render_drawn_plan(drawn_plan)
    if plan_image is None:
        return result

    points = _normalize_trajectory(result.get("trajectory"))
    if len(points) < 2:
        return result

    grid, clearance, skeleton, cols, rows, width, height = _build_occupancy_grid(plan_image)
    auto_scale = _select_plan_scale(points, grid, cols, rows, width, height, reference_point, direction_point)
    transformed = _build_transformed_points(points, width, height, reference_point, direction_point, auto_scale)
    corrected, matching_info = _correct_path(transformed, result.get("turn_points") or [], grid, clearance, cols, rows, skeleton)
    corrected_turns = _detect_turns_on_map_path(corrected)

    updated = dict(result)
    updated["map_trajectory"] = corrected
    updated["map_turn_points"] = corrected_turns
    updated["final_turn_points"] = updated["map_turn_points"] or updated.get("turn_points") or []
    updated["map_metadata"] = {
        "plan_width": width,
        "plan_height": height,
        "grid_cell": GRID_CELL,
        "auto_scale": round(auto_scale, 4),
        "skeleton_points": int(np.count_nonzero(skeleton)),
        "turn_anchor_count": len(result.get("turn_points") or []),
        "loop_closures": int(matching_info.get("loop_closures", 0)),
        "state_count": int(matching_info.get("state_count", 0)),
        "source": "drawn_plan" if drawn_plan and plan_image is not None and floor_plan_data is None else "floor_plan",
        "trajectory_contract": "heading_deg",
    }
    processing_stats = dict(updated.get("processing_stats") or {})
    processing_stats["map_matching_applied"] = True
    processing_stats["map_trajectory_points"] = len(corrected)
    processing_stats["map_auto_scale"] = round(auto_scale, 4)
    processing_stats["map_loop_closures"] = int(matching_info.get("loop_closures", 0))
    updated["processing_stats"] = processing_stats
    return updated
