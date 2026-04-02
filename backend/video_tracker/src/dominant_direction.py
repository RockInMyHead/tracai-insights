# src/dominant_direction.py
# Dominant corridor axis from Hough lines (Manhattan-style) for straight-line lock.
# Structured indoor: long lines → vertical/horizontal in aligned frame → corridor axis.

import logging
import math
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Угол "вертикаль" в кадре (направление коридора вперёд), градусы
CORRIDOR_AXIS_TARGET_DEG = 90.0
# Допуск для Manhattan (горизонталь/вертикаль), градусы
MANHATTAN_TOLERANCE_DEG = 18.0
# Минимальная длина линии (доля от min(w,h))
MIN_LINE_LENGTH_RATIO = 0.08
# Минимум линий для уверенной оценки
MIN_LINES_FOR_CONFIDENCE = 8


def _angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Угол отрезка от горизонтали, [0, 180)."""
    dx = x2 - x1
    dy = y2 - y1
    return float(np.degrees(np.mod(np.arctan2(dy, dx) + np.pi, np.pi)))


def _wrap_deg(a: float) -> float:
    """Привести угол к [-180, 180]."""
    a = float(a) % 360.0
    if a > 180.0:
        a -= 360.0
    return a


def get_dominant_corridor_heading(
    gray: np.ndarray,
    current_heading_deg: float,
) -> Tuple[float, float]:
    """
    Оценка направления коридора по длинным линиям (Hough), Manhattan-style.
    Возвращает (corridor_heading_deg, confidence in [0,1]).
    """
    if gray is None or gray.size == 0:
        return current_heading_deg, 0.0

    h, w = gray.shape[:2]
    min_side = min(h, w)
    min_len = max(15, int(min_side * MIN_LINE_LENGTH_RATIO))

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=min_len,
        maxLineGap=12,
    )

    if lines is None or len(lines) == 0:
        return current_heading_deg, 0.0

    angles = []
    lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_len:
            continue
        angle = _angle_deg(x1, y1, x2, y2)
        # Близко к горизонтали (0°) или вертикали (90°)
        near_0 = abs(angle) < MANHATTAN_TOLERANCE_DEG or abs(angle - 180.0) < MANHATTAN_TOLERANCE_DEG
        near_90 = abs(angle - 90.0) < MANHATTAN_TOLERANCE_DEG
        if near_0:
            angles.append(0.0)
            lengths.append(length)
        elif near_90:
            angles.append(90.0)
            lengths.append(length)

    if len(angles) < 4:
        return current_heading_deg, 0.0

    # Доминантная ось: вертикаль (90°) типична для коридора вперёд
    angles_arr = np.array(angles)
    lengths_arr = np.array(lengths)
    vertical_mask = np.abs(angles_arr - 90.0) < MANHATTAN_TOLERANCE_DEG
    horizontal_mask = (np.abs(angles_arr) < MANHATTAN_TOLERANCE_DEG) | (np.abs(angles_arr - 180.0) < MANHATTAN_TOLERANCE_DEG)

    # Предпочитаем вертикальные линии как ось коридора (стены)
    if np.any(vertical_mask):
        vert_angles = angles_arr[vertical_mask]
        vert_lengths = lengths_arr[vertical_mask]
        # Взвешенное по длине среднее
        dominant_image_angle = float(np.average(vert_angles, weights=vert_lengths))
        n_used = int(np.sum(vertical_mask))
    else:
        # Иначе горизонтальные
        hor_angles = angles_arr[horizontal_mask]
        hor_lengths = lengths_arr[horizontal_mask]
        dominant_image_angle = float(np.average(hor_angles, weights=hor_lengths))
        n_used = int(np.sum(horizontal_mask))

    # Ось коридора в мире: когда камера смотрит вдоль коридора, в кадре доминирует 90°
    # Отклонение dominant_image_angle от 90° даёт поправку к heading
    correction_deg = CORRIDOR_AXIS_TARGET_DEG - dominant_image_angle
    corridor_heading_deg = current_heading_deg + correction_deg
    corridor_heading_deg = _wrap_deg(corridor_heading_deg)

    # Уверенность: по количеству и длине линий
    n_lines = len(angles)
    conf = min(1.0, n_lines / 24.0) * (0.5 + 0.5 * min(1.0, np.sum(lengths_arr) / (min_side * 4)))
    if n_lines < MIN_LINES_FOR_CONFIDENCE:
        conf *= 0.6

    return corridor_heading_deg, float(np.clip(conf, 0.0, 1.0))
