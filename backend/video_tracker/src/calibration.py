# src/calibration.py
import cv2
import numpy as np
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CameraCalibrator:
    """Калибровка камеры и устранение дисторсии"""
    
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.is_calibrated = camera_matrix is not None
        
    @staticmethod
    def calibrate_from_chessboard(images_paths, board_size=(9, 6)):
        """
        Калибровка камеры по изображениям шахматной доски.
        
        Args:
            images_paths: Список путей к изображениям
            board_size: Размер шахматной доски (внутренние углы)
            
        Returns:
            tuple: (camera_matrix, dist_coeffs, image_size)
        """
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        
        objpoints = []
        imgpoints = []
        
        for img_path in images_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)
            if ret:
                objpoints.append(objp)
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                imgpoints.append(corners_refined)
        
        if len(objpoints) < 3:
            logger.warning("Недостаточно изображений для калибровки")
            return None, None, None
            
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if ret:
            logger.info(f"Калибровка успешна. Средняя ошибка: {ret:.4f}")
            return mtx, dist, gray.shape[::-1]
        
        return None, None, None
    
    @staticmethod
    def estimate_distortion_auto(frame, roi_percent=0.1):
        """
        Автоматическая оценка дисторсии по краям кадра.
        Полезно, когда нет калибровочных данных.
        
        Args:
            frame: Входной кадр
            roi_percent: Процент края для анализа линий
            
        Returns:
            tuple: (camera_matrix, dist_coeffs)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Упрощённая модель камеры (без дисторсии)
        focal_length = max(h, w)
        cx, cy = w / 2, h / 2
        
        camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Пытаемся найти дисторсию по линиям на краях
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100)
        
        dist_coeffs = np.zeros(5)  # Начальная оценка - без дисторсии
        
        if lines is not None:
            # Анализируем линии - ищем искривления
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                
                if abs(slope) < 0.3:  # Горизонтальные линии
                    if x1 < w * 0.3 or x2 < w * 0.3:
                        left_lines.append((y1 + y2) / 2)
                    if x1 > w * 0.7 or x2 > w * 0.7:
                        right_lines.append((y1 + y2) / 2)
            
            # Оцениваем бочкообразную дисторсию
            if left_lines and right_lines:
                left_avg = np.mean(left_lines)
                right_avg = np.mean(right_lines)
                
                # Если края "выгибаются" наружу - бочкообразная дисторсия
                center_y = h / 2
                k1 = 0.0001  # Небольшая бочкообразная
                dist_coeffs = np.array([k1, 0, 0, 0, 0])
        
        return camera_matrix, dist_coeffs
    
    def undistort(self, frame):
        """Устранение дисторсии кадра"""
        if not self.is_calibrated:
            return frame
        
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        undistorted = cv2.undistort(
            frame, self.camera_matrix, self.dist_coeffs, 
            None, new_camera_matrix
        )
        
        return undistorted
    
    def save_calibration(self, filepath):
        """Сохранение калибровочных данных"""
        data = {
            'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            'dist_coeffs': self.dist_coeffs.tolist() if self.dist_coeffs is not None else None
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Калибровка сохранена: {filepath}")
    
    @staticmethod
    def load_calibration(filepath):
        """Загрузка калибровочных данных"""
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        camera_matrix = np.array(data['camera_matrix']) if data['camera_matrix'] else None
        dist_coeffs = np.array(data['dist_coeffs']) if data['dist_coeffs'] else None
        
        calibrator = CameraCalibrator(camera_matrix, dist_coeffs)
        logger.info(f"Калибровка загружена: {filepath}")
        return calibrator


class ScaleEstimator:
    """
    Автоматическая оценка масштаба траектории.
    Production: предпочтительно estimate_from_known_distance() по известному отрезку.
    Fallback: estimate_from_corridor() / estimate_from_corridor_local() по ширине коридора.
    """

    @staticmethod
    def estimate_from_known_distance(trajectory_points, real_distance_meters):
        """
        Оценка масштаба по известному расстоянию (предпочтительный production mode).
        
        Args:
            trajectory_points: Список точек траектории [[x, y, z], ...] или [dict с x,y]
            real_distance_meters: Реальное расстояние в метрах
            
        Returns:
            float: Коэффициент масштабирования
        """
        if len(trajectory_points) < 2:
            return 1.0
        
        # Вычисляем расстояние по траектории
        pixel_distance = sum(
            np.linalg.norm(
                np.array(trajectory_points[i+1][:2]) - np.array(trajectory_points[i][:2])
            )
            for i in range(len(trajectory_points) - 1)
        )
        
        if pixel_distance > 0:
            return real_distance_meters / pixel_distance
        
        return 1.0
    
    @staticmethod
    def estimate_from_person_height(avg_pixel_height, person_height_meters=1.7):
        """
        Оценка масштаба по высоте человека в кадре.
        
        Args:
            avg_pixel_height: Средняя высота человека в пикселях
            person_height_meters: Реальный рост человека
            
        Returns:
            float: Коэффициент масштабирования
        """
        if avg_pixel_height > 0:
            return person_height_meters / avg_pixel_height
        return 1.0
    
    @staticmethod
    def _to_xy_array(trajectory):
        """Привести траекторию к np.array shape (N, 2)."""
        pts = []
        for p in trajectory:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append([float(p[0]), float(p[1])])
            elif isinstance(p, dict):
                pts.append([float(p.get("x", p.get(0, 0))), float(p.get("y", p.get(1, 0)))])
            else:
                pts.append([0.0, 0.0])
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def estimate_from_corridor_local(trajectory, corridor_width_meters):
        """
        Оценка масштаба по ширине коридора ортогонально локальному heading.
        Ширина = размах проекций точек на normal к касательной в середине сегмента.
        Подходит для прямых участков; на П-маршруте/диагонали max(x)-min(x) даёт ложный scale.
        """
        if len(trajectory) < 20:
            return 1.0

        pts = ScaleEstimator._to_xy_array(trajectory)
        center = len(pts) // 2
        a = pts[max(0, center - 8)]
        b = pts[min(len(pts) - 1, center + 8)]
        tangent = b - a
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            return 1.0

        tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

        projections = pts @ normal
        width_units = float(np.max(projections) - np.min(projections))
        if width_units <= 1e-6:
            return 1.0

        return corridor_width_meters / width_units

    @staticmethod
    def estimate_from_corridor(trajectory, corridor_width_meters):
        """
        Оценка масштаба по ширине коридора (fallback).
        Использует ширину ортогонально локальному heading, а не размах по global X.
        Для production предпочтителен estimate_from_known_distance() по известному отрезку.
        
        Args:
            trajectory: Список точек траектории
            corridor_width_meters: Известная ширина коридора в метрах
            
        Returns:
            float: Коэффициент масштабирования
        """
        return ScaleEstimator.estimate_from_corridor_local(trajectory, corridor_width_meters)
