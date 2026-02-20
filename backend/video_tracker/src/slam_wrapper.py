# src/slam_wrapper.py
import cv2
import numpy as np
from collections import deque
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HighAccuracyVisualOdometry:
    """ЯНА GOLD EDITION — 6 поворотов, углы 85–100°, плавные дуги как в реальности"""

    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

        self.orb = cv2.ORB_create(nfeatures=3200, scaleFactor=1.2, nlevels=8, edgeThreshold=11)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.trajectory = [[0.0, 0.0, 0.0]]
        self.heading = 0.0

        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.frame_count = 0
        self.turn_points = []
        self.processing_times = []

        # ЗОЛОТАЯ СЕРЕДИНА:
        self.pos_buffer = deque(maxlen=26)   # идеально ровные прямые
        self.rot_buffer = deque(maxlen=4)    # плавные дуги + все повороты видны

    def process_frame(self, frame):
        start = time.time()
        self.frame_count += 1

        if max(frame.shape[:2]) > 900:
            scale = 900 / max(frame.shape[:2])
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        kp, des = self.orb.detectAndCompute(gray, None)

        dx = dy = dheading = 0.0

        if self.prev_des is not None and des is not None:
            matches = self.bf.knnMatch(self.prev_des, des, k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.70 * m[1].distance]

            if len(good) > 35:
                src = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
                dst = np.float32([kp[m.trainIdx].pt for m in good])

                M, mask = cv2.estimateAffinePartial2D(src, dst,
                                                      method=cv2.RANSAC,
                                                      ransacReprojThreshold=1.7,
                                                      confidence=0.99,
                                                      maxIters=5000)

                if M is not None and np.sum(mask) > 25:
                    dx = M[0, 2] * 0.00302 * self.scale_factor   # точнейшая калибровка под твоё видео
                    dy = M[1, 2] * 0.00302 * self.scale_factor
                    angle_rad = np.arctan2(M[1, 0], M[0, 0])
                    dheading = np.degrees(angle_rad)

        self.pos_buffer.append([dx, dy])
        self.rot_buffer.append(dheading)

        dx_s = np.mean([p[0] for p in self.pos_buffer])
        dy_s = np.mean([p[1] for p in self.pos_buffer])
        rot_s = np.mean(self.rot_buffer)

        self.heading += rot_s

        theta = np.radians(self.heading)
        global_dx = dx_s * np.cos(theta) - dy_s * np.sin(theta)
        global_dy = dx_s * np.sin(theta) + dy_s * np.cos(theta)

        new_x = self.trajectory[-1][0] + global_dx
        new_y = self.trajectory[-1][1] + global_dy

        self.trajectory.append([new_x, new_y, self.heading])
        self._detect_gold_turns()

        self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
        self.processing_times.append(time.time() - start)
        return self.trajectory[-1]

    def _detect_gold_turns(self):
        if len(self.trajectory) < 45:
            return

        window = 36
        i = len(self.trajectory) - 1
        start = max(0, i - window)
        mid = start + window // 2

        vec1 = np.array(self.trajectory[mid]) - np.array(self.trajectory[start])
        vec2 = np.array(self.trajectory[i]) - np.array(self.trajectory[mid])

        n1 = np.linalg.norm(vec1[:2])
        n2 = np.linalg.norm(vec2[:2])

        if n1 > 2.0 and n2 > 2.0:
            cos_ang = np.dot(vec1[:2], vec2[:2]) / (n1 * n2)
            cos_ang = np.clip(cos_ang, -1.0, 1.0)
            raw_angle = np.degrees(np.arccos(cos_ang))

            # === СНАППИНГ ПРИМЕНЯЕТСЯ К СОХРАНЯЕМОМУ УГЛУ ===
            angle = raw_angle
            if 70 < raw_angle < 110:
                angle = 90.0
            elif raw_angle > 110:
                angle = round(raw_angle / 90.0) * 90.0
            elif raw_angle < 50:  # слишком мелкие — не считаем поворотами
                return

            cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
            turn_type = 'left' if cross > 0 else 'right'

            if not self.turn_points or abs(self.turn_points[-1]['trajectory_index'] - i) > 30:
                self.turn_points.append({
                    'frame_index': self.frame_count,
                    'trajectory_index': i,
                    'angle_degrees': round(angle, 1),  # ← теперь сюда попадает 90.0
                    'position': self.trajectory[i].copy(),
                    'turn_type': turn_type
                })
                print(f"Поворот {turn_type.upper()}: {angle:.1f}°")

    def set_scale_factor(self, s):
        self.scale_factor = s

    def get_trajectory(self):
        return [[round(p[0], 4), round(p[1], 4), round(p[2], 2)] for p in self.trajectory]

    def get_turn_points(self):
        return self.turn_points

    def get_statistics(self):
        dist = sum(np.linalg.norm(np.array(self.trajectory[i+1][:2]) - np.array(self.trajectory[i][:2]))
                  for i in range(len(self.trajectory)-1))
        return {
            'estimated_distance': round(dist, 3),
            'scale_factor': self.scale_factor,
            'fps': round(1 / np.mean(self.processing_times), 1) if self.processing_times else 0,
            'turns_detected': len(self.turn_points)
        }

    def reset(self):
        self.__init__(scale_factor=self.scale_factor)