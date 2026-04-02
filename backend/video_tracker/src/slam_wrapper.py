import cv2
import logging
import numpy as np
from collections import deque
import time
import sys
import os
from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

from video_tracker.src.calibration import CameraCalibrator, ScaleEstimator
from video_tracker.src.ml_roi import MLDetector, SimpleTracker
from video_tracker.src.dominant_direction import get_dominant_corridor_heading


def _wrap_deg(a: float) -> float:
    a = float(a) % 360.0
    if a > 180.0:
        a -= 360.0
    return a


class KalmanFilter2D:
    """Simple CV Kalman for XY only."""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.state_dim = 4
        self.state = np.zeros(self.state_dim, dtype=np.float64)
        self.P = np.eye(self.state_dim, dtype=np.float64) * 1000.0
        self.F = np.eye(self.state_dim, dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.Q_base = np.eye(self.state_dim, dtype=np.float64) * process_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        self.initialized = False

    def predict(self, dt: float = 1.0):
        dt = max(1e-3, float(dt))
        self.F[:] = np.eye(self.state_dim, dtype=np.float64)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        q_scale = max(1.0, dt)
        Q = self.Q_base * q_scale

        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + Q
        return self.state[:2].copy()

    def update(self, measurement):
        z = np.array(measurement, dtype=np.float64)
        if not self.initialized:
            self.state[0] = z[0]
            self.state[1] = z[1]
            self.state[2] = 0.0
            self.state[3] = 0.0
            self.initialized = True
            return self.state[:2].copy()

        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
        return self.state[:2].copy()

    def get_velocity(self):
        return np.array([self.state[2], self.state[3]], dtype=np.float64)

    def anchor(self, position, zero_velocity: bool = True):
        z = np.array(position, dtype=np.float64)
        self.state[0] = z[0]
        self.state[1] = z[1]
        if zero_velocity:
            self.state[2] = 0.0
            self.state[3] = 0.0
        self.P = np.eye(self.state_dim, dtype=np.float64)
        self.initialized = True
        return self.state[:2].copy()

    def reset(self):
        self.state.fill(0.0)
        self.P = np.eye(self.state_dim, dtype=np.float64) * 1000.0
        self.initialized = False


class EnhancedVisualOdometry:
    """Enhanced monocular VO with explicit rotation/translation fusion."""

    def __init__(
        self,
        scale_factor=1.0,
        use_homography=False,
        use_kalman=False,
        use_akaze=False,
        calibrator=None,
        use_optical_flow=False,
        corridor_width_m: Optional[float] = None,
        detect_interval: int = 3,
        invert_turn_direction: bool = False,
        turn_vote_threshold: int = 3,
        use_ml_roi: bool = True,
        ml_model_path: str = "models/detector.onnx",
        tracker_max_age: int = 8,
    ):
        self._init_config = dict(
            scale_factor=scale_factor,
            use_homography=use_homography,
            use_kalman=use_kalman,
            use_akaze=use_akaze,
            calibrator=calibrator,
            use_optical_flow=use_optical_flow,
            corridor_width_m=corridor_width_m,
            detect_interval=detect_interval,
            invert_turn_direction=invert_turn_direction,
            turn_vote_threshold=turn_vote_threshold,
            use_ml_roi=use_ml_roi,
            ml_model_path=ml_model_path,
            tracker_max_age=tracker_max_age,
        )

        self.scale_factor = float(scale_factor)
        self.raw_scale_candidate = None
        self._last_vp_confidence = 0.0
        self.use_homography = bool(use_homography)
        self.use_kalman = bool(use_kalman)
        self.use_akaze = bool(use_akaze)
        self.calibrator = calibrator
        self.use_optical_flow = bool(use_optical_flow)
        self.corridor_width_m = corridor_width_m
        self.detect_interval = max(1, int(detect_interval))
        self.yaw_sign = -1.0 if invert_turn_direction else 1.0
        self.invert_turn_direction = invert_turn_direction
        self.turn_vote_threshold = max(1, min(5, int(turn_vote_threshold)))

        # Quality / gating parameters
        self.min_good_matches = 45
        self.min_inlier_ratio = 0.32
        self.min_inliers_abs = 22
        self.max_heading_step_deg = 35.0
        self.max_translation_px = 80.0
        self.deadzone_speed = 0.05
        self.motion_scale = 0.00302
        self.scene_texture_threshold = 9
        self.min_sparse_tracks = 80
        self.sparse_reseed_threshold = 120
        self.traj_smooth_alpha = 0.40
        self.turn_smooth_alpha = 0.70
        self.rotation_decay_alpha = 0.85
        self.rotation_flip_alpha = 0.92
        self.rotation_decay_threshold = 0.35
        self.rotation_sign_flip_threshold = 1.0
        self.motion_vec_buffer = deque(maxlen=20)
        self.straight_mode_min_speed = 0.04
        self.straight_mode_consistency = 0.92
        self.lateral_damping = 0.08
        self.stationary_translation_threshold = 0.018
        self.stationary_rotation_threshold = 0.45
        self.stationary_flow_threshold = 0.012
        self.stationary_hysteresis_frames = 3
        self.turn_in_place_translation_threshold = 0.035
        self.turn_in_place_rotation_threshold = 2.0
        self.turn_in_place_min_conf = 0.14
        self.motion_resume_confidence = 0.18
        self.max_flow_assist = 0.05
        self.max_kalman_step = 0.35
        self.max_kalman_residual = 0.25
        self.tracking_mode = "auto"
        self.person_mode_votes = 0
        self.ego_mode_votes = 0
        self.active_tracking_mode = "ego"
        self.stationary_state = False
        self.stationary_frames = 0
        self.turn_in_place_frames = 0
        self.no_pose_frames = 0

        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=5,
            poly_n=5,
            poly_sigma=1.5,
            flags=0,
        )
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.feature_detector = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,
            descriptor_channels=3,
            threshold=0.0001,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.kalman = KalmanFilter2D(process_noise=0.01, measurement_noise=0.5)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        self.trajectory = [[0.0, 0.0, 0.0]]
        self.heading = 0.0

        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_masked_gray = None
        self.prev_track_pts = None
        self.prev_tracking_mask = None
        self.prev_shape = None
        self.prev_frame_index = None
        self.prev_timestamp_sec = None
        self.last_dt_sec = 1.0
        self.frame_count = 0

        self.raw_turn_points = []
        self.trajectory_turn_points = []
        self.processing_times = []
        self.frame_debug = {}

        self.pos_buffer = deque(maxlen=30)
        self.rot_buffer = deque(maxlen=8)

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)

        self.total_matches = 0
        self.total_ransac_failures = 0
        self.total_gated_failures = 0
        self.total_deadzone_skips = 0
        self.total_sparse_tracks = 0
        self.total_sparse_successes = 0
        self.total_feature_successes = 0
        self.feature_insufficient_matches = 0
        self.feature_model_failures = 0
        self.global_pose_failures = 0
        self.flow_assist_frames = 0
        self.flow_rejection_count = 0
        self.stationary_lock_frames = 0
        self.kalman_lock_frames = 0
        self.sign_conflict_frames = 0
        self.turn_conflict_count = 0
        self._pose_essential_frames = 0
        self._pose_affine_frames = 0
        self._pose_sparse_only_frames = 0
        self._pose_feature_only_frames = 0
        self._feature_used_essential = False
        self._sparse_used_essential = False

        self.camera_matrix = None
        self.dist_coeffs = None
        if calibrator and getattr(calibrator, "is_calibrated", False):
            self.camera_matrix = calibrator.camera_matrix
            self.dist_coeffs = calibrator.dist_coeffs

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.last_bbox = None
        self.last_bbox_ts = 0
        self.turn_sign_buffer = deque(maxlen=5)
        self.delta_yaw_history = deque(maxlen=240)
        self.turn_step_history = deque(maxlen=240)
        self.active_turn = None
        self.turn_events_raw = []

        self.use_ml_roi = use_ml_roi
        self.ml_detector = None
        self.ml_tracker = None
        if self.use_ml_roi:
            try:
                self.ml_detector = MLDetector(ml_model_path)
                if self.ml_detector.session is not None:
                    self.ml_tracker = SimpleTracker(max_age=tracker_max_age, iou_thres=0.4)
                    logger.info("ML ROI enabled with model %s", ml_model_path)
                else:
                    self.use_ml_roi = False
            except Exception as e:
                logger.warning("Failed to init ML ROI: %s", e)
                self.use_ml_roi = False

    def _clamp_bbox(self, bbox, shape):
        h, w = shape[:2]
        x, y, bw, bh = bbox
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        bw = max(1, min(int(bw), w - x))
        bh = max(1, min(int(bh), h - y))
        return x, y, bw, bh

    def _bbox_iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return (inter / union) if union > 0 else 0.0

    def _bbox_center_distance(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        acx, acy = ax + aw * 0.5, ay + ah * 0.5
        bcx, bcy = bx + bw * 0.5, by + bh * 0.5
        return float(np.hypot(acx - bcx, acy - bcy))

    def _build_person_mask(self, gray):
        bbox_mask = None
        if self.use_ml_roi and self.ml_detector and ((self.frame_count - self.last_bbox_ts) >= self.detect_interval or self.last_bbox is None):
            dets = self.ml_detector.detect(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            if dets:
                _ = self.ml_tracker.update(dets) if self.ml_tracker else []
                best = self.ml_tracker.get_best_track() if self.ml_tracker else None
                if best:
                    x1, y1, x2, y2 = best["bbox"]
                    w, h = x2 - x1, y2 - y1
                    self.last_bbox = self._clamp_bbox((x1, y1, w, h), gray.shape)
                    self.last_bbox_ts = self.frame_count

        if (not self.use_ml_roi) and ((self.frame_count - self.last_bbox_ts) >= self.detect_interval or self.last_bbox is None):
            small_w = 480
            img_for_det = gray
            scale_det = 1.0
            if gray.shape[1] > small_w:
                scale_det = small_w / gray.shape[1]
                img_for_det = cv2.resize(gray, (small_w, int(gray.shape[0] * scale_det)))
            rects, _ = self.hog.detectMultiScale(
                img_for_det,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
            if len(rects) > 0:
                candidates = []
                for r in rects:
                    x, y, w, h = r
                    if scale_det != 1.0:
                        x = int(x / scale_det)
                        y = int(y / scale_det)
                        w = int(w / scale_det)
                        h = int(h / scale_det)
                    mx = int(w * 0.2)
                    my = int(h * 0.2)
                    cand = self._clamp_bbox((x - mx, y - my, w + 2 * mx, h + 2 * my), gray.shape)
                    candidates.append(cand)
                if self.last_bbox is not None:
                    def score(c):
                        iou = self._bbox_iou(c, self.last_bbox)
                        dist = self._bbox_center_distance(c, self.last_bbox)
                        area = c[2] * c[3]
                        return (2.0 * iou) + (1.0 / (1.0 + dist)) + (1e-6 * area)
                    self.last_bbox = max(candidates, key=score)
                else:
                    self.last_bbox = max(candidates, key=lambda c: c[2] * c[3])
                self.last_bbox_ts = self.frame_count

        if self.last_bbox:
            x, y, w, h = self.last_bbox
            bbox_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.rectangle(bbox_mask, (x, y), (x + w, y + h), 255, thickness=-1)

        mask_fg = self.bg_subtractor.apply(gray)
        mask_fg = cv2.medianBlur(mask_fg, 5)
        mask_fg = cv2.threshold(mask_fg, 32, 255, cv2.THRESH_BINARY)[1]

        if bbox_mask is not None:
            mask = cv2.bitwise_and(mask_fg, bbox_mask)
            if cv2.countNonZero(mask) < 250:
                mask = bbox_mask
        else:
            mask = mask_fg
        return mask

    def _resolve_tracking_mode(self, gray):
        if self.tracking_mode != "auto":
            self.active_tracking_mode = self.tracking_mode
            return self.active_tracking_mode

        if not self.use_ml_roi:
            self.active_tracking_mode = "ego"
            return self.active_tracking_mode

        h, w = gray.shape[:2]
        if self.last_bbox is not None:
            x, y, bw, bh = self.last_bbox
            area_ratio = (bw * bh) / max(float(h * w), 1.0)
            cx = (x + bw * 0.5) / max(float(w), 1.0)
            cy = (y + bh * 0.5) / max(float(h), 1.0)
            aspect_ratio = bh / max(float(bw), 1.0)
            centered = 0.2 <= cx <= 0.8 and 0.15 <= cy <= 0.85
            if 0.04 <= area_ratio <= 0.35 and centered and bh >= h * 0.28 and aspect_ratio >= 1.15:
                self.person_mode_votes += 1
            else:
                self.ego_mode_votes += 1
        else:
            self.ego_mode_votes += 1

        if self.frame_count >= 12:
            if self.person_mode_votes >= max(6, int(self.ego_mode_votes * 1.5)):
                self.active_tracking_mode = "person"
            elif self.ego_mode_votes >= 10:
                self.active_tracking_mode = "ego"
        return self.active_tracking_mode

    def _build_scene_mask(self, gray):
        h, w = gray.shape[:2]
        base_mask = np.full((h, w), 255, dtype=np.uint8)

        top_cut = int(h * 0.04)
        bottom_cut = int(h * 0.02)
        side_cut = int(w * 0.02)
        if top_cut > 0:
            base_mask[:top_cut, :] = 0
        if bottom_cut > 0:
            base_mask[h - bottom_cut:, :] = 0
        if side_cut > 0:
            base_mask[:, :side_cut] = 0
            base_mask[:, w - side_cut:] = 0

        base_mask[int(h * 0.84):, :int(w * 0.28)] = 0

        if self.last_bbox is not None:
            x, y, bw, bh = self.last_bbox
            pad_x = int(bw * 0.2)
            pad_y = int(bh * 0.15)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)
            base_mask[y1:y2, x1:x2] = 0

        texture = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        texture = np.uint8(np.clip(np.abs(texture), 0, 255))
        texture_mask = cv2.threshold(texture, self.scene_texture_threshold, 255, cv2.THRESH_BINARY)[1]
        texture_mask = cv2.medianBlur(texture_mask, 5)
        strict_mask = cv2.bitwise_and(base_mask, texture_mask)

        if cv2.countNonZero(strict_mask) >= int(strict_mask.size * 0.05):
            return strict_mask
        if cv2.countNonZero(base_mask) >= int(base_mask.size * 0.08):
            return base_mask
        return np.full((h, w), 255, dtype=np.uint8)

    def _detect_sparse_points(self, gray, mask):
        h, w = gray.shape[:2]
        rows = 4
        cols = 4
        max_per_cell = 45
        points = []

        quality = 0.01
        if self.prev_track_pts is not None and len(self.prev_track_pts) < self.min_sparse_tracks:
            quality = 0.005

        for gy in range(rows):
            for gx in range(cols):
                y1 = int(gy * h / rows)
                y2 = int((gy + 1) * h / rows)
                x1 = int(gx * w / cols)
                x2 = int((gx + 1) * w / cols)
                roi = gray[y1:y2, x1:x2]
                roi_mask = mask[y1:y2, x1:x2] if mask is not None else None
                corners = cv2.goodFeaturesToTrack(
                    roi,
                    maxCorners=max_per_cell,
                    qualityLevel=quality,
                    minDistance=8,
                    blockSize=7,
                    mask=roi_mask,
                )
                if corners is None:
                    continue
                corners[:, 0, 0] += x1
                corners[:, 0, 1] += y1
                points.append(corners)

        if not points:
            return None
        return np.vstack(points).astype(np.float32)

    def _estimate_sparse_flow_motion(self, prev_gray, gray, mask):
        if prev_gray is None:
            return 0.0, 0.0, 0.0, 0.0, False

        if self.prev_track_pts is None or len(self.prev_track_pts) < self.sparse_reseed_threshold:
            self.prev_track_pts = self._detect_sparse_points(prev_gray, self.prev_tracking_mask)

        if self.prev_track_pts is None or len(self.prev_track_pts) < self.min_sparse_tracks:
            return 0.0, 0.0, 0.0, 0.0, False

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, self.prev_track_pts, None, **self.lk_params)
        if next_pts is None or status is None:
            self.prev_track_pts = self._detect_sparse_points(gray, mask)
            return 0.0, 0.0, 0.0, 0.0, False

        valid = status.reshape(-1) == 1
        src = self.prev_track_pts[valid].reshape(-1, 2)
        dst = next_pts[valid].reshape(-1, 2)
        if len(src) < self.min_sparse_tracks:
            self.prev_track_pts = self._detect_sparse_points(gray, mask)
            return 0.0, 0.0, 0.0, 0.0, False

        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, dst.reshape(-1, 1, 2), None, **self.lk_params)
        if back_pts is not None and back_status is not None:
            fb_ok = back_status.reshape(-1) == 1
            fb_err = np.linalg.norm(src[fb_ok] - back_pts.reshape(-1, 2)[fb_ok], axis=1)
            good_fb = np.zeros(len(src), dtype=bool)
            good_fb[np.where(fb_ok)[0][fb_err < 1.5]] = True
            src = src[good_fb]
            dst = dst[good_fb]

        self.total_sparse_tracks += len(src)
        if len(src) < self.min_sparse_tracks:
            self.prev_track_pts = self._detect_sparse_points(gray, mask)
            return 0.0, 0.0, 0.0, 0.0, False

        src_f = src.astype(np.float32)
        dst_f = dst.astype(np.float32)
        self._sparse_used_essential = False

        dx, dy, dtheta, ratio, ok = self._estimate_pose_essential(src_f, dst_f)
        if ok:
            self._sparse_used_essential = True
            # essential -> rotation only
            aff_dx, aff_dy, _, aff_ratio, aff_ok = self._estimate_motion(src_f, dst_f)
            if aff_ok:
                dx, dy = aff_dx, aff_dy
                ratio = max(ratio, aff_ratio)
        else:
            dx, dy, dtheta, ratio, ok = self._estimate_motion(src_f, dst_f)

        self.prev_track_pts = dst.reshape(-1, 1, 2) if len(dst) >= self.sparse_reseed_threshold else self._detect_sparse_points(gray, mask)
        if not ok:
            return 0.0, 0.0, 0.0, ratio, False

        self.total_sparse_successes += 1
        conf = float(np.clip(ratio * min(1.0, len(dst) / 240.0), 0.0, 1.0))
        return dx, dy, dtheta, conf, True

    def _estimate_motion(self, src, dst):
        M = None
        inlier_mask = None
        if self.use_homography:
            method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
            M, inlier_mask = cv2.findHomography(src, dst, method, 2.0)
            if M is None or inlier_mask is None:
                return 0.0, 0.0, 0.0, 0.0, False
            inliers = int(np.sum(inlier_mask))
            ratio = inliers / max(len(src), 1)
            if inliers < self.min_inliers_abs or ratio < self.min_inlier_ratio:
                return 0.0, 0.0, 0.0, ratio, False
            tx = float(M[0, 2])
            ty = float(M[1, 2])
            a, b = float(M[0, 0]), float(M[0, 1])
            c, _d = float(M[1, 0]), float(M[1, 1])
            scale = np.sqrt(a * a + b * b)
            dtheta = float(np.degrees(np.arctan2(c / scale, a / scale))) if scale > 1e-6 else 0.0
        else:
            M, inlier_mask = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                confidence=0.995,
                maxIters=3000,
            )
            if M is None or inlier_mask is None:
                return 0.0, 0.0, 0.0, 0.0, False
            inliers = int(np.sum(inlier_mask))
            ratio = inliers / max(len(src), 1)
            if inliers < self.min_inliers_abs or ratio < self.min_inlier_ratio:
                return 0.0, 0.0, 0.0, ratio, False
            tx = float(M[0, 2])
            ty = float(M[1, 2])
            a, c = float(M[0, 0]), float(M[1, 0])
            scale = np.sqrt(a * a + c * c)
            dtheta = float(np.degrees(np.arctan2(c / scale, a / scale))) if scale > 1e-6 else 0.0

        if self.prev_shape is not None:
            h, w = self.prev_shape[:2]
            diag = float(np.hypot(w, h))
            if float(np.hypot(tx, ty)) > 0.08 * diag:
                return 0.0, 0.0, 0.0, ratio, False
        else:
            if abs(tx) > self.max_translation_px or abs(ty) > self.max_translation_px:
                return 0.0, 0.0, 0.0, ratio, False

        if abs(dtheta) > 90.0:
            return 0.0, 0.0, 0.0, ratio, False

        dx = tx * self.motion_scale * self.scale_factor
        dy = ty * self.motion_scale * self.scale_factor
        return dx, dy, dtheta, ratio, True

    def _estimate_pose_essential(self, src, dst):
        if self.camera_matrix is None or len(src) < 8:
            return 0.0, 0.0, 0.0, 0.0, False

        E, mask = cv2.findEssentialMat(
            src,
            dst,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if E is None or mask is None:
            return 0.0, 0.0, 0.0, 0.0, False

        inliers = int(np.sum(mask))
        ratio = inliers / max(len(src), 1)
        min_essential_inliers = max(8, self.min_inliers_abs)
        if inliers < min_essential_inliers or ratio < self.min_inlier_ratio:
            return 0.0, 0.0, 0.0, ratio, False

        _, R, _t, _ = cv2.recoverPose(E, src, dst, self.camera_matrix, mask=mask)
        yaw_deg = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
        if abs(yaw_deg) > 90.0:
            return 0.0, 0.0, 0.0, ratio, False

        conf = float(np.clip(ratio, 0.0, 1.0))
        # Rotation only. Translation from essential is intentionally not used as metric step.
        return 0.0, 0.0, yaw_deg, conf, True

    def _undistort_frame(self, frame):
        if self.camera_matrix is None:
            return frame
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def _rotation_weight(self, conf: float, used_essential: bool) -> float:
        return max(0.05, float(conf)) + (0.20 if used_essential else 0.0)

    def _classify_motion_state(
        self,
        dx: float,
        dy: float,
        dheading: float,
        flow_mag: float,
        feature_conf: float,
        sparse_conf: float,
        feature_ok: bool,
        sparse_ok: bool,
        mode: str,
    ) -> str:
        conf = max(float(feature_conf), float(sparse_conf))
        speed = float(np.hypot(dx, dy))
        rot = abs(float(dheading))
        no_pose = not feature_ok and not sparse_ok

        if no_pose:
            self.no_pose_frames += 1
        else:
            self.no_pose_frames = 0

        if mode != "ego":
            self.stationary_frames = 0
            self.turn_in_place_frames = 0
            self.stationary_state = False
            return "moving"

        low_flow = flow_mag <= self.stationary_flow_threshold
        low_speed = speed <= self.stationary_translation_threshold
        low_rot = rot <= self.stationary_rotation_threshold

        stationary_candidate = (
            (self.no_pose_frames >= 1 and low_flow and low_rot)
            or (conf < 0.10 and low_speed and low_rot and low_flow)
            or (conf < 0.06 and low_speed and low_rot)
        )
        if stationary_candidate:
            self.stationary_frames = min(self.stationary_hysteresis_frames + 2, self.stationary_frames + 1)
        else:
            self.stationary_frames = max(0, self.stationary_frames - 1)

        if self.stationary_frames >= self.stationary_hysteresis_frames:
            self.stationary_state = True
        elif self.stationary_frames == 0:
            self.stationary_state = False

        strong_rot_source = (
            (feature_ok and self._feature_used_essential)
            or (sparse_ok and self._sparse_used_essential)
            or conf >= self.turn_in_place_min_conf
        )
        turn_in_place_candidate = (
            strong_rot_source
            and rot >= self.turn_in_place_rotation_threshold
            and speed <= self.turn_in_place_translation_threshold
        )
        if turn_in_place_candidate and not self.stationary_state:
            self.turn_in_place_frames = min(4, self.turn_in_place_frames + 1)
        else:
            self.turn_in_place_frames = 0

        if self.stationary_state:
            return "stationary"
        if self.turn_in_place_frames >= 2:
            return "turn_in_place"
        return "moving"

    def _final_motion_gate(self, dx, dy, dheading, feature_conf, sparse_conf, mode, dt_sec):
        speed = float(np.hypot(dx, dy))
        rot = abs(dheading)
        conf = max(float(feature_conf), float(sparse_conf))
        dt_scale = max(1.0, float(dt_sec))

        if mode == "ego":
            if rot > self.max_heading_step_deg * dt_scale and speed > 0.20 * dt_scale:
                self.total_gated_failures += 1
                return 0.0, 0.0, 0.0, False
            if conf < 0.15 and speed > 0.08 * dt_scale:
                self.total_gated_failures += 1
                return 0.0, 0.0, 0.0, False
            if conf < 0.10 and rot > 8.0 * dt_scale:
                self.total_gated_failures += 1
                return 0.0, 0.0, 0.0, False
        return dx, dy, dheading, True

    def process_frame(self, frame, frame_index=None, timestamp_sec=None, dt_sec: Optional[float] = None, dt_frames: Optional[int] = None):
        start = time.time()
        self.frame_count += 1
        is_key_frame = (self.frame_count % 30 == 0)

        if self.camera_matrix is None and self.frame_count == 1:
            cam, dist = CameraCalibrator.estimate_distortion_auto(frame)
            if cam is not None:
                self.camera_matrix = cam
                self.dist_coeffs = dist

        if self.camera_matrix is not None:
            frame = self._undistort_frame(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        self.prev_shape = gray.shape

        if dt_sec is None:
            if timestamp_sec is not None and self.prev_timestamp_sec is not None:
                dt_sec = max(1e-3, float(timestamp_sec) - float(self.prev_timestamp_sec))
            elif dt_frames is not None:
                dt_sec = float(max(1, dt_frames))
            else:
                dt_sec = 1.0
        dt_sec = max(1e-3, float(dt_sec))
        self.last_dt_sec = dt_sec

        person_mask = self._build_person_mask(gray)
        mode = self._resolve_tracking_mode(gray)
        mask = person_mask if mode == "person" else self._build_scene_mask(gray)
        gray_masked = cv2.bitwise_and(gray, mask)

        kp, des = self.feature_detector.detectAndCompute(gray, mask)
        if des is None or len(kp) < 80:
            kp, des = self.feature_detector.detectAndCompute(gray, None)

        dx = dy = dheading = 0.0
        feature_conf = 0.0
        sparse_conf = 0.0
        sparse_dx = sparse_dy = sparse_heading = 0.0
        feat_dx = feat_dy = feat_heading = 0.0
        sparse_ok = False
        feature_ok = False
        self._feature_used_essential = False
        self.frame_debug = {
            "mode": mode,
            "feature_ok": False,
            "sparse_ok": False,
            "fusion_mode": "none",
            "source_rot": None,
            "source_trans": None,
        }

        if mode == "ego" and self.prev_gray is not None:
            sparse_dx, sparse_dy, sparse_heading, sparse_conf, sparse_ok = self._estimate_sparse_flow_motion(self.prev_gray, gray, mask)
            self.frame_debug["sparse_ok"] = sparse_ok

        good = []
        if self.prev_des is not None and des is not None:
            matches = self.matcher.knnMatch(self.prev_des, des, k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.70 * m[1].distance]
            self.total_matches += len(good)

            if len(good) > self.min_good_matches:
                src = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
                dst = np.float32([kp[m.trainIdx].pt for m in good])
                if mode == "ego" and self.camera_matrix is not None and len(src) >= 8:
                    _zero_dx, _zero_dy, feat_heading, feature_conf, ok_rot = self._estimate_pose_essential(src, dst)
                    if ok_rot:
                        self._feature_used_essential = True
                        feat_dx, feat_dy, _aff_heading, aff_conf, ok_trans = self._estimate_motion(src, dst)
                        feature_ok = True
                        if ok_trans:
                            feature_conf = max(feature_conf, aff_conf)
                        else:
                            feat_dx = feat_dy = 0.0
                    else:
                        feat_dx, feat_dy, feat_heading, feature_conf, feature_ok = self._estimate_motion(src, dst)
                else:
                    feat_dx, feat_dy, feat_heading, feature_conf, feature_ok = self._estimate_motion(src, dst)

                if feature_ok:
                    dx, dy, dheading = feat_dx, feat_dy, feat_heading
                    self.total_feature_successes += 1
                    self.frame_debug["feature_ok"] = True
                else:
                    self.feature_model_failures += 1
            else:
                self.feature_insufficient_matches += 1
        else:
            self.feature_insufficient_matches += 1

        # Separate fusion for rotation and translation
        rot_source = None
        trans_source = None
        if sparse_ok and feature_ok:
            sparse_rot_weight = self._rotation_weight(sparse_conf, self._sparse_used_essential)
            feature_rot_weight = self._rotation_weight(feature_conf, self._feature_used_essential)
            sign_conflict = (
                abs(sparse_heading) > 1.0 and abs(feat_heading) > 1.0 and np.sign(sparse_heading) != np.sign(feat_heading)
            )
            if sign_conflict:
                self.sign_conflict_frames += 1
                if sparse_rot_weight > feature_rot_weight + 0.12:
                    dheading = sparse_heading
                    dx, dy = sparse_dx, sparse_dy
                    rot_source, trans_source = "sparse", "sparse"
                    self.frame_debug["fusion_mode"] = "conflict_sparse_wins"
                elif feature_rot_weight > sparse_rot_weight + 0.12:
                    dheading = feat_heading
                    dx, dy = feat_dx, feat_dy
                    rot_source, trans_source = "feature", "feature"
                    self.frame_debug["fusion_mode"] = "conflict_feature_wins"
                else:
                    dx = dy = dheading = 0.0
                    sparse_ok = feature_ok = False
                    self.global_pose_failures += 1
                    self.frame_debug["fusion_mode"] = "conflict_zero"
            else:
                sparse_trans_weight = max(0.05, sparse_conf)
                feature_trans_weight = max(0.05, feature_conf)
                dheading = (
                    (sparse_heading * sparse_rot_weight) + (feat_heading * feature_rot_weight)
                ) / (sparse_rot_weight + feature_rot_weight)
                dx = (
                    (sparse_dx * sparse_trans_weight) + (feat_dx * feature_trans_weight)
                ) / (sparse_trans_weight + feature_trans_weight)
                dy = (
                    (sparse_dy * sparse_trans_weight) + (feat_dy * feature_trans_weight)
                ) / (sparse_trans_weight + feature_trans_weight)
                rot_source, trans_source = "fused", "fused"
                self.frame_debug["fusion_mode"] = "weighted"
        elif sparse_ok:
            dx, dy, dheading = sparse_dx, sparse_dy, sparse_heading
            rot_source, trans_source = "sparse", "sparse"
            self._pose_sparse_only_frames += 1
            self.frame_debug["fusion_mode"] = "sparse_only"
        elif feature_ok:
            dx, dy, dheading = feat_dx, feat_dy, feat_heading
            rot_source, trans_source = ("feature_essential_rot" if self._feature_used_essential else "feature"), "feature"
            self._pose_feature_only_frames += 1
            self.frame_debug["fusion_mode"] = "feature_only"
        else:
            self.total_ransac_failures += 1
            self.global_pose_failures += 1
            self.frame_debug["fusion_mode"] = "no_tracking"

        if sparse_ok and feature_ok:
            if self._feature_used_essential or self._sparse_used_essential:
                self._pose_essential_frames += 1
            else:
                self._pose_affine_frames += 1

        speed = float(np.hypot(dx, dy))
        if abs(dheading) > 3.0:
            dx *= 0.35
            dy *= 0.35
        if abs(dheading) > 6.0:
            dx *= 0.15
            dy *= 0.15
        if abs(dheading) > 8.0 and speed < 0.03:
            dx = 0.0
            dy = 0.0

        dheading *= self.yaw_sign
        raw_dheading = dheading

        flow_dx = flow_dy = 0.0
        flow_mag = 0.0
        if self.use_optical_flow and self.prev_masked_gray is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(self.prev_masked_gray, gray_masked, None, **self.flow_params)
                h, w = flow.shape[:2]
                cx1, cx2 = w // 4, 3 * w // 4
                cy1, cy2 = h // 4, 3 * h // 4
                center_region = flow[cy1:cy2, cx1:cx2]
                flow_dx = float(np.median(center_region[..., 0])) * self.motion_scale * self.scale_factor
                flow_dy = float(np.median(center_region[..., 1])) * self.motion_scale * self.scale_factor
                flow_mag = float(np.hypot(flow_dx, flow_dy))
                pose_available = feature_ok or sparse_ok
                pose_mag = float(np.hypot(dx, dy))
                flow_consistent = True
                if pose_available and pose_mag > 1e-6 and flow_mag > 1e-6:
                    flow_consistent = (
                        ((dx * flow_dx) + (dy * flow_dy)) / ((pose_mag * flow_mag) + 1e-6)
                    ) > -0.15

                if pose_available and flow_consistent and flow_mag <= self.max_flow_assist:
                    flow_weight = 0.08 if (sparse_ok and feature_ok) else 0.12
                    dx = dx * (1 - flow_weight) + flow_dx * flow_weight
                    dy = dy * (1 - flow_weight) + flow_dy * flow_weight
                    trans_source = "fused_flow"
                    self.flow_assist_frames += 1
                elif flow_mag > 0.0:
                    self.flow_rejection_count += 1
            except Exception:
                pass

        dx, dy, dheading, motion_ok = self._final_motion_gate(dx, dy, raw_dheading, feature_conf, sparse_conf, mode, dt_sec)
        if not motion_ok:
            self.total_gated_failures += 1

        motion_state = self._classify_motion_state(
            dx=dx,
            dy=dy,
            dheading=dheading,
            flow_mag=flow_mag,
            feature_conf=feature_conf,
            sparse_conf=sparse_conf,
            feature_ok=feature_ok,
            sparse_ok=sparse_ok,
            mode=mode,
        )
        if motion_state == "stationary":
            dx = 0.0
            dy = 0.0
            dheading = 0.0
            raw_dheading = 0.0
            self.pos_buffer.clear()
            self.rot_buffer.clear()
            self.stationary_lock_frames += 1
            trans_source = "stationary_lock"
            rot_source = rot_source or "stationary_lock"
        elif motion_state == "turn_in_place":
            dx = 0.0
            dy = 0.0
            self.pos_buffer.clear()
            self.rot_buffer.clear()
            trans_source = "turn_in_place_lock"

        self.prev_masked_gray = gray_masked.copy()

        prev_dx = self.pos_buffer[-1][0] if len(self.pos_buffer) > 0 else dx
        prev_dy = self.pos_buffer[-1][1] if len(self.pos_buffer) > 0 else dy
        prev_rot = self.rot_buffer[-1] if len(self.rot_buffer) > 0 else dheading

        alpha_pos = self.traj_smooth_alpha if motion_state == "moving" else 1.0
        alpha_rot = self.traj_smooth_alpha
        if motion_state == "stationary":
            alpha_rot = 1.0
        elif motion_state == "turn_in_place":
            alpha_rot = self.rotation_decay_alpha
        elif abs(raw_dheading) < self.rotation_decay_threshold:
            alpha_rot = self.rotation_decay_alpha
        elif (
            abs(prev_rot) > self.rotation_sign_flip_threshold
            and abs(raw_dheading) > self.rotation_sign_flip_threshold
            and np.sign(prev_rot) != np.sign(raw_dheading)
        ):
            alpha_rot = self.rotation_flip_alpha
        dx_s = alpha_pos * dx + (1 - alpha_pos) * prev_dx
        dy_s = alpha_pos * dy + (1 - alpha_pos) * prev_dy
        rot_s = alpha_rot * dheading + (1 - alpha_rot) * prev_rot
        if motion_state == "stationary":
            turn_dyaw = 0.0
        elif motion_state == "turn_in_place":
            turn_dyaw = raw_dheading
        else:
            turn_dyaw = self.turn_smooth_alpha * raw_dheading + (1 - self.turn_smooth_alpha) * prev_rot

        self.pos_buffer.append([dx_s, dy_s])
        self.rot_buffer.append(rot_s)

        kalman_mode = "disabled"
        if self.use_kalman:
            current_xy = np.array(self.trajectory[-1][:2], dtype=np.float64)
            max_conf = max(float(feature_conf), float(sparse_conf))
            kalman_can_update = (
                motion_state == "moving"
                and motion_ok
                and (feature_ok or sparse_ok)
                and max_conf >= self.motion_resume_confidence
            )
            if motion_state in ("stationary", "turn_in_place"):
                self.kalman.anchor(current_xy, zero_velocity=True)
                dx_s = 0.0
                dy_s = 0.0
                kalman_mode = "locked"
                self.kalman_lock_frames += 1
            elif kalman_can_update:
                measurement = current_xy + np.array([dx_s, dy_s], dtype=np.float64)
                if not self.kalman.initialized:
                    self.kalman.anchor(current_xy, zero_velocity=True)
                self.kalman.predict(dt=dt_sec)
                if np.linalg.norm(measurement - self.kalman.state[:2]) > self.max_kalman_residual and max_conf < 0.35:
                    self.kalman.anchor(current_xy, zero_velocity=True)
                    dx_s = 0.0
                    dy_s = 0.0
                    kalman_mode = "reanchor"
                    self.kalman_lock_frames += 1
                else:
                    filtered_pos = self.kalman.update(measurement)
                    dx_s = float(filtered_pos[0] - current_xy[0])
                    dy_s = float(filtered_pos[1] - current_xy[1])
                    max_step = max(
                        self.stationary_translation_threshold * 2.0,
                        min(self.max_kalman_step, (float(np.hypot(dx_s, dy_s)) * 2.0) + 0.03),
                    )
                    dx_s = float(np.clip(dx_s, -max_step, max_step))
                    dy_s = float(np.clip(dy_s, -max_step, max_step))
                    kalman_mode = "update"
            else:
                self.kalman.anchor(current_xy, zero_velocity=True)
                kalman_mode = "locked"
                self.kalman_lock_frames += 1

        speed_s = float(np.hypot(dx_s, dy_s))
        if speed_s < self.deadzone_speed and abs(rot_s) < 0.2:
            self.total_deadzone_skips += 1
            if motion_state != "turn_in_place":
                dx_s = 0.0
                dy_s = 0.0
            if motion_state == "stationary":
                rot_s = 0.0

        prev_heading = self.heading
        prev_heading_rad = np.radians(prev_heading)
        global_dx = dx_s * np.cos(prev_heading_rad) - dy_s * np.sin(prev_heading_rad)
        global_dy = dx_s * np.sin(prev_heading_rad) + dy_s * np.cos(prev_heading_rad)
        self.heading = _wrap_deg(self.heading + rot_s)

        straight_mode = (
            self.active_turn is None
            and len(self.delta_yaw_history) >= 5
            and float(np.mean([abs(d) for d in list(self.delta_yaw_history)[-5:]])) < 0.6
        )
        if straight_mode:
            corridor_heading_deg, vp_confidence = get_dominant_corridor_heading(gray, self.heading)
            self._last_vp_confidence = vp_confidence
            if vp_confidence > 0.75:
                heading_error = _wrap_deg(self.heading - corridor_heading_deg)
                self.heading = _wrap_deg(self.heading - 0.08 * heading_error)
        else:
            self._last_vp_confidence = 0.0

        new_x = self.trajectory[-1][0] + global_dx
        new_y = self.trajectory[-1][1] + global_dy
        if abs(new_x) > 1e6 or np.isnan(new_x):
            new_x = self.trajectory[-1][0]
        if abs(new_y) > 1e6 or np.isnan(new_y):
            new_y = self.trajectory[-1][1]

        self.trajectory.append([float(new_x), float(new_y), float(self.heading)])

        self.delta_yaw_history.append(float(turn_dyaw))
        self.turn_step_history.append(
            {
                "frame_index": int(frame_index if frame_index is not None else self.frame_count),
                "trajectory_index": len(self.trajectory) - 1,
                "dyaw": float(turn_dyaw),
                "speed": float(np.hypot(dx_s, dy_s)),
                "x": float(new_x),
                "y": float(new_y),
            }
        )

        self._detect_turns_from_heading()
        self._detect_enhanced_turns()

        self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
        self.prev_tracking_mask = mask.copy()
        self.prev_frame_index = frame_index
        self.prev_timestamp_sec = timestamp_sec
        self.processing_times.append(time.time() - start)

        self.frame_debug.update(
            {
                "source_rot": rot_source,
                "source_trans": trans_source,
                "raw_dheading": float(raw_dheading),
                "filtered_dheading": float(rot_s),
                "dx": float(dx),
                "dy": float(dy),
                "dx_s": float(dx_s),
                "dy_s": float(dy_s),
                "heading": float(self.heading),
                "feature_conf": float(feature_conf),
                "sparse_conf": float(sparse_conf),
                "flow_mag": float(flow_mag),
                "motion_state": motion_state,
                "kalman_mode": kalman_mode,
                "no_pose_frames": int(self.no_pose_frames),
                "dt_sec": float(dt_sec),
            }
        )

        if self.corridor_width_m and len(self.trajectory) > 50 and (len(self.trajectory) % 50 == 0):
            n_straight = 30
            straight_segment = (
                self.active_turn is None
                and len(self.delta_yaw_history) >= n_straight
                and float(np.mean([abs(d) for d in list(self.delta_yaw_history)[-n_straight:]])) < 0.5
            )
            lateral_low = False
            if straight_segment and len(self.trajectory) >= n_straight:
                pts = self.trajectory[-n_straight:]
                path_len = sum(
                    float(np.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])) for i in range(1, len(pts))
                )
                straight_dist = float(np.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]))
                lateral_low = path_len > 2.0 and (straight_dist / path_len) >= 0.92
            if straight_segment and lateral_low:
                new_scale = ScaleEstimator.estimate_from_corridor(self.trajectory, self.corridor_width_m)
                self.raw_scale_candidate = new_scale
                if self._last_vp_confidence > 0.7 and 0.2 < new_scale < 10:
                    self.scale_factor = 0.9 * self.scale_factor + 0.1 * new_scale
            else:
                self.raw_scale_candidate = None

        return self.trajectory[-1]

    def _detect_turns_from_heading(self):
        if len(self.turn_step_history) < 6:
            return

        recent = list(self.turn_step_history)
        tail = recent[-5:]
        mean_abs_yaw = float(np.mean([abs(s["dyaw"]) for s in tail]))
        mean_speed = float(np.mean([s["speed"] for s in tail]))

        start_thresh = 0.8
        end_thresh = 0.25
        min_speed = 0.01

        if self.active_turn is None:
            turn_in_place_signal = max(abs(s["dyaw"]) for s in tail) >= self.turn_in_place_rotation_threshold
            if mean_abs_yaw >= start_thresh and (mean_speed >= min_speed or turn_in_place_signal):
                self.active_turn = {"samples": tail.copy()}
            return

        self.active_turn["samples"].append(recent[-1])
        tail2 = self.active_turn["samples"][-5:]
        if float(np.mean([abs(s["dyaw"]) for s in tail2])) <= end_thresh:
            samples = self.active_turn["samples"]
            total_angle = float(sum(s["dyaw"] for s in samples))
            if abs(total_angle) >= 20.0:
                mid = samples[len(samples) // 2]
                self.raw_turn_points.append(
                    {
                        "frame_index": mid["frame_index"],
                        "trajectory_index": mid["trajectory_index"],
                        "angle_degrees": round(abs(total_angle), 1),
                        "raw_angle_degrees": round(total_angle, 1),
                        "position": [mid["x"], mid["y"], self.heading],
                        "turn_type": "left" if total_angle > 0 else "right",
                        "confidence": round(min(1.0, abs(total_angle) / 90.0), 3),
                        "source": "yaw_integration",
                    }
                )
            self.active_turn = None

    def _detect_enhanced_turns(self):
        if len(self.trajectory) < 60:
            return

        window = 48
        i = len(self.trajectory) - 1
        start = max(0, i - window)
        mid = start + window // 2

        p0 = np.array(self.trajectory[start][:2], dtype=np.float32)
        p1 = np.array(self.trajectory[mid][:2], dtype=np.float32)
        p2 = np.array(self.trajectory[i][:2], dtype=np.float32)
        v1 = p1 - p0
        v2 = p2 - p1
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 <= 3.0 or n2 <= 3.0:
            return

        cross = v1[0] * v2[1] - v1[1] * v2[0]
        chord_raw_angle = float(np.degrees(np.arctan2(cross, np.dot(v1, v2))))
        abs_chord = abs(chord_raw_angle)
        angle_thresh = max(30, 25 + 2.5 * np.std(self.rot_buffer)) if self.rot_buffer else 30
        if abs_chord < angle_thresh:
            return

        min_gap = 25
        if not self.trajectory_turn_points or abs((self.trajectory_turn_points[-1].get("trajectory_index") or 0) - i) > min_gap:
            pt = self.trajectory[i]
            self.trajectory_turn_points.append(
                {
                    "frame_index": self.frame_count,
                    "trajectory_index": i,
                    "angle_degrees": round(abs_chord, 1),
                    "raw_angle_degrees": round(chord_raw_angle, 1),
                    "position": [pt[0], pt[1], self.heading],
                    "turn_type": "left" if chord_raw_angle > 0 else "right",
                    "confidence": round(min(1.0, abs_chord / 90.0), 3),
                    "source": "chord",
                }
            )

        if not self.raw_turn_points:
            return
        last = self.raw_turn_points[-1]
        if abs(last["trajectory_index"] - mid) > 40:
            return
        integrated_raw = float(last.get("raw_angle_degrees", 0))
        sign_agree = np.sign(chord_raw_angle) == np.sign(integrated_raw)
        mag_close = abs(abs_chord - abs(integrated_raw)) <= 18.0
        if sign_agree and mag_close:
            last["confidence"] = round(min(1.0, last.get("confidence", 0.5) * 1.25), 3)
        else:
            last["confidence"] = round(min(last.get("confidence", 0.5), 0.35), 3)
            last["needs_review"] = True
            self.turn_conflict_count += 1

    def set_scale_factor(self, s):
        self.scale_factor = float(s)

    def set_calibrator(self, calibrator):
        self.calibrator = calibrator
        if calibrator and getattr(calibrator, "is_calibrated", False):
            self.camera_matrix = calibrator.camera_matrix
            self.dist_coeffs = calibrator.dist_coeffs

    def set_video_context(self, fps: Optional[float] = None, frame_skip: int = 1):
        self.video_fps = fps
        self.video_frame_skip = frame_skip

    def calibrate_yaw_sign(self, accumulated_yaw_degrees: float, known_turn_right: bool):
        if known_turn_right and accumulated_yaw_degrees < 0:
            self.yaw_sign = -1.0
        elif (not known_turn_right) and accumulated_yaw_degrees > 0:
            self.yaw_sign = -1.0
        else:
            self.yaw_sign = 1.0

    def get_trajectory(self):
        return [[round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 2)] for p in self.trajectory]

    def get_turn_points(self):
        return self.raw_turn_points

    def get_raw_turn_points(self):
        return self.raw_turn_points

    def get_trajectory_turn_points(self):
        return self.trajectory_turn_points

    def get_last_frame_debug(self):
        return dict(self.frame_debug)

    def get_statistics(self):
        dist = sum(
            np.linalg.norm(np.array(self.trajectory[i + 1][:2]) - np.array(self.trajectory[i][:2]))
            for i in range(len(self.trajectory) - 1)
        )
        processed_frames = max(len(self.trajectory) - 1, 1)
        avg_matches = self.total_matches / processed_frames
        ransac_rate = self.total_ransac_failures / processed_frames
        mean_time = np.mean(self.processing_times) if self.processing_times else 0
        fps = round(1 / mean_time, 1) if mean_time and mean_time > 0 else 0

        pose_modes = [
            (self._pose_essential_frames, "essential"),
            (self._pose_affine_frames, "affine_fallback"),
            (self._pose_sparse_only_frames, "sparse_only"),
            (self._pose_feature_only_frames, "feature_only"),
        ]
        pose_estimation_mode = max(pose_modes, key=lambda x: x[0])[1] if pose_modes else "unknown"

        return {
            "estimated_distance": round(dist, 3),
            "scale_factor": self.scale_factor,
            "fps": fps,
            "turns_detected": len(self.raw_turn_points),
            "avg_matches_per_processed_frame": round(avg_matches, 1),
            "ransac_failure_rate": round(ransac_rate, 3),
            "gating_failure_rate": round(self.total_gated_failures / processed_frames, 3),
            "deadzone_skip_rate": round(self.total_deadzone_skips / processed_frames, 3),
            "sparse_track_density": round(self.total_sparse_tracks / processed_frames, 1),
            "sparse_success_rate": round(self.total_sparse_successes / processed_frames, 3),
            "feature_success_rate": round(self.total_feature_successes / processed_frames, 3),
            "feature_insufficient_matches_rate": round(self.feature_insufficient_matches / processed_frames, 3),
            "feature_model_failure_rate": round(self.feature_model_failures / processed_frames, 3),
            "global_pose_failure_rate": round(self.global_pose_failures / processed_frames, 3),
            "flow_assist_rate": round(self.flow_assist_frames / processed_frames, 3),
            "flow_rejection_rate": round(self.flow_rejection_count / processed_frames, 3),
            "stationary_lock_rate": round(self.stationary_lock_frames / processed_frames, 3),
            "kalman_lock_rate": round(self.kalman_lock_frames / processed_frames, 3),
            "tracking_mode_used": self.active_tracking_mode,
            "homography_used": self.use_homography,
            "kalman_filter_used": self.use_kalman,
            "akaze_used": self.use_akaze,
            "yaw_sign_used": float(self.yaw_sign),
            "turn_detection_mode": "yaw_and_chord",
            "turn_conflict_count": self.turn_conflict_count,
            "sign_conflict_frames": self.sign_conflict_frames,
            "pose_estimation_mode": pose_estimation_mode,
            "last_dt_sec": round(float(self.last_dt_sec), 4),
        }

    def reset(self):
        yaw_sign_saved = self.yaw_sign
        init_cfg = dict(self._init_config)
        self.__init__(**init_cfg)
        self.yaw_sign = yaw_sign_saved


class HighAccuracyVisualOdometry(EnhancedVisualOdometry):
    def __init__(self, scale_factor=1.0, ml_model_path: str = "models/detector.onnx"):
        super().__init__(
            scale_factor=scale_factor,
            use_homography=True,
            use_kalman=True,
            use_akaze=False,
            calibrator=None,
            use_ml_roi=True,
            ml_model_path=ml_model_path,
        )


def create_enhanced_vo(
    scale_factor=1.0,
    use_homography=True,
    use_kalman=True,
    use_akaze=False,
    calibration_path=None,
    use_ml_roi: bool = True,
    ml_model_path: str = "models/detector.onnx",
):
    calibrator = None
    if calibration_path and os.path.exists(calibration_path):
        calibrator = CameraCalibrator.load_calibration(calibration_path)

    return EnhancedVisualOdometry(
        scale_factor=scale_factor,
        use_homography=use_homography,
        use_kalman=use_kalman,
        use_akaze=use_akaze,
        calibrator=calibrator,
        use_ml_roi=use_ml_roi,
        ml_model_path=ml_model_path,
    )
