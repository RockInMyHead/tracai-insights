import os
import math
import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


class MLDetector:
    """YOLOv8/RT-DETR ONNX detector (person class only)."""

    def __init__(self, model_path: str, conf: float = 0.20, iou: float = 0.50, multi_scales=(640, 960)):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.multi_scales = multi_scales
        self.session = None
        self.input_name = None
        self.input_shape = None
        if ort is None:
            logger.warning("onnxruntime not installed; ML ROI disabled")
            return
        if not os.path.exists(model_path):
            logger.warning(f"Detector model not found: {model_path}")
            return
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape  # [batch,3,h,w] maybe dynamic
        logger.info(f"MLDetector loaded: {model_path}, input shape {self.input_shape}")

    def _preprocess(self, img: np.ndarray, target: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        # letterbox to target size (square)
        h, w = img.shape[:2]
        inp_h = target
        inp_w = target
        scale = min(inp_w / w, inp_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((inp_h, inp_w, 3), dtype=np.uint8)
        top = (inp_h - nh) // 2
        left = (inp_w - nw) // 2
        canvas[top:top+nh, left:left+nw, :] = resized
        img = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize
        img = np.transpose(img, (2, 0, 1))
        return img[None, ...], scale, (left, top)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / ( (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]) + (boxes[order[1:],2]-boxes[order[1:],0])*(boxes[order[1:],3]-boxes[order[1:],1]) - inter + 1e-6 )
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        return keep

    def detect(self, img: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Returns list of (x1,y1,x2,y2,score) in original image coords."""
        if self.session is None:
            return []
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img
        all_boxes = []
        all_scores = []
        for target in self.multi_scales:
            blob, scale, (left, top) = self._preprocess(img_bgr, target)
            inp = {self.input_name: blob}
            outputs = self.session.run(None, inp)
            out = outputs[0]
            if out.ndim == 3:
                out = out[0]
            # YOLOv8 ONNX: (84, N) → transpose to (N, 84); legacy: (N, 4+1+num_classes)
            if out.shape[0] < out.shape[1]:
                out = out.T
            if out.shape[-1] < 5:
                continue
            # 84 = 4 box + 80 classes (no objectness); 85 = 4 + objectness + 80
            has_objectness = out.shape[1] == 85
            boxes = out[:, :4]
            scores = out[:, 5:] if has_objectness else out[:, 4:]
            if scores.size == 0:
                continue
            class_ids = scores.argmax(axis=1)
            obj = out[:, 4] if has_objectness else 1.0
            confs = scores[np.arange(scores.shape[0]), class_ids] * (obj if has_objectness else 1.0)
            mask = (class_ids == 0) & (confs >= self.conf)
            boxes = boxes[mask]
            confs = confs[mask]
            if boxes.size == 0:
                continue
            xyxy = np.zeros_like(boxes)
            xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
            xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / scale
            xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / scale
            all_boxes.append(xyxy)
            all_scores.append(confs)

        if not all_boxes:
            return []
        xyxy = np.concatenate(all_boxes, axis=0)
        confs = np.concatenate(all_scores, axis=0)
        keep = self._nms(xyxy, confs, self.iou)
        xyxy = xyxy[keep]
        confs = confs[keep]
        results = []
        for b, s in zip(xyxy, confs):
            x1, y1, x2, y2 = b.tolist()
            results.append((x1, y1, x2, y2, float(s)))
        return results


class SimpleTracker:
    """Minimal IoU+velocity tracker (ByteTrack-lite)."""

    def __init__(self, max_age: int = 8, iou_thres: float = 0.4):
        self.max_age = max_age
        self.iou_thres = iou_thres
        self.tracks = []
        self.next_id = 1

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def update(self, detections: List[Tuple[float, float, float, float, float]]):
        # tracks: dict {id, bbox, score, age, miss}
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append({
                    "id": self.next_id,
                    "bbox": det[:4],
                    "score": det[4],
                    "age": 1,
                    "miss": 0,
                    "vel": (0.0, 0.0)
                })
                self.next_id += 1
            return self.tracks

        for t in self.tracks:
            t["miss"] += 1

        for det in detections:
            best_score = -1
            best_idx = None
            dx_det = (det[0] + det[2]) * 0.5
            dy_det = (det[1] + det[3]) * 0.5
            for idx, t in enumerate(self.tracks):
                iou = self._iou(det[:4], t["bbox"])
                if iou < self.iou_thres:
                    continue
                tx = (t["bbox"][0] + t["bbox"][2]) * 0.5
                ty = (t["bbox"][1] + t["bbox"][3]) * 0.5
                vx, vy = t["vel"]
                # Предсказанный центр
                px, py = tx + vx, ty + vy
                dist = math.hypot(dx_det - px, dy_det - py)
                score = iou + 1.0 / (1.0 + dist)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None:
                t = self.tracks[best_idx]
                prev_cx = (t["bbox"][0] + t["bbox"][2]) * 0.5
                prev_cy = (t["bbox"][1] + t["bbox"][3]) * 0.5
                cx = (det[0] + det[2]) * 0.5
                cy = (det[1] + det[3]) * 0.5
                vx = cx - prev_cx
                vy = cy - prev_cy
                t["vel"] = (0.7 * t["vel"][0] + 0.3 * vx, 0.7 * t["vel"][1] + 0.3 * vy)
                t["bbox"] = det[:4]
                t["score"] = det[4]
                t["age"] += 1
                t["miss"] = 0
            else:
                self.tracks.append({
                    "id": self.next_id,
                    "bbox": det[:4],
                    "score": det[4],
                    "age": 1,
                    "miss": 0,
                    "vel": (0.0, 0.0)
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["miss"] <= self.max_age]
        return self.tracks

    def get_best_track(self) -> Optional[dict]:
        if not self.tracks:
            return None
        # choose by score * age penalty
        return max(self.tracks, key=lambda t: (t["score"], -t["miss"]))
