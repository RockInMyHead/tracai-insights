import cv2
import numpy as np


def get_video_info(video_path):
    """Получение информации о видео"""
    cap = cv2.VideoCapture(video_path)
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)) if cap.get(
            cv2.CAP_PROP_FPS) > 0 else 0
    }
    cap.release()
    return info


def extract_frames_efficiently(video_path, frame_interval=5, target_width=640):
    """Эффективное извлечение кадров из видео"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Ресайз для экономии памяти
            if frame.shape[1] > target_width:
                scale = target_width / frame.shape[1]
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (target_width, new_height))
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames, frame_count