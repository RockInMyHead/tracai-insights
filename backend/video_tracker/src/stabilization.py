import cv2
from pathlib import Path
import logging
import subprocess
import os

logger = logging.getLogger(__name__)

FFMPEG_STAB_TIMEOUT = 1800  # 30 минут для больших видео


def _video_duration_sec(path: Path) -> float:
    ffprobe_path = '/usr/bin/ffprobe' if os.path.exists('/usr/bin/ffprobe') else 'ffprobe'
    try:
        result = subprocess.run(
            [ffprobe_path, '-v', 'error', '-show_entries', 'format=duration', '-of',
             'default=noprint_wrappers=1:nokey=1', str(path)],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return 0.0


def _compute_smoothing(duration_sec: float) -> int:
    if duration_sec <= 60:
        return 40
    if duration_sec <= 180:
        return 55
    if duration_sec <= 600:
        return 70
    return 90

def stabilize_video(input_path: Path, output_path: Path = None, progress_callback=None, timeout: int = FFMPEG_STAB_TIMEOUT,
                    dynamic_smoothing: bool = True):
    """
    Высокоскоростная стабилизация видео с использованием FFmpeg (vid.stab).
    Это в 5-10 раз быстрее, чем решение на чистом Python.
    
    Также выполняет автоматическое уменьшение разрешения до 720p для 
    ускорения последующего SLAM-анализа.
    """
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_stabilized.mp4"

    logger.info(f"🚀 Запуск аппаратной стабилизации: {input_path.name}")
    
    # Путь к ffmpeg
    ffmpeg_path = '/usr/bin/ffmpeg'
    if not os.path.exists(ffmpeg_path):
        ffmpeg_path = 'ffmpeg'

    # Временный файл для векторов движения
    transforms_path = input_path.parent / f"{input_path.stem}_transforms.trf"
    
    try:
        duration_sec = _video_duration_sec(input_path) if dynamic_smoothing else 0
        smoothing_val = _compute_smoothing(duration_sec) if duration_sec > 0 else 45

        # ШАГ 1: Анализ движения (Pass 1)
        # Мы также уменьшаем разрешение до 720p для ускорения всех последующих этапов
        if progress_callback:
            progress_callback(10)
            
        logger.info("Шаг 1: Анализ векторов движения...")
        subprocess.run([
            ffmpeg_path, '-v', 'error', '-i', str(input_path),
            '-vf', 'scale=-2:720,vidstabdetect=shakiness=10:accuracy=18:result=' + str(transforms_path),
            '-f', 'null', '-'
        ], check=True, timeout=timeout)

        if progress_callback:
            progress_callback(50)

        # ШАГ 2: Применение стабилизации (Pass 2)
        logger.info("Шаг 2: Применение стабилизации и рендеринг...")
        subprocess.run([
            ffmpeg_path, '-v', 'error', '-i', str(input_path),
            '-vf', f'scale=-2:720,vidstabtransform=smoothing={smoothing_val}:tripod=1:input={str(transforms_path)},unsharp=5:5:0.8:3:3:0.4',
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '21',
            '-threads', '0',
            '-c:a', 'copy', '-y', str(output_path)
        ], check=True, timeout=timeout)

        # Очистка
        if transforms_path.exists():
            transforms_path.unlink()

        if progress_callback:
            progress_callback(100)

        logger.info(f"✅ Стабилизация успешно завершена: {output_path.name}")
        return output_path

    except Exception as e:
        logger.error(f"❌ Ошибка при FFmpeg стабилизации: {e}")
        if transforms_path.exists():
            transforms_path.unlink()
        # Возвращаем оригинал в случае неудачи
        return input_path
