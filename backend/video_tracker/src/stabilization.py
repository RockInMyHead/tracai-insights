import cv2
from pathlib import Path
import logging
import subprocess
import os

logger = logging.getLogger(__name__)

FFMPEG_STAB_TIMEOUT = 1800  # 30 минут для больших видео

def stabilize_video(input_path: Path, output_path: Path = None, progress_callback=None, timeout: int = FFMPEG_STAB_TIMEOUT):
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
        # ШАГ 1: Анализ движения (Pass 1)
        # Мы также уменьшаем разрешение до 720p для ускорения всех последующих этапов
        if progress_callback:
            progress_callback(10)
            
        logger.info("Шаг 1: Анализ векторов движения...")
        subprocess.run([
            ffmpeg_path, '-v', 'error', '-i', str(input_path),
            '-vf', 'scale=-2:720,vidstabdetect=shakiness=10:accuracy=15:result=' + str(transforms_path),
            '-f', 'null', '-'
        ], check=True, timeout=timeout)

        if progress_callback:
            progress_callback(50)

        # ШАГ 2: Применение стабилизации (Pass 2)
        logger.info("Шаг 2: Применение стабилизации и рендеринг...")
        subprocess.run([
            ffmpeg_path, '-v', 'error', '-i', str(input_path),
            '-vf', f'scale=-2:720,vidstabtransform=smoothing=30:input={str(transforms_path)},unsharp=5:5:0.8:3:3:0.4',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22',
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
