import cv2
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Any

from video_tracker.src.slam_wrapper import HighAccuracyVisualOdometry, EnhancedVisualOdometry
from video_tracker.src.calibration import CameraCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/processing.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _to_json_serializable(obj: Any):
    """
    Приводит numpy-типы и другие вложенные структуры к стандартным JSON-совместимым типам.
    Нужен, чтобы избежать ошибок вида \"Object of type bool is not JSON serializable\".
    """
    # Numpy scalars / arrays
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Контейнеры
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_serializable(v) for v in obj]

    return obj


def get_video_info(video_path):
    """Получение информации о видео."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(
            f"Не удалось открыть видеофайл: {video_path}. Возможно, неподдерживаемый кодек или поврежденный файл."
        )

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise ValueError(
            f"Не удалось прочитать кадры из видео: {video_path}. Файл может быть поврежден или использовать неподдерживаемый кодек."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': fps,
        'frame_count': frame_count,
        'duration': (frame_count / fps) if fps > 0 else 0.0,
    }
    cap.release()

    if info['width'] == 0 or info['height'] == 0:
        raise ValueError(f"Некорректные параметры видео: {video_path} (ширина или высота = 0)")

    return info


class FullFeatureProcessor:
    """Полнофункциональный процессор видео с улучшенным orchestration layer."""

    def __init__(
        self,
        input_dir,
        output_dir,
        scale_factor=12.306,
        progress_callback=None,
        use_homography=False,
        use_kalman=False,
        use_akaze=False,
        calibration_path=None,
        frame_skip: int = 1,
        target_width: int = 900,
        use_optical_flow: bool = False,
        detect_interval: int = 3,
        turn_vote_threshold: int = 3,
        use_ml_roi: bool = True,
        ml_model_path: str = None,
        tracker_max_age: int = 8,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback
        self.frame_skip = max(1, int(frame_skip))
        self.target_width = max(160, int(target_width))

        self.use_homography = use_homography
        self.use_kalman = use_kalman
        self.use_akaze = use_akaze
        self.use_optical_flow = use_optical_flow
        self.detect_interval = detect_interval
        self.turn_vote_threshold = turn_vote_threshold
        self.use_ml_roi = use_ml_roi
        self.tracker_max_age = tracker_max_age
        self.ml_model_path = ml_model_path

        self.calibrator = None
        if calibration_path and Path(calibration_path).exists():
            self.calibrator = CameraCalibrator.load_calibration(calibration_path)
            logger.info("Калибровка загружена: %s", calibration_path)

        if self.ml_model_path is None:
            self.ml_model_path = str(
                Path(__file__).resolve().parent.parent.parent / "models" / "detector.onnx"
            )

        self.vo = self._create_vo(scale_factor)
        logger.info("Инициализирован FullFeatureProcessor с scale_factor=%s", scale_factor)

    def _create_vo(self, scale_factor: float):
        """Единая фабрика VO без разъезда preset-веток."""
        if self.use_homography or self.use_kalman or self.use_akaze or self.calibrator or self.use_optical_flow:
            vo = EnhancedVisualOdometry(
                scale_factor=scale_factor,
                use_homography=self.use_homography,
                use_kalman=self.use_kalman,
                use_akaze=self.use_akaze,
                calibrator=self.calibrator,
                use_optical_flow=self.use_optical_flow,
                detect_interval=self.detect_interval,
                turn_vote_threshold=self.turn_vote_threshold,
                use_ml_roi=self.use_ml_roi,
                ml_model_path=self.ml_model_path,
                tracker_max_age=self.tracker_max_age,
            )
            logger.info(
                "Инициализирована EnhancedVisualOdometry: homography=%s, kalman=%s, akaze=%s, optical_flow=%s, ml_roi=%s",
                self.use_homography,
                self.use_kalman,
                self.use_akaze,
                self.use_optical_flow,
                self.use_ml_roi,
            )
            return vo

        logger.info("Используется compatibility preset HighAccuracyVisualOdometry")
        return HighAccuracyVisualOdometry(scale_factor=scale_factor, ml_model_path=self.ml_model_path)

    def _calculate_distance(self, trajectory):
        if len(trajectory) < 2:
            return 0.0
        distance = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i - 1][0]
            dy = trajectory[i][1] - trajectory[i - 1][1]
            distance += float((dx ** 2 + dy ** 2) ** 0.5)
        return distance

    def set_scale_factor(self, scale_factor):
        self.vo.set_scale_factor(scale_factor)
        logger.info("Установлен scale_factor=%s", scale_factor)

    def process_video(self, video_path):
        """Обработка конкретного видеофайла."""
        start_time = time.time()
        video_path = Path(video_path)

        if not video_path.exists():
            logger.error("Файл не найден: %s", video_path)
            return None

        logger.info("🚀 Начало обработки: %s", video_path.name)

        try:
            video_info = get_video_info(str(video_path))
        except ValueError as e:
            logger.error(str(e))
            return None

        logger.info(
            "📹 Информация о видео: %sx%s, %.2f FPS, %.2f сек",
            video_info['width'],
            video_info['height'],
            video_info['fps'],
            video_info['duration'],
        )

        # ВАЖНО: отдельный lifecycle на каждое видео
        self.vo.reset()
        if hasattr(self.vo, 'set_video_context'):
            self.vo.set_video_context(fps=video_info['fps'], frame_skip=self.frame_skip)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Не удалось открыть видео: %s", video_path)
            return None

        frame_skip = self.frame_skip
        frame_count = 0
        frames_processed = 0
        fps = float(video_info.get('fps') or 0.0)
        effective_input_fps = (fps / frame_skip) if fps > 1e-6 else None
        dt_sec = (frame_skip / fps) if fps > 1e-6 else 1.0

        print("⏳ Обработка видео...")
        debug_samples = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[1] > self.target_width:
                scale = self.target_width / frame.shape[1]
                frame = cv2.resize(frame, (self.target_width, int(frame.shape[0] * scale)))

            if frame_count % frame_skip == 0:
                timestamp_sec = (frame_count / fps) if fps > 1e-6 else None
                self.vo.process_frame(
                    frame,
                    frame_index=frame_count,
                    timestamp_sec=timestamp_sec,
                    dt_sec=dt_sec,
                    dt_frames=frame_skip,
                )
                frames_processed += 1

                if hasattr(self.vo, 'get_last_frame_debug') and (frames_processed % 30 == 0):
                    debug_samples.append(self.vo.get_last_frame_debug())

            frame_count += 1

            if frame_count % 20 == 0:
                progress = (frame_count / max(video_info['frame_count'], 1)) * 100
                logger.info("📊 Прогресс: %s/%s кадров (%.1f%%)", frame_count, video_info['frame_count'], progress)
                if self.progress_callback:
                    self.progress_callback(progress)

        cap.release()

        if frames_processed == 0:
            logger.error(
                "Не удалось обработать ни одного кадра из видео: %s (всего кадров в метаданных: %s)",
                video_path,
                video_info.get('frame_count', 0),
            )
            raise ValueError(
                "Не удалось прочитать кадры из видео. Если включена стабилизация — попробуйте отключить её и запустить анализ снова. "
                "Иначе проверьте формат файла (рекомендуется MP4 с кодеком H.264)."
            )

        trajectory = self.vo.get_trajectory()
        raw_turn_points = self.vo.get_raw_turn_points()
        trajectory_turn_points = self.vo.get_trajectory_turn_points()
        turn_points = raw_turn_points
        stats = self.vo.get_statistics()

        degraded_flags = {
            "low_motion_output": stats.get("estimated_distance", 0) < 1.0,
            "high_ransac_failure": stats.get("ransac_failure_rate", 0) > 0.35,
            "high_gating_failure": stats.get("gating_failure_rate", 0) > 0.35,
            "high_sign_conflict": stats.get("sign_conflict_frames", 0) > 25,
            "turn_conflict_heavy": stats.get("turn_conflict_count", 0) > 5,
        }

        method_name = "enhanced_vo"
        if self.use_homography:
            method_name += "_homography"
        if self.use_kalman:
            method_name += "_kalman"
        if self.use_akaze:
            method_name += "_akaze"
        if self.calibrator:
            method_name += "_calibrated"
        if self.use_optical_flow:
            method_name += "_flow"

        result = {
            "method": method_name,
            "trajectory": trajectory,
            "turn_points": turn_points,
            "raw_turn_points": raw_turn_points,
            "trajectory_turn_points": trajectory_turn_points,
            "frame_count": frame_count,
            "trajectory_points": len(trajectory),
            "processing_stats": stats,
            "total_processing_time": time.time() - start_time,
            "video_info": video_info,
            "execution_context": {
                "frame_skip": frame_skip,
                "target_width": self.target_width,
                "processed_frame_count": frames_processed,
                "raw_video_frame_count": frame_count,
                "effective_input_fps": effective_input_fps,
                "video_fps": fps,
                "dt_sec": dt_sec,
            },
            "degraded_flags": degraded_flags,
            "debug_samples": debug_samples,
            "enhancement_features": {
                "homography": self.use_homography,
                "kalman_filter": self.use_kalman,
                "akaze": self.use_akaze,
                "calibration": self.calibrator is not None,
                "optical_flow": self.use_optical_flow,
                "ml_roi": self.use_ml_roi,
            },
        }

        self._save_detailed_results(video_path, result)

        logger.info("✅ Обработка завершена: %s", video_path.name)
        logger.info("📊 Результаты: %s точек траектории", result['trajectory_points'])
        logger.info("📏 Дистанция: %.2f единиц (масштаб: %s)", stats['estimated_distance'], stats['scale_factor'])
        logger.info("🔄 Обнаружено поворотов: %s", len(turn_points))
        logger.info(
            "VO summary | processed=%s/%s | proc_fps=%.2f | dist=%.2f | turns=%s | ransac_fail=%.3f | gating_fail=%.3f | sign_conflicts=%s | pose_mode=%s",
            frames_processed,
            frame_count,
            stats.get('fps', 0),
            stats.get('estimated_distance', 0),
            len(raw_turn_points),
            stats.get('ransac_failure_rate', 0),
            stats.get('gating_failure_rate', 0),
            stats.get('sign_conflict_frames', 0),
            stats.get('pose_estimation_mode', 'unknown'),
        )

        return result

    def _save_detailed_results(self, video_path, result):
        """Сохранение детализированных результатов с корректной сериализацией heading."""

        def _build_debug_fields(res):
            stats = res.get("processing_stats") or {}
            if res.get("map_turn_points") is not None:
                turn_angle_source = "map_segments"
            elif res.get("trajectory_turn_points"):
                turn_angle_source = "hybrid"
            else:
                turn_angle_source = "integrated_yaw"
            return {
                "yaw_sign_used": stats.get("yaw_sign_used", 1.0),
                "turn_detection_mode": stats.get("turn_detection_mode", "yaw_and_chord"),
                "turn_conflict_count": stats.get("turn_conflict_count", 0),
                "sign_conflict_frames": stats.get("sign_conflict_frames", 0),
                "pose_estimation_mode": stats.get("pose_estimation_mode", "unknown"),
                "turn_angle_source": turn_angle_source,
                "degraded_flags": res.get("degraded_flags", {}),
            }

        def _pos_heading(pos):
            if isinstance(pos, dict):
                return [pos.get("x", 0), pos.get("y", 0), pos.get("heading_deg", 0)]
            if not isinstance(pos, (list, tuple)):
                return [0, 0, 0]
            if len(pos) >= 3:
                return [pos[0], pos[1], pos[2]]
            if len(pos) == 2:
                return [pos[0], pos[1], 0]
            if len(pos) == 1:
                return [pos[0], 0, 0]
            return [0, 0, 0]

        def _turns_to_analysis(turns, default_key="turn_points"):
            lst = result.get(default_key) if turns is None else turns
            if not lst:
                return {"turns": [], "total_turns": 0}
            data = []
            for turn in lst:
                p = _pos_heading(turn.get("position"))
                data.append({
                    "frame_index": turn.get("frame_index", 0),
                    "trajectory_index": turn.get("trajectory_index", 0),
                    "angle_degrees": turn.get("angle_degrees", 0),
                    "position": {
                        "x": round(float(p[0]), 4),
                        "y": round(float(p[1]), 4),
                        "heading_deg": round(float(p[2]), 2),
                    },
                    "turn_type": turn.get("turn_type", "unknown"),
                    "source": turn.get("source"),
                })
            return {"turns": data, "total_turns": len(data)}

        turn_analysis = _turns_to_analysis(result["turn_points"], "turn_points")
        raw_turn_analysis = _turns_to_analysis(result.get("raw_turn_points"), "raw_turn_points")
        trajectory_turn_analysis = _turns_to_analysis(result.get("trajectory_turn_points"), "trajectory_turn_points")
        map_turn_analysis = _turns_to_analysis(result.get("map_turn_points"), "map_turn_points") if result.get("map_turn_points") is not None else None

        output_data = {
            "analysis_info": {
                "camera_id": video_path.stem,
                "video_file": str(video_path),
                "processing_method": result["method"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "2.1",
            },
            "video_statistics": {
                "total_frames": result["frame_count"],
                "processed_frames": result.get("execution_context", {}).get("processed_frame_count", 0),
                "trajectory_points": result["trajectory_points"],
                "estimated_distance": round(result["processing_stats"]["estimated_distance"], 3),
                "total_processing_time": round(result["total_processing_time"], 2),
                "processing_fps": round(result["processing_stats"].get('fps', 0), 1),
                "video_fps": round(result.get("video_info", {}).get("fps", 0), 2),
                "effective_input_fps": result.get("execution_context", {}).get("effective_input_fps"),
                "scale_factor": result["processing_stats"]["scale_factor"],
                "turns_detected": len(result["turn_points"]),
                "raw_turns_detected": len(result.get("raw_turn_points", [])),
                "trajectory_turns_detected": len(result.get("trajectory_turn_points", [])),
            },
            "trajectory_data": {
                "points": [
                    {
                        "x": round(float(p[0]), 4),
                        "y": round(float(p[1]), 4),
                        "heading_deg": round(float(p[2]), 2),
                    }
                    for p in result["trajectory"]
                ]
            },
            "turn_analysis": turn_analysis,
            "raw_turn_analysis": raw_turn_analysis,
            "trajectory_turn_analysis": trajectory_turn_analysis,
            "processing_details": result["processing_stats"],
            "execution_context": result.get("execution_context", {}),
            "debug": _build_debug_fields(result),
            "debug_samples": result.get("debug_samples", []),
        }
        if map_turn_analysis is not None:
            output_data["map_turn_analysis"] = map_turn_analysis
            output_data["video_statistics"]["map_turns_detected"] = map_turn_analysis["total_turns"]

        output_path = self.output_dir / f"{video_path.stem}_analysis.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                _to_json_serializable(output_data),
                f,
                indent=2,
                ensure_ascii=False,
                default=_to_json_serializable,
            )

        trajectory = result["trajectory"]
        turn_points = result["turn_points"]

        if len(trajectory) > 50:
            dx = trajectory[50][0] - trajectory[0][0]
            dy = trajectory[50][1] - trajectory[0][1]
            angle_rad = np.arctan2(dx, dy)
            cos_a = np.cos(-angle_rad)
            sin_a = np.sin(-angle_rad)

            rotated_traj = []
            for p in trajectory:
                rx = p[0] * cos_a - p[1] * sin_a
                ry = p[0] * sin_a + p[1] * cos_a
                rotated_traj.append([rx, ry, p[2]])

            rotated_turns = []
            for t in turn_points:
                pos = _pos_heading(t.get('position'))
                rx = pos[0] * cos_a - pos[1] * sin_a
                ry = pos[0] * sin_a + pos[1] * cos_a
                new_pos = pos.copy()
                new_pos[0], new_pos[1] = rx, ry
                new_t = t.copy()
                new_t['position'] = new_pos
                rotated_turns.append(new_t)
        else:
            rotated_traj = trajectory
            rotated_turns = turn_points

        self._create_enhanced_visualization(rotated_traj, rotated_turns, video_path.stem)

        logger.info("💾 Результаты сохранены: %s", output_path)
        print(f"💾 Результаты сохранены: {output_path}")

    def _create_enhanced_visualization(self, trajectory, turn_points, video_name):
        """Создание квадратной визуализации траектории."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec

            x = [p[0] for p in trajectory]
            y = [p[1] for p in trajectory]
            if not x or not y:
                print("Траектория пуста — график не создан")
                return

            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            range_x = x_max - x_min
            range_y = y_max - y_min
            max_range = max(range_x, range_y, 1.0)
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            fig = plt.figure(figsize=(14, 14))
            gs = GridSpec(2, 2, width_ratios=[7, 3], height_ratios=[3, 1], wspace=0.3, hspace=0.4)

            ax1 = plt.subplot(gs[0, 0])
            ax1.plot(x, y, 'b-', linewidth=2.5, alpha=0.8)
            ax1.plot(x[0], y[0], 'go', markersize=12, label='Старт')
            ax1.plot(x[-1], y[-1], 'ro', markersize=12, label='Финиш')

            if turn_points:
                turn_x = [t['position'][0] for t in turn_points]
                turn_y = [t['position'][1] for t in turn_points]
                colors = ['orange' if t['turn_type'] == 'left' else 'purple' for t in turn_points]
                ax1.scatter(turn_x, turn_y, c=colors, s=100, zorder=5, edgecolors='black', linewidth=1)
                for i, (tx, ty) in enumerate(zip(turn_x, turn_y), 1):
                    ax1.annotate(
                        str(i),
                        (tx, ty),
                        xytext=(8, 8),
                        textcoords='offset points',
                        fontsize=11,
                        fontweight='bold',
                        bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", lw=1),
                    )

            pad = max_range * 0.05
            ax1.set_xlim(center_x - max_range / 2 - pad, center_x + max_range / 2 + pad)
            ax1.set_ylim(center_y - max_range / 2 - pad, center_y + max_range / 2 + pad)
            ax1.set_aspect('equal', adjustable='box')
            ax1.grid(True, alpha=0.4, linestyle='--')
            ax1.set_xlabel('X (метры)', fontsize=12)
            ax1.set_ylabel('Y (метры)', fontsize=12)
            ax1.set_title(f'Траектория движения: {video_name}', fontsize=14, fontweight='bold', pad=20)

            ax_info = plt.subplot(gs[0, 1])
            ax_info.axis('off')
            total_distance = self._calculate_distance(trajectory)
            info_text = (
                f"Дистанция: {total_distance:.1f} м\n"
                f"Точек траектории: {len(trajectory)}\n"
                f"Обнаружено поворотов: {len(turn_points)}"
            )
            ax_info.text(
                0.05,
                0.95,
                info_text,
                transform=ax_info.transAxes,
                fontsize=13,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=1", fc="lightblue", alpha=0.9),
            )

            legend_elements = [
                plt.Line2D([0], [0], color='blue', lw=4, label='Траектория'),
                plt.Line2D([0], [0], marker='o', color='green', markersize=12, linestyle='None', label='Старт'),
                plt.Line2D([0], [0], marker='o', color='red', markersize=12, linestyle='None', label='Финиш'),
            ]
            if turn_points:
                left_cnt = sum(1 for t in turn_points if t['turn_type'] == 'left')
                right_cnt = len(turn_points) - left_cnt
                legend_elements += [
                    plt.Line2D([0], [0], marker='o', color='orange', markersize=12, linestyle='None', label=f'Левые ({left_cnt})'),
                    plt.Line2D([0], [0], marker='o', color='purple', markersize=12, linestyle='None', label=f'Правые ({right_cnt})'),
                ]
            ax_info.legend(handles=legend_elements, loc='center left', fontsize=12, frameon=True, fancybox=True, shadow=True)

            ax2 = plt.subplot(gs[1, :])
            if turn_points:
                nums = list(range(1, len(turn_points) + 1))
                angles = [t['angle_degrees'] for t in turn_points]
                colors = ['orange' if t['turn_type'] == 'left' else 'purple' for t in turn_points]
                bars = ax2.bar(nums, angles, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
                for bar, ang in zip(bars, angles):
                    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{ang:.0f}°', ha='center', va='bottom', fontweight='bold')
                ax2.set_xlabel('Номер поворота')
                ax2.set_ylabel('Угол (°)')
                ax2.set_title('Углы поворотов', fontweight='bold')
                ax2.grid(True, axis='y', alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Повороты не обнаружены', ha='center', va='center', transform=ax2.transAxes, fontsize=14, fontweight='bold')

            plt.tight_layout()
            plot_path = self.output_dir / f"{video_name}_trajectory_square.png"
            plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"Квадратный график сохранён: {plot_path.name}")
        except Exception as e:
            print(f"Ошибка при создании графика: {e}")

    def _create_text_report(self, trajectory, turn_points, video_name):
        """Создание текстового отчета о траектории."""
        report_path = self.output_dir / f"{video_name}_report.txt"
        total_distance = self._calculate_distance(trajectory)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("📊 ОТЧЕТ О ТРАЕКТОРИИ ДВИЖЕНИЯ\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"📹 Видеофайл: {video_name}\n")
            f.write(f"📏 Общее пройденное расстояние: {total_distance:.1f} м\n")
            f.write(f"📍 Начальная точка: ({trajectory[0][0]:.1f}, {trajectory[0][1]:.1f}) м\n")
            f.write(f"🎯 Конечная точка: ({trajectory[-1][0]:.1f}, {trajectory[-1][1]:.1f}) м\n")
            f.write(f"🧭 Начальный курс: {trajectory[0][2]:.1f}°\n")
            f.write(f"🎯 Конечный курс: {trajectory[-1][2]:.1f}°\n")
            f.write(f"🔄 Обнаружено поворотов: {len(turn_points)}\n\n")

        print(f"📄 Текстовый отчет сохранен: {report_path}")
        logger.info("Текстовый отчет сохранен: %s", report_path)
