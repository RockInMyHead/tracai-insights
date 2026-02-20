import cv2
import numpy as np
import json
import time
import logging
from pathlib import Path
from matplotlib.gridspec import GridSpecFromSubplotSpec

# ИСПРАВЛЕННЫЙ ИМПОРТ
from video_tracker.src.slam_wrapper import HighAccuracyVisualOdometry

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_video_info(video_path):
    """Получение информации о видео"""
    cap = cv2.VideoCapture(video_path)
    
    # Проверка, что видео успешно открыто
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Не удалось открыть видеофайл: {video_path}. Возможно, неподдерживаемый кодек или поврежденный файл.")
    
    # Проверка, что можно прочитать хотя бы один кадр
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise ValueError(f"Не удалось прочитать кадры из видео: {video_path}. Файл может быть поврежден или использовать неподдерживаемый кодек.")
    
    # Возвращаемся к началу
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)) if cap.get(
            cv2.CAP_PROP_FPS) > 0 else 0
    }
    cap.release()
    
    # Дополнительная проверка валидности данных
    if info['width'] == 0 or info['height'] == 0:
        raise ValueError(f"Некорректные параметры видео: {video_path} (ширина или высота = 0)")
    
    return info


class FullFeatureProcessor:
    """Полнофункциональный процессор видео с повышенной точностью"""

    def __init__(self, input_dir, output_dir, scale_factor=12.306, progress_callback=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback

        # ИСПРАВЛЕННАЯ ИНИЦИАЛИЗАЦИЯ
        self.vo = HighAccuracyVisualOdometry(scale_factor=scale_factor)

        logger.info(f"Инициализирован FullFeatureProcessor с scale_factor={scale_factor}")

    def _calculate_distance(self, trajectory):
        """Вычисление пройденной дистанции"""
        if len(trajectory) < 2:
            return 0.0

        distance = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i - 1][0]
            dy = trajectory[i][1] - trajectory[i - 1][1]
            dz = trajectory[i][2] - trajectory[i - 1][2]
            segment_distance = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            distance += segment_distance

        return distance

    def set_scale_factor(self, scale_factor):
        """Изменение масштаба во время работы"""
        self.vo.set_scale_factor(scale_factor)
        logger.info(f"Установлен scale_factor={scale_factor}")

    def process_video(self, video_path):
        """Обработка конкретного видеофайла"""
        start_time = time.time()
        video_path = Path(video_path)

        if not video_path.exists():
            logger.error(f"Файл не найден: {video_path}")
            return None

        logger.info(f"🚀 Начало обработки: {video_path.name}")

        # Получаем информацию о видео
        try:
            video_info = get_video_info(str(video_path))
        except ValueError as e:
            logger.error(str(e))
            return None
        
        logger.info(
            f"📹 Информация о видео: {video_info['width']}x{video_info['height']}, {video_info['fps']:.1f} FPS, {video_info['duration']:.1f} сек")

        # Обработка видео
        cap = cv2.VideoCapture(str(video_path))
        
        # Проверка открытия
        if not cap.isOpened():
            logger.error(f"Не удалось открыть видео: {video_path}")
            return None
        
        frame_skip = 3  # Обрабатываем каждый 3-й кадр
        frame_count = 0
        frames_processed = 0

        print(f"⏳ Обработка видео...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                self.vo.process_frame(frame)
                frames_processed += 1

            frame_count += 1

            # Прогресс для длинных видео
            if frame_count % 20 == 0:
                progress = (frame_count / video_info['frame_count']) * 100
                logger.info(f"📊 Прогресс: {frame_count}/{video_info['frame_count']} кадров ({progress:.1f}%)")
                if self.progress_callback:
                    self.progress_callback(progress)

        cap.release()
        
        # Проверка, что обработаны кадры
        if frames_processed == 0:
            logger.error(f"Не удалось обработать ни одного кадра из видео: {video_path}")
            return None

        # Получаем результаты
        trajectory = self.vo.get_trajectory()
        turn_points = self.vo.get_turn_points()
        stats = self.vo.get_statistics()

        # Формируем результат
        result = {
            "method": "advanced_vo_scaled",
            "trajectory": trajectory,
            "turn_points": turn_points,
            "frame_count": frame_count,
            "trajectory_points": len(trajectory),
            "processing_stats": stats,
            "total_processing_time": time.time() - start_time,
            "video_info": video_info
        }

        # Сохраняем результаты
        self._save_detailed_results(video_path, result)

        logger.info(f"✅ Обработка завершена: {video_path.name}")
        logger.info(f"📊 Результаты: {result['trajectory_points']} точек траектории")
        logger.info(f"📏 Дистанция: {stats['estimated_distance']:.2f} единиц (масштаб: {stats['scale_factor']})")
        logger.info(f"🔄 Обнаружено поворотов: {len(turn_points)}")

        return result

    def _save_detailed_results(self, video_path, result):
        """Сохранение детализированных результатов с информацией о поворотах"""

        # Подготовка данных о поворотах
        turn_data = []
        for turn in result["turn_points"]:
            turn_data.append({
                "frame_index": turn["frame_index"],
                "trajectory_index": turn["trajectory_index"],
                "angle_degrees": turn["angle_degrees"],
                "position": {
                    "x": round(turn["position"][0], 4),
                    "y": round(turn["position"][1], 4),
                    "z": round(turn["position"][2], 4)
                },
                "turn_type": turn["turn_type"]
            })

        output_data = {
            "analysis_info": {
                "camera_id": video_path.stem,
                "video_file": str(video_path),
                "processing_method": result["method"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "2.0"
            },
            "video_statistics": {
                "total_frames": result["frame_count"],
                "trajectory_points": result["trajectory_points"],
                "estimated_distance": round(result["processing_stats"]["estimated_distance"], 3),
                "total_processing_time": round(result["total_processing_time"], 2),
                "processing_fps": round(result["processing_stats"].get('fps', 0), 1),
                "scale_factor": result["processing_stats"]["scale_factor"],
                "turns_detected": len(result["turn_points"])
            },
            "trajectory_data": {
                "points": [{"x": round(p[0], 4), "y": round(p[1], 4), "z": round(p[2], 4)}
                           for p in result["trajectory"]]
            },
            "turn_analysis": {
                "turns": turn_data,
                "total_turns": len(turn_data)
            },
            "processing_details": result["processing_stats"]
        }

        # Сохраняем JSON
        output_path = self.output_dir / f"{video_path.stem}_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Создаем улучшенную визуализацию
        # === ПОВОРАЧИВАЕМ ТРАЕКТОРИЮ ТАК, ЧТОБЫ СТАРТ СМОТРЕЛ ВВЕРХ (0,1) ===
        trajectory = result["trajectory"]
        turn_points = result["turn_points"]

        if len(trajectory) > 50:
            # берём вектор от старта к 50-й точке (чтобы не брать шум на первых кадрах)
            dx = trajectory[50][0] - trajectory[0][0]
            dy = trajectory[50][1] - trajectory[0][1]

            # угол поворота всей траектории, чтобы этот вектор стал (0,1)
            angle_rad = np.arctan2(dx, dy)  # atan2(x, y) → угол от вектора (dy, dx) к (0,1)
            cos_a = np.cos(-angle_rad)
            sin_a = np.sin(-angle_rad)

            # поворачиваем все точки траектории
            rotated_traj = []
            for p in trajectory:
                rx = p[0] * cos_a - p[1] * sin_a
                ry = p[0] * sin_a + p[1] * cos_a
                rotated_traj.append([rx, ry, p[2]])

            # поворачиваем точки поворотов тоже
            rotated_turns = []
            for t in turn_points:
                rx = t['position'][0] * cos_a - t['position'][1] * sin_a
                ry = t['position'][0] * sin_a + t['position'][1] * cos_a
                new_pos = t['position'].copy()
                new_pos[0], new_pos[1] = rx, ry
                new_t = t.copy()
                new_t['position'] = new_pos
                rotated_turns.append(new_t)
        else:
            rotated_traj = trajectory
            rotated_turns = turn_points

        # теперь рисуем уже повёрнутую траекторию
        self._create_enhanced_visualization(rotated_traj, rotated_turns, video_path.stem)

        logger.info(f"💾 Результаты сохранены: {output_path}")
        print(f"💾 Результаты сохранены: {output_path}")

    def _create_enhanced_visualization(self, trajectory, turn_points, video_name):
        """Создание ИДЕАЛЬНО КВАДРАТНОЙ визуализации траектории с легендой снаружи"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.gridspec import GridSpec

            # Данные
            x = [p[0] for p in trajectory]
            y = [p[1] for p in trajectory]

            if not x or not y:
                print("Траектория пуста — график не создан")
                return

            # Вычисляем максимальный диапазон, чтобы сделать график квадратным
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            range_x = x_max - x_min
            range_y = y_max - y_min
            max_range = max(range_x, range_y, 1.0)  # избежим деления на 0

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            # Создаём квадратную фигуру
            fig = plt.figure(figsize=(14, 14))
            gs = GridSpec(2, 2, width_ratios=[7, 3], height_ratios=[3, 1], wspace=0.3, hspace=0.4)

            # Основной график — квадратный
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
                    ax1.annotate(str(i), (tx, ty),
                                 xytext=(8, 8), textcoords='offset points',
                                 fontsize=11, fontweight='bold',
                                 bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", lw=1))

            # Делаем оси строго квадратными
            ax1.set_xlim(center_x - max_range/2 - max_range*0.05, center_x + max_range/2 + max_range*0.05)
            ax1.set_ylim(center_y - max_range/2 - max_range*0.05, center_y + max_range/2 + max_range*0.05)
            ax1.set_aspect('equal', adjustable='box')

            ax1.grid(True, alpha=0.4, linestyle='--')
            ax1.set_xlabel('X (метры)', fontsize=12)
            ax1.set_ylabel('Y (метры)', fontsize=12)
            ax1.set_title(f'Траектория движения: {video_name}', fontsize=14, fontweight='bold', pad=20)

            # Легенда и информация справа
            ax_info = plt.subplot(gs[0, 1])
            ax_info.axis('off')

            total_distance = self._calculate_distance(trajectory)
            info_text = f"Дистанция: {total_distance:.1f} м\n"
            info_text += f"Точек траектории: {len(trajectory)}\n"
            info_text += f"Обнаружено поворотов: {len(turn_points)}"

            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=13,
                         verticalalignment='top', bbox=dict(boxstyle="round,pad=1", fc="lightblue", alpha=0.9))

            legend_elements = [
                plt.Line2D([0], [0], color='blue', lw=4, label='Траектория'),
                plt.Line2D([0], [0], marker='o', color='green', markersize=12, linestyle='None', label='Старт'),
                plt.Line2D([0], [0], marker='o', color='red', markersize=12, linestyle='None', label='Финиш')
            ]
            if turn_points:
                left_cnt = sum(1 for t in turn_points if t['turn_type'] == 'left')
                right_cnt = len(turn_points) - left_cnt
                legend_elements += [
                    plt.Line2D([0], [0], marker='o', color='orange', markersize=12, linestyle='None', label=f'Левые ({left_cnt})'),
                    plt.Line2D([0], [0], marker='o', color='purple', markersize=12, linestyle='None', label=f'Правые ({right_cnt})')
                ]

            ax_info.legend(handles=legend_elements, loc='center left', fontsize=12, frameon=True, fancybox=True, shadow=True)

            # Гистограмма поворотов снизу
            ax2 = plt.subplot(gs[1, :])
            if turn_points:
                nums = list(range(1, len(turn_points) + 1))
                angles = [t['angle_degrees'] for t in turn_points]
                colors = ['orange' if t['turn_type'] == 'left' else 'purple' for t in turn_points]
                bars = ax2.bar(nums, angles, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
                for bar, ang in zip(bars, angles):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                             f'{ang:.0f}°', ha='center', va='bottom', fontweight='bold')
                ax2.set_xlabel('Номер поворота')
                ax2.set_ylabel('Угол (°)')
                ax2.set_title('Углы поворотов', fontweight='bold')
                ax2.grid(True, axis='y', alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Повороты не обнаружены', ha='center', va='center',
                         transform=ax2.transAxes, fontsize=14, fontweight='bold')

            plt.tight_layout()

            plot_path = self.output_dir / f"{video_name}_trajectory_square.png"
            plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            print(f"Квадратный график сохранён: {plot_path.name}")

        except Exception as e:
            print(f"Ошибка при создании графика: {e}")

    def _calculate_grid_step(self, range_size):
        """Рассчитывает оптимальный шаг сетки"""
        if range_size > 200:
            return 50
        elif range_size > 100:
            return 20
        elif range_size > 50:
            return 10
        elif range_size > 20:
            return 5
        elif range_size > 10:
            return 2
        else:
            return 1

    def _create_text_report(self, trajectory, turn_points, video_name):
        """Создание текстового отчета о траектории"""
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
            f.write(f"🔄 Обнаружено поворотов: {len(turn_points)}\n\n")

            f.write("🧭 ДЕТАЛИЗИРОВАННАЯ ТРАЕКТОРИЯ:\n")
            f.write("-" * 60 + "\n")

            # Анализ общего направления
            total_dx = trajectory[-1][0] - trajectory[0][0]
            total_dy = trajectory[-1][1] - trajectory[0][1]

            if abs(total_dx) > abs(total_dy):
                main_direction = "Запад" if total_dx < 0 else "Восток"
            else:
                main_direction = "Юг" if total_dy < 0 else "Север"

            f.write(f"Основное направление: {main_direction}\n")
            f.write(f"Смещение: {abs(total_dx):.1f} м по X, {abs(total_dy):.1f} м по Y\n\n")

            # Анализ поворотов
            if turn_points:
                f.write("🔄 ОБНАРУЖЕННЫЕ ПОВОРОТЫ:\n")
                f.write("-" * 60 + "\n")

                for i, turn in enumerate(turn_points, 1):
                    # Вычисляем расстояние от начала до этого поворота
                    dist_to_turn = self._calculate_distance(trajectory[:turn['trajectory_index'] + 1])

                    f.write(f"Поворот {i}:\n")
                    f.write(f"  • Тип: {'↰ Левый' if turn['turn_type'] == 'left' else '↱ Правый'}\n")
                    f.write(f"  • Угол: {abs(turn['angle_degrees']):.1f}°\n")
                    f.write(f"  • Координаты: ({turn['position'][0]:.1f}, {turn['position'][1]:.1f}) м\n")
                    f.write(f"  • Пройдено до поворота: {dist_to_turn:.1f} м\n")

                    # Определяем направление после поворота
                    if i < len(turn_points):
                        next_turn = turn_points[i]
                        dx = next_turn['position'][0] - turn['position'][0]
                        dy = next_turn['position'][1] - turn['position'][1]
                    else:
                        dx = trajectory[-1][0] - turn['position'][0]
                        dy = trajectory[-1][1] - turn['position'][1]

                    # Определяем направление
                    if abs(dx) > abs(dy):
                        direction = "Запад" if dx < 0 else "Восток"
                    else:
                        direction = "Юг" if dy < 0 else "Север"

                    f.write(f"  • Направление после: {direction}\n")
                    f.write("\n")

            # Статистика по квадрантам
            f.write("📈 СТАТИСТИКА ПО КВАДРАНТАМ:\n")
            f.write("-" * 60 + "\n")

            quadrants = {"I": 0, "II": 0, "III": 0, "IV": 0}  # счетчики точек

            for point in trajectory:
                x, y = point[0], point[1]
                if x >= 0 and y >= 0:
                    quadrants["I"] += 1
                elif x < 0 and y >= 0:
                    quadrants["II"] += 1
                elif x < 0 and y < 0:
                    quadrants["III"] += 1
                else:
                    quadrants["IV"] += 1

            total_points = len(trajectory)
            for quad, count in quadrants.items():
                percentage = (count / total_points) * 100
                f.write(f"Квадрант {quad}: {count} точек ({percentage:.1f}%)\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("🎯 ВЫВОДЫ:\n")
            f.write("=" * 60 + "\n")

            # Основные выводы
            f.write(f"• Маршрут составляет {total_distance:.1f} метров\n")
            f.write(f"• Начинается в точке ({trajectory[0][0]:.1f}, {trajectory[0][1]:.1f})\n")
            f.write(f"• Заканчивается в точке ({trajectory[-1][0]:.1f}, {trajectory[-1][1]:.1f})\n")

            if turn_points:
                left_turns = sum(1 for t in turn_points if t['turn_type'] == 'left')
                right_turns = len(turn_points) - left_turns
                f.write(f"• Совершено {left_turns} левых и {right_turns} правых поворотов\n")

                avg_turn_angle = sum(abs(t['angle_degrees']) for t in turn_points) / len(turn_points)
                f.write(f"• Средний угол поворота: {avg_turn_angle:.1f}°\n")

            f.write(f"• Основное направление движения: {main_direction}\n")

            # Определяем самый активный квадрант
            main_quadrant = max(quadrants.items(), key=lambda x: x[1])[0]
            quadrant_names = {"I": "северо-восток", "II": "северо-запад",
                              "III": "юго-запад", "IV": "юго-восток"}
            f.write(f"• Основная зона движения: {quadrant_names[main_quadrant]}\n")

        print(f"📄 Текстовый отчет сохранен: {report_path}")
        logger.info(f"Текстовый отчет сохранен: {report_path}")