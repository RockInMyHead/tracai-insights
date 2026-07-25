# TrackAI - Анализ траектории движения

Интеллектуальная система анализа траектории движения с использованием компьютерного зрения и машинного обучения.

## Воспроизводимый эталон траектории

Перед изменением алгоритмов сохраните все данные проблемного запуска:

```bash
python3 backend/tools/capture_trajectory_baseline.py \
  --r3-output /path/to/r3_output \
  --analysis /path/to/video_analysis.json \
  --map-context /path/to/map_context.json \
  --ground-truth /path/to/operator_route.json \
  --output /path/to/new-baseline-directory
```

Map context должен содержать точные `reference_point`, `direction_point`,
`floorplan_id` и рисунок плана, использованные при запуске. Эталонный маршрут
передаётся отдельно как массив точек либо объект:

```json
{"trajectory": [[2222.6, 684.2], [2100.0, 700.0], [1950.0, 850.0]]}
```

Команда никогда не перезаписывает выходной каталог. Она копирует camera poses,
pose graph, robust/scale candidates, frame selection, run parameters, итоговый
analysis, операторские точки и эталонный маршрут. `pose_conf.npy` и
`pose_edge_log.json` добавляются при наличии. В `manifest.json` сохраняются
SHA-256 и размер каждого файла.

`trajectory_report.json` содержит одинаковые метрики для `raw`,
`robust_candidate`, `scale_aware_candidate` и `final_map`. Метрики R3 остаются
в единицах реконструкции, а `final_map` переводится в метры при наличии
`meters_per_pixel`. Отклонение от raw вычисляется после arc-length resampling
и одной best-fit 2D similarity без отражения.

Общие геометрические проверки находятся в
`backend/trajectory_geometry.py`:

```python
from trajectory_geometry import (
    trajectory_metrics,
    compare_trajectories,
    trajectory_acceptance,
)

metrics = trajectory_metrics(points_2d_or_3d)
comparison = compare_trajectories(raw, candidate)
decision = trajectory_acceptance(raw, candidate, {
    "verified_loop_closure": False,
    "thresholds": {"maximum_normalized_frechet": 0.08},
})
```

`compare_trajectories()` использует одну similarity без reflection и сообщает
нормированные Fréchet/Chamfer distance, согласованность поворотов и локального
направления, искажение длин сегментов, endpoint/span ratios и расхождение
распределений кривизны. `trajectory_acceptance()` возвращает все причины
отклонения и фактически использованные пороги.

## 🚀 Возможности

- **Анализ траектории движения** из видео файлов
- **Программная стабилизация** видео для точного анализа
- **Множественная обработка** видео с разными пользователями
- **Визуализация траекторий** на интерактивной карте
- **Поддержка планов помещений** (изображения и PDF)
- **Админ-панель** для мониторинга предприятия
- **Экспорт результатов** анализа

## 🛠️ Технологии

- **Frontend:** React, TypeScript, Vite
- **UI Framework:** shadcn/ui, Tailwind CSS
- **Backend:** FastAPI, Python
- **Computer Vision:** OpenCV, Video Tracker
- **Database:** JSON-based storage

## 📦 Установка и запуск

### Требования
- Node.js 18+
- Python 3.8+
- npm или yarn

### Frontend

```bash
# Установка зависимостей
npm install

# Запуск в режиме разработки
npm run dev

# Сборка для продакшена
npm run build
```

### Backend

```bash
# Переход в директорию backend
cd backend

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

# Запуск сервера
python -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

## 🎯 Использование

1. **Загрузите план помещения** (изображение или PDF)
2. **Установите точку отсчета** кликом на плане
3. **Добавьте видео** пользователей с именами
4. **Запустите анализ** траекторий
5. **Просмотрите результаты** на интерактивной карте

## 📁 Структура проекта

```
trackAI/
├── src/
│   ├── components/     # React компоненты
│   ├── pages/         # Страницы приложения
│   ├── lib/           # Утилиты и API клиенты
│   └── hooks/         # Пользовательские хуки
├── backend/
│   ├── main.py        # FastAPI приложение
│   └── video_tracker/ # Модуль анализа видео
├── public/            # Статические файлы
└── docs/              # Документация
```
