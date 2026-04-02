# Модель детектора для ML ROI

Файл **detector.onnx** используется для ML ROI (выделение области человека в кадре).

- Поддерживается формат YOLOv8 ONNX: один выход `(1, 84, N)` — 4 координаты + 80 классов COCO (person = класс 0).
- Рекомендуемая модель: **YOLOv8s** (точнее, чем YOLOv8n). Экспорт из репозитория:
  ```bash
  python backend/scripts/export_detector_onnx.py
  ```
  Скрипт скачает `yolov8s.pt`, экспортирует в ONNX и сохранит как `backend/models/detector.onnx`.
- Если файла нет, трекер работает без ML ROI (используется весь кадр).

Файл должен называться: `detector.onnx`
