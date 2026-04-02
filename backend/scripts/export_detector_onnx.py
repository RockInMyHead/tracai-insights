#!/usr/bin/env python3
"""
Экспорт YOLOv8s в ONNX для ML ROI (детекция человека, класс 0 в COCO).
Результат: backend/models/detector.onnx (совместим с ml_roi.MLDetector).
Требуется: pip install ultralytics (уже в requirements.txt).
"""
from pathlib import Path

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Установите ultralytics: pip install ultralytics")
        return 1

    # YOLOv8s — баланс точности и скорости (точнее, чем YOLOv8n)
    model_name = "yolov8s.pt"
    script_dir = Path(__file__).resolve().parent
    backend_dir = script_dir.parent
    models_dir = backend_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "detector.onnx"

    print(f"Загрузка {model_name}...")
    model = YOLO(model_name)

    print("Экспорт в ONNX (imgsz=640, simplify=True)...")
    # export() сохраняет рядом с моделью или в cwd; вернёт путь к .onnx
    exported = model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        dynamic=True,  # для multi-scale 640/960 в ml_roi
    )
    exported_path = Path(exported)
    if not exported_path.is_absolute():
        exported_path = Path.cwd() / exported_path

    if exported_path.resolve() != out_path.resolve():
        import shutil
        shutil.copy2(exported_path, out_path)
        print(f"Скопировано: {out_path}")
    else:
        print(f"Сохранено: {out_path}")

    print("Готово. Модель: backend/models/detector.onnx (YOLOv8s, класс person=0).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
