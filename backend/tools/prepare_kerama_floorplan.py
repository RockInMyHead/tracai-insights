#!/usr/bin/env python3
"""Build the immutable Kerama floor-plan image, obstacle mask and metadata.

Usage:
  python backend/tools/prepare_kerama_floorplan.py ORIGINAL.pdf MARKED.pdf

The marked PDF is the canonical plan displayed by the service.  The original
PDF is used only to isolate the user's added red no-go annotation from red
labels that were already present in the CAD drawing.
"""

from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "public" / "floorplans"
BACKEND = ROOT / "backend" / "assets" / "floorplans"
MAP_ID = "kerama_marazzi_2025"
OFFICE_INTERIOR = (2190, 662, 2260, 720)  # x1, y1, x2, y2 at 160 dpi
OFFICE_AREA_M2 = 10.0


def render(pdf: Path, output_prefix: Path) -> Path:
    subprocess.run(
        ["pdftoppm", "-f", "1", "-singlefile", "-png", "-r", "160", str(pdf), str(output_prefix)],
        check=True,
    )
    return output_prefix.with_suffix(".png")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Expected ORIGINAL.pdf and MARKED.pdf")
    original_pdf = Path(sys.argv[1]).resolve()
    marked_pdf = Path(sys.argv[2]).resolve()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    BACKEND.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        original = np.asarray(Image.open(render(original_pdf, temporary / "original")).convert("RGB"))
        marked_image = Image.open(render(marked_pdf, temporary / "marked")).convert("RGB")
        marked = np.asarray(marked_image)
        if original.shape != marked.shape:
            raise RuntimeError(f"Rendered pages differ: {original.shape} vs {marked.shape}")

        red = (marked[:, :, 0] > 190) & (marked[:, :, 1] < 130) & (marked[:, :, 2] < 130)
        original_red = (original[:, :, 0] > 190) & (original[:, :, 1] < 130) & (original[:, :, 2] < 130)
        changed = np.max(np.abs(original.astype(np.int16) - marked.astype(np.int16)), axis=2) > 40
        seed = red & changed & ~original_red
        neighbourhood = ndimage.binary_dilation(seed, iterations=2)
        mask = red & changed & (~original_red | neighbourhood)
        labels, count = ndimage.label(mask)
        ids, sizes = np.unique(labels, return_counts=True)
        keep = ids[(ids != 0) & (sizes >= 20)]
        mask = np.isin(labels, keep)

        display_path = PUBLIC / "kerama-marazzi-2025.png"
        marked_image.save(display_path, optimize=True)
        obstacle_path = BACKEND / "kerama_marazzi_2025_obstacles.png"
        Image.fromarray((mask * 255).astype(np.uint8)).save(obstacle_path, optimize=True)

    source_pdf = PUBLIC / "kerama-marazzi-2025.pdf"
    source_pdf.write_bytes(marked_pdf.read_bytes())
    x1, y1, x2, y2 = OFFICE_INTERIOR
    office_pixels = (x2 - x1) * (y2 - y1)
    metadata = {
        "map_id": MAP_ID,
        "width": int(marked.shape[1]),
        "height": int(marked.shape[0]),
        "meters_per_pixel": math.sqrt(OFFICE_AREA_M2 / office_pixels),
        "scale_calibration": {
            "source": "office_area",
            "office_area_m2": OFFICE_AREA_M2,
            "office_interior_bbox_pixels": list(OFFICE_INTERIOR),
            "office_pixel_area": office_pixels,
        },
        "grid_cell_pixels": 4,
        "person_radius_meters": 0.28,
        "walking_speed_mps": 1.20,
        "obstacle_mask_file": "kerama_marazzi_2025_obstacles.png",
        "obstacle_mask_sha256": hashlib.sha256(obstacle_path.read_bytes()).hexdigest(),
        "source_pdf_sha256": hashlib.sha256(source_pdf.read_bytes()).hexdigest(),
        "display_image_sha256": hashlib.sha256(display_path.read_bytes()).hexdigest(),
        "source_pdf": source_pdf.name,
        "display_image": display_path.name,
        "annotation_component_count": int(len(keep)),
        "annotation_pixel_count": int(mask.sum()),
    }
    (BACKEND / f"{MAP_ID}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
