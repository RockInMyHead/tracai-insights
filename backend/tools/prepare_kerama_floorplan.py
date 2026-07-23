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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        # The free-space engine adds a physical 0.28 m safety halo later.
        # Keeping the full antialiased brush fringe here applies the boundary
        # margin twice and can close a visibly green doorway.  Remove only two
        # raster pixels from annotation edges; filled machines remain intact
        # and regain the physical halo during grid construction.
        red_boundary_cleanup_pixels = 2
        mask = ndimage.binary_erosion(
            mask,
            iterations=red_boundary_cleanup_pixels,
        )

        # Green operator paint is the positive counterpart of the red no-go
        # annotation: it marks verified walkable floor even when the original
        # CAD flood-fill classified that area as exterior or as a separate
        # island.  Isolate only newly added green pixels so native green CAD
        # layers cannot accidentally make machinery walkable.  A small close
        # and dilation bridge antialiasing/text holes in the painted corridor;
        # the red obstacle mask is still applied afterwards and therefore has
        # final authority wherever the annotations overlap.
        green = (
            (marked[:, :, 1] > 190)
            & (marked[:, :, 0] < 130)
            & (marked[:, :, 2] < 130)
        )
        original_green = (
            (original[:, :, 1] > 190)
            & (original[:, :, 0] < 130)
            & (original[:, :, 2] < 130)
        )
        green_seed = green & changed & ~original_green
        green_labels, _ = ndimage.label(green_seed)
        green_ids, green_sizes = np.unique(green_labels, return_counts=True)
        green_keep = green_ids[(green_ids != 0) & (green_sizes >= 20)]
        green_mask = np.isin(green_labels, green_keep)
        # Operator paint is intentionally rough.  Convert it into a regular
        # passage network instead of preserving every brush notch: close gaps
        # smaller than about one metre, remove isolated edge whiskers, and add
        # enough width for the downstream 0.28 m person-radius inflation.  No
        # route coordinates are involved in this operation.
        green_support_close_pixels = 10
        green_support_open_pixels = 3
        green_support_dilation_pixels = 6

        def disk(radius: int) -> np.ndarray:
            axis = np.arange(-radius, radius + 1)
            yy, xx = np.meshgrid(axis, axis, indexing="ij")
            return xx * xx + yy * yy <= radius * radius

        green_support = ndimage.binary_closing(
            green_mask,
            structure=disk(green_support_close_pixels),
        )
        green_support = ndimage.binary_opening(
            green_support,
            structure=disk(green_support_open_pixels),
        )
        green_support = ndimage.binary_dilation(
            green_support,
            iterations=green_support_dilation_pixels,
        )

        display_path = PUBLIC / "kerama-marazzi-2025.png"
        marked_image.save(display_path, optimize=True)
        obstacle_path = BACKEND / "kerama_marazzi_2025_obstacles.png"

        # A distance halo around every CAD primitive is not a walkability
        # model.  In particular it creates an artificial corridor on the
        # *outside* of a long wall/roof line.  That failure is especially bad
        # for map matching because A* can replace a real indoor route with a
        # visually plausible straight line through blank PDF canvas.
        #
        # Treat page-border-connected white space as exterior instead.  CAD
        # ink is used only as a flood-fill barrier; it is not made an obstacle
        # (the immutable red annotation remains the no-go source of truth).
        # Two pixels of dilation close antialiasing pinholes while preserving
        # the large entrances and courtyards represented by the drawing.
        support_barrier_dilation_pixels = 2
        original_gray = np.asarray(Image.fromarray(original).convert("L"))
        ink = original_gray < 242
        barrier = ndimage.binary_dilation(
            ink, iterations=support_barrier_dilation_pixels
        )
        blank = ~barrier
        blank_labels, _ = ndimage.label(blank)
        border_labels = np.unique(np.concatenate((
            blank_labels[0, :],
            blank_labels[-1, :],
            blank_labels[:, 0],
            blank_labels[:, -1],
        )))
        exterior = np.isin(blank_labels, border_labels)
        # Positive-only walkability: CAD enclosures describe rooms,
        # equipment, roofs and site boundaries, not traversable floor.  The
        # previous ``~exterior`` union silently made every enclosed polygon a
        # legal route and let A* escape through machinery or along the outer
        # site contour.  Only the operator's green annotation is affirmative
        # evidence of a corridor.  Red remains a hard obstacle in the engine.
        green_corridor_margin_pixels = 24
        support = ndimage.binary_dilation(
            green_support,
            iterations=green_corridor_margin_pixels,
        )
        # Red is an immutable obstacle. Green is affirmative walkability
        # evidence only outside red; it must never erase an obstacle.
        support = support & ~mask
        meters_per_pixel = math.sqrt(
            OFFICE_AREA_M2
            / ((OFFICE_INTERIOR[2] - OFFICE_INTERIOR[0])
               * (OFFICE_INTERIOR[3] - OFFICE_INTERIOR[1]))
        )
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            obstacle_path, optimize=True
        )
        support_path = BACKEND / "kerama_marazzi_2025_support.png"
        Image.fromarray((support * 255).astype(np.uint8)).save(
            support_path, optimize=True
        )

    source_pdf = PUBLIC / "kerama-marazzi-2025.pdf"
    source_pdf.write_bytes(marked_pdf.read_bytes())
    x1, y1, x2, y2 = OFFICE_INTERIOR
    office_pixels = (x2 - x1) * (y2 - y1)
    metadata = {
        "map_id": MAP_ID,
        "width": int(marked.shape[1]),
        "height": int(marked.shape[0]),
        "meters_per_pixel": meters_per_pixel,
        "scale_calibration": {
            "source": "office_area",
            "office_area_m2": OFFICE_AREA_M2,
            "office_interior_bbox_pixels": list(OFFICE_INTERIOR),
            "office_pixel_area": office_pixels,
        },
        "grid_cell_pixels": 4,
        "person_radius_meters": 0.28,
        "walking_speed_mps": 1.20,
        "default_start_anchor": {
            "source": "fixed_plan_start_left_heading_red_safe_v3",
            "reference_pixels": [2190.0, 686.0],
            "direction_pixels": [2170.0, 666.0],
            "trajectory_points_used": False,
        },
        "obstacle_mask_file": "kerama_marazzi_2025_obstacles.png",
        "obstacle_mask_sha256": hashlib.sha256(obstacle_path.read_bytes()).hexdigest(),
        "support_mask_file": support_path.name,
        "support_mask_sha256": hashlib.sha256(support_path.read_bytes()).hexdigest(),
        "support_mask_generation": {
            "method": "positive_green_corridors_only",
            "ink_threshold": 242,
            "barrier_dilation_pixels": support_barrier_dilation_pixels,
            "coverage_ratio": float(support.mean()),
            "green_corridor_margin_pixels": green_corridor_margin_pixels,
        },
        "source_pdf_sha256": hashlib.sha256(source_pdf.read_bytes()).hexdigest(),
        "display_image_sha256": hashlib.sha256(display_path.read_bytes()).hexdigest(),
        "source_pdf": source_pdf.name,
        "display_image": display_path.name,
        "annotation_component_count": int(len(keep)),
        "annotation_pixel_count": int(mask.sum()),
        "obstacle_annotation": {
            "method": "changed_red_operator_annotation",
            "boundary_cleanup_pixels": red_boundary_cleanup_pixels,
            "physical_person_halo_applied_by_engine": True,
            "remaining_red_is_absolute": True,
            "overlap_policy": "red_obstacles_have_absolute_precedence",
            "route_specific_overrides": False,
        },
        "walkable_annotation": {
            "method": "changed_green_operator_annotation",
            "component_count": int(len(green_keep)),
            "pixel_count": int(green_mask.sum()),
            "support_pixels_added": int(np.count_nonzero(green_support)),
            "closing_pixels": green_support_close_pixels,
            "opening_pixels": green_support_open_pixels,
            "dilation_pixels": green_support_dilation_pixels,
            "red_obstacles_take_precedence": True,
            "precedence": "red_obstacles_over_positive_green",
            "reconciliation": "absolute_red_priority_v2",
            "route_specific_overrides": False,
        },
        "reference_mask": {
            "method": "none",
            "route_specific_overrides": False,
        },
    }
    (BACKEND / f"{MAP_ID}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
