#!/usr/bin/env python3
"""Apply the verified Kerama route override to the checked-in v5 assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kerama_reference_route import apply_reference_route_overrides


ASSET_ROOT = ROOT / "backend" / "assets" / "floorplans"
METADATA_PATH = ASSET_ROOT / "kerama_marazzi_2025.json"


def main() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    obstacle_path = ASSET_ROOT / metadata["obstacle_mask_file"]
    support_path = ASSET_ROOT / metadata["support_mask_file"]
    obstacles = np.asarray(Image.open(obstacle_path).convert("L")) >= 128
    support = np.asarray(Image.open(support_path).convert("L")) >= 128
    obstacles, support, override = apply_reference_route_overrides(
        obstacles,
        support,
        meters_per_pixel=float(metadata["meters_per_pixel"]),
    )
    # The command is deliberately idempotent.  Once the checked-in mask has
    # been overridden there are no blocked corridor pixels left to count, but
    # rerunning the tool must not erase the provenance of the original v5
    # correction.
    previous_override = metadata.get("reference_mask") or {}
    if previous_override.get("method") == override.get("method"):
        for key in ("obstacle_pixels_cleared", "support_pixels_added"):
            override[key] = max(
                int(override.get(key, 0)), int(previous_override.get(key, 0))
            )
    Image.fromarray((obstacles * 255).astype(np.uint8)).save(
        obstacle_path, optimize=True
    )
    Image.fromarray((support * 255).astype(np.uint8)).save(
        support_path, optimize=True
    )
    metadata.update({
        "obstacle_mask_sha256": hashlib.sha256(obstacle_path.read_bytes()).hexdigest(),
        "support_mask_sha256": hashlib.sha256(support_path.read_bytes()).hexdigest(),
        "annotation_pixel_count": int(obstacles.sum()),
        "reference_mask": override,
    })
    metadata["support_mask_generation"]["coverage_ratio"] = float(support.mean())
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(override, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
