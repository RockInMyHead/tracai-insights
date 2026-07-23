#!/usr/bin/env python3
"""Enforce the Kerama floor-plan mask precedence invariant.

Red pixels in the operator obstacle layer are immutable. Green is positive
walkability evidence only where it does not overlap that layer. Native red CAD
ink is deliberately excluded by ``prepare_kerama_floorplan.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "floorplans"
MAP_ID = "kerama_marazzi_2025"
OBSTACLE_PATH = ASSET_ROOT / f"{MAP_ID}_obstacles.png"
SUPPORT_PATH = ASSET_ROOT / f"{MAP_ID}_support.png"
METADATA_PATH = ASSET_ROOT / f"{MAP_ID}.json"


def main() -> None:
    obstacle = np.asarray(Image.open(OBSTACLE_PATH).convert("L")) >= 128
    support = np.asarray(Image.open(SUPPORT_PATH).convert("L")) >= 128
    reconciled_obstacle = obstacle
    reconciled_support = support & ~reconciled_obstacle
    if np.any(reconciled_obstacle & reconciled_support):
        raise AssertionError("Red obstacle/support overlap survived reconciliation")

    Image.fromarray((reconciled_obstacle * 255).astype(np.uint8)).save(
        OBSTACLE_PATH, optimize=True
    )
    Image.fromarray((reconciled_support * 255).astype(np.uint8)).save(
        SUPPORT_PATH, optimize=True
    )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata["obstacle_mask_sha256"] = hashlib.sha256(
        OBSTACLE_PATH.read_bytes()
    ).hexdigest()
    metadata["support_mask_sha256"] = hashlib.sha256(
        SUPPORT_PATH.read_bytes()
    ).hexdigest()
    metadata["support_mask_generation"] = {
        **dict(metadata.get("support_mask_generation") or {}),
        "coverage_ratio": float(reconciled_support.mean()),
    }
    metadata["walkable_annotation"] = {
        **dict(metadata.get("walkable_annotation") or {}),
        "red_obstacles_take_precedence": True,
        "precedence": "red_obstacles_over_positive_green",
        "reconciliation": "absolute_red_priority_v2",
        "route_specific_overrides": False,
    }
    metadata["obstacle_annotation"] = {
        **dict(metadata.get("obstacle_annotation") or {}),
        "method": "changed_red_operator_annotation",
        "annotation_pixel_count": int(reconciled_obstacle.sum()),
        "remaining_red_is_absolute": True,
        "remaining_red_is_absolute_outside_positive_green": False,
        "overlap_policy": "red_obstacles_have_absolute_precedence",
        "route_specific_overrides": False,
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
