#!/usr/bin/env python3
"""Reconcile the fixed Kerama walkability layers.

The green annotation is the operator's affirmative statement that a part of
the plan is traversable.  Red paint outside that layer remains a no-go area,
but red CAD/markup strokes which overlap a verified green corridor must not
split that corridor.  The previous global red-over-green rule did exactly
that: it even made the approved initial heading point an obstacle.

This is a plan-layer reconciliation, not a trajectory-specific edit.
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

    # The final engine already excludes every pixel outside ``support``.
    # Keeping an obstacle there is harmless, while an obstacle *inside* the
    # positive layer contradicts the operator-provided corridor topology.
    reconciled = obstacle & ~support
    Image.fromarray((reconciled * 255).astype(np.uint8)).save(
        OBSTACLE_PATH, optimize=True
    )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata["obstacle_mask_sha256"] = hashlib.sha256(
        OBSTACLE_PATH.read_bytes()
    ).hexdigest()
    metadata["walkable_annotation"] = {
        **dict(metadata.get("walkable_annotation") or {}),
        "red_obstacles_take_precedence": False,
        "precedence": "positive_green_corridor_over_overlapping_red_markup",
        "reconciliation": "green_support_authoritative_v1",
        "route_specific_overrides": False,
    }
    metadata["obstacle_annotation"] = {
        **dict(metadata.get("obstacle_annotation") or {}),
        "remaining_red_is_absolute_outside_positive_green": True,
        "overlap_policy": "positive_green_corridor_has_precedence",
        "route_specific_overrides": False,
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
