#!/usr/bin/env python3
"""Open two reviewed Kerama passage gaps and refresh immutable asset hashes.

The red/green annotation contains two small overlaps where the red markup
closed a visibly green doorway. The production engine correctly treats red as
absolute, so those annotation mistakes split one real corridor into three
components. This correction removes red only inside the reviewed doorway
boxes; it does not paint the example trajectory into the walkable mask.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "floorplans"
MASK_PATH = ASSET_ROOT / "kerama_marazzi_2025_obstacles.png"
METADATA_PATH = ASSET_ROOT / "kerama_marazzi_2025.json"

# x1, y1, x2, y2 in the canonical 5298 x 3743 plan. Both boxes are confined
# to existing positive-green support, so clearing red outside a passage cannot
# make blank CAD space walkable.
REVIEWED_PASSAGE_BOXES = (
    (2485, 925, 2575, 1005),
    (2890, 1890, 2990, 1990),
)


def main() -> None:
    obstacle = np.asarray(Image.open(MASK_PATH).convert("L")) >= 128
    corrected = obstacle.copy()
    for x1, y1, x2, y2 in REVIEWED_PASSAGE_BOXES:
        corrected[y1:y2, x1:x2] = False
    Image.fromarray((corrected * 255).astype(np.uint8)).save(MASK_PATH, optimize=True)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata["obstacle_mask_sha256"] = hashlib.sha256(MASK_PATH.read_bytes()).hexdigest()
    annotation = dict(metadata.get("obstacle_annotation") or {})
    annotation.update({
        "reviewed_passage_openings": [list(box) for box in REVIEWED_PASSAGE_BOXES],
        "reviewed_passage_policy": "remove_conflicting_red_inside_positive_green_only",
        "remaining_red_is_absolute": True,
        "route_specific_overrides": False,
    })
    metadata["obstacle_annotation"] = annotation
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
