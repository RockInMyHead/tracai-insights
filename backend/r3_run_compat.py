"""Pure compatibility checks for cached R3 inference artifacts."""

from __future__ import annotations

import math
from typing import Any, Mapping


def sampling_contract_matches(
    params: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    frame_stride: int,
    max_frames: int,
    size: int,
    long_target_fps: float,
) -> bool:
    """Return whether cached poses used the requested information density."""
    try:
        saved_stride = int(selection.get("requested_frame_stride") or 0)
        saved_max_frames = int(selection.get("requested_max_frames") or 0)
        saved_size = int(
            params.get("wrapper_input_size")
            or params.get("size")
            or params.get("image_size")
            or 0
        )
        if (
            saved_stride != int(frame_stride)
            or saved_max_frames != int(max_frames)
            or saved_size != int(size)
        ):
            return False
        if selection.get("long_video_sampling"):
            return math.isclose(
                float(selection.get("long_target_fps")),
                float(long_target_fps),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        return True
    except (TypeError, ValueError):
        return False
