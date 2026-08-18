#!/usr/bin/env python3
"""Standalone R³ job worker — runs outside the uvicorn process."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("r3_job_worker")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: r3_job_worker.py <job.json>", file=sys.stderr)
        return 2
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    video_id = str(job["video_id"])
    video_path_raw = job.get("video_path")
    video_path = Path(video_path_raw) if video_path_raw else None

    # Import after path setup so worker uses the same backend package.
    backend_dir = Path(__file__).resolve().parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    import main as api_main  # noqa: WPS433

    logger.info("[%s] worker start path=%s", video_id, video_path)
    asyncio.run(
        api_main.process_video_r3_background(
            video_id,
            video_path,
            str(job.get("original_filename") or "video.avi"),
            float(job.get("scale_factor") or 1.0),
            int(job.get("frame_stride") or 5),
            int(job.get("max_frames") or 1500),
            str(job.get("ckpt") or "r3_long.safetensors"),
            int(job.get("size") or 392),
            str(job.get("mode") or "strided"),
            job.get("map_context") or {},
        )
    )
    logger.info("[%s] worker done", video_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
