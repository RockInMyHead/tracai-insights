"""TrackAI localhost-only Windows R³ worker bundled with Electron."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

RUNTIME_DIR = Path(os.environ["TRACKAI_RUNTIME_DIR"]).resolve()
DATA_DIR = Path(os.environ["TRACKAI_LOCAL_DATA_DIR"]).resolve()
BACKEND_DIR = RUNTIME_DIR / "backend"
R3_DIR = RUNTIME_DIR / "R3"
WEIGHT = R3_DIR / "ckpt" / "r3_long.safetensors"
WRAPPER = BACKEND_DIR / "r3_worker_wrapper.py"

sys.path.insert(0, str(BACKEND_DIR))
from floorplan_constraints import apply_floorplan_constraints  # noqa: E402
from r3_trajectory import build_r3_trajectory  # noqa: E402

app = FastAPI(title="TrackAI Local Worker", docs_url=None, redoc_url=None)


def _last_complete(stdout: str) -> dict:
    complete = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "complete":
            complete = event.get("result")
    if not isinstance(complete, dict):
        raise RuntimeError("R³ did not emit a complete result")
    return complete


@app.get("/health")
def health() -> dict:
    import torch

    return {
        "ok": bool(torch.cuda.is_available() and WEIGHT.exists()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": str(torch.version.cuda or ""),
        "gpu_count": int(torch.cuda.device_count()),
        "weight_ready": WEIGHT.exists(),
    }


@app.post("/process/{video_id}")
async def process(video_id: str, request: Request) -> dict:
    if not WEIGHT.exists():
        raise HTTPException(503, "R³ weights are missing")
    import torch
    if not torch.cuda.is_available():
        raise HTTPException(503, "CUDA GPU is unavailable")

    safe_id = "".join(c for c in video_id if c.isalnum() or c in "-_") or str(uuid.uuid4())
    job_dir = DATA_DIR / "jobs" / safe_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / "input.mp4"
    with video_path.open("wb") as output:
        async for chunk in request.stream():
            output.write(chunk)

    env = os.environ.copy()
    env.update({
        "TRACKAI_R3_DIR": str(R3_DIR),
        "TRACKAI_R3_PYTHON": sys.executable,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "R3_USE_RELEASE_PRESET": "true",
    })
    command = [
        sys.executable, str(WRAPPER),
        "--video_path", str(video_path),
        "--output_dir", str(job_dir / "r3"),
        "--frame_stride", "5",
        "--max_frames", "0",
        "--ckpt", WEIGHT.name,
        "--size", "392",
        "--mode", "long",
    ]
    completed = subprocess.run(
        command, cwd=str(R3_DIR), env=env, capture_output=True,
        text=True, timeout=4 * 60 * 60,
    )
    if completed.returncode:
        raise HTTPException(500, (completed.stderr or completed.stdout)[-4000:])

    r3 = _last_complete(completed.stdout)
    bundle = build_r3_trajectory(
        r3.get("camera_poses") or [],
        r3.get("pose_confidence"),
        r3.get("frame_selection"),
        r3.get("run_params"),
    )
    result = {
        "method": "r3_reconstruction_local_cuda",
        "trajectory": bundle["plan_trajectory"],
        "plan_trajectory": bundle["plan_trajectory"],
        "raw_trajectory_3d": bundle["raw_trajectory_3d"],
        "turn_points": bundle["turn_points"],
        "processing_stats": {
            "trajectory_points": len(bundle["plan_trajectory"]),
            "local_cuda": True,
            "r3_total_time_s": r3.get("total_time_s"),
        },
    }
    result = apply_floorplan_constraints(
        result,
        {"floorplan_id": "kerama_marazzi_2025"},
    )
    (job_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )
    return {"success": True, "status": "completed", "video_id": safe_id, "data": result}

