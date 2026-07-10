from __future__ import annotations

import json
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import config
from .schemas import CreateSessionRequest, CreateSessionResponse, HealthResponse, SessionStatus
from .service import LingBotSessionService


app = FastAPI(title="TrackAI LingBot-Map GPU Worker", version="0.1.0")
service = LingBotSessionService()


def _cuda_health() -> tuple[bool, Optional[str], Optional[float], Optional[float]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, None, None, None
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        return (
            True,
            props.name,
            round(total_bytes / 1024**3, 2),
            round(free_bytes / 1024**3, 2),
        )
    except Exception:
        return False, None, None, None


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    cuda_available, gpu_name, vram_total_gb, vram_free_gb = _cuda_health()
    return HealthResponse(
        ok=True,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        vram_total_gb=vram_total_gb,
        vram_free_gb=vram_free_gb,
        model_path=str(config.LINGBOT_MODEL_PATH),
        model_exists=config.LINGBOT_MODEL_PATH.exists(),
        repo_path=str(config.LINGBOT_REPO_PATH),
        repo_exists=config.LINGBOT_REPO_PATH.exists(),
        sessions_dir=str(config.LINGBOT_SESSIONS_DIR),
    )


@app.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    try:
        session_id = service.create_session(request)
        return CreateSessionResponse(session_id=session_id, status=SessionStatus.queued)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/sessions/upload", response_model=CreateSessionResponse)
async def create_session_upload(
    file: UploadFile = File(...),
    fps: int = Form(config.DEFAULT_FPS),
    target_frames: int = Form(config.DEFAULT_TARGET_FRAMES),
    keyframe_interval: int = Form(config.DEFAULT_KEYFRAME_INTERVAL),
    use_sdpa: bool = Form(config.DEFAULT_USE_SDPA),
    mask_sky: bool = Form(config.DEFAULT_MASK_SKY),
) -> CreateSessionResponse:
    incoming_dir = config.LINGBOT_SESSIONS_DIR / "_incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1].lower()
    tmp_path = incoming_dir / f"{uuid.uuid4()}{suffix or '.mp4'}"
    try:
        with tmp_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        request = CreateSessionRequest(
            video_path=str(tmp_path),
            fps=fps,
            target_frames=target_frames,
            keyframe_interval=keyframe_interval,
            # Keep accepting the multipart field for API compatibility, but
            # force SDPA in this deployment because FlashInfer is unavailable.
            use_sdpa=True,
            mask_sky=mask_sky,
        )
        session_id = service.create_session(request)
        return CreateSessionResponse(session_id=session_id, status=SessionStatus.queued)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/sessions/{session_id}/status")
async def get_status(session_id: str):
    try:
        return service.get_status(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/sessions/{session_id}/trajectory")
async def get_trajectory(session_id: str):
    try:
        path = service.trajectory_file(session_id)
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/sessions/{session_id}/pointcloud")
async def get_pointcloud(session_id: str):
    try:
        path = service.pointcloud_file(session_id)
        media_type = "application/octet-stream"
        if path.suffix.lower() == ".ply":
            media_type = "application/octet-stream"
        return FileResponse(str(path), media_type=media_type, filename=path.name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/sessions/{session_id}/metadata")
async def get_metadata(session_id: str):
    try:
        return service.get_metadata(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
