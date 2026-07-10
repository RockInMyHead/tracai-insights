from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from . import config


class SessionStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class CreateSessionRequest(BaseModel):
    video_path: str = Field(..., description="Absolute path to an existing video file")
    fps: int = Field(default=config.DEFAULT_FPS, ge=1, le=120)
    target_frames: int = Field(default=config.DEFAULT_TARGET_FRAMES, ge=0, le=10000)
    keyframe_interval: int = Field(default=config.DEFAULT_KEYFRAME_INTERVAL, ge=1, le=240)
    use_sdpa: bool = config.DEFAULT_USE_SDPA
    mask_sky: bool = config.DEFAULT_MASK_SKY


class CreateSessionResponse(BaseModel):
    session_id: str
    status: SessionStatus


class StatusResponse(BaseModel):
    session_id: str
    status: SessionStatus
    progress: float = 0.0
    fps: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ok: bool
    cuda_available: bool
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    model_path: str
    model_exists: bool
    repo_path: str
    repo_exists: bool
    sessions_dir: str


class MetadataResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    session_id: str
    status: SessionStatus
    video_path: str
    fps: int
    target_frames: int
    keyframe_interval: int
    use_sdpa: bool
    mask_sky: bool
    model_path: str
    repo_path: str
    outputs: Dict[str, Any] = Field(default_factory=dict)
    timings: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
