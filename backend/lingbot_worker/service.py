from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from . import config
from .lingbot_adapter import LingBotMapAdapter, LingBotRunOptions
from .schemas import CreateSessionRequest, MetadataResponse, SessionStatus, StatusResponse
from .storage import SessionStorage


class LingBotSessionService:
    def __init__(self, storage: SessionStorage | None = None, adapter: LingBotMapAdapter | None = None):
        self.storage = storage or SessionStorage()
        self.adapter = adapter or LingBotMapAdapter()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lingbot-worker")
        self.futures: Dict[str, Future] = {}

    def create_session(self, request: CreateSessionRequest) -> str:
        video_path = Path(request.video_path).expanduser()
        if not video_path.is_absolute():
            raise ValueError("video_path must be absolute")
        if not video_path.exists() or not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Validate early so failed configuration is visible before queueing.
        self.adapter.validate_environment()

        session_id = self.storage.create_session()
        session_dir = self.storage.session_dir(session_id)
        input_video = self.storage.copy_input_video(session_id, video_path)
        use_sdpa = True

        metadata = {
            "session_id": session_id,
            "status": SessionStatus.queued.value,
            "video_path": str(video_path),
            "input_path": str(input_video),
            "fps": request.fps,
            "target_frames": request.target_frames,
            "keyframe_interval": request.keyframe_interval,
            "use_sdpa": use_sdpa,
            "mask_sky": request.mask_sky,
            "model_path": str(config.LINGBOT_MODEL_PATH),
            "repo_path": str(config.LINGBOT_REPO_PATH),
            "outputs": {},
            "timings": {"queued_at": time.time()},
            "error": None,
        }
        self.storage.write_json(self.storage.metadata_path(session_id), metadata)
        self.storage.write_status(session_id, SessionStatus.queued, progress=0.0)

        options = LingBotRunOptions(
            session_id=session_id,
            input_video=input_video,
            output_dir=session_dir,
            log_path=self.storage.log_path(session_id),
            fps=request.fps,
            target_frames=request.target_frames,
            keyframe_interval=request.keyframe_interval,
            use_sdpa=use_sdpa,
            mask_sky=request.mask_sky,
        )
        self.futures[session_id] = self.executor.submit(self._run_session, options)
        return session_id

    def _run_session(self, options: LingBotRunOptions) -> None:
        metadata_path = self.storage.metadata_path(options.session_id)
        metadata = self.storage.read_json(metadata_path)
        metadata["status"] = SessionStatus.running.value
        metadata.setdefault("timings", {})["started_at"] = time.time()
        self.storage.write_json(metadata_path, metadata)
        self.storage.write_status(options.session_id, SessionStatus.running, progress=0.05)

        try:
            artifacts = self.adapter.run(options)
            metadata = self.storage.read_json(metadata_path)
            metadata["status"] = SessionStatus.completed.value
            metadata["outputs"] = artifacts
            metadata["timings"].update(artifacts.get("timings", {}))
            metadata["timings"]["completed_at"] = time.time()
            metadata["error"] = None
            self.storage.write_json(metadata_path, metadata)
            self.storage.write_status(options.session_id, SessionStatus.completed, progress=1.0)
        except Exception as exc:
            error = str(exc)
            self.storage.append_log(options.session_id, f"[lingbot] ERROR: {error}")
            metadata = self.storage.read_json(metadata_path, default={})
            metadata["status"] = SessionStatus.failed.value
            metadata["error"] = error
            metadata.setdefault("timings", {})["failed_at"] = time.time()
            self.storage.write_json(metadata_path, metadata)
            self.storage.write_status(options.session_id, SessionStatus.failed, progress=0.0, error=error)

    def get_status(self, session_id: str) -> StatusResponse:
        self.storage.ensure_session(session_id)
        data = self.storage.read_json(self.storage.status_path(session_id))
        return StatusResponse(**data)

    def get_metadata(self, session_id: str) -> MetadataResponse:
        self.storage.ensure_session(session_id)
        data = self.storage.read_json(self.storage.metadata_path(session_id))
        return MetadataResponse(**data)

    def trajectory_file(self, session_id: str) -> Path:
        self.storage.ensure_session(session_id)
        path = self.storage.trajectory_path(session_id)
        if not path.exists():
            raise FileNotFoundError("trajectory.json is not ready")
        return path

    def pointcloud_file(self, session_id: str) -> Path:
        self.storage.ensure_session(session_id)
        npz = self.storage.pointcloud_npz_path(session_id)
        ply = self.storage.pointcloud_ply_path(session_id)
        if ply.exists():
            return ply
        if npz.exists():
            return npz
        raise FileNotFoundError("point cloud is not ready")
