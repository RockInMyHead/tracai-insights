from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from . import config
from .schemas import SessionStatus


class SessionStorage:
    def __init__(self, sessions_dir: Path = config.LINGBOT_SESSIONS_DIR):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.session_dir(session_id).mkdir(parents=True, exist_ok=False)
        return session_id

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def input_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "input.mp4"

    def status_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "status.json"

    def trajectory_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "trajectory.json"

    def pointcloud_npz_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "pointcloud.npz"

    def pointcloud_ply_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "pointcloud.ply"

    def metadata_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "metadata.json"

    def log_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "logs.txt"

    def ensure_session(self, session_id: str) -> Path:
        path = self.session_dir(session_id)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Unknown session_id: {session_id}")
        return path

    def copy_input_video(self, session_id: str, source: Path) -> Path:
        target = self.input_path(session_id)
        shutil.copy2(source, target)
        return target

    def write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def read_json(self, path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not path.exists():
            if default is None:
                raise FileNotFoundError(str(path))
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def write_status(
        self,
        session_id: str,
        status: SessionStatus,
        progress: float = 0.0,
        fps: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        self.write_json(
            self.status_path(session_id),
            {
                "session_id": session_id,
                "status": status.value,
                "progress": max(0.0, min(1.0, float(progress))),
                "fps": fps,
                "error": error,
                "updated_at": time.time(),
            },
        )

    def append_log(self, session_id: str, line: str) -> None:
        with self.log_path(session_id).open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

