from __future__ import annotations

import os
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parents[1]

LINGBOT_REPO_PATH = Path(os.getenv("LINGBOT_REPO_PATH", BASE_DIR / "third_party" / "lingbot-map")).expanduser()
LINGBOT_MODEL_PATH = Path(os.getenv("LINGBOT_MODEL_PATH", BASE_DIR / "checkpoints" / "lingbot-map.pt")).expanduser()
LINGBOT_SESSIONS_DIR = Path(os.getenv("LINGBOT_SESSIONS_DIR", BASE_DIR / "data" / "lingbot_sessions")).expanduser()

DEFAULT_FPS = int(os.getenv("LINGBOT_DEFAULT_FPS", "10"))
DEFAULT_TARGET_FRAMES = int(os.getenv("LINGBOT_TARGET_FRAMES", "1500"))
DEFAULT_KEYFRAME_INTERVAL = int(os.getenv("LINGBOT_DEFAULT_KEYFRAME_INTERVAL", "6"))
DEFAULT_USE_SDPA = _bool_env("LINGBOT_USE_SDPA", True)
DEFAULT_MASK_SKY = _bool_env("LINGBOT_MASK_SKY", False)
DEFAULT_DEVICE = os.getenv("LINGBOT_DEVICE", "cuda")
DEFAULT_MODE = os.getenv("LINGBOT_MODE", "windowed")
DEFAULT_IMAGE_SIZE = int(os.getenv("LINGBOT_IMAGE_SIZE", "518"))

LINGBOT_PYTHON = os.getenv("LINGBOT_PYTHON", "")
LINGBOT_TIMEOUT_SECONDS = int(os.getenv("LINGBOT_TIMEOUT_SECONDS", str(12 * 60 * 60)))
