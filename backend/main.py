from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import os
import shutil
import tempfile
import time
import asyncio
import aiohttp
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set, TextIO
import threading
import logging
import sqlite3
import json
import uuid
import math as _math
import fitz  # PyMuPDF

try:
    from r3_trajectory import build_r3_trajectory
except ImportError:  # pragma: no cover - allows `uvicorn backend.main:app`
    from backend.r3_trajectory import build_r3_trajectory

try:
    from floorplan_constraints import (
        DEFAULT_FLOORPLAN_ID,
        FLOORPLAN_CONSTRAINT_REVISION,
        apply_floorplan_constraints,
    )
except ImportError:  # pragma: no cover - allows `uvicorn backend.main:app`
    from backend.floorplan_constraints import (
        DEFAULT_FLOORPLAN_ID,
        FLOORPLAN_CONSTRAINT_REVISION,
        apply_floorplan_constraints,
    )

try:
    from lingbot_fusion import (
        attach_lingbot_fusion_candidate,
        should_restore_lingbot_fusion_candidate,
    )
except ImportError:  # pragma: no cover - allows `uvicorn backend.main:app`
    from backend.lingbot_fusion import (
        attach_lingbot_fusion_candidate,
        should_restore_lingbot_fusion_candidate,
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="TrackAI Video Analysis API", version="1.0.0")

# Global storage for processing status
processing_status = {}
convert_dwg_status: Dict[str, Dict] = {}

# Employee batch scheduling
EMPLOYEE_BATCH_TS: Dict[str, float] = {}
BATCH_WAIT_SECONDS = 1  # минимальная задержка — обработка начинается почти мгновенно

# GPU Worker — тяжёлый CV-пайплайн вынесен на отдельный сервер с RTX 3090
GPU_WORKER_URL = os.getenv("GPU_WORKER_URL", "http://79.137.227.106:8003")
LINGBOT_WORKER_URL = os.getenv("LINGBOT_WORKER_URL", "http://79.137.227.106:8004")
LINGBOT_FUSION_ENABLED = os.getenv("LINGBOT_FUSION_ENABLED", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
LINGBOT_FUSION_TARGET_FRAMES = int(os.getenv("LINGBOT_FUSION_TARGET_FRAMES", "3000"))
LINGBOT_FUSION_KEYFRAME_INTERVAL = int(os.getenv("LINGBOT_FUSION_KEYFRAME_INTERVAL", "6"))
LINGBOT_FUSION_TIMEOUT_SECONDS = int(
    os.getenv("LINGBOT_FUSION_TIMEOUT_SECONDS", str(3 * 60 * 60))
)

#
# Helper: safely convert numpy/scipy types to plain JSON-serializable objects
try:
    import numpy as _np  # type: ignore
except Exception:  # numpy may be missing in some minimal environments
    _np = None  # type: ignore


def _to_json_serializable(obj: Any):
    """
    Recursively convert common non-serializable types (numpy, sets, etc.)
    into plain Python types so that json.dumps / FastAPI can encode them.
    """
    # Numpy scalars / arrays
    if _np is not None:
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()

    # Containers
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_json_serializable(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, complex):
        return [obj.real, obj.imag]

    # Fallback: leave as-is (bool, int, float, str, None, etc.)
    # Anything else that FastAPI cannot encode becomes a string.
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _status_response_payload(status_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a poll-safe status payload with live timing for the UI dashboard."""
    safe = _to_json_serializable(status_data)
    if not isinstance(safe, dict):
        return {"status": "unknown", "progress": 0, "message": "invalid status"}
    now = time.time()
    start_raw = safe.get("start_time")
    try:
        start_time = float(start_raw) if start_raw is not None else None
    except (TypeError, ValueError):
        start_time = None
    if start_time is None or not _math.isfinite(start_time) or start_time <= 0:
        start_time = None
    elapsed = max(0.0, now - start_time) if start_time is not None else None
    progress = int(round(_finite_status_progress(safe.get("progress"))))
    status = str(safe.get("status") or "unknown").lower()
    eta_seconds = None
    if (
        elapsed is not None
        and elapsed >= 8.0
        and 3 <= progress < 100
        and status not in {"completed", "error", "failed", "unknown"}
    ):
        # Extrapolate from observed rate; clamp so the UI never shows absurd ETAs.
        rate = progress / max(elapsed, 1e-6)
        remaining = (100.0 - progress) / max(rate, 1e-6)
        eta_seconds = float(min(max(remaining, 5.0), 3 * 60 * 60))
    stage = str(safe.get("stage") or _infer_processing_stage(status, safe.get("message"), progress))
    enriched = dict(safe)
    enriched["progress"] = progress
    enriched["stage"] = stage
    if start_time is not None:
        enriched["start_time"] = start_time
    if elapsed is not None:
        enriched["elapsed_seconds"] = round(elapsed, 1)
    if eta_seconds is not None:
        enriched["eta_seconds"] = round(eta_seconds, 1)
    else:
        enriched.pop("eta_seconds", None)
    return enriched


def _finite_status_progress(value: Any) -> float:
    try:
        progress = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not _math.isfinite(progress):
        return 0.0
    return max(0.0, min(100.0, progress))


def _infer_processing_stage(status: str, message: Any, progress: float) -> str:
    text = str(message or "").lower()
    status = str(status or "").lower()
    if status in {"completed", "done", "success"} or progress >= 100:
        return "done"
    if status in {"error", "failed"}:
        return "error"
    if "загруз" in text or "upload" in text or status in {"uploading_to_gpu"}:
        return "upload"
    if "lingbot" in text or status in {"lingbot_fusion", "lingbot_queued", "lingbot_running"}:
        return "lingbot"
    if "план" in text or "floorplan" in text or "map" in text:
        return "map"
    if "gpu" in text or "r³" in text or "r3" in text or status in {"gpu_processing", "processing"}:
        return "gpu"
    if status in {"queued"}:
        return "queued"
    return "processing"


async def _gpu_progress_heartbeat(
    video_id: str,
    *,
    base_progress: int = 15,
    ceiling: int = 88,
    time_constant_seconds: float = 780.0,
    label: str = "R³ реконструкция на GPU",
) -> None:
    """Soft-live progress while the blocking GPU HTTP call is in flight."""
    try:
        while True:
            await asyncio.sleep(2.0)
            state = processing_status.get(video_id)
            if not isinstance(state, dict):
                return
            status = str(state.get("status") or "")
            if status in {"completed", "error", "failed"}:
                return
            current = int(_finite_status_progress(state.get("progress")))
            if current >= 90:
                return
            start = float(state.get("start_time") or time.time())
            elapsed = max(0.0, time.time() - start)
            soft = int(
                base_progress
                + (ceiling - base_progress)
                * (1.0 - _math.exp(-elapsed / max(time_constant_seconds, 1.0)))
            )
            soft = max(base_progress, min(ceiling, soft))
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            update = {
                "elapsed_seconds": round(elapsed, 1),
                "stage": "gpu",
                "message": f"{label} · {mins:02d}:{secs:02d}",
            }
            if soft > current:
                update["progress"] = soft
            processing_status[video_id].update(update)
            # Persist only occasionally to avoid SQLite churn every 2s.
            if int(elapsed) % 10 < 2:
                _update_task_status(video_id, "processing", int(update.get("progress", current)))
    except asyncio.CancelledError:
        return


def _load_completed_analysis_status(video_id: str) -> Optional[Dict[str, Any]]:
    """If analysis is already on disk / DB-completed, expose it even after restart."""
    analysis_file = OUTPUT_DIR / f"{video_id}_analysis.json"
    db_completed = False
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT status, progress FROM tracking_tasks WHERE id = ?",
            (video_id,),
        ).fetchone()
        conn.close()
        if row and str(row[0]).lower() in {"completed", "done", "success"}:
            db_completed = True
    except Exception:
        db_completed = False
    if not analysis_file.exists() and not db_completed:
        return None
    if not analysis_file.exists():
        return {
            "id": video_id,
            "status": "completed",
            "progress": 100,
            "message": "Анализ завершён",
        }
    try:
        with open(analysis_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result = payload.get("analysis_result") or payload
        constraint = result.get("floorplan_constraint") if isinstance(result, dict) else None
        saved_revision = (
            constraint.get("constraint_revision")
            if isinstance(constraint, dict) else None
        )
        if saved_revision != FLOORPLAN_CONSTRAINT_REVISION:
            map_context = _load_task_map_context(video_id)
            if map_context.get("reference_point") and map_context.get("direction_point"):
                refreshed = apply_floorplan_constraints(result, map_context)
                refreshed_constraint = refreshed.get("floorplan_constraint") or {}
                if refreshed_constraint.get("constraint_revision") == FLOORPLAN_CONSTRAINT_REVISION:
                    result = refreshed
                    if isinstance(payload, dict) and "analysis_result" in payload:
                        payload["analysis_result"] = result
                    else:
                        payload = result
                    analysis_file.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=analysis_file.parent,
                        prefix=f".{analysis_file.name}.",
                        suffix=".tmp",
                        delete=False,
                    ) as temporary:
                        json.dump(
                            payload,
                            temporary,
                            indent=2,
                            ensure_ascii=False,
                            default=_to_json_serializable,
                        )
                        temporary.flush()
                        os.fsync(temporary.fileno())
                        temporary_path = Path(temporary.name)
                    os.replace(temporary_path, analysis_file)
                    logger.info(
                        "[%s] Refreshed persisted floorplan result to %s",
                        video_id,
                        FLOORPLAN_CONSTRAINT_REVISION,
                    )
        return {
            "id": video_id,
            "status": "completed",
            "progress": 100,
            "message": "Анализ завершён",
            "result": result,
        }
    except Exception as exc:
        logger.warning("[%s] Could not load completed analysis for status: %s", video_id, exc)
        return {
            "id": video_id,
            "status": "completed",
            "progress": 100,
            "message": "Анализ завершён",
        }
# Configure CORS (Safari лучше работает с 127.0.0.1, не localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:3000",
        "http://localhost:8081",
        "http://localhost:8082",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "http://127.0.0.1:8082",
        "http://127.0.0.1:29483",
        "http://localhost:29483",
        "null",
        "http://45.67.57.72",
        "https://45.67.57.72",
        "https://trackai.eu.ngrok.io",
        "https://trackai-app.eu.ngrok.io",
        "https://trackai-backend.loca.lt",
        "https://trackai-frontend.loca.lt",
        "https://fa44db5269c86bf8-185-104-115-196.serveousercontent.com",
        "https://14e265884d57c1eb-185-104-115-196.serveousercontent.com",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|45\.67\.57\.72)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Увеличиваем лимит загрузки для DWG/PDF/видео (Starlette 0.40+)
UPLOAD_MAX_PART_SIZE = 1024 * 1024 * 500  # 500 MB

@app.middleware("http")
async def preflight_ok(request: Request, call_next):
    # Preflight (OPTIONS) всегда 200 — избегаем 400 из-за CORS/валидации
    if request.method == "OPTIONS":
        origin = request.headers.get("origin") or "*"
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "86400",
            },
        )
    return await call_next(request)


@app.middleware("http")
async def increase_upload_limit(request: Request, call_next):
    # Только convert-dwg: middleware парсит form с большим лимитом (upload-video и analyze-video парсят в своих хендлерах)
    if request.method == "POST" and request.url.path == "/api/convert-dwg":
        await request.form()
    return await call_next(request)

# #region agent log
_AGENT_DEBUG_LOG = Path("/Users/artembutko/Desktop/trackAI/.cursor/debug-64890b.log")


def _agent_debug_ndjson(
    hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "run1"
) -> None:
    try:
        payload = {
            "sessionId": "64890b",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        _AGENT_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as _f:
            _f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


@app.middleware("http")
async def agent_debug_http_log(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    _t0 = time.time()
    try:
        response = await call_next(request)
        _agent_debug_ndjson(
            "H1",
            "main.py:agent_debug_http_log",
            "response",
            {
                "path": request.url.path,
                "method": request.method,
                "status": getattr(response, "status_code", None),
                "ms": round((time.time() - _t0) * 1000, 2),
            },
        )
        return response
    except Exception as _e:
        _agent_debug_ndjson(
            "H2",
            "main.py:agent_debug_http_log",
            "exception",
            {
                "path": request.url.path,
                "method": request.method,
                "exc": type(_e).__name__,
            },
        )
        raise


# #endregion

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
VIDEOS_DIR = Path("videos")  # Для хранения оригинальных видео
VIDEO_PREVIEWS_DIR = Path("video_previews")
MANUAL_TRAJECTORIES_PATH = Path("backend/data/manual_trajectories.json")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
VIDEO_PREVIEWS_DIR.mkdir(exist_ok=True)
MANUAL_TRAJECTORIES_PATH.parent.mkdir(exist_ok=True, parents=True)

# Database initialization
DB_PATH = Path(__file__).parent / "data" / "database.db"
DB_PATH.parent.mkdir(exist_ok=True, parents=True)

_video_preview_lock = threading.Lock()
_video_preview_jobs: Set[str] = set()


def _find_uploaded_video_path(video_id: str) -> Optional[Path]:
    video_filename = UPLOADED_VIDEOS.get(video_id)
    video_path = (VIDEOS_DIR / video_filename) if video_filename else None
    if video_path and video_path.exists():
        return video_path

    matches = list(VIDEOS_DIR.glob(f"{video_id}_*"))
    for m in matches:
        if m.is_file():
            UPLOADED_VIDEOS[video_id] = m.name
            return m
    return None


def _video_preview_path(video_id: str) -> Path:
    return VIDEO_PREVIEWS_DIR / f"{video_id}.mp4"


def _ensure_video_preview(video_id: str) -> Optional[Path]:
    preview_path = _video_preview_path(video_id)
    if preview_path.exists() and preview_path.stat().st_size > 0:
        return preview_path

    source_path = _find_uploaded_video_path(video_id)
    if not source_path:
        return None

    with _video_preview_lock:
        if video_id in _video_preview_jobs:
            return None
        _video_preview_jobs.add(video_id)

    tmp_path = preview_path.with_suffix(".tmp.mp4")
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-vf",
            "scale='min(1280,iw)':-2",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ]
        logger.info(f"[{video_id}] Creating admin MP4 preview from {source_path.name}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=1800)
        tmp_path.replace(preview_path)
        logger.info(f"[{video_id}] Admin MP4 preview ready: {preview_path}")
        return preview_path
    except Exception as e:
        logger.warning(f"[{video_id}] Admin MP4 preview failed: {e}")
        tmp_path.unlink(missing_ok=True)
        return None
    finally:
        with _video_preview_lock:
            _video_preview_jobs.discard(video_id)


def _schedule_video_preview(video_id: str) -> None:
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(asyncio.to_thread(_ensure_video_preview, video_id))
    except RuntimeError:
        threading.Thread(target=_ensure_video_preview, args=(video_id,), daemon=True).start()


def _map_context_summary(map_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Keep /api/admin/tasks light; full map_context is loaded by /api/admin/tasks/{id}."""
    summary: Dict[str, Any] = {}
    if "client_source" in map_ctx:
        summary["client_source"] = map_ctx.get("client_source")
    if map_ctx.get("floorplan_id"):
        summary["floorplan_id"] = map_ctx.get("floorplan_id")
    if map_ctx.get("floor_plan_data"):
        summary["has_floor_plan_data"] = True
    if map_ctx.get("drawn_plan"):
        summary["has_drawn_plan"] = True
    return summary

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            data TEXT NOT NULL,
            preview_svg TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracking_tasks (
            id TEXT PRIMARY KEY,
            employee_name TEXT,
            video_filename TEXT NOT NULL,
            original_filename TEXT,
            map_context TEXT,
            status TEXT,
            progress INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()

def _update_task_status(task_id: str, status: str, progress: int = None):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if progress is not None:
            cursor.execute("UPDATE tracking_tasks SET status = ?, progress = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, progress, task_id))
        else:
            cursor.execute("UPDATE tracking_tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (status, task_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to update task status {task_id}: {e}")



def _parse_json_field(value: Any, default: Any = None) -> Any:
    if value in (None, "", "null", "undefined"):
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def _extract_map_context(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Re-analysis payloads commonly contain only ``floorplan_id``.  Do not
    # turn absent keys into explicit nulls: doing so erased the durable map
    # anchor saved by the original upload and made an otherwise valid mask
    # fail with ``missing_start_or_direction``.  Explicit null is still kept
    # when a UI intentionally sends the key to clear an anchor.
    map_context = {
        "floorplan_id": payload.get("floorplan_id") or DEFAULT_FLOORPLAN_ID,
    }
    if "floor_plan_data" in payload:
        map_context["floor_plan_data"] = payload.get("floor_plan_data")
    for key in ("drawn_plan", "reference_point", "direction_point"):
        if key in payload:
            map_context[key] = _parse_json_field(payload.get(key))
    return map_context


def _load_task_map_context(video_id: str) -> Dict[str, Any]:
    """Load the durable start/direction used for map-constrained R3 queries."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT map_context FROM tracking_tasks WHERE id = ?", (video_id,))
        row = cursor.fetchone()
        conn.close()
        parsed = json.loads(row[0]) if row and row[0] else {}
        if isinstance(parsed, dict):
            parsed.setdefault("floorplan_id", DEFAULT_FLOORPLAN_ID)
            return parsed
    except Exception as exc:
        logger.warning(f"[{video_id}] Failed to load map context: {exc}")
    return {"floorplan_id": DEFAULT_FLOORPLAN_ID}


def _merge_task_map_context(video_id: str, map_context: Optional[Dict[str, Any]]) -> None:
    """Persist the exact map anchor before an asynchronous analysis starts."""
    if not map_context:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT map_context FROM tracking_tasks WHERE id = ?", (video_id,))
        row = cursor.fetchone()
        try:
            existing = json.loads(row[0]) if row and row[0] else {}
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}
        # None is meaningful for anchor fields: clearing a direction in the
        # UI must not resurrect the previous database value when the
        # lightweight R3 trajectory endpoint is queried later.
        existing.update(map_context)
        existing.setdefault("floorplan_id", DEFAULT_FLOORPLAN_ID)
        cursor.execute(
            "UPDATE tracking_tasks SET map_context = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(existing), video_id),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning(f"[{video_id}] Failed to persist map context: {exc}")


def _load_manual_trajectories() -> Dict[str, Any]:
    if not MANUAL_TRAJECTORIES_PATH.exists():
        return {}
    try:
        with open(MANUAL_TRAJECTORIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to load manual trajectories: {e}")
        return {}


def _save_manual_trajectories(data: Dict[str, Any]) -> None:
    with open(MANUAL_TRAJECTORIES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _normalize_manual_trajectory_to_plan_space(trajectory: Any) -> Any:
    """Админка раньше сохраняла точки в 0–100; TrajectoryMap — 800×600."""
    if not trajectory or not isinstance(trajectory, list):
        return trajectory
    pts_xy: List[Tuple[float, float]] = []
    for p in trajectory:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            pts_xy.append((float(p[0]), float(p[1])))
    if not pts_xy:
        return trajectory
    max_x = max(x for x, _ in pts_xy)
    max_y = max(y for _, y in pts_xy)
    if max_x > 100.5 or max_y > 100.5:
        return trajectory
    out: List[List[float]] = []
    for p in trajectory:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            px = float(p[0]) / 100.0 * 800.0
            py = float(p[1]) / 100.0 * 600.0
            ph = float(p[2]) if len(p) >= 3 else 0.0
            out.append([px, py, ph])
    return out


def _make_manual_result(trajectory: Any, turn_points: Optional[List[Any]] = None) -> Dict[str, Any]:
    trajectory = _normalize_manual_trajectory_to_plan_space(trajectory)
    normalized_traj: List[List[float]] = []
    for p in trajectory or []:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x = float(p[0])
            y = float(p[1])
            h = float(p[2]) if len(p) >= 3 else 0.0
            normalized_traj.append([round(x, 4), round(y, 4), round(h, 2)])

    turns = turn_points or []
    return {
        "method": "manual_admin",
        "trajectory": normalized_traj,
        "turn_points": turns,
        "raw_turn_points": turns,
        "trajectory_turn_points": turns,
        "frame_count": 0,
        "trajectory_points": len(normalized_traj),
        "processing_stats": {
            "estimated_distance": 0.0,
            "scale_factor": 1.0,
            "fps": 0,
            "turns_detected": len(turns),
            "manual_override": True,
        },
        "total_processing_time": 0.0,
        "video_info": {
            "width": 0,
            "height": 0,
            "fps": 0,
            "frame_count": 0,
            "duration": 0,
        },
    }

# Telegram bot configuration
TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
DESKTOP_UPLOAD_BOT_TOKEN = (os.getenv("TRACKAI_DESKTOP_UPLOAD_BOT_TOKEN") or "").strip()
TELEGRAM_DESKTOP_SUBSCRIBERS_PATH = DB_PATH.parent / "telegram_desktop_subscribers.json"
_tg_desktop_sub_lock = threading.Lock()

async def get_telegram_chat_id():
    """Получить chat_id из Telegram updates или переменной окружения"""
    chat_id = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f'{TELEGRAM_API_URL}/getUpdates',
                params={'offset': -1, 'limit': 10}
            ) as updates_resp:
                if updates_resp.status == 200:
                    updates_data = await updates_resp.json()
                    if updates_data.get('result') and len(updates_data['result']) > 0:
                        # Ищем последнее сообщение от пользователя
                        for update in reversed(updates_data['result']):
                            if 'message' in update:
                                chat_id = update['message'].get('chat', {}).get('id')
                                if chat_id:
                                    logger.info(f"Found chat_id from Telegram updates: {chat_id}")
                                    break
    except Exception as e:
        logger.warning(f"Failed to get chat_id from Telegram: {e}")

    # Если не удалось получить chat_id, пробуем использовать переменную окружения
    if not chat_id:
        import os
        env_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if env_chat_id:
            try:
                chat_id = int(env_chat_id)
                logger.info(f"Using chat_id from environment: {chat_id}")
            except:
                logger.warning(f"Invalid TELEGRAM_CHAT_ID format: {env_chat_id}")
    
    return chat_id

async def send_error_to_telegram(error_message: str, context: str = ""):
    """Отправить ошибку в Telegram канал"""
    try:
        chat_id = await get_telegram_chat_id()
        if not chat_id:
            logger.warning("Skipping Telegram error notification: chat_id not found")
            return
        
        # Формируем сообщение об ошибке
        telegram_message = f"❌ ОШИБКА TrackAI\n\n"
        if context:
            telegram_message += f"📋 Контекст: {context}\n\n"
        telegram_message += f"🔴 Ошибка: {error_message}"
        
        # Ограничиваем длину сообщения
        telegram_message = telegram_message[:4096]
        
        # Преобразуем chat_id в int
        try:
            chat_id_int = int(chat_id) if isinstance(chat_id, str) and chat_id.isdigit() else chat_id
        except:
            chat_id_int = chat_id

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{TELEGRAM_API_URL}/sendMessage',
                json={
                    'chat_id': chat_id_int,
                    'text': telegram_message
                }
            ) as send_resp:
                if send_resp.status == 200:
                    logger.info(f"Error sent to Telegram successfully (chat_id: {chat_id_int})")
                else:
                    error_data = await send_resp.json() if send_resp.content_type == 'application/json' else await send_resp.text()
                    logger.warning(f"Failed to send error to Telegram: {send_resp.status} - {error_data}")
    except Exception as e:
        logger.error(f"Failed to send error to Telegram: {e}")


async def _telegram_send_with_token(token: str, chat_id: int, text: str) -> Optional[Dict[str, Any]]:
    if not token or not chat_id:
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text[:4096]},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                return await resp.json(content_type=None)
    except Exception as e:
        logger.warning(f"Telegram desktop upload send failed: {e}")
        return None


async def _telegram_latest_chat_id_for_token(token: str) -> Optional[int]:
    if not token:
        return None
    env_chat_id = (os.getenv("TRACKAI_DESKTOP_UPLOAD_CHAT_ID") or "").strip()
    if env_chat_id:
        try:
            return int(env_chat_id)
        except ValueError:
            logger.warning(f"Invalid TRACKAI_DESKTOP_UPLOAD_CHAT_ID: {env_chat_id}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"offset": -1, "limit": 10},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json(content_type=None)
        for update in reversed(data.get("result") or []):
            message = update.get("message") or update.get("edited_message") or {}
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            if chat_id is not None:
                return int(chat_id)
    except Exception as e:
        logger.warning(f"Telegram desktop upload getUpdates failed: {e}")
    return None


def _load_desktop_upload_subscribers() -> Set[int]:
    subscribers: Set[int] = set()
    env_chat_id = (os.getenv("TRACKAI_DESKTOP_UPLOAD_CHAT_ID") or "").strip()
    if env_chat_id:
        for raw in env_chat_id.replace(";", ",").split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                subscribers.add(int(raw))
            except ValueError:
                logger.warning(f"Invalid TRACKAI_DESKTOP_UPLOAD_CHAT_ID item: {raw}")

    with _tg_desktop_sub_lock:
        if TELEGRAM_DESKTOP_SUBSCRIBERS_PATH.exists():
            try:
                data = json.loads(TELEGRAM_DESKTOP_SUBSCRIBERS_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        try:
                            subscribers.add(int(item))
                        except (TypeError, ValueError):
                            continue
            except Exception as e:
                logger.warning(f"desktop Telegram subscribers load failed: {e}")

    return subscribers


def _save_desktop_upload_subscribers(ids: Set[int]) -> None:
    with _tg_desktop_sub_lock:
        TELEGRAM_DESKTOP_SUBSCRIBERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        TELEGRAM_DESKTOP_SUBSCRIBERS_PATH.write_text(
            json.dumps(sorted(ids), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


async def _refresh_desktop_upload_subscribers_from_updates(token: str) -> Set[int]:
    subscribers = _load_desktop_upload_subscribers()
    if not token:
        return subscribers
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"limit": 100},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json(content_type=None)
        for update in data.get("result") or []:
            message = update.get("message") or update.get("edited_message") or {}
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            if chat_id is not None:
                subscribers.add(int(chat_id))
        if subscribers:
            _save_desktop_upload_subscribers(subscribers)
    except Exception as e:
        logger.warning(f"desktop Telegram getUpdates refresh failed: {e}")
    return subscribers


async def send_desktop_upload_notification(
    video_id: str,
    original_filename: str,
    file_size: int,
    gpu_url: str,
    client_source: str = "unknown",
) -> None:
    """Уведомление о загрузке после успешной отправки файла на GPU worker."""
    token = DESKTOP_UPLOAD_BOT_TOKEN
    if not token:
        return
    size_mb = file_size / (1024 * 1024) if file_size else 0
    text = (
        "Новая загрузка видео TrackAI\n"
        f"Источник: {client_source or 'unknown'}\n"
        f"Видео: {original_filename}\n"
        f"video_id: {video_id}\n"
        f"Размер: {size_mb:.1f} MB\n"
        f"GPU: {gpu_url}\n"
        "Статус: видео отправлено на 3090"
    )

    sent_to: Set[int] = set()
    desktop_subscribers = await _refresh_desktop_upload_subscribers_from_updates(token)
    if not desktop_subscribers:
        logger.warning(
            "Desktop upload Telegram notification skipped: no chat_id. "
            "Open @ceoscrepka_bot and send /start, or set TRACKAI_DESKTOP_UPLOAD_CHAT_ID."
        )

    for cid in desktop_subscribers:
        res = await _telegram_send_with_token(token, cid, text)
        if res and res.get("ok"):
            sent_to.add(cid)
        else:
            logger.warning(f"desktop upload Telegram notify to desktop subscriber {cid} failed: {res}")

    latest_chat_id = await _telegram_latest_chat_id_for_token(token)
    if latest_chat_id and latest_chat_id not in sent_to:
        res = await _telegram_send_with_token(token, latest_chat_id, text)
        if not res or not res.get("ok"):
            logger.warning(f"desktop upload Telegram notify to latest chat failed: {res}")
        elif latest_chat_id:
            desktop_subscribers.add(latest_chat_id)
            _save_desktop_upload_subscribers(desktop_subscribers)


@app.post("/api/telegram/desktop-test")
async def telegram_desktop_test():
    token = DESKTOP_UPLOAD_BOT_TOKEN
    if not token:
        raise HTTPException(status_code=500, detail="TRACKAI_DESKTOP_UPLOAD_BOT_TOKEN is empty")
    subscribers = await _refresh_desktop_upload_subscribers_from_updates(token)
    if not subscribers:
        return {
            "success": False,
            "reason": "no_chat_id",
            "message": "Откройте @ceoscrepka_bot в Telegram и отправьте /start",
        }
    results = []
    for cid in sorted(subscribers):
        res = await _telegram_send_with_token(token, cid, "Тест TrackAI desktop upload notifications")
        results.append({"chat_id": cid, "result": res})
    return {"success": any((r["result"] or {}).get("ok") for r in results), "subscribers": sorted(subscribers), "results": results}


# --- Telegram-бот «обработка видео»: подписчики + webhook ---
# Токен: только TELEGRAM_PROCESSING_BOT_TOKEN в окружении (не коммитить в репозиторий).
# Webhook: HTTPS POST на https://<ваш-домен>/api/telegram/processing-webhook
#   Рекомендуется secret_token в setWebhook и переменная TELEGRAM_WEBHOOK_SECRET (совпадает с токеном из BotFather).
TELEGRAM_PROCESSING_SUBSCRIBERS_PATH = DB_PATH.parent / "telegram_processing_subscribers.json"
_tg_processing_sub_lock = threading.Lock()


def _get_processing_bot_token() -> str:
    return (os.getenv("TELEGRAM_PROCESSING_BOT_TOKEN") or "").strip()


def _load_processing_subscribers() -> Set[int]:
    with _tg_processing_sub_lock:
        if not TELEGRAM_PROCESSING_SUBSCRIBERS_PATH.exists():
            return set()
        try:
            raw = TELEGRAM_PROCESSING_SUBSCRIBERS_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, list):
                return set()
            out: Set[int] = set()
            for x in data:
                try:
                    out.add(int(x))
                except (TypeError, ValueError):
                    continue
            return out
        except Exception as e:
            logger.warning(f"telegram subscribers load failed: {e}")
            return set()


def _save_processing_subscribers(ids: Set[int]) -> None:
    with _tg_processing_sub_lock:
        TELEGRAM_PROCESSING_SUBSCRIBERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        TELEGRAM_PROCESSING_SUBSCRIBERS_PATH.write_text(
            json.dumps(sorted(ids), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _add_processing_subscriber(chat_id: int) -> None:
    s = _load_processing_subscribers()
    s.add(chat_id)
    _save_processing_subscribers(s)


async def _telegram_processing_api(method: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    token = _get_processing_bot_token()
    if not token:
        return None
    url = f"https://api.telegram.org/bot{token}/{method}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                return await resp.json(content_type=None)
    except Exception as e:
        logger.warning(f"Telegram processing API {method} failed: {e}")
        return None


async def broadcast_new_processing_to_all_subscribers() -> None:
    """Всем, кто нажал /start у бота обработки — уведомление о новой задаче."""
    token = _get_processing_bot_token()
    if not token:
        return
    subs = _load_processing_subscribers()
    if not subs:
        logger.info("Telegram broadcast: нет подписчиков (/start не доходил до сервера или long poll выключен)")
        return
    text = "Новая Обработка"
    for cid in subs:
        res = await _telegram_processing_api(
            "sendMessage",
            {"chat_id": cid, "text": text},
        )
        if not res or not res.get("ok"):
            logger.warning(f"broadcast to {cid} failed: {res}")


async def broadcast_new_processing_for_video(video_id: str) -> None:
    """
    Notify processing subscribers with a message that includes the position of this video
    among all videos for the same employee: "Загружено видео i/n\nНовая обработка — <filename>"
    Falls back to a generic message if employee or DB data is missing.
    """
    token = _get_processing_bot_token()
    if not token:
        return
    subs = _load_processing_subscribers()
    if not subs:
        logger.info("Telegram broadcast (per-video): нет подписчиков")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Fetch row and map_context to detect optional batch_id provided by client
        cursor.execute(
            "SELECT employee_name, original_filename, created_at, video_filename, map_context FROM tracking_tasks WHERE id = ?",
            (video_id,),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            text = "Новая Обработка"
        else:
            employee_name, original_filename, created_at, video_filename, map_ctx_raw = row
            batch_id = None
            batch_size = None
            if map_ctx_raw:
                try:
                    mc = json.loads(map_ctx_raw)
                    batch_id = mc.get("batch_id")
                    batch_size = int(mc.get("batch_size")) if mc.get("batch_size") is not None else None
                except Exception:
                    batch_id = None

            # If client provided a batch_id (frontend should set when uploading multiple files at once),
            # count items in that batch to form i/n = position/total.
            if batch_id:
                try:
                    cursor.execute(
                        "SELECT id FROM tracking_tasks WHERE map_context LIKE ? ORDER BY created_at",
                        (f'%"batch_id":"{batch_id}"%',),
                    )
                    rows = [r[0] for r in cursor.fetchall()]
                    conn.close()
                    total = len(rows) if rows else 1
                    try:
                        idx = rows.index(video_id) + 1
                    except ValueError:
                        idx = 1
                    fname = original_filename or video_id
                    text = f"Загружено видео {idx}/{total}\nНовая обработка — {fname}"
                except Exception:
                    conn.close()
                    text = f"Новая Обработка — {original_filename or video_id}"
            elif employee_name:
                # Consider only recent uploads for the same employee as the "batch".
                # This prevents counting all historical uploads. Window: last 10 minutes.
                try:
                    cursor.execute(
                        """SELECT id FROM tracking_tasks
                           WHERE employee_name = ? AND created_at >= datetime('now', '-10 minutes')
                           ORDER BY created_at""",
                        (employee_name,),
                    )
                    rows = [r[0] for r in cursor.fetchall()]
                except Exception:
                    # Fallback to full history if datetime function not supported
                    cursor.execute(
                        "SELECT id FROM tracking_tasks WHERE employee_name = ? ORDER BY created_at",
                        (employee_name,),
                    )
                    rows = [r[0] for r in cursor.fetchall()]
                conn.close()
                total = len(rows) if rows else 1
                try:
                    idx = rows.index(video_id) + 1
                except ValueError:
                    idx = 1
                fname = original_filename or video_id
                text = f"Загружено видео {idx}/{total}\nНовая обработка — {fname}"
            else:
                conn.close()
                fname = original_filename or video_id
                text = f"Новая Обработка — {fname}"

        # Send to subscribers
        for cid in subs:
            res = await _telegram_processing_api(
                "sendMessage",
                {"chat_id": cid, "text": text},
            )
            if not res or not res.get("ok"):
                logger.warning(f"broadcast per-video to {cid} failed: {res}")
    except Exception as e:
        logger.warning(f"broadcast_new_processing_for_video failed: {e}")


# ──────────────────────────────────────────────
# R³ helpers — конвертация camera poses в траекторию
# ──────────────────────────────────────────────

def _project_r3_camera_path_to_2d(raw_points: List[List[float]]) -> List[List[float]]:
    """Project arbitrary R3 3D camera translations to the dominant 2D motion plane."""
    if len(raw_points) < 2:
        return raw_points
    try:
        import numpy as _np2
        pts = _np2.array(raw_points, dtype=_np2.float64)
        finite = _np2.isfinite(pts).all(axis=1)
        if finite.sum() < 2:
            return raw_points
        valid = pts[finite]
        centered = valid - valid[0]
        if valid.shape[0] >= 3:
            _, _, vh = _np2.linalg.svd(centered - centered.mean(axis=0, keepdims=True), full_matrices=False)
            basis = vh[:2].T
            coords = centered @ basis
        else:
            direction = centered[-1]
            norm = _np2.linalg.norm(direction)
            if norm < 1e-9:
                return [[0.0, 0.0, 0.0] for _ in raw_points]
            e1 = direction / norm
            aux = _np2.array([0.0, 1.0, 0.0])
            if abs(float(_np2.dot(e1, aux))) > 0.95:
                aux = _np2.array([1.0, 0.0, 0.0])
            e2 = _np2.cross(e1, aux)
            e2 = e2 / max(_np2.linalg.norm(e2), 1e-9)
            coords = centered @ _np2.stack([e1, e2], axis=1)

        full = _np2.zeros((len(raw_points), 3), dtype=_np2.float64)
        full[finite, 0] = coords[:, 0]
        full[finite, 1] = coords[:, 1]
        if valid.shape[1] >= 3:
            full[finite, 2] = valid[:, 1] - valid[0, 1]
        return [[round(float(x), 6), round(float(y), 6), round(float(z), 6)] for x, y, z in full]
    except Exception:
        # Conservative fallback: OpenCV-style camera paths usually move on X/Z; Y is often height.
        return [[p[0], p[2] if len(p) > 2 else p[1], p[1] if len(p) > 1 else 0.0] for p in raw_points]


def _clean_r3_plan_trajectory(points: List[List[float]]) -> List[List[float]]:
    """Remove pose jumps and smooth the R3 path before it is projected onto a floor plan."""
    if len(points) < 5:
        return points
    try:
        import numpy as _np2

        pts = _np2.array(points, dtype=_np2.float64)
        finite = _np2.isfinite(pts).all(axis=1)
        if finite.sum() < 5:
            return points

        # Fill occasional invalid values from nearest previous valid point.
        for i in range(len(pts)):
            if not finite[i]:
                pts[i] = pts[i - 1] if i > 0 else pts[finite][0]

        steps = _np2.linalg.norm(_np2.diff(pts[:, :2], axis=0), axis=1)
        positive_steps = steps[steps > 1e-9]
        if positive_steps.size:
            median_step = float(_np2.median(positive_steps))
            mad = float(_np2.median(_np2.abs(positive_steps - median_step)))
            jump_limit = max(median_step * 8.0, median_step + 6.0 * mad, 1e-6)
            for i, step in enumerate(steps, start=1):
                if step > jump_limit:
                    # Keep continuity: replace isolated pose teleport with previous point.
                    pts[i] = pts[i - 1]

        # Small centered moving average. Edge padding keeps start/end anchored enough for map alignment.
        window = min(9, len(pts) if len(pts) % 2 == 1 else len(pts) - 1)
        if window >= 5:
            pad = window // 2
            padded = _np2.pad(pts, ((pad, pad), (0, 0)), mode="edge")
            kernel = _np2.ones(window, dtype=_np2.float64) / window
            smoothed = _np2.empty_like(pts)
            for dim in range(pts.shape[1]):
                smoothed[:, dim] = _np2.convolve(padded[:, dim], kernel, mode="valid")
            # Preserve exact first point as the local origin for frontend alignment.
            smoothed -= smoothed[0] - pts[0]
            pts = smoothed

        return [[round(float(x), 6), round(float(y), 6), round(float(z), 6)] for x, y, z in pts]
    except Exception:
        return points


def _r3_poses_to_trajectory(r3_result: dict, scale_factor: float = 1.0) -> dict:
    """Convert R3 c2w poses into matching plan and 3D trajectory products."""
    camera_poses = r3_result.get("camera_poses", r3_result.get("camera_poses", []))
    if not camera_poses:
        # Fallback: читаем из output_dir
        output_dir = r3_result.get("output_dir", "")
        if output_dir:
            camera_dir = Path(output_dir) / "camera"
            if camera_dir.exists():
                camera_files = sorted(camera_dir.glob("*.npz"))
                camera_poses = []
                for cf in camera_files:
                    try:
                        import numpy as _np2
                        data = _np2.load(str(cf))
                        pose = data["pose"].tolist() if "pose" in data else None
                        if pose:
                            camera_poses.append({"frame": int(cf.stem), "pose": pose})
                    except Exception:
                        pass

    trajectory_bundle = build_r3_trajectory(
        camera_poses,
        pose_confidence=r3_result.get("pose_confidence"),
        frame_selection=r3_result.get("frame_selection"),
        run_params=r3_result.get("run_params"),
    )
    trajectory = trajectory_bundle["plan_trajectory"]
    raw_trajectory_3d = trajectory_bundle["raw_trajectory_3d"]
    turn_points = trajectory_bundle["turn_points"]
    num_frames = len(trajectory)
    confidence = trajectory_bundle.get("pose_confidence") or []
    valid_conf = [value for value in confidence if value is not None]

    run_params = r3_result.get("run_params", {})
    inference_time = run_params.get("inference_time_s", 0) or 0

    # Distance is measured in plan-space, never from the display-only 3D path.
    estimated_distance = 0.0
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]
        dz = trajectory[i][2] - trajectory[i - 1][2]
        estimated_distance += (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

    # Средняя уверенность
    avg_conf = None
    if valid_conf:
        try:
            if _np is not None:
                avg_conf = round(float(_np.mean(valid_conf)), 2)
            else:
                avg_conf = round(sum(valid_conf) / len(valid_conf), 2)
        except Exception:
            avg_conf = None

    return {
        "method": "r3_reconstruction",
        "trajectory": trajectory,
        "plan_trajectory": trajectory,
        "raw_trajectory_3d": raw_trajectory_3d,
        "turn_points": turn_points,
        "frame_count": num_frames,
        "trajectory_points": len(trajectory),
        "r3_camera_points": raw_trajectory_3d,
        "r3_raw_camera_points": trajectory_bundle["raw_camera_points"],
        "r3_source_frame_indices": trajectory_bundle["source_frame_indices"],
        "r3_source_timestamps_seconds": trajectory_bundle.get("source_timestamps_seconds", []),
        "r3_pose_confidence": confidence or None,
        "r3_pose_graph": r3_result.get("pose_graph"),
        "r3_pose_graph_candidate": r3_result.get("pose_graph_candidate"),
        "r3_scale_aware_candidate": r3_result.get("scale_aware_candidate"),
        "pointcloud_status": r3_result.get("pointcloud_status"),
        "r3_projection": trajectory_bundle["trajectory_quality"].get("projection", {}).get("method", "robust_floor_plane"),
        "processing_stats": {
            "estimated_distance": round(estimated_distance, 2),
            "scale_factor": 1.0,
            "input_scale_factor": scale_factor,
            "r3_map_scale_disabled": True,
            "fps": round(num_frames / max(inference_time, 0.1), 1) if inference_time > 0 else 0,
            "turns_detected": len(turn_points),
            "avg_pose_confidence": avg_conf,
            "r3_trajectory_quality": trajectory_bundle["trajectory_quality"],
        },
        "total_processing_time": inference_time,
        "video_info": {
            "width": 0, "height": 0, "fps": 0, "frame_count": num_frames, "duration": 0,
        },
    }


def _merge_r3_production_trajectory(base: dict, selected: dict) -> dict:
    """Promote the worker's accepted production pose source into saved output.

    Raw camera poses remain available for 3D diagnostics; only the floor-plan
    trajectory and its turn/source metadata are selected here.
    """
    if not isinstance(selected, dict) or not selected.get("success"):
        return base
    plan = selected.get("plan_trajectory") or selected.get("trajectory") or []
    if not isinstance(plan, list) or len(plan) < 2:
        return base
    source = str(selected.get("trajectory_source") or "raw")
    method = {
        "scale_aware_candidate": "r3_reconstruction_scale_aware",
        "robust_candidate": "r3_reconstruction_robust_candidate",
    }.get(source, "r3_reconstruction")
    updated = dict(base)
    updated.update({
        "method": method,
        "trajectory": plan,
        "plan_trajectory": plan,
        "turn_points": selected.get("turn_points") or [],
        "trajectory_points": len(plan),
        "r3_source_frame_indices": selected.get("source_frame_indices") or [],
        "r3_source_timestamps_seconds": selected.get("source_timestamps_seconds") or [],
        "r3_pose_graph": selected.get("pose_graph"),
        "r3_pose_graph_candidate": selected.get("pose_graph_candidate"),
        "r3_scale_aware_candidate": selected.get("scale_aware_candidate"),
    })
    # Keep confidence samples index-aligned with the promoted plan. Prefer the
    # selected source's own confidence; otherwise resample/clear the base array.
    selected_confidence = selected.get("r3_pose_confidence") or selected.get("pose_confidence")
    if isinstance(selected_confidence, list) and len(selected_confidence) == len(plan):
        updated["r3_pose_confidence"] = selected_confidence
    else:
        base_confidence = base.get("r3_pose_confidence")
        if isinstance(base_confidence, list) and len(base_confidence) >= 2 and len(plan) >= 2:
            import numpy as _np
            # Keep name distinct from trajectory `source` below — rebinding it
            # previously wrote pose-confidence arrays into r3_trajectory_source
            # and broke LingBot restore on /api/r3-trajectory refresh.
            confidence_series = _np.asarray(
                [
                    float(value) if value is not None else _math.nan
                    for value in base_confidence
                ],
                dtype=float,
            )
            finite = _np.flatnonzero(_np.isfinite(confidence_series))
            if len(finite) >= 2:
                confidence_series = _np.interp(
                    _np.arange(len(confidence_series)), finite, confidence_series[finite]
                )
                updated["r3_pose_confidence"] = _np.interp(
                    _np.linspace(0.0, 1.0, len(plan)),
                    _np.linspace(0.0, 1.0, len(confidence_series)),
                    confidence_series,
                ).tolist()
            else:
                updated["r3_pose_confidence"] = []
        else:
            updated["r3_pose_confidence"] = []
    quality = selected.get("trajectory_quality") or {}
    stats = dict(base.get("processing_stats") or {})
    stats.update({
        "turns_detected": len(updated["turn_points"]),
        "r3_trajectory_quality": quality,
        "r3_trajectory_source": source,
        "r3_trajectory_source_requested": selected.get(
            "trajectory_source_requested", "scale_aware_candidate"
        ),
        "r3_trajectory_source_fallback_reason": selected.get(
            "trajectory_source_fallback_reason"
        ),
        "r3_trajectory_source_selection": selected.get(
            "trajectory_source_selection"
        ) or {},
    })
    updated["processing_stats"] = stats
    return updated


def _schedule_process_video_background(
    background_tasks: Optional[BackgroundTasks],
    video_id: str,
    video_path: Optional[Path],
    original_filename: str,
    scale_factor: float,
    stabilize: bool,
    detect_interval: int,
    turn_vote_threshold: int,
    use_ml_roi: bool,
    map_context: Optional[Dict[str, Any]],
) -> None:
    if background_tasks is not None:
        background_tasks.add_task(
            process_video_background,
            video_id,
            video_path,
            original_filename,
            scale_factor,
            stabilize,
            detect_interval,
            turn_vote_threshold,
            use_ml_roi,
            map_context,
        )
    else:
        asyncio.create_task(
            process_video_background(
                video_id,
                video_path,
                original_filename,
                scale_factor,
                stabilize,
                detect_interval,
                turn_vote_threshold,
                use_ml_roi,
                map_context,
            )
        )


async def _tg_get_webhook_info_raw() -> Dict[str, Any]:
    token = _get_processing_bot_token()
    if not token:
        return {}
    url = f"https://api.telegram.org/bot{token}/getWebhookInfo"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                return await resp.json(content_type=None)
    except Exception as e:
        logger.warning(f"getWebhookInfo: {e}")
        return {}


async def _tg_should_use_long_polling() -> bool:
    """Без HTTPS webhook /start не работает — включаем long poll (если не отключено env)."""
    if not _get_processing_bot_token():
        return False
    if os.getenv("TELEGRAM_PROCESSING_POLLING", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    info = await _tg_get_webhook_info_raw()
    if not info.get("ok"):
        return True
    res = info.get("result") or {}
    url = (res.get("url") or "").strip()
    return len(url) == 0


_tg_poll_offset = 0
_tg_poll_lock_file: Optional[TextIO] = None

try:
    import fcntl as _fcntl_mod

    _HAS_FCNTL = True
except ImportError:
    _fcntl_mod = None
    _HAS_FCNTL = False


async def telegram_processing_poll_updates_loop() -> None:
    """Long polling getUpdates — работает без HTTPS (в отличие от webhook)."""
    global _tg_poll_offset
    token = _get_processing_bot_token()
    if not token:
        return
    logger.info("Telegram processing bot: long polling активен (webhook пустой или недоступен)")
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.telegram.org/bot{token}/getUpdates",
                    params={
                        "timeout": 50,
                        "offset": _tg_poll_offset,
                        "allowed_updates": json.dumps(["message", "edited_message"]),
                    },
                    timeout=aiohttp.ClientTimeout(total=55),
                ) as resp:
                    data = await resp.json(content_type=None)
            if not data.get("ok"):
                logger.warning(f"getUpdates: {data}")
                await asyncio.sleep(3)
                continue
            for upd in data.get("result", []):
                _tg_poll_offset = max(_tg_poll_offset, int(upd.get("update_id", 0)) + 1)
                try:
                    await dispatch_telegram_processing_update(upd, None)
                except Exception as ex:
                    logger.error(f"dispatch telegram update: {ex}", exc_info=True)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Telegram poll loop: {e}")
            await asyncio.sleep(3)


async def dispatch_telegram_processing_update(
    update: Dict[str, Any],
    background_tasks: Optional[BackgroundTasks],
) -> None:
    """Общая логика: /start → подписка; видео → очередь + рассылка."""
    token = _get_processing_bot_token()
    if not token:
        return

    message = update.get("message") or update.get("edited_message")
    if not message:
        return

    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return

    text = (message.get("text") or "").strip()
    if text.startswith("/start"):
        _add_processing_subscriber(int(chat_id))
        await _telegram_processing_api(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": "Вы подписаны на уведомления. Когда кто-то отправит видео на обработку (сайт или этот чат), вам придёт «Новая Обработка».",
            },
        )
        return

    picked = _telegram_message_pick_video(message)
    if not picked:
        return

    file_id, orig_name = picked
    if not orig_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        await _telegram_processing_api(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": "Пришлите видео в формате MP4, AVI, MOV или MKV (как файл или как видео).",
            },
        )
        return

    from_user = message.get("from") or {}
    employee_name = (
        from_user.get("username")
        or from_user.get("first_name")
        or "Telegram"
    )
    if isinstance(employee_name, str):
        employee_name = employee_name.strip() or "Telegram"

    video_id = str(uuid.uuid4())
    video_filename = f"{video_id}_{orig_name}"
    video_path = VIDEOS_DIR / video_filename

    try:
        await _telegram_download_file(token, file_id, video_path)
    except Exception as e:
        logger.error(f"Telegram video download failed: {e}", exc_info=True)
        await _telegram_processing_api(
            "sendMessage",
            {"chat_id": chat_id, "text": f"Не удалось скачать файл: {e}"},
        )
        return

    UPLOADED_VIDEOS[video_id] = video_filename
    processing_status[video_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Поставлено в очередь на обработку (Telegram)",
        "start_time": time.time(),
    }

    map_context: Optional[Dict[str, Any]] = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO tracking_tasks (id, employee_name, video_filename, original_filename, map_context, status) VALUES (?, ?, ?, ?, ?, ?)",
            (video_id, employee_name, video_filename, orig_name, json.dumps(map_context), "queued"),
        )
        conn.commit()
        conn.close()
    except Exception as db_err:
        logger.error(f"Telegram task DB save failed: {db_err}")

    scale_factor = 12.306
    stabilize = True
    detect_interval = 3
    turn_vote_threshold = 3
    use_ml_roi = True

    # Немедленно запускаем обработку на GPU Worker
    _schedule_process_video_background(
        background_tasks,
        video_id,
        video_path,
        original_filename or file.filename,
        scale_factor,
        stabilize,
        detect_interval,
        turn_vote_threshold,
        use_ml_roi,
        None,  # map_context
    )

    await broadcast_new_processing_for_video(video_id)

    await _telegram_processing_api(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": "Видео принято, обработка запущена.",
        },
    )


def _telegram_message_pick_video(message: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """(file_id, original_filename) или None."""
    if "video" in message:
        v = message["video"]
        fid = v.get("file_id")
        if not fid:
            return None
        fn = v.get("file_name") or "telegram_video.mp4"
        return fid, fn
    if "document" in message:
        d = message["document"]
        mime = (d.get("mime_type") or "").lower()
        fn = (d.get("file_name") or "").strip() or "telegram_video.bin"
        ext_ok = fn.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        if not ext_ok and not mime.startswith("video/"):
            return None
        fid = d.get("file_id")
        if not fid:
            return None
        return fid, fn
    return None


async def _telegram_download_file(token: str, file_id: str, dest: Path) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.telegram.org/bot{token}/getFile",
            params={"file_id": file_id},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as r:
            gj = await r.json(content_type=None)
        if not gj or not gj.get("ok"):
            raise RuntimeError(f"getFile failed: {gj}")
        fp = gj["result"]["file_path"]
        url = f"https://api.telegram.org/file/bot{token}/{fp}"
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=600),
        ) as r2:
            r2.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as out:
                async for chunk in r2.content.iter_chunked(1024 * 1024):
                    out.write(chunk)


@app.post("/api/telegram/processing-webhook")
async def telegram_processing_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Входящие обновления от Telegram Bot API (setWebhook).
    Логика совпадает с long polling; см. dispatch_telegram_processing_update.
    """
    secret = (os.getenv("TELEGRAM_WEBHOOK_SECRET") or "").strip()
    if secret and request.headers.get("X-Telegram-Bot-Api-Secret-Token") != secret:
        raise HTTPException(status_code=403, detail="Forbidden")

    if not _get_processing_bot_token():
        return {"ok": True, "skipped": True, "reason": "TELEGRAM_PROCESSING_BOT_TOKEN not set"}

    try:
        update = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    await dispatch_telegram_processing_update(update, background_tasks)
    return {"ok": True}


FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dist")

@app.get("/")
async def root():
    # Serve frontend if built, otherwise API info
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"), media_type="text/html")
    return {"message": "TrackAI Video Analysis API", "version": "1.0.0"}


async def _fetch_lingbot_json(path: str, timeout_seconds: int = 30) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{LINGBOT_WORKER_URL}{path}",
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        ) as resp:
            text = await resp.text()
            if resp.status >= 300:
                raise RuntimeError(f"LingBot worker error ({resp.status}): {text[:500]}")
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError("LingBot worker returned invalid JSON") from exc


def _lingbot_to_trackai_result(
    video_id: str,
    session_id: str,
    metadata: Dict[str, Any],
    trajectory_payload: Dict[str, Any],
) -> Dict[str, Any]:
    poses = trajectory_payload.get("poses") if isinstance(trajectory_payload, dict) else []
    trajectory: List[List[float]] = []
    raw_trajectory_3d: List[List[float]] = []
    pose_confidence: List[Optional[float]] = []
    source_timestamps_seconds: List[Optional[float]] = []
    camera_poses: List[Dict[str, Any]] = []

    target_frames = int(metadata.get("target_frames") or 0)
    fps = float(metadata.get("fps") or 0.0)

    if isinstance(poses, list):
        for pose in poses:
            position = None
            if isinstance(pose, dict):
                position = pose.get("position")
                if position is None and isinstance(pose.get("c2w"), list):
                    c2w = pose.get("c2w")
                    try:
                        position = [c2w[0][3], c2w[1][3], c2w[2][3]]
                    except Exception:
                        position = None
            elif isinstance(pose, list):
                position = pose

            if not isinstance(position, list) or len(position) < 3:
                continue
            try:
                x = float(position[0])
                y = float(position[1])
                z = float(position[2])
            except Exception:
                continue
            if all(_math.isfinite(v) for v in (x, y, z)):
                # TrackAI plan UI expects x/y/z-like points. For LingBot camera
                # poses, horizontal movement is better represented by x/z; y is height.
                trajectory.append([x, z, y])
                raw_trajectory_3d.append([x, y, z])
                confidence = pose.get("confidence") if isinstance(pose, dict) else None
                try:
                    confidence_value = float(confidence) if confidence is not None else None
                    if confidence_value is not None and not _math.isfinite(confidence_value):
                        confidence_value = None
                except (TypeError, ValueError):
                    confidence_value = None
                pose_confidence.append(confidence_value)
                timestamp_value: Optional[float] = None
                if isinstance(pose, dict):
                    for key in ("timestamp", "timestamp_seconds", "time_seconds", "t"):
                        if pose.get(key) is None:
                            continue
                        try:
                            candidate = float(pose.get(key))
                        except (TypeError, ValueError):
                            candidate = float("nan")
                        if _math.isfinite(candidate):
                            timestamp_value = candidate
                            break
                    if timestamp_value is None and fps > 1e-9:
                        frame_idx = pose.get("frame_idx", len(trajectory) - 1)
                        try:
                            frame_value = float(frame_idx)
                        except (TypeError, ValueError):
                            frame_value = float("nan")
                        if _math.isfinite(frame_value):
                            timestamp_value = frame_value / fps
                source_timestamps_seconds.append(timestamp_value)
                if isinstance(pose, dict) and isinstance(pose.get("c2w"), list):
                    camera_poses.append({
                        "frame_idx": pose.get("frame_idx", len(trajectory) - 1),
                        "source_frame_idx": pose.get("source_frame_idx"),
                        "c2w": pose.get("c2w"),
                        "confidence": confidence_value,
                        "timestamp": timestamp_value,
                    })

    estimated_distance = 0.0
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]
        dz = trajectory[i][2] - trajectory[i - 1][2]
        estimated_distance += (dx * dx + dy * dy + dz * dz) ** 0.5

    target_frames = int(metadata.get("target_frames") or len(trajectory) or 0)
    fps = float(metadata.get("fps") or 0.0)
    video_info = _get_video_info_for_ui(
        _find_uploaded_video_path(video_id),
        fallback_frames=target_frames,
        fallback_fps=fps,
    )

    timings = metadata.get("timings") if isinstance(metadata.get("timings"), dict) else {}
    total_seconds = 0.0
    if isinstance(metadata.get("outputs"), dict):
        output_timings = metadata["outputs"].get("timings")
        if isinstance(output_timings, dict):
            total_seconds = float(output_timings.get("total_seconds") or 0.0)
    total_seconds = total_seconds or float(timings.get("total_seconds") or 0.0)
    processing_fps = (len(trajectory) / total_seconds) if total_seconds > 0 else 0.0
    if (
        all(value is None for value in source_timestamps_seconds)
        and len(trajectory) >= 2
        and fps > 1e-9
    ):
        source_timestamps_seconds = [
            round(index / fps, 6) for index in range(len(trajectory))
        ]

    return {
        "method": "lingbot_map",
        "trajectory": trajectory,
        "plan_trajectory": trajectory,
        "raw_trajectory_3d": raw_trajectory_3d,
        "turn_points": [],
        "trajectory_turn_points": [],
        "frame_count": len(trajectory),
        "trajectory_points": len(trajectory),
        "lingbot_session_id": session_id,
        "lingbot_metadata": metadata,
        "lingbot_trajectory": trajectory_payload,
        "lingbot_camera_poses": camera_poses,
        "lingbot_pose_confidence": pose_confidence,
        "lingbot_source_timestamps_seconds": source_timestamps_seconds,
        "source_timestamps_seconds": source_timestamps_seconds,
        "lingbot_frame_selection": (
            trajectory_payload.get("frame_selection", {})
            if isinstance(trajectory_payload, dict)
            else {}
        ),
        "processing_stats": {
            "algorithm": "LingBot-Map",
            "session_id": session_id,
            "status": "completed",
            "estimated_distance": round(estimated_distance, 3),
            "total_distance": round(estimated_distance, 3),
            "turns_detected": 0,
            "processing_fps": round(processing_fps, 2),
            "analysis_time": round(total_seconds, 2),
            "target_frames": target_frames,
            "source_poses": len(poses) if isinstance(poses, list) else 0,
        },
        "video_info": video_info,
    }


async def _merge_lingbot_status(video_id: str, status_data: Dict[str, Any]) -> Dict[str, Any]:
    session_id = status_data.get("lingbot_session_id")
    if not session_id:
        return status_data

    try:
        worker_status = await _fetch_lingbot_json(f"/sessions/{session_id}/status")
    except Exception as exc:
        merged = dict(status_data)
        merged["message"] = f"LingBot-Map статус временно недоступен: {exc}"
        merged.setdefault("progress", 0)
        return merged

    worker_state = worker_status.get("status", status_data.get("status", "queued"))
    progress = int(round(float(worker_status.get("progress") or 0.0) * 100))
    if worker_state == "queued":
        message = "LingBot-Map реконструкция в очереди"
    elif worker_state == "running":
        message = "LingBot-Map реконструкция выполняется на RTX 3090"
    elif worker_state == "completed":
        message = "LingBot-Map реконструкция завершена"
        progress = 100
    elif worker_state == "failed":
        message = worker_status.get("error") or "LingBot-Map реконструкция завершилась ошибкой"
        progress = 0
    else:
        message = status_data.get("message") or "LingBot-Map реконструкция"

    merged = {
        **status_data,
        "status": "error" if worker_state == "failed" else worker_state,
        "progress": progress,
        "message": message,
        "lingbot_session_id": session_id,
        "lingbot_status": worker_status,
    }

    if worker_state == "completed":
        metadata: Dict[str, Any] = {}
        trajectory: Dict[str, Any] = {"poses": []}
        try:
            metadata = await _fetch_lingbot_json(f"/sessions/{session_id}/metadata")
        except Exception as exc:
            metadata = {"error": str(exc)}
        try:
            trajectory = await _fetch_lingbot_json(f"/sessions/{session_id}/trajectory")
        except Exception as exc:
            trajectory = {"poses": [], "error": str(exc)}

        result = _lingbot_to_trackai_result(video_id, session_id, metadata, trajectory)
        merged["result"] = result
        processing_status[video_id] = {
            **merged,
            "result": result,
        }

    if worker_state == "failed":
        processing_status[video_id] = merged

    return merged


def _persist_lingbot_session_id(video_id: str, session_id: Optional[str]) -> None:
    if not session_id:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT map_context FROM tracking_tasks WHERE id = ?", (video_id,))
        row = cursor.fetchone()
        map_context: Dict[str, Any] = {}
        if row and row[0]:
            try:
                parsed = json.loads(row[0])
                if isinstance(parsed, dict):
                    map_context = parsed
            except Exception:
                map_context = {}
        map_context["lingbot_session_id"] = session_id
        map_context["analysis_method"] = "lingbot"
        cursor.execute(
            """
            UPDATE tracking_tasks
            SET map_context = ?, status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (json.dumps(map_context), "lingbot_queued", video_id),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning(f"[{video_id}] Failed to persist LingBot session id: {exc}")


def _load_lingbot_session_id(video_id: str) -> Optional[str]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT map_context FROM tracking_tasks WHERE id = ?", (video_id,))
        row = cursor.fetchone()
        conn.close()
        if not row or not row[0]:
            return None
        parsed = json.loads(row[0])
        if isinstance(parsed, dict):
            value = parsed.get("lingbot_session_id")
            return str(value) if value else None
    except Exception as exc:
        logger.warning(f"[{video_id}] Failed to load LingBot session id: {exc}")
    return None


@app.get("/api/status/{video_id}")
async def get_video_status(video_id: str):
    """
    Get the current status and progress of video analysis
    """
    manual_store = _load_manual_trajectories()
    manual_item = manual_store.get(video_id)
    if manual_item:
        manual_result = _make_manual_result(
            manual_item.get("trajectory") or [],
            manual_item.get("turn_points") or [],
        )
        return JSONResponse(_status_response_payload({
            "status": "completed",
            "progress": 100,
            "message": "Ручная траектория готова",
            "result": manual_result,
            "manual_updated_at": manual_item.get("updated_at"),
        }))

    live = processing_status.get(video_id)
    live_state = str((live or {}).get("status") or "").lower()
    live_busy = live_state in {
        "queued",
        "processing",
        "uploading_to_gpu",
        "gpu_processing",
        "running",
        "lingbot_queued",
        "lingbot_running",
    }
    # A stale in-memory payload (often with a non-serializable prior result)
    # must not block a finished analysis after restart / re-open.
    if not live_busy:
        completed = _load_completed_analysis_status(video_id)
        if completed is not None:
            processing_status[video_id] = {
                "status": "completed",
                "progress": 100,
                "message": completed.get("message") or "Анализ завершён",
                "start_time": (live or {}).get("start_time") or time.time(),
            }
            return JSONResponse(_status_response_payload(completed))

    if video_id not in processing_status:
        lingbot_session_id = _load_lingbot_session_id(video_id)
        if lingbot_session_id:
            processing_status[video_id] = {
                "status": "queued",
                "progress": 0,
                "message": "LingBot-Map реконструкция поставлена в очередь",
                "lingbot_session_id": lingbot_session_id,
                "start_time": time.time(),
            }
            merged = await _merge_lingbot_status(video_id, processing_status[video_id])
            return JSONResponse(_status_response_payload(merged))
        return JSONResponse({"id": video_id, "status": "unknown", "progress": 0})

    # Never re-emit a previous heavy result while a new GPU job is running —
    # FastAPI jsonable_encoder can 500 on leftover numpy / graph objects.
    if live_busy and "result" in processing_status[video_id]:
        processing_status[video_id].pop("result", None)

    merged = await _merge_lingbot_status(video_id, processing_status[video_id])
    return JSONResponse(_status_response_payload(merged))
@app.post("/api/reset-status/{video_id}")
async def reset_video_status(video_id: str):
    """Reset processing status for a video (to allow re-processing)."""
    if video_id in processing_status:
        del processing_status[video_id]
    return {"id": video_id, "status": "reset"}

def _update_dwg_progress(job_id: str, progress: int, message: str):
    if job_id in convert_dwg_status:
        convert_dwg_status[job_id]["progress"] = progress
        convert_dwg_status[job_id]["message"] = message
        logger.info(f"DWG job {job_id}: {progress}% - {message}")

def _run_dwg_conversion(job_id: str, tmp_path: Path, filename: str):
    from fastapi.concurrency import run_in_threadpool
    import base64
    svg_path = tmp_path.with_suffix('.svg')
    dxf_path = tmp_path.with_suffix('.dxf')
    png_path = tmp_path.with_suffix('.png')
    try:
        # Путь A: DWG -> DXF -> PNG (ezdxf) — часто лучше для сложных DWG
        _update_dwg_progress(job_id, 5, "DWG→DXF: попытка...")
        for cmd in ['/usr/local/bin/dwgread', '/usr/bin/dwgread', '/usr/local/bin/dwg2dxf', '/usr/bin/dwg2dxf']:
            if Path(cmd).exists():
                try:
                    if 'dwg2dxf' in cmd:
                        result = subprocess.run([cmd, '-y', '-o', str(dxf_path), str(tmp_path)], capture_output=True, text=True, timeout=300)
                    else:
                        result = subprocess.run(
                            [cmd, '-O', 'dxf', '-o', str(dxf_path), str(tmp_path)],
                            capture_output=True, text=True, timeout=300
                        )
                    if result.returncode != 0 or not dxf_path.exists() or dxf_path.stat().st_size < 500:
                        continue
                    if dxf_path.stat().st_size > 500:
                        _update_dwg_progress(job_id, 30, "DXF→PNG: рендеринг (ezdxf)...")
                        try:
                            import ezdxf
                            from ezdxf.addons.drawing import Frontend, RenderContext, config, layout
                            from ezdxf.addons.drawing import pymupdf
                            doc = ezdxf.readfile(str(dxf_path))
                            msp = doc.modelspace()
                            ctx = RenderContext(doc)
                            backend = pymupdf.PyMuPdfBackend()
                            cfg = config.Configuration(background_policy=config.BackgroundPolicy.WHITE)
                            frontend = Frontend(ctx, backend, config=cfg)
                            frontend.draw_layout(msp)
                            page = layout.Page(0, 0, layout.Units.mm, margins=layout.Margins.all(10))
                            png_bytes = backend.get_pixmap_bytes(page, fmt="png", dpi=150)
                            if png_bytes and len(png_bytes) > 1000:
                                png_b64 = base64.b64encode(png_bytes).decode('ascii')
                                convert_dwg_status[job_id] = {"status": "done", "progress": 100, "message": "Готово (DXF)", "png": png_b64, "filename": filename}
                                return
                        except Exception as ex:
                            logger.warning(f"ezdxf DXF→PNG: {ex}")
                    break
                except subprocess.TimeoutExpired:
                    continue

        # Путь B: DWG -> SVG -> PNG (0-35%)
        # Пробуем оба варианта (model space и всё) — берём тот, где больше контента
        _update_dwg_progress(job_id, 5, "DWG→SVG: обработка...")
        svg_ok = False
        best_svg = None
        best_size = 0
        for use_mspace in [False, True]:  # сначала всё (paper+model), потом только model
            for cmd in ['/usr/local/bin/dwg2SVG', '/usr/local/bin/dwg2svg', '/usr/bin/dwg2SVG']:
                if Path(cmd).exists():
                    try:
                        args = [cmd]
                        if use_mspace:
                            args.append('--mspace')
                        args.append(str(tmp_path))
                        result = subprocess.run(args, capture_output=True, text=True, timeout=300)
                        if result.returncode == 0 and result.stdout and len(result.stdout) > 500:
                            if len(result.stdout) > best_size:
                                best_size = len(result.stdout)
                                best_svg = result.stdout
                    except subprocess.TimeoutExpired:
                        continue
        if best_svg:
            svg_path.write_text(best_svg, encoding='utf-8', errors='replace')
            _update_dwg_progress(job_id, 35, f"DWG→SVG: готово ({best_size//1024} KB)")
            svg_ok = True
        if not svg_ok:
            for cmd in ['/usr/local/bin/dwgread', '/usr/bin/dwgread']:
                if Path(cmd).exists():
                    try:
                        result = subprocess.run(
                            [cmd, '-O', 'svg', '-o', str(svg_path), str(tmp_path)],
                            capture_output=True, text=True, timeout=300
                        )
                        if result.returncode == 0 and svg_path.exists() and svg_path.stat().st_size > 1000:
                            _update_dwg_progress(job_id, 35, "DWG→SVG: готово (dwgread)")
                            svg_ok = True
                            break
                    except subprocess.TimeoutExpired:
                        continue
        if not svg_ok:
            convert_dwg_status[job_id] = {"status": "error", "progress": 0, "message": "DWG→SVG не удалось", "error": "Нет инструмента"}
            return
        if not svg_path.exists():
            convert_dwg_status[job_id] = {"status": "error", "progress": 0, "message": "SVG не создан", "error": "Ошибка"}
            return
        # Step 2: Санитизация (35-45%)
        _update_dwg_progress(job_id, 40, "Санитизация SVG...")
        try:
            data = svg_path.read_bytes()
            data = data.replace(b'\x00', b'')
            for bad, good in [(b'&#0;', b'&#32;'), (b'&#00;', b'&#32;'), (b'&#000;', b'&#32;'),
                              (b'&#x0;', b'&#x20;'), (b'&#x00;', b'&#x20;'), (b'&#x000;', b'&#x20;')]:
                data = data.replace(bad, good)
            svg_path.write_bytes(data)
        except Exception as ex:
            logger.warning(f"SVG sanitization: {ex}")
        _update_dwg_progress(job_id, 45, "Санитизация: готово")
        # Step 3: SVG -> PNG (45-100%) — Inkscape: сначала drawing, при пустом — page
        _update_dwg_progress(job_id, 50, "SVG→PNG: рендеринг...")
        for export_area in ['--export-area-drawing', '--export-area-page']:
            for inkscape in ['/usr/bin/inkscape', 'inkscape']:
                try:
                    result = subprocess.run(
                        [inkscape, '-b', 'white', '-y', '1',
                         '--export-type=png', f'--export-filename={png_path}',
                         export_area, '--export-width=2048',
                         str(svg_path)],
                        capture_output=True, text=True, timeout=600
                    )
                    if result.returncode == 0 and png_path.exists():
                        png_bytes = png_path.read_bytes()
                        png_b64 = base64.b64encode(png_bytes).decode('ascii')
                        convert_dwg_status[job_id] = {"status": "done", "progress": 100, "message": "Готово", "png": png_b64, "filename": filename}
                        return
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
        for rsvg in ['/usr/bin/rsvg-convert', 'rsvg-convert']:
            try:
                result = subprocess.run(
                    [rsvg, '--width=2048', '--height=2048', '--keep-aspect-ratio',
                     '--output', str(png_path), str(svg_path)],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0 and png_path.exists():
                    png_bytes = png_path.read_bytes()
                    png_b64 = base64.b64encode(png_bytes).decode('ascii')
                    convert_dwg_status[job_id] = {"status": "done", "progress": 100, "message": "Готово", "png": png_b64, "filename": filename}
                    return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        for conv in ['/usr/bin/convert', 'convert']:
            try:
                result = subprocess.run(
                    [conv, '-density', '300', '-resize', '2048x2048>', '-background', 'white', '-flatten', str(svg_path), str(png_path)],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0 and png_path.exists():
                    png_bytes = png_path.read_bytes()
                    png_b64 = base64.b64encode(png_bytes).decode('ascii')
                    convert_dwg_status[job_id] = {"status": "done", "progress": 100, "message": "Готово", "png": png_b64, "filename": filename}
                    return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        convert_dwg_status[job_id] = {"status": "error", "progress": 0, "message": "SVG→PNG не удалось", "error": "rsvg и convert не сработали"}
    except Exception as e:
        logger.exception("DWG conversion error")
        convert_dwg_status[job_id] = {"status": "error", "progress": 0, "message": str(e), "error": str(e)}
    finally:
        tmp_path.unlink(missing_ok=True)
        dxf_path.unlink(missing_ok=True)
        svg_path.unlink(missing_ok=True)
        png_path.unlink(missing_ok=True)

@app.post("/api/convert-dwg")
async def convert_dwg_start(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Start DWG conversion, returns job_id for polling status."""
    if not file.filename or not file.filename.lower().endswith('.dwg'):
        raise HTTPException(status_code=400, detail="Требуется файл .dwg")
    job_id = str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(suffix='.dwg', delete=False) as tmp_in:
        content = await file.read()
        tmp_in.write(content)
        tmp_path = Path(tmp_in.name)
    convert_dwg_status[job_id] = {"status": "processing", "progress": 0, "message": "Загрузка: получено"}
    async def run_in_thread():
        await asyncio.to_thread(_run_dwg_conversion, job_id, tmp_path, file.filename)
    background_tasks.add_task(run_in_thread)
    return {"job_id": job_id}

@app.get("/api/convert-dwg-status/{job_id}")
async def convert_dwg_status_get(job_id: str):
    """Get DWG conversion progress and result."""
    if job_id not in convert_dwg_status:
        return {"status": "unknown", "progress": 0, "message": "Задача не найдена"}
    return convert_dwg_status[job_id]

@app.post("/api/convert-pdf")
async def convert_pdf_to_png(request: Request):
    """Convert PDF to PNG (первая страница) для отображения плана."""
    import base64
    try:
        form = await request.form()

        file = form.get("file")
        if not file or not hasattr(file, "filename") or not file.filename:
            raise HTTPException(status_code=400, detail="Требуется файл .pdf")
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Требуется файл .pdf")
        
        content = await file.read()
        logger.info(f"Processing PDF for conversion: {file.filename} ({len(content)} bytes)")
        
        # Step 1: Попробуем использовать fitz (PyMuPDF), он в requirements и работает быстрее/стабильнее
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            if len(doc) == 0:
                raise Exception("PDF файл пуст или поврежден")
            
            page = doc[0]  # Первая страница
            # Масштабируем до 2048px по ширине/высоте
            zoom = 2048 / max(page.rect.width, page.rect.height)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_bytes = pix.tobytes("png")
            doc.close()
            
            png_b64 = base64.b64encode(png_bytes).decode('ascii')
            logger.info("Successfully converted PDF to PNG using PyMuPDF")
            return {"success": True, "png": png_b64, "filename": file.filename}
            
        except Exception as fitz_err:
            logger.warning(f"PyMuPDF conversion failed: {fitz_err}. Falling back to pdftoppm.")
            
            # Step 2: Fallback на pdftoppm (вдруг fitz не справился)
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_in:
                tmp_in.write(content)
                tmp_path = Path(tmp_in.name)
            
            out_prefix = tmp_path.with_suffix('')
            try:
                pdftoppm_path = shutil.which('pdftoppm') or '/usr/bin/pdftoppm'
                result = subprocess.run(
                    [pdftoppm_path, '-png', '-singlefile', '-scale-to', '2048', '-f', '1', '-l', '1', str(tmp_path), str(out_prefix)],
                    capture_output=True, text=True, timeout=120
                )
                
                png_path = out_prefix.with_suffix('.png') if out_prefix.with_suffix('.png').exists() else next(out_prefix.parent.glob(out_prefix.name + '-*.png'), None)
                
                if result.returncode != 0 or not png_path:
                    stderr = result.stderr if result else "No stderr"
                    logger.error(f"pdftoppm failed: returncode={result.returncode}, stderr={stderr}")
                    raise HTTPException(status_code=500, detail=f"Не удалось конвертировать PDF. Попробуйте другой файл или убедитесь, что он не поврежден.")
                
                png_data = png_path.read_bytes()
                png_b64 = base64.b64encode(png_data).decode('ascii')
                return {"success": True, "png": png_b64, "filename": file.filename}
            finally:
                tmp_path.unlink(missing_ok=True)
                if 'png_path' in locals() and png_path:
                    png_path.unlink(missing_ok=True)
                for f in out_prefix.parent.glob(out_prefix.name + '-*.png'):
                    f.unlink(missing_ok=True)
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in convert_pdf_to_png")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.get("/api/plans")
async def list_plans():
    """List all saved plans from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, data, preview_svg, created_at FROM plans ORDER BY created_at DESC")
        plans = []
        for r in cursor.fetchall():
            try:
                data_obj = json.loads(r[2])
            except:
                data_obj = []
            plans.append({
                "id": r[0], 
                "name": r[1], 
                "data": data_obj, 
                "preview_svg": r[3], 
                "created_at": r[4]
            })
        conn.close()
        return plans
    except Exception as e:
        logger.error(f"Error listing plans: {e}")
        return []

@app.post("/api/plans")
async def save_plan(plan_data: Dict[str, Any]):
    """Save a new drawn plan"""
    try:
        name = plan_data.get("name", "Unnamed Plan")
        data = json.dumps(plan_data.get("data", []))
        preview_svg = plan_data.get("preview_svg", "")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO plans (name, data, preview_svg) VALUES (?, ?, ?)", (name, data, preview_svg))
        plan_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return {"id": plan_id, "name": name, "status": "success"}
    except Exception as e:
        logger.error(f"Error saving plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/plans/{id}")
async def delete_plan(id: int):
    """Delete a plan by ID"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM plans WHERE id = ?", (id,))
        conn.commit()
        conn.close()
        return {"status": "success", "id": id}
    except Exception as e:
        logger.error(f"Error deleting plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

FFMPEG_TIMEOUT = 1800  # 30 минут для больших AVI/видео

def _get_video_duration_sec(path: Path) -> float:
    """Получить длительность видео в секундах через ffprobe"""
    ffprobe_path = '/usr/bin/ffprobe' if os.path.exists('/usr/bin/ffprobe') else 'ffprobe'
    try:
        result = subprocess.run(
            [ffprobe_path, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return 0.0


def _get_video_info_for_ui(path: Optional[Path], fallback_frames: int = 0, fallback_fps: float = 0.0) -> Dict[str, Any]:
    info = {
        "width": 0,
        "height": 0,
        "fps": float(fallback_fps or 0.0),
        "frame_count": int(fallback_frames or 0),
        "duration": 0.0,
    }
    if not path or not path.exists():
        return info

    ffprobe_path = "/usr/bin/ffprobe" if os.path.exists("/usr/bin/ffprobe") else "ffprobe"
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return info
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        stream = streams[0] if streams else {}

        def _rate(value: Any) -> float:
            if not isinstance(value, str) or "/" not in value:
                try:
                    return float(value or 0)
                except Exception:
                    return 0.0
            num, den = value.split("/", 1)
            try:
                den_f = float(den)
                return float(num) / den_f if den_f else 0.0
            except Exception:
                return 0.0

        fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate")) or info["fps"]
        duration = float(stream.get("duration") or 0.0)
        frame_count = int(float(stream.get("nb_frames") or 0))
        if not frame_count and duration and fps:
            frame_count = int(round(duration * fps))
        info.update(
            {
                "width": int(stream.get("width") or 0),
                "height": int(stream.get("height") or 0),
                "fps": float(fps or 0.0),
                "frame_count": frame_count or info["frame_count"],
                "duration": duration,
            }
        )
    except Exception:
        return info
    return info


def _validate_video_readable(path: Path) -> bool:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return False
        ok, frame = cap.read()
        cap.release()
        return bool(ok and frame is not None)
    except Exception:
        return False

def _run_ffmpeg_with_progress(cmd: list, video_id: str, duration_sec: float, progress_min: int, progress_max: int, message: str):
    """Запуск ffmpeg с обновлением прогресса по stderr (time=)"""
    import re
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, universal_newlines=True)
    time_pattern = re.compile(r'time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})')
    for line in proc.stderr:
        m = time_pattern.search(line)
        if m and duration_sec > 0:
            h, m_, s, cs = map(int, m.groups())
            current_sec = h * 3600 + m_ * 60 + s + cs / 100
            pct = min(99, int(current_sec / duration_sec * 100))
            overall = progress_min + int((progress_max - progress_min) * pct / 100)
            processing_status[video_id].update({"progress": overall, "message": f"{message} ({pct}%)"})
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

async def process_video_background(
    video_id: str,
    video_path: Optional[Path],
    original_filename: str,
    scale_factor: float,
    stabilize: bool,
    detect_interval: int = 3,
    turn_vote_threshold: int = 3,
    use_ml_roi: bool = True,
    map_context: Optional[Dict[str, Any]] = None,
):
    """Background task — отправляет видео на GPU Worker (RTX 3090), получает результат.
    
    Если video_path=None — видео уже на GPU Worker (загружено через upload-video-stream).
    """
    _update_task_status(video_id, "processing", 10)
    try:
        from fastapi.concurrency import run_in_threadpool

        # Инициализируем статус, если ещё не задан (upload-video не создаёт его)
        if video_id not in processing_status:
            processing_status[video_id] = {
                "status": "queued", "progress": 0, "message": "Подготовка к отправке на GPU",
                "start_time": time.time(),
            }

        processing_status[video_id].update({
            "status": "uploading_to_gpu",
            "progress": 5,
            "message": "Отправка видео на GPU-сервер для анализа..."
        })
        processing_status[video_id].pop("result", None)

        timeout_sec = 7200  # 2 hours max
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec)) as session:
            params = {
                'original_filename': original_filename,
                'scale_factor': str(scale_factor),
                'stabilize': str(stabilize).lower(),
                'detect_interval': str(detect_interval),
                'turn_vote_threshold': str(turn_vote_threshold),
                'use_ml_roi': str(use_ml_roi).lower(),
            }
            if map_context:
                params['map_context'] = json.dumps(map_context)

            if video_path and video_path.exists():
                # ─── Видео есть локально — отправляем файл ──────────
                logger.info(f"[{video_id}] Sending local file to GPU Worker: {video_path.name}")
                processing_status[video_id].update({
                    "status": "gpu_processing",
                    "progress": 15,
                    "message": "GPU-сервер обрабатывает видео..."
                })
                async with session.post(
                    f"{GPU_WORKER_URL}/api/process-video-raw/{video_id}",
                    params=params,
                    data=open(video_path, 'rb'),
                ) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        raise Exception(f"GPU Worker error (HTTP {resp.status}): {err_text[:500]}")
                    gpu_result = await resp.json()
            else:
                # ─── Видео уже на GPU — триггерим обработку ────────
                logger.info(f"[{video_id}] Video already on GPU, triggering processing")
                params['use_uploaded'] = 'true'
                async with session.post(
                    f"{GPU_WORKER_URL}/api/process-video-raw/{video_id}",
                    params=params,
                    data=b'',  # empty body
                    headers={'Content-Length': '0'},
                ) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        raise Exception(f"GPU Worker error (HTTP {resp.status}): {err_text[:500]}")
                    gpu_result = await resp.json()

        if not gpu_result.get("success"):
            raise Exception(gpu_result.get("error", "GPU Worker returned unsuccessful status"))

        result = gpu_result["result"]
        # The GPU worker supplies visual motion.  The VPS owns the immutable
        # Kerama plan and applies the same production constraints to every
        # reconstruction backend, even when an older worker is deployed.
        result = apply_floorplan_constraints(result, map_context)
        processing_time = gpu_result.get("processing_time", 0)
        logger.info(f"[{video_id}] GPU Worker completed in {processing_time}s")

        # ─── Сохраняем результат (общая часть) ─────────────────────
        analysis_data = {
            "video_id": video_id,
            "video_filename": video_path.name,
            "original_filename": original_filename,
            "scale_factor": scale_factor,
            "stabilized": stabilize,
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(video_path.stat().st_mtime)),
            "analysis_result": result
        }

        analysis_file = OUTPUT_DIR / f"{video_id}_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=_to_json_serializable)

        processing_status[video_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Обработка завершена успешно",
            "result": result
        })
        _update_task_status(video_id, "completed", 100)
        logger.info(f"[{video_id}] Analysis saved and status set to completed")

    except Exception as e:
        error_msg = str(e)
        _update_task_status(video_id, "error", 0)
        logger.error(f"[{video_id}] Background processing error: {error_msg}")
        if video_id in processing_status:
            processing_status[video_id].update({
                "status": "error",
                "progress": 0,
                "message": f"Ошибка: {error_msg}"
            })
        await send_error_to_telegram(error_msg, f"Фон. обработка: {original_filename}")

# ──────────────────────────────────────────────
# R³ background processor
# ──────────────────────────────────────────────

async def process_video_r3_background(
    video_id: str,
    video_path: Optional[Path],
    original_filename: str,
    scale_factor: float = 1.0,
    frame_stride: int = 5,
    max_frames: int = 1500,
    ckpt: str = "r3_long.safetensors",
    size: int = 392,
    mode: str = "strided",
    map_context: Optional[Dict[str, Any]] = None,
):
    """Background task — отправляет видео на GPU Worker для R³ реконструкции.

    Если video_path=None — видео уже на GPU Worker (загружено через upload-video-stream).
    """
    _update_task_status(video_id, "processing", 10)
    try:
        if video_id not in processing_status:
            processing_status[video_id] = {
                "status": "queued", "progress": 0, "message": "Подготовка к R³ анализу",
                "start_time": time.time(),
                "stage": "queued",
            }

        processing_status[video_id].update({
            "status": "uploading_to_gpu",
            "progress": 5,
            "stage": "upload",
            "message": "Отправка видео на GPU-сервер для R³ реконструкции...",
            "start_time": processing_status[video_id].get("start_time") or time.time(),
        })
        # Drop any previous analysis payload so /api/status stays JSON-safe
        # while this long GPU upload/inference is in flight.
        processing_status[video_id].pop("result", None)
        logger.info(f"[{video_id}] Sending to GPU Worker R³ at {GPU_WORKER_URL}")
        timeout_sec = 7200
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_sec)) as session:
            params = {
                'original_filename': original_filename,
                'frame_stride': str(frame_stride),
                'ckpt': ckpt,
                'size': str(size),
                'mode': mode,
                'max_frames': str(max_frames),
            }

            processing_status[video_id].update({
                "status": "gpu_processing",
                "progress": 15,
                "message": "R³ реконструкция на GPU...",
                "stage": "gpu",
                "start_time": processing_status[video_id].get("start_time") or time.time(),
            })
            heartbeat = asyncio.create_task(
                _gpu_progress_heartbeat(
                    video_id,
                    base_progress=15,
                    ceiling=88,
                    label="R³ реконструкция на GPU",
                )
            )

            try:
                if video_path and video_path.exists():
                    # ─── Видео есть локально — отправляем файл ──────────
                    logger.info(f"[{video_id}] Sending local file to GPU Worker R³: {video_path.name}")
                    async with session.post(
                        f"{GPU_WORKER_URL}/api/r3-process-video-raw/{video_id}",
                        params=params,
                        data=open(video_path, 'rb'),
                    ) as resp:
                        if resp.status != 200:
                            err_text = await resp.text()
                            raise Exception(f"GPU Worker R³ error (HTTP {resp.status}): {err_text[:500]}")
                        gpu_result = await resp.json()
                else:
                    # ─── Видео уже на GPU — триггерим обработку ────────
                    logger.info(f"[{video_id}] Video already on GPU, triggering R³ processing")
                    params['use_uploaded'] = 'true'
                    async with session.post(
                        f"{GPU_WORKER_URL}/api/r3-process-video-raw/{video_id}",
                        params=params,
                        data=b'',  # empty body
                        headers={'Content-Length': '0'},
                    ) as resp:
                        if resp.status != 200:
                            err_text = await resp.text()
                            raise Exception(f"GPU Worker R³ error (HTTP {resp.status}): {err_text[:500]}")
                        gpu_result = await resp.json()
            finally:
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass

        if not gpu_result.get("success"):
            raise Exception(gpu_result.get("error", "GPU Worker R³ returned unsuccessful status"))

        processing_status[video_id].update({
            "status": "processing",
            "progress": 90,
            "stage": "trajectory",
            "message": "Сборка production-траектории R³...",
        })
        _update_task_status(video_id, "processing", 90)

        # Конвертируем R³ camera poses в формат траектории
        r3_result = gpu_result["result"]
        trajectory_data = _r3_poses_to_trajectory(r3_result, scale_factor=scale_factor)
        try:
            async with aiohttp.ClientSession() as source_session:
                async with source_session.get(
                    f"{GPU_WORKER_URL}/api/r3-trajectory/{video_id}",
                    params={"trajectory_source": "scale_aware_candidate"},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as source_response:
                    if source_response.status == 200:
                        selected_trajectory = await source_response.json()
                        trajectory_data = _merge_r3_production_trajectory(
                            trajectory_data,
                            selected_trajectory,
                        )
                    else:
                        logger.warning(
                            f"[{video_id}] Production trajectory selection returned "
                            f"HTTP {source_response.status}; keeping raw R3"
                        )
        except Exception as source_error:
            logger.warning(
                f"[{video_id}] Production trajectory selection failed; keeping raw R3: "
                f"{source_error}"
            )

        # LingBot runs after R3 on the shared RTX 3090 and acts as an
        # independent geometry observer. Failure is deliberately non-fatal:
        # production falls back to the already selected R3 trajectory.
        if LINGBOT_FUSION_ENABLED and video_path and video_path.exists():
            try:
                processing_status[video_id].update({
                    "status": "lingbot_fusion",
                    "progress": 92,
                    "stage": "lingbot",
                    "message": "LingBot-Map проверяет геометрию R³...",
                })
                _update_task_status(video_id, "processing", 92)
                submission = await _submit_lingbot_session(
                    video_path,
                    fps=10,
                    target_frames=LINGBOT_FUSION_TARGET_FRAMES,
                    keyframe_interval=LINGBOT_FUSION_KEYFRAME_INTERVAL,
                    use_sdpa=True,
                    mask_sky=False,
                )
                fusion_session_id = str(submission.get("session_id") or "")
                if not fusion_session_id:
                    raise RuntimeError("LingBot worker did not return session_id")
                processing_status[video_id]["lingbot_fusion_session_id"] = fusion_session_id
                lingbot_result = await _await_lingbot_session_result(
                    video_id,
                    fusion_session_id,
                    timeout_seconds=LINGBOT_FUSION_TIMEOUT_SECONDS,
                )
                trajectory_data = attach_lingbot_fusion_candidate(
                    trajectory_data,
                    lingbot_result,
                )
                logger.info(
                    "[%s] LingBot fusion candidate accepted=%s",
                    video_id,
                    trajectory_data.get("lingbot_fusion_candidate", {}).get("accepted"),
                )
            except Exception as lingbot_error:
                logger.warning(
                    "[%s] LingBot shadow/fusion unavailable; keeping R3: %s",
                    video_id,
                    lingbot_error,
                )
                stats = dict(trajectory_data.get("processing_stats") or {})
                stats["lingbot_shadow_available"] = False
                stats["lingbot_fusion"] = {
                    "accepted": False,
                    "reason": "worker_unavailable",
                    "error": str(lingbot_error),
                }
                trajectory_data["processing_stats"] = stats
        else:
            stats = dict(trajectory_data.get("processing_stats") or {})
            stats["lingbot_shadow_available"] = False
            stats["lingbot_fusion"] = {
                "accepted": False,
                "reason": (
                    "disabled"
                    if not LINGBOT_FUSION_ENABLED
                    else "video_not_available_on_vps"
                ),
            }
            trajectory_data["processing_stats"] = stats

        processing_status[video_id].update({
            "status": "processing",
            "progress": 96,
            "stage": "map",
            "message": "Сопоставление маршрута с планом Kerama...",
        })
        _update_task_status(video_id, "processing", 96)
        trajectory_data = apply_floorplan_constraints(trajectory_data, map_context)

        # Сохраняем результат
        video_filename = video_path.name if video_path and video_path.exists() else f"{video_id}_{original_filename}"
        analysis_data = {
            "video_id": video_id,
            "video_filename": video_filename,
            "original_filename": original_filename,
            "scale_factor": scale_factor,
            "stabilized": False,
            "analysis_method": "r3",
            "analysis_result": trajectory_data,
        }

        analysis_file = OUTPUT_DIR / f"{video_id}_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=_to_json_serializable)

        processing_status[video_id].update({
            "status": "completed",
            "progress": 100,
            "stage": "done",
            "message": "R³ + LingBot + план: обработка завершена",
            "result": trajectory_data,
            "eta_seconds": 0,
        })
        _update_task_status(video_id, "completed", 100)
        logger.info(f"[{video_id}] R³ analysis saved and status set to completed")

    except Exception as e:
        error_msg = str(e)
        _update_task_status(video_id, "error", 0)
        logger.error(f"[{video_id}] R³ background error: {error_msg}")
        if video_id in processing_status:
            processing_status[video_id].update({
                "status": "error",
                "progress": 0,
                "message": f"Ошибка R³: {error_msg}"
            })


def _schedule_r3_process_background(
    background_tasks: Optional[BackgroundTasks],
    video_id: str,
    video_path: Optional[Path],
    original_filename: str,
    scale_factor: float,
    frame_stride: int = 5,
    max_frames: int = 1500,
    ckpt: str = "r3_long.safetensors",
    size: int = 392,
    mode: str = "strided",
    map_context: Optional[Dict[str, Any]] = None,
) -> None:
    if background_tasks is not None:
        background_tasks.add_task(
            process_video_r3_background,
            video_id, video_path, original_filename,
            scale_factor, frame_stride, max_frames, ckpt, size, mode, map_context,
        )
    else:
        asyncio.create_task(
            process_video_r3_background(
                video_id, video_path, original_filename,
                scale_factor, frame_stride, max_frames, ckpt, size, mode, map_context,
            )
        )


# Метаданные загруженных видео (video_id -> filename) для анализа по id
UPLOADED_VIDEOS: Dict[str, str] = {}

def _load_uploaded_videos():
    """При старте загружаем маппинг video_id -> filename из VIDEOS_DIR"""
    for f in VIDEOS_DIR.glob("*_*"):
        if f.is_file():
            # Формат: {uuid}_{original_name}
            name = f.name
            if "_" in name:
                vid = name.split("_", 1)[0]
                if len(vid) == 36 and vid.count("-") == 4:  # uuid format
                    UPLOADED_VIDEOS[vid] = name
_load_uploaded_videos()


@app.on_event("startup")
async def _startup_telegram_processing_long_poll() -> None:
    """Без HTTPS webhook: один worker держит flock и крутит getUpdates."""
    global _tg_poll_lock_file
    try:
        if not await _tg_should_use_long_polling():
            return
        if _HAS_FCNTL and _fcntl_mod is not None:
            try:
                _tg_poll_lock_file = open("/tmp/trackai_tg_processing_poll.lock", "w")
                _fcntl_mod.flock(_tg_poll_lock_file.fileno(), _fcntl_mod.LOCK_EX | _fcntl_mod.LOCK_NB)
            except BlockingIOError:
                logger.info("Telegram processing poll: другой worker уже опрашивает API")
                return
            except OSError as e:
                logger.warning(f"Telegram poll flock: {e}")
                return
        asyncio.create_task(telegram_processing_poll_updates_loop())
    except Exception as e:
        logger.warning(f"Telegram processing startup poll: {e}")


@app.post("/api/init-upload")
async def init_upload(request: Request) -> Dict[str, Any]:
    """Зарегистрировать видео для загрузки (без файла). Возвращает video_id."""
    try:
        body = await request.json()
        filename = body.get("filename", "video.avi")
        employee_name = body.get("employee_name") or None
        client_source = (body.get("client_source") or request.headers.get("X-TrackAI-Client") or "web").strip()

        if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Неподдерживаемый формат видео")

        video_id = str(uuid.uuid4())
        video_filename = f"{video_id}_{filename}"

        # Регистрируем
        UPLOADED_VIDEOS[video_id] = video_filename

        map_context = {"client_source": client_source} if client_source else {}

        # Создаём задачу в tracking_tasks
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO tracking_tasks (id, employee_name, video_filename, original_filename, map_context, status) VALUES (?, ?, ?, ?, ?, ?)",
                (video_id, employee_name, video_filename, filename, json.dumps(map_context), "registered")
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            logger.error(f"Failed to save tracking task: {db_err}")

        logger.info(f"[{video_id}] Initialized upload for {filename}")
        return {
            "success": True,
            "video_id": video_id,
            "filename": video_filename,
            "original_filename": filename,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in init_upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-video/{video_id}")
async def upload_video_proxy(video_id: str, request: Request, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Принять raw-байты видео, отправить на GPU и сохранить локально для R³ анализа.
    
    Тело запроса — сырые байты видеофайла (Content-Type: application/octet-stream).
    Буферизирует во временный файл, затем отправляет на GPU и сохраняет локально.
    """
    try:
        video_info = UPLOADED_VIDEOS.get(video_id)
        if not video_info:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not registered")
        
        original_filename = video_info[len(video_id)+1:] if '_' in video_info else "video.avi"
        client_source = (request.headers.get("X-TrackAI-Client") or "").strip().lower()
        is_desktop_upload = client_source == "desktop"
        
        logger.info(f"[{video_id}] Raw upload started: {original_filename} source={client_source or 'web'}")

        ext = Path(original_filename).suffix or '.mp4'
        local_video_path = VIDEOS_DIR / f"{video_id}_{original_filename}"
        local_tmp = VIDEOS_DIR / f".{video_id}_uploading{ext}"

        # ─── Буферизируем во временный файл ──────────────────────
        file_size = 0
        with open(local_tmp, "wb") as tmp:
            async for chunk in request.stream():
                if chunk:
                    tmp.write(chunk)
                    file_size += len(chunk)

        if file_size == 0:
            local_tmp.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Пустой файл видео")

        # Перемещаем из временного в постоянное место
        local_tmp.rename(local_video_path)

        # Регистрируем локальный файл
        UPLOADED_VIDEOS[video_id] = local_video_path.name
        logger.info(f"[{video_id}] Saved locally: {local_video_path} ({file_size} bytes)")
        _schedule_video_preview(video_id)

        # ─── Отправляем на GPU ────────────────────────────────────
        gpu_url = f"{GPU_WORKER_URL}/api/upload-video-stream/{video_id}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                gpu_url,
                params={'original_filename': original_filename},
                data=open(local_video_path, 'rb'),
                timeout=aiohttp.ClientTimeout(total=7200),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"[{video_id}] GPU Worker upload failed ({resp.status}): {error_text[:200]}")
                    raise HTTPException(status_code=500, detail=f"GPU Worker upload failed: {error_text[:200]}")
                result = await resp.json()
                file_size = result.get("file_size", file_size)

        logger.info(f"[{video_id}] Uploaded {file_size} bytes to GPU Worker")

        # Обновляем статус задачи
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT map_context FROM tracking_tasks WHERE id = ?", (video_id,))
            row = cursor.fetchone()
            try:
                map_context = json.loads(row[0]) if row and row[0] else {}
            except Exception:
                map_context = {}
            if is_desktop_upload:
                map_context["client_source"] = "desktop"
                map_context["gpu_upload_url"] = gpu_url
            cursor.execute(
                "UPDATE tracking_tasks SET status = 'uploaded', map_context = ? WHERE id = ?",
                (json.dumps(map_context), video_id)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        upload_source = client_source or "web"
        if background_tasks is not None:
            background_tasks.add_task(
                send_desktop_upload_notification,
                video_id,
                original_filename,
                file_size,
                gpu_url,
                upload_source,
            )
        else:
            asyncio.create_task(
                send_desktop_upload_notification(video_id, original_filename, file_size, gpu_url, upload_source)
            )

        # Запускаем SLAM обработку (с video_path=None — use_uploaded на GPU)
        logger.info(f"[{video_id}] Starting GPU processing")
        _schedule_process_video_background(
            background_tasks,
            video_id,
            None,
            original_filename,
            12.306, True, 3, 3, True, None,
        )

        # Notify subscribers
        if background_tasks is not None:
            background_tasks.add_task(broadcast_new_processing_for_video, video_id)
        else:
            asyncio.create_task(broadcast_new_processing_for_video(video_id))

        return {
            "success": True,
            "video_id": video_id,
            "filename": video_info,
            "original_filename": original_filename,
            "file_size": file_size,
            "message": "Видео отправлено на обработку на GPU-сервер"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload_video_proxy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/tasks/{task_id}/context")
async def update_task_context(task_id: str, request: Request) -> Dict[str, Any]:
    """Обновить контекст задачи (чертеж, план, имя сотрудника) — вызывается фронтом при каждом изменении."""
    try:
        body = await request.json()
        map_context = {
            "floorplan_id": body.get("floorplan_id") or DEFAULT_FLOORPLAN_ID,
        }
        if body.get("floor_plan_data"):
            map_context["floor_plan_data"] = body["floor_plan_data"]
        if body.get("drawn_plan"):
            map_context["drawn_plan"] = body["drawn_plan"]
        if body.get("reference_point"):
            map_context["reference_point"] = body["reference_point"]
        if body.get("direction_point"):
            map_context["direction_point"] = body["direction_point"]

        employee_name = body.get("employee_name")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Проверяем существует ли задача и сохраняем уже записанный контекст
        cursor.execute("SELECT id, map_context FROM tracking_tasks WHERE id = ?", (task_id,))
        existing = cursor.fetchone()
        if not existing:
            conn.close()
            raise HTTPException(status_code=404, detail="Задача не найдена")

        try:
            existing_context = json.loads(existing[1]) if existing[1] else {}
            if not isinstance(existing_context, dict):
                existing_context = {}
        except Exception:
            existing_context = {}
        existing_context.update(map_context)

        updates = ["map_context = ?", "updated_at = CURRENT_TIMESTAMP"]
        params = [json.dumps(existing_context)]

        if employee_name:
            updates.append("employee_name = ?")
            params.append(employee_name.strip())

        params.append(task_id)
        cursor.execute(f"UPDATE tracking_tasks SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
        conn.close()

        logger.info(f"Updated context for task {task_id}")
        return {"success": True, "message": "Контекст задачи обновлён"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task context {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/tasks/{video_id}/register-existing")
async def register_existing_video_task(video_id: str, request: Request) -> Dict[str, Any]:
    """Поднять уже загруженное видео как свежий запрос для админки."""
    try:
        body = await request.json()
        employee_name = (body.get("employee_name") or "").strip() or None
        client_source = (
            body.get("client_source")
            or request.headers.get("X-TrackAI-Client")
            or "web"
        ).strip()

        video_filename = UPLOADED_VIDEOS.get(video_id)
        video_path = (VIDEOS_DIR / video_filename) if video_filename else None
        if not video_filename or not video_path.exists():
            matches = list(VIDEOS_DIR.glob(f"{video_id}_*"))
            if matches:
                video_path = matches[0]
                video_filename = video_path.name
                UPLOADED_VIDEOS[video_id] = video_filename
        if not video_filename or not video_path or not video_path.exists():
            raise HTTPException(status_code=404, detail="Загруженное видео не найдено")

        original_filename = (
            video_filename[len(video_id) + 1:]
            if video_filename.startswith(f"{video_id}_")
            else video_filename
        )

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT map_context FROM tracking_tasks WHERE id = ?", (video_id,))
        row = cursor.fetchone()
        try:
            map_context = json.loads(row[0]) if row and row[0] else {}
            if not isinstance(map_context, dict):
                map_context = {}
        except Exception:
            map_context = {}

        map_context["client_source"] = client_source
        map_context["selected_existing_video"] = True

        cursor.execute(
            """
            INSERT INTO tracking_tasks (
                id, employee_name, video_filename, original_filename, map_context, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                employee_name = excluded.employee_name,
                video_filename = excluded.video_filename,
                original_filename = excluded.original_filename,
                map_context = excluded.map_context,
                status = excluded.status,
                created_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                video_id,
                employee_name,
                video_filename,
                original_filename,
                json.dumps(map_context),
                "selected",
            ),
        )
        conn.commit()
        conn.close()

        logger.info(f"[{video_id}] Registered existing upload for admin source={client_source} employee={employee_name}")
        return {
            "success": True,
            "video_id": video_id,
            "filename": video_filename,
            "original_filename": original_filename,
            "status": "selected",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering existing video {video_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _submit_lingbot_session(
    video_path: Optional[Path],
    fps: int = 10,
    target_frames: int = 1500,
    keyframe_interval: int = 6,
    use_sdpa: bool = True,
    mask_sky: bool = False,
) -> Dict[str, Any]:
    if not video_path or not video_path.exists():
        raise HTTPException(
            status_code=422,
            detail="LingBot-Map MVP requires a local video file on the VPS for upload to the GPU worker",
        )

    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field("fps", str(fps))
        form.add_field("target_frames", str(target_frames))
        form.add_field("keyframe_interval", str(keyframe_interval))
        # The deployed LingBot environment does not have FlashInfer installed.
        # Always force SDPA so stale frontend bundles or old clients cannot
        # trigger the FlashInfer attention path and fail reconstruction.
        form.add_field("use_sdpa", "true")
        form.add_field("mask_sky", str(mask_sky).lower())
        with video_path.open("rb") as video_file:
            form.add_field(
                "file",
                video_file,
                filename=video_path.name,
                content_type="application/octet-stream",
            )
            async with session.post(
                f"{LINGBOT_WORKER_URL}/sessions/upload",
                data=form,
                timeout=aiohttp.ClientTimeout(total=60 * 60),
            ) as resp:
                text = await resp.text()
                if resp.status >= 300:
                    raise HTTPException(
                        status_code=502,
                        detail=f"LingBot worker error ({resp.status}): {text[:500]}",
                    )
                try:
                    result = json.loads(text)
                except Exception:
                    raise HTTPException(status_code=502, detail="LingBot worker returned invalid JSON")

    return result


async def _start_lingbot_session(
    video_id: str,
    video_path: Optional[Path],
    fps: int = 10,
    target_frames: int = 1500,
    keyframe_interval: int = 6,
    use_sdpa: bool = True,
    mask_sky: bool = False,
) -> Dict[str, Any]:
    result = await _submit_lingbot_session(
        video_path,
        fps=fps,
        target_frames=target_frames,
        keyframe_interval=keyframe_interval,
        use_sdpa=use_sdpa,
        mask_sky=mask_sky,
    )
    processing_status[video_id] = {
        "status": "queued",
        "progress": 0,
        "message": "LingBot-Map реконструкция поставлена в очередь",
        "lingbot_session_id": result.get("session_id"),
        "lingbot_use_sdpa": True,
        "start_time": time.time(),
    }
    _persist_lingbot_session_id(video_id, result.get("session_id"))
    return result


async def _await_lingbot_session_result(
    video_id: str,
    session_id: str,
    *,
    timeout_seconds: int,
    poll_seconds: float = 2.0,
) -> Dict[str, Any]:
    """Wait for a shadow LingBot run without replacing the R3 status result."""
    deadline = time.monotonic() + max(1, timeout_seconds)
    while time.monotonic() < deadline:
        status = await _fetch_lingbot_json(f"/sessions/{session_id}/status")
        state = str(status.get("status") or "unknown")
        progress = float(status.get("progress") or 0.0)
        if video_id in processing_status:
            processing_status[video_id].update({
                "status": "lingbot_fusion",
                "progress": min(94, 70 + int(round(progress * 24))),
                "message": "LingBot-Map строит второе геометрическое наблюдение...",
                "lingbot_fusion_session_id": session_id,
            })
        if state == "completed":
            metadata = await _fetch_lingbot_json(f"/sessions/{session_id}/metadata")
            trajectory = await _fetch_lingbot_json(f"/sessions/{session_id}/trajectory")
            return _lingbot_to_trackai_result(video_id, session_id, metadata, trajectory)
        if state == "failed":
            raise RuntimeError(status.get("error") or "LingBot shadow reconstruction failed")
        await asyncio.sleep(max(0.25, poll_seconds))
    raise TimeoutError(f"LingBot shadow reconstruction timed out after {timeout_seconds}s")


@app.post("/api/analyze-video-by-id")
async def analyze_video_by_id(background_tasks: BackgroundTasks, request: Request) -> Dict[str, Any]:
    """Запустить анализ уже загруженного видео по video_id."""
    try:
        body = await request.json()
        video_id = body.get("video_id")
        if not video_id:
            raise HTTPException(status_code=422, detail="Требуется video_id")
        scale_factor = float(body.get("scale_factor", 12.306))
        stabilize = body.get("stabilize", True)
        original_filename = body.get("original_filename", "video")
        detect_interval = int(body.get("detect_interval", 3))
        turn_vote_threshold = int(body.get("turn_vote_threshold", 3))
        use_ml_roi = bool(body.get("use_ml_roi", True))
        map_context = _extract_map_context(body)
        _merge_task_map_context(video_id, map_context)
        force_reprocess = bool(body.get("force_reprocess", False))

        # При явном новом анализе ручная траектория больше не должна подменять результат.
        # Иначе выбранное с сервера видео с прежней админ-разметкой сразу возвращает manual_result.
        if force_reprocess:
            manual_store = _load_manual_trajectories()
            if video_id in manual_store:
                del manual_store[video_id]
                _save_manual_trajectories(manual_store)
                logger.info(f"[{video_id}] Removed stale manual trajectory before forced reprocess")

        # Если администратор задал ручную траекторию и новый прогон не запрошен, возвращаем её сразу.
        manual_store = _load_manual_trajectories()
        manual_item = manual_store.get(video_id)
        if manual_item:
            manual_result = _make_manual_result(
                manual_item.get("trajectory") or [],
                manual_item.get("turn_points") or [],
            )
            processing_status[video_id] = {
                "status": "completed",
                "progress": 100,
                "message": "Обработка завершена",
                "result": manual_result,
                "start_time": time.time(),
            }
            return {
                "success": True,
                "video_id": video_id,
                "status": "completed",
                "message": "Обработка завершена",
                "data": manual_result,
            }

        # Ищем файл: UPLOADED_VIDEOS, потом glob; если файл не найден — возможно,
        # видео было загружено напрямую на GPU через upload-video-stream.
        video_filename = UPLOADED_VIDEOS.get(video_id)
        video_path = (VIDEOS_DIR / video_filename) if video_filename else None

        if not video_path or not video_path.exists():
            matches = list(VIDEOS_DIR.glob(f"{video_id}_*"))
            for m in matches:
                if m.is_file():
                    video_filename = m.name
                    video_path = VIDEOS_DIR / video_filename
                    UPLOADED_VIDEOS[video_id] = video_filename
                    break
            if not video_path or not video_path.exists():
                # Видео может быть только на GPU (стриминговая загрузка)
                if video_id in UPLOADED_VIDEOS:
                    logger.info(f"[{video_id}] Video not found locally but registered — it's on GPU, using use_uploaded flow")
                    video_path = None
                else:
                    raise HTTPException(status_code=404, detail=f"Видео {video_id} не найдено на сервере")

        processing_status[video_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Поставлено в очередь на обработку",
            "start_time": time.time()
        }

        # ─── НЕМЕДЛЕННО запускаем обработку на GPU Worker ────────
        employee_name = body.get("employee_name")
        analysis_method = body.get("analysis_method", "slam")

        if analysis_method == "lingbot":
            lingbot_result = await _start_lingbot_session(
                video_id,
                video_path,
                int(body.get("lingbot_fps", 10)),
                int(body.get("lingbot_target_frames", 1500)),
                int(body.get("lingbot_keyframe_interval", 6)),
                True,
                bool(body.get("lingbot_mask_sky", False)),
            )
            return {
                "success": True,
                "video_id": video_id,
                "status": "queued",
                "message": "LingBot-Map анализ запущен",
                "lingbot_session_id": lingbot_result.get("session_id"),
            }
        elif analysis_method == "r3":
            frame_stride = int(body.get("frame_stride", 5))
            max_frames = int(body.get("max_frames", 1500))
            ckpt = body.get("ckpt", "r3_long.safetensors")
            size = int(body.get("size", 392))
            mode = body.get("mode", "strided")
            _schedule_r3_process_background(
                background_tasks,
                video_id, video_path, original_filename,
                scale_factor,
                frame_stride, max_frames, ckpt, size, mode, map_context,
            )
        else:
            _schedule_process_video_background(
                background_tasks,
                video_id,
                video_path,
                original_filename,
                scale_factor,
                stabilize,
                detect_interval,
                turn_vote_threshold,
                use_ml_roi,
                map_context,
            )

        # Уведомление Telegram (фоновое, не блокирует)
        if background_tasks is not None:
            background_tasks.add_task(broadcast_new_processing_for_video, video_id)
        else:
            asyncio.create_task(broadcast_new_processing_for_video(video_id))

        return {
            "success": True,
            "video_id": video_id,
            "status": "queued",
            "message": "Анализ запущен на GPU-сервере"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_video_by_id: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-video")
async def analyze_video(background_tasks: BackgroundTasks, request: Request) -> Dict[str, Any]:
    """Analyze uploaded video file and return trajectory data."""
    form = await request.form()

    file = form.get("file")
    if not file or not hasattr(file, "filename") or not file.filename:
        raise HTTPException(status_code=422, detail="Требуется файл видео")
    scale_factor = float(form.get("scale_factor", 12.306))
    stabilize = form.get("stabilize", "true").lower() in ("true", "1", "yes")
    detect_interval = int(form.get("detect_interval", 3))
    turn_vote_threshold = int(form.get("turn_vote_threshold", 3))
    use_ml_roi = form.get("use_ml_roi", "true").lower() in ("true", "1", "yes")
    map_context = _extract_map_context(dict(form))
    client_id = form.get("client_id") or None
    video_id = client_id or str(uuid.uuid4())

    try:
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        video_filename = f"{video_id}_{file.filename}"
        video_path = VIDEOS_DIR / video_filename

        processing_status[video_id] = {
            "status": "uploading",
            "progress": 5,
            "message": "Сохранение видео на сервере...",
            "start_time": time.time()
        }

        from fastapi.concurrency import run_in_threadpool
        content = await file.read()
        def save_file():
            with open(video_path, "wb") as f:
                f.write(content)
        await run_in_threadpool(save_file)

        logger.info(f"Saved video: {video_filename}")
        
        # Save task to DB
        try:
            employee_name = form.get("employee_name")
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO tracking_tasks (id, employee_name, video_filename, original_filename, map_context, status) VALUES (?, ?, ?, ?, ?, ?)",
                (video_id, employee_name, video_filename, file.filename, json.dumps(map_context), "queued")
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            logger.error(f"Failed to save task to DB: {db_err}")

        # ─── НЕМЕДЛЕННО запускаем обработку на GPU Worker ────────
        employee_name = form.get("employee_name")
        _schedule_process_video_background(
            background_tasks,
            video_id,
            video_path,
            file.filename,
            scale_factor,
            stabilize,
            detect_interval,
            turn_vote_threshold,
            use_ml_roi,
            map_context,
        )

        # Уведомление Telegram (фоновое, не блокирует)
        if background_tasks is not None:
            background_tasks.add_task(broadcast_new_processing_for_video, video_id)
        else:
            asyncio.create_task(broadcast_new_processing_for_video(video_id))

        return {
            "success": True,
            "video_id": video_id,
            "status": "queued",
            "message": "Видео отправлено на GPU-обработку"
        }

    except Exception as e:
        logger.error(f"Error in analyze_video POST: {e}")
        processing_status[video_id] = {"status": "error", "message": str(e), "progress": 0}
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up processing status after some time (keep for 1 hour)
        if video_id is not None:
            import threading
            def cleanup_status():
                time.sleep(3600)  # Keep status for 1 hour
                processing_status.pop(video_id, None)

            cleanup_thread = threading.Thread(target=cleanup_status, daemon=True)
            cleanup_thread.start()

@app.post("/api/test-chat")
async def test_chat():
    """Test chat endpoint"""
    return {"response": "Test chat working"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "TrackAI Video Analysis API"}

@app.get("/api/processing-status/{video_id}")
async def get_processing_status(video_id: str):
    """Get processing status for a video"""
    manual_store = _load_manual_trajectories()
    manual_item = manual_store.get(video_id)
    if manual_item:
        manual_result = _make_manual_result(
            manual_item.get("trajectory") or [],
            manual_item.get("turn_points") or [],
        )
        return {
            "status": "completed",
            "progress": 100,
            "message": "Ручная траектория готова",
            "result": manual_result,
            "manual_updated_at": manual_item.get("updated_at"),
        }
    if video_id in processing_status:
        return processing_status[video_id]
    else:
        raise HTTPException(status_code=404, detail="Video processing not found")

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint"""
    return {"message": "Test endpoint working"}

@app.get("/api/uploaded-videos")
async def get_uploaded_videos_list():
    """Список всех загруженных на сервер видео (для выбора перед анализом)"""
    videos = []
    try:
        for f in VIDEOS_DIR.glob("*_*"):
            if f.is_file():
                name = f.name
                if "_" in name:
                    vid = name.split("_", 1)[0]
                    if len(vid) == 36 and vid.count("-") == 4:
                        try:
                            stat = f.stat()
                            videos.append({
                                "video_id": vid,
                                "filename": name,
                                "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
                                "file_size": stat.st_size,
                                "scale_factor": 12.306,
                                "stabilized": False,
                                "has_analysis": False
                            })
                        except Exception as e:
                            logger.warning(f"Error stat {f}: {e}")
        return {"success": True, "videos": videos}
    except Exception as e:
        logger.error(f"Error getting uploaded videos: {e}")
        return {"success": False, "videos": [], "error": str(e)}


@app.get("/api/admin/tasks")
async def list_tracking_tasks():
    """List all tracking tasks for admin panel"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, employee_name, original_filename, status, created_at, map_context FROM tracking_tasks ORDER BY created_at DESC")
        tasks = []
        for r in cursor.fetchall():
            try:
                map_ctx = json.loads(r[5]) if r[5] else {}
            except:
                map_ctx = {}
            tasks.append({
                "id": r[0],
                "employee_name": r[1],
                "original_filename": r[2],
                "status": r[3],
                "created_at": r[4],
                "map_context": _map_context_summary(map_ctx)
            })
        conn.close()
        return tasks
    except Exception as e:
        logger.error(f"Error listing tracking tasks: {e}")
        return []

@app.get("/api/admin/tasks/{task_id}")
async def get_tracking_task(task_id: str):
    """Get specific tracking task details"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, employee_name, original_filename, status, created_at, map_context FROM tracking_tasks WHERE id = ?", (task_id,))
        r = cursor.fetchone()
        conn.close()
        if not r:
            raise HTTPException(status_code=404, detail="Task not found")
        
        try:
            map_ctx = json.loads(r[5]) if r[5] else {}
        except:
            map_ctx = {}
            
        return {
            "id": r[0],
            "employee_name": r[1],
            "original_filename": r[2],
            "status": r[3],
            "created_at": r[4],
            "map_context": map_ctx
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/tasks/{task_id}")
async def delete_tracking_task(task_id: str) -> Dict[str, Any]:
    """Удалить задачу из админки: запись в БД, файл видео, ручная траектория, результат анализа."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, video_filename FROM tracking_tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Задача не найдена")
        _, video_filename = row[0], row[1]

        if video_filename:
            vp = VIDEOS_DIR / video_filename
            if vp.is_file():
                try:
                    vp.unlink()
                except OSError as e:
                    logger.warning(f"Не удалось удалить файл видео {vp}: {e}")
        for f in VIDEOS_DIR.glob(f"{task_id}_*"):
            if f.is_file():
                try:
                    f.unlink()
                except OSError as e:
                    logger.warning(f"Не удалить {f}: {e}")

        UPLOADED_VIDEOS.pop(task_id, None)
        processing_status.pop(task_id, None)

        analysis_file = OUTPUT_DIR / f"{task_id}_analysis.json"
        if analysis_file.is_file():
            try:
                analysis_file.unlink()
            except OSError as e:
                logger.warning(f"Не удалить {analysis_file}: {e}")

        manual_store = _load_manual_trajectories()
        if task_id in manual_store:
            del manual_store[task_id]
            _save_manual_trajectories(manual_store)

        cursor.execute("DELETE FROM tracking_tasks WHERE id = ?", (task_id,))
        conn.commit()
        conn.close()
        logger.info(f"Admin deleted task/video {task_id}")
        return {"success": True, "id": task_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"delete_tracking_task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/clear-database")
async def admin_clear_database() -> Dict[str, Any]:
    """Очистить все строки в SQLite: tracking_tasks и plans."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tracking_tasks")
        cursor.execute("DELETE FROM plans")
        try:
            cursor.execute(
                "DELETE FROM sqlite_sequence WHERE name IN ('plans', 'tracking_tasks')"
            )
        except Exception:
            pass
        conn.commit()
        try:
            conn.execute("VACUUM")
        except Exception as ve:
            logger.warning(f"VACUUM after clear: {ve}")
        conn.close()
        logger.warning("Admin: полная очистка таблиц БД (tracking_tasks, plans)")
        return {"success": True, "message": "База данных очищена"}
    except Exception as e:
        logger.error(f"admin_clear_database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/uploaded-video/{video_id}/stream")
async def stream_uploaded_video(video_id: str, request: Request):
    """Return uploaded video file by video_id (no analysis required)."""
    video_path = _find_uploaded_video_path(video_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Uploaded video not found")

    return FileResponse(
        str(video_path),
        filename=video_path.name,
        content_disposition_type="attachment",
        headers={
            "Cache-Control": "private, max-age=0, must-revalidate",
        },
    )


@app.get("/api/uploaded-video/{video_id}/preview.mp4")
async def stream_uploaded_video_preview(video_id: str):
    """Browser-friendly MP4 preview for admin video player."""
    preview_path = _video_preview_path(video_id)
    if not preview_path.exists() or preview_path.stat().st_size == 0:
        _schedule_video_preview(video_id)
        raise HTTPException(status_code=202, detail="Video preview is preparing")

    return FileResponse(
        str(preview_path),
        media_type="video/mp4",
        filename=f"{video_id}.mp4",
        content_disposition_type="inline",
        headers={"Cache-Control": "private, max-age=0, must-revalidate"},
    )


@app.get("/api/manual-trajectory/{video_id}")
async def get_manual_trajectory(video_id: str):
    manual_store = _load_manual_trajectories()
    item = manual_store.get(video_id)
    if not item:
        return {"success": False, "video_id": video_id, "exists": False}
    return {
        "success": True,
        "video_id": video_id,
        "exists": True,
        "trajectory": item.get("trajectory") or [],
        "turn_points": item.get("turn_points") or [],
        "updated_at": item.get("updated_at"),
    }


@app.post("/api/manual-trajectory/{video_id}")
async def save_manual_trajectory(video_id: str, payload: Dict[str, Any] = Body(default={})):
    trajectory = payload.get("trajectory") or []
    turn_points = payload.get("turn_points") or []
    if not isinstance(trajectory, list) or len(trajectory) < 2:
        raise HTTPException(status_code=400, detail="Требуется траектория минимум из 2 точек")

    manual_store = _load_manual_trajectories()
    updated_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    manual_store[video_id] = {
        "trajectory": trajectory,
        "turn_points": turn_points,
        "updated_at": updated_at,
    }
    _save_manual_trajectories(manual_store)

    manual_result = _make_manual_result(trajectory, turn_points)
    processing_status[video_id] = {
        "status": "completed",
        "progress": 100,
        "message": "Готово",
        "result": manual_result,
        "start_time": time.time(),
    }

    return {
        "success": True,
        "video_id": video_id,
        "updated_at": updated_at,
        "trajectory_points": len(trajectory),
    }

@app.get("/api/videos")
async def get_videos_list():
    """Get list of processed videos"""
    videos = []
    try:
        for json_file in OUTPUT_DIR.glob("*_analysis.json"):
            name = json_file.name
            if ".pre-" in name or ".before_" in name:
                continue
            try:
                import json
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                video_id = str(data.get("video_id") or "").strip()
                if not video_id:
                    continue
                video_filename = str(data.get("video_filename") or "")
                video_path = VIDEOS_DIR / video_filename if video_filename else None
                uploaded_at = data.get("uploaded_at")
                if not uploaded_at:
                    # Prefer the video file mtime, then the analysis JSON mtime.
                    stamp = (
                        video_path.stat().st_mtime
                        if video_path and video_path.exists()
                        else json_file.stat().st_mtime
                    )
                    uploaded_at = time.strftime(
                        "%Y-%m-%dT%H:%M:%S",
                        time.localtime(stamp),
                    )
                videos.append({
                    "video_id": video_id,
                    "filename": data.get("original_filename") or video_filename or video_id,
                    "uploaded_at": uploaded_at,
                    "file_size": video_path.stat().st_size if video_path and video_path.exists() else 0,
                    "scale_factor": data.get("scale_factor", 1.0),
                    "stabilized": bool(data.get("stabilized", False)),
                    "has_analysis": True
                })
            except Exception as e:
                logger.warning(f"Error reading analysis file {json_file}: {e}")

        videos.sort(key=lambda item: item.get("uploaded_at") or "", reverse=True)
        return {"success": True, "videos": videos}
    except Exception as e:
        logger.error(f"Error getting videos list: {e}")
        return {"success": False, "videos": [], "error": str(e)}

@app.get("/api/video/{video_id}")
async def get_video_analysis(video_id: str):
    """Get analysis results for a specific video"""
    try:
        analysis_file = OUTPUT_DIR / f"{video_id}_analysis.json"
        if not analysis_file.exists():
            raise HTTPException(status_code=404, detail="Analysis not found")

        import json
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {
            "success": True,
            "data": data["analysis_result"],
            "video_info": {
                "filename": data["original_filename"],
                "scale_factor": data["scale_factor"],
                "stabilized": data["stabilized"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving analysis: {str(e)}")

@app.get("/api/video/{video_id}/download")
async def download_video(video_id: str):
    """Download original video file"""
    try:
        # Find video file by video_id
        for json_file in OUTPUT_DIR.glob("*_analysis.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get("video_id") == video_id:
                    video_path = VIDEOS_DIR / data.get("video_filename", "")
                    if video_path.exists():
                        return FileResponse(str(video_path), filename=data.get("video_filename", "video.mp4"))
                    raise HTTPException(status_code=404, detail="Video file not found")
            except Exception as e:
                logger.warning(f"Error reading analysis file {json_file}: {e}")

        raise HTTPException(status_code=404, detail="Video not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/chat")
async def chat_with_ai(request: Dict[str, Any]):
    """Chat with OpenAI for technical support"""
    try:
        message = request.get("message", "")
        system_prompt = request.get("system_prompt", "")
        history = request.get("history", [])

        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Get chat_id
        chat_id = await get_telegram_chat_id()

        # Send user message to Telegram (только если chat_id валидный)
        if chat_id:
            telegram_message = f"👤 Пользователь: {message}"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',
                        json={
                            'chat_id': chat_id,
                            'text': telegram_message[:4096]
                        }
                    ) as send_resp:
                        if send_resp.status == 200:
                            logger.info(f"User message sent to Telegram successfully (chat_id: {chat_id})")
                        else:
                            error_data = await send_resp.text()
                            logger.warning(f"Failed to send user message to Telegram: {send_resp.status} - {error_data}")
            except Exception as e:
                logger.error(f"Failed to send message to Telegram: {e}")
        else:
            logger.warning("Skipping Telegram notification: chat_id not found")

        # DeepSeek API configuration
        deepseek_api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {deepseek_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *history,  # Добавляем историю сообщений
                        {"role": "user", "content": message}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"DeepSeek API error: {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error detail: {error_text}")
                    # Fallback response
                    ai_response = "Извините, в данный момент служба ИИ поддержки недоступна. Пожалуйста, опишите вашу проблему, и мы постараемся помочь."
                else:
                    data = await response.json()
                    ai_response = data["choices"][0]["message"]["content"]

                # Send AI response to Telegram (только если chat_id валидный)
                if chat_id:
                    try:
                        telegram_response = f"🤖 Бот поддержки: {ai_response}"
                        # Преобразуем chat_id в int, если это число
                        try:
                            chat_id_int = int(chat_id) if isinstance(chat_id, str) and chat_id.isdigit() else chat_id
                        except:
                            chat_id_int = chat_id

                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                f"{TELEGRAM_API_URL}/sendMessage",
                                json={
                                    'chat_id': chat_id_int,
                                    'text': telegram_response[:4096]  # Ограничение Telegram API
                                }
                            ) as bot_resp:
                                if bot_resp.status == 200:
                                    logger.info(f"Bot response sent to Telegram successfully (chat_id: {chat_id_int})")
                                else:
                                    error_data = await bot_resp.json() if bot_resp.content_type == 'application/json' else await bot_resp.text()
                                    logger.warning(f"Failed to send bot response to Telegram: {bot_resp.status} - {error_data}")
                    except Exception as e:
                        logger.error(f"Failed to send AI response to Telegram: {e}")

                return {"response": ai_response}

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return {"response": "Извините, произошла ошибка. Попробуйте позже или обратитесь к разработчикам."}

@app.get("/api/sample-data")
async def get_sample_data():
    """Get sample trajectory data for testing"""
    import json
    from pathlib import Path

    try:
        # Try to find any analysis file
        analysis_files = list(OUTPUT_DIR.glob("*_analysis.json"))
        if not analysis_files:
            # Return mock data if no real data exists
            return {
                "success": True,
                "data": {
                    "method": "SLAM",
                    "trajectory": [[0, 0], [10, 5], [20, 15], [30, 25]],
                    "turn_points": [
                        {
                            "frame_index": 50,
                            "trajectory_index": 2,
                            "angle_degrees": 45.0,
                            "position": [20, 15, 0],
                            "turn_type": "right_turn"
                        }
                    ],
                    "frame_count": 100,
                    "trajectory_points": 4,
                    "processing_stats": {
                        "estimated_distance": 42.426,
                        "scale_factor": 12.306,
                        "fps": 30.0,
                        "turns_detected": 1
                    },
                    "total_processing_time": 5.2,
                    "video_info": {
                        "width": 1920,
                        "height": 1080,
                        "fps": 30.0,
                        "frame_count": 100,
                        "duration": 3.33
                    }
                }
            }

        # Return real data from the first analysis file
        # Пробуем разные кодировки для чтения файла
        data = None
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1251']:
            try:
                with open(analysis_files[0], 'r', encoding=encoding) as f:
                    data = json.load(f)
                    logger.info(f"Successfully read analysis file with encoding: {encoding}")
                    break
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to read file with encoding {encoding}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error reading file with encoding {encoding}: {e}")
                continue

        if data and "analysis_result" in data:
            return {
                "success": True,
                "data": data["analysis_result"]
            }
        else:
            # Если не удалось прочитать файл, возвращаем mock данные
            logger.warning("Could not read analysis file, returning mock data")
            return {
                "success": True,
                "data": {
                    "method": "SLAM",
                    "trajectory": [[0, 0], [10, 5], [20, 15], [30, 25]],
                    "turn_points": [
                        {
                            "frame_index": 50,
                            "trajectory_index": 2,
                            "angle_degrees": 45.0,
                            "position": [20, 15, 0],
                            "turn_type": "right_turn"
                        }
                    ],
                    "frame_count": 100,
                    "trajectory_points": 4,
                    "processing_stats": {
                        "estimated_distance": 42.426,
                        "scale_factor": 12.306,
                        "fps": 30.0,
                        "turns_detected": 1
                    },
                    "total_processing_time": 5.2,
                    "video_info": {
                        "width": 1920,
                        "height": 1080,
                        "fps": 30.0,
                        "frame_count": 100,
                        "duration": 3.33
                    }
                }
            }
    except Exception as e:
        logger.error(f"Error getting sample data: {e}")
        # В случае ошибки возвращаем mock данные вместо исключения
        return {
            "success": True,
            "data": {
                "method": "SLAM",
                "trajectory": [[0, 0], [10, 5], [20, 15], [30, 25]],
                "turn_points": [
                    {
                        "frame_index": 50,
                        "trajectory_index": 2,
                        "angle_degrees": 45.0,
                        "position": [20, 15, 0],
                        "turn_type": "right_turn"
                    }
                ],
                "frame_count": 100,
                "trajectory_points": 4,
                "processing_stats": {
                    "estimated_distance": 42.426,
                    "scale_factor": 12.306,
                    "fps": 30.0,
                    "turns_detected": 1
                },
                "total_processing_time": 5.2,
                "video_info": {
                    "width": 1920,
                    "height": 1080,
                    "fps": 30.0,
                    "frame_count": 100,
                    "duration": 3.33
                }
            }
        }


# ──────────────────────────────────────────────
# SSE proxy — real-time R³ streaming from GPU Worker
# ──────────────────────────────────────────────

@app.get("/api/r3-stream/{video_id}")
async def r3_stream_proxy(video_id: str, request: Request):
    """Proxy SSE stream from GPU Worker R³ process to the frontend.
    
    Клиент подключается через EventSource (GET), а мы:
    1. Ищем локальное видео (если есть — отправляем на GPU)
    2. Если видео нет — пробуем replay (GPU сам обнаружит .npz)
    3. Форвардим SSE события обратно
    4. После завершения стрима GPU держим соединение keepalive
    """
    # Find video file locally
    video_filename = UPLOADED_VIDEOS.get(video_id)
    video_path = (VIDEOS_DIR / video_filename) if video_filename else None
    if not video_path or not video_path.exists():
        matches = list(VIDEOS_DIR.glob(f"{video_id}_*"))
        for m in matches:
            if m.is_file():
                video_path = m
                break

    gpu_url = f"{GPU_WORKER_URL}/api/r3-process-stream/{video_id}"
    params = {
        'original_filename': f"{video_id}.mp4",
        'frame_stride': '5',
        'size': '392',
        'max_frames': '1500',
        'ckpt': 'r3_long.safetensors',
        'mode': 'strided',
    }

    async def event_generator():
        try:
            async with aiohttp.ClientSession() as session:
                if video_path and video_path.exists():
                    # Видео есть — POST с телом
                    file_size = video_path.stat().st_size
                    params['original_filename'] = video_path.name
                    async with session.post(
                        gpu_url, params=params,
                        data=open(video_path, 'rb'),
                        headers={'Content-Length': str(file_size)},
                        timeout=aiohttp.ClientTimeout(total=7200),
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            yield f"event: error\ndata: {json.dumps({'message': f'GPU Worker error: {error_text[:200]}'})}\n\n"
                            return
                        while True:
                            line = await resp.content.readline()
                            if not line:
                                break
                            decoded = line.decode('utf-8', errors='replace')
                            if await request.is_disconnected():
                                logger.info(f"[{video_id}] SSE proxy: client disconnected")
                                return
                            yield decoded
                else:
                    # Видео нет — пробуем replay (GPU сам найдёт .npz)
                    logger.info(f"[{video_id}] Video not found locally, trying replay on GPU")
                    async with session.post(
                        gpu_url, params=params,
                        data=b'', headers={'Content-Length': '0'},
                        timeout=aiohttp.ClientTimeout(total=7200),
                    ) as resp:
                        if resp.status == 404:
                            yield f"event: error\ndata: {json.dumps({'message': f'Видео и результаты R³ для {video_id} не найдены'})}\n\n"
                            return
                        if resp.status != 200:
                            error_text = await resp.text()
                            yield f"event: error\ndata: {json.dumps({'message': f'GPU Worker error: {error_text[:200]}'})}\n\n"
                            return
                        while True:
                            line = await resp.content.readline()
                            if not line:
                                break
                            decoded = line.decode('utf-8', errors='replace')
                            if await request.is_disconnected():
                                return
                            yield decoded

            # ─── GPU стрим завершён — держим keepalive, чтобы EventSource не переподключался ──
            while True:
                if await request.is_disconnected():
                    break
                await asyncio.sleep(30)
                yield ": keepalive\n\n"

        except asyncio.CancelledError:
            logger.warning(f"[{video_id}] SSE proxy cancelled")
        except Exception as e:
            logger.error(f"[{video_id}] SSE proxy error ({type(e).__name__}): {e}", exc_info=True)
            try:
                yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            except Exception:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/r3-pointcloud-status/{video_id}")
async def r3_pointcloud_status_proxy(video_id: str):
    """Proxy background point-cloud progress from GPU Worker."""
    gpu_url = f"{GPU_WORKER_URL}/api/r3-pointcloud-status/{video_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(gpu_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                text = await resp.text()
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = {"detail": "GPU Worker returned invalid point-cloud status JSON"}
                return JSONResponse(payload, status_code=resp.status)
    except Exception as exc:
        logger.error(f"[{video_id}] R³ pointcloud status proxy error: {exc}", exc_info=True)
        return JSONResponse({"detail": str(exc)}, status_code=502)


@app.get("/api/r3-pointcloud/{video_id}")
async def r3_pointcloud_proxy(video_id: str, max_points: int = 100000, min_conf: float = 1.0):
    """Proxy completed R³ point cloud from GPU Worker.

    The SSE stream only sends a small preview to keep the live connection light.
    The viewer calls this endpoint after completion to load the full RGB cloud.
    """
    gpu_url = f"{GPU_WORKER_URL}/api/r3-pointcloud/{video_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                gpu_url,
                params={"max_points": str(max_points), "min_conf": str(min_conf)},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return JSONResponse(
                        {"detail": text[:500] or "GPU Worker point cloud error"},
                        status_code=resp.status,
                    )
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    return JSONResponse(
                        {"detail": "GPU Worker returned invalid point cloud JSON"},
                        status_code=502,
                    )
                return JSONResponse(payload)
    except Exception as e:
        logger.error(f"[{video_id}] R³ pointcloud proxy error: {e}", exc_info=True)
        return JSONResponse({"detail": str(e)}, status_code=502)


@app.get("/api/r3-pointcloud-filtered/{video_id}")
async def r3_pointcloud_filtered_proxy(
    video_id: str,
    max_points: int = 100000,
    min_conf: float = 1.4,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    sampling_strategy: str = "random",
    include_trajectory: bool = True,
    include_cameras: bool = True,
):
    """Proxy server-side filtered R³ point cloud from GPU Worker."""
    gpu_url = f"{GPU_WORKER_URL}/api/r3-pointcloud-filtered/{video_id}"
    params = {
        "max_points": str(max_points),
        "min_conf": str(min_conf),
        "sampling_strategy": sampling_strategy,
        "include_trajectory": str(include_trajectory).lower(),
        "include_cameras": str(include_cameras).lower(),
    }
    if frame_start is not None:
        params["frame_start"] = str(frame_start)
    if frame_end is not None:
        params["frame_end"] = str(frame_end)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                gpu_url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return JSONResponse({"detail": text[:500]}, status_code=resp.status)
                try:
                    return JSONResponse(json.loads(text))
                except json.JSONDecodeError:
                    return JSONResponse({"detail": "GPU Worker returned invalid filtered point cloud JSON"}, status_code=502)
    except Exception as e:
        logger.error(f"[{video_id}] R³ filtered pointcloud proxy error: {e}", exc_info=True)
        return JSONResponse({"detail": str(e)}, status_code=502)


@app.get("/api/r3-diagnostics/{video_id}")
async def r3_diagnostics_proxy(video_id: str):
    """Proxy R³ diagnostics from GPU Worker."""
    gpu_url = f"{GPU_WORKER_URL}/api/r3-diagnostics/{video_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(gpu_url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return JSONResponse({"detail": text[:500]}, status_code=resp.status)
                try:
                    return JSONResponse(json.loads(text))
                except json.JSONDecodeError:
                    return JSONResponse({"detail": "GPU Worker returned invalid diagnostics JSON"}, status_code=502)
    except Exception as e:
        logger.error(f"[{video_id}] R³ diagnostics proxy error: {e}", exc_info=True)
        return JSONResponse({"detail": str(e)}, status_code=502)


@app.get("/api/r3-trajectory/{video_id}")
async def r3_trajectory_proxy(
    video_id: str,
    trajectory_source: str = Query("raw"),
):
    """Proxy the current lightweight R3 trajectory post-processing result."""
    gpu_url = f"{GPU_WORKER_URL}/api/r3-trajectory/{video_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                gpu_url,
                params={"trajectory_source": trajectory_source},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return JSONResponse({"detail": text[:500]}, status_code=resp.status)
                try:
                    worker_result = json.loads(text)
                    # The LingBot shadow run is owned by the VPS and therefore
                    # is not present in the R3 GPU worker response. Re-attach
                    # the persisted guarded candidate before map selection so
                    # frontend refreshes keep the production fusion result.
                    analysis_path = OUTPUT_DIR / f"{video_id}_analysis.json"
                    if analysis_path.exists():
                        try:
                            saved_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
                            saved_result = saved_payload.get("analysis_result") or {}
                            saved_stats = saved_result.get("processing_stats") or {}
                            requested_source = str(trajectory_source or "raw")
                            saved_source = saved_stats.get("r3_trajectory_source")
                            saved_source_requested = saved_stats.get(
                                "r3_trajectory_source_requested"
                            )
                            if not isinstance(saved_source, str) or not saved_source.strip():
                                saved_source = str(
                                    saved_source_requested
                                    or "scale_aware_candidate"
                                )
                            candidate = saved_result.get("lingbot_fusion_candidate")
                            if should_restore_lingbot_fusion_candidate(
                                candidate,
                                requested_source=requested_source,
                                saved_source=str(saved_source),
                                saved_source_requested=str(
                                    saved_source_requested or ""
                                ),
                            ):
                                worker_result["lingbot_fusion_candidate"] = candidate
                                worker_result["lingbot_shadow"] = saved_result.get("lingbot_shadow")
                                # Fragmentation lives on the saved VPS result; the
                                # GPU worker payload alone would skip independent.
                                if saved_result.get("r3_pose_graph") is not None:
                                    worker_result["r3_pose_graph"] = saved_result.get(
                                        "r3_pose_graph"
                                    )
                                if saved_result.get("pose_graph") is not None:
                                    worker_result["pose_graph"] = saved_result.get(
                                        "pose_graph"
                                    )
                                stats = dict(worker_result.get("processing_stats") or {})
                                stats["lingbot_fusion"] = (
                                    (candidate.get("diagnostics") if isinstance(candidate, dict) else None)
                                    or saved_stats.get("lingbot_fusion")
                                    or {}
                                )
                                stats["lingbot_shadow_available"] = True
                                for graph_key in (
                                    "component_count",
                                    "largest_component_coverage",
                                    "largest_component_ratio",
                                    "connected_pose_count",
                                    "connected_poses",
                                ):
                                    if graph_key in saved_stats:
                                        stats[graph_key] = saved_stats[graph_key]
                                # Repair corrupted historical saves.
                                if not isinstance(stats.get("r3_trajectory_source"), str):
                                    stats["r3_trajectory_source"] = str(saved_source)
                                worker_result["processing_stats"] = stats
                        except Exception as saved_error:
                            logger.warning(
                                "[%s] Could not restore persisted LingBot fusion: %s",
                                video_id,
                                saved_error,
                            )
                    constrained = apply_floorplan_constraints(
                        worker_result,
                        _load_task_map_context(video_id),
                    )
                    return JSONResponse(_to_json_serializable(constrained))
                except json.JSONDecodeError:
                    return JSONResponse(
                        {"detail": "GPU Worker returned invalid trajectory JSON"},
                        status_code=502,
                    )
    except Exception as e:
        logger.error(f"[{video_id}] R3 trajectory proxy error: {e}", exc_info=True)
        return JSONResponse({"detail": str(e)}, status_code=502)


async def _proxy_lingbot_json(path: str, timeout_seconds: int = 60) -> JSONResponse:
    url = f"{LINGBOT_WORKER_URL}{path}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_seconds)) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return JSONResponse({"detail": text[:500]}, status_code=resp.status)
                try:
                    return JSONResponse(json.loads(text))
                except json.JSONDecodeError:
                    return JSONResponse({"detail": "LingBot worker returned invalid JSON"}, status_code=502)
    except Exception as e:
        logger.error(f"LingBot proxy error for {path}: {e}", exc_info=True)
        return JSONResponse({"detail": str(e)}, status_code=502)


@app.get("/api/lingbot-health")
async def lingbot_health_proxy():
    """Proxy LingBot-Map worker health from the RTX 3090 host."""
    return await _proxy_lingbot_json("/health")


@app.get("/api/lingbot-sessions/{session_id}/status")
async def lingbot_session_status_proxy(session_id: str):
    """Proxy LingBot-Map session status."""
    return await _proxy_lingbot_json(f"/sessions/{session_id}/status")


@app.get("/api/lingbot-sessions/{session_id}/trajectory")
async def lingbot_session_trajectory_proxy(session_id: str):
    """Proxy LingBot-Map trajectory JSON."""
    return await _proxy_lingbot_json(f"/sessions/{session_id}/trajectory")


@app.get("/api/lingbot-sessions/{session_id}/metadata")
async def lingbot_session_metadata_proxy(session_id: str):
    """Proxy LingBot-Map session metadata."""
    return await _proxy_lingbot_json(f"/sessions/{session_id}/metadata")


@app.get("/api/lingbot-sessions/{session_id}/pointcloud")
async def lingbot_session_pointcloud_proxy(session_id: str):
    """Proxy LingBot-Map point cloud artifact."""
    url = f"{LINGBOT_WORKER_URL}/sessions/{session_id}/pointcloud"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=180)) as resp:
                data = await resp.read()
                if resp.status != 200:
                    return JSONResponse(
                        {"detail": data.decode("utf-8", errors="replace")[:500]},
                        status_code=resp.status,
                    )
                content_type = resp.headers.get("content-type", "application/octet-stream")
                filename = resp.headers.get("content-disposition", "")
                headers = {}
                if filename:
                    headers["Content-Disposition"] = filename
                return Response(content=data, media_type=content_type, headers=headers)
    except Exception as e:
        logger.error(f"LingBot pointcloud proxy error for {session_id}: {e}", exc_info=True)
        return JSONResponse({"detail": str(e)}, status_code=502)


@app.get("/api/r3-projection-debug/{video_id}")
async def r3_projection_debug_proxy(
    video_id: str,
    max_points: int = 250000,
    min_conf: float = 1.4,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    sampling_strategy: str = "per_frame_uniform",
):
    """Proxy server-side R³ top/front/right projection debug generation."""
    gpu_url = f"{GPU_WORKER_URL}/api/r3-projection-debug/{video_id}"
    params = {
        "max_points": str(max_points),
        "min_conf": str(min_conf),
        "sampling_strategy": sampling_strategy,
    }
    if frame_start is not None:
        params["frame_start"] = str(frame_start)
    if frame_end is not None:
        params["frame_end"] = str(frame_end)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(gpu_url, params=params, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return JSONResponse({"detail": text[:500]}, status_code=resp.status)
                try:
                    return JSONResponse(json.loads(text))
                except json.JSONDecodeError:
                    return JSONResponse({"detail": "GPU Worker returned invalid projection debug JSON"}, status_code=502)
    except Exception as e:
        logger.error(f"[{video_id}] R³ projection debug proxy error: {e}", exc_info=True)
        return JSONResponse({"detail": str(e)}, status_code=502)


# ──────────────────────────────────────────────
# Static files for production frontend
# ──────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "dist"

# Serve static assets
if FRONTEND_DIR.exists():
    logger.info(f"Mounting frontend from {FRONTEND_DIR}")
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")
    app.mount("/downloads", StaticFiles(directory=str(FRONTEND_DIR / "downloads")), name="downloads")

    # SPA fallback: any non-API, non-asset path → index.html
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str = ""):
        if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("openapi"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path), media_type="text/html")
        return JSONResponse({"detail": "Not Found"}, status_code=404)
else:
    logger.warning(f"Frontend dist directory not found at {FRONTEND_DIR}, serving API only")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
