"""
GPU Worker — лёгкий сервис для тяжёлого CV-пайплайна (SLAM, YOLO, стабилизация).
Работает на RTX 3090 (или любом GPU-сервере), порт 8003.

Принимает видео + параметры через multipart POST /process-video,
запускает полный пайплайн и возвращает JSON с результатами анализа траектории.
"""
import os, sys, time, json, shutil, subprocess, asyncio, threading, math
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import threading

# --- video_tracker imports (CV pipeline) ---
from video_tracker.src.processor import FullFeatureProcessor
from video_tracker.src.map_postprocessing import apply_map_postprocessing
from video_tracker.src.stabilization import stabilize_video

try:
    from r3_trajectory import build_r3_trajectory, summarize_fallback_edges
except ImportError:  # pragma: no cover - supports package-style startup
    from backend.r3_trajectory import build_r3_trajectory, summarize_fallback_edges

try:
    from r3_pointcloud import PointCloudBuildCancelled, build_sampled_pointcloud
except ImportError:  # pragma: no cover - supports package-style startup
    from backend.r3_pointcloud import PointCloudBuildCancelled, build_sampled_pointcloud

try:
    from r3_pose_graph import load_pose_graph_summary
except ImportError:  # pragma: no cover - supports package-style startup
    from backend.r3_pose_graph import load_pose_graph_summary

try:
    from r3_pose_graph_optimizer import load_pose_graph_candidate_summary
except ImportError:  # pragma: no cover - supports package-style startup
    from backend.r3_pose_graph_optimizer import load_pose_graph_candidate_summary

try:
    from r3_trajectory_sources import select_r3_trajectory_camera_poses
except ImportError:  # pragma: no cover - supports package-style startup
    from backend.r3_trajectory_sources import select_r3_trajectory_camera_poses

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger("gpu_worker")

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = FastAPI(title="TrackAI GPU Worker", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK_DIR = Path("/home/artem/trackai/gpu_worker_data")
WORK_DIR.mkdir(parents=True, exist_ok=True)
FFMPEG_TIMEOUT = 1800
CONDA_TRACKAI_BIN = Path("/home/artem/miniconda3/envs/trackai/bin")

# ──────────────────────────────────────────────
# Helpers (copied from main.py for self-containment)
# ──────────────────────────────────────────────
try:
    import numpy as _np
except Exception:
    _np = None

def _to_json_serializable(obj: Any):
    if _np is not None:
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_serializable(v) for v in obj]
    return obj

def _resolve_binary(name: str) -> str:
    for candidate in (Path("/usr/bin") / name, Path("/usr/local/bin") / name, CONDA_TRACKAI_BIN / name):
        if candidate.exists():
            return str(candidate)
    return name


def _get_video_duration_sec(video_path: Path) -> float:
    ffprobe = _resolve_binary("ffprobe")
    try:
        r = subprocess.run([ffprobe, '-v', 'error', '-show_entries', 'format=duration',
                            '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0

def _validate_video_readable(video_path: Path) -> bool:
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return False
        ok, frame = cap.read()
        cap.release()
        return bool(ok and frame is not None)
    except Exception:
        return False

def _normalize_turn_list(turns):
    if not turns:
        return turns
    out = []
    for turn in turns:
        nt = turn.copy()
        if isinstance(turn.get("position"), dict):
            pos = turn["position"]
            nt["position"] = [pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)]
        elif not isinstance(turn.get("position"), list):
            nt["position"] = [0, 0, 0]
        out.append(nt)
    return out

# ──────────────────────────────────────────────
# Progress callback factory
# ──────────────────────────────────────────────
def _make_progress_callback(status_dict: dict, offset: float, span: float):
    def cb(p: float):
        status_dict["progress"] = int(offset + p * span)
    return cb

# ──────────────────────────────────────────────
# Processing pipeline
# ──────────────────────────────────────────────
async def run_analysis_pipeline(
    video_id: str,
    video_path: Path,
    original_filename: str,
    scale_factor: float = 12.306,
    stabilize: bool = True,
    detect_interval: int = 3,
    turn_vote_threshold: int = 3,
    use_ml_roi: bool = True,
    map_context: Optional[Dict] = None,
    frame_skip: int = 1,
) -> dict:
    """Run the full trajectory analysis on GPU.

    Returns the analysis result dict (same structure as local process_video_background).
    Raises on error.
    """
    from fastapi.concurrency import run_in_threadpool

    ffmpeg = _resolve_binary("ffmpeg")
    status = {}  # local status for progress tracking

    # Work directory
    temp_dir = WORK_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Copy input file to temp dir first (never modify the original)
    work_path = temp_dir / f"proc_{video_id}{video_path.suffix}"
    shutil.copy2(video_path, work_path)
    processing_path = work_path

    # Step 0: AVI → MP4
    if video_path.suffix.lower() == '.avi':
        logger.info(f"[{video_id}] Converting AVI to MP4...")
        duration_sec = await run_in_threadpool(_get_video_duration_sec, video_path)
        avi_mp4 = temp_dir / f"proc_{video_id}.from_avi.mp4"
        cmd = [ffmpeg, '-i', str(work_path), '-c:v', 'libx264', '-preset', 'fast',
               '-crf', '23', '-c:a', 'aac', '-movflags', '+faststart', '-y', str(avi_mp4)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
        if _validate_video_readable(avi_mp4):
            work_path.unlink()
            work_path = avi_mp4
            processing_path = work_path
        else:
            raise Exception("AVI→MP4 output is not readable")

    # Step 1: Normalize to 720p
    logger.info(f"[{video_id}] Normalizing to 720p...")
    normalized = temp_dir / f"proc_{video_id}.optimized.mp4"
    try:
        subprocess.run([
            ffmpeg, '-i', str(work_path), '-vf', 'scale=-2:min(ih\\,720)',
            '-c:v', 'libx264', '-preset', 'superfast', '-crf', '23',
            '-c:a', 'aac', '-movflags', '+faststart', '-y', str(normalized)
        ], check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
        if _validate_video_readable(normalized):
            work_path.unlink()
            work_path = normalized
            processing_path = work_path
        else:
            normalized.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"[{video_id}] Normalization failed: {e}, using original")

    # Step 3: Stabilization
    if stabilize:
        logger.info(f"[{video_id}] Stabilizing...")
        stab_path = temp_dir / f"stab_{video_id}{processing_path.suffix}"
        try:
            processing_path = await run_in_threadpool(
                stabilize_video, processing_path, stab_path,
                progress_callback=_make_progress_callback(status, 0, 1),
                dynamic_smoothing=True
            )
            if not _validate_video_readable(processing_path):
                processing_path = work_path if work_path.is_file() else temp_dir / f"proc_{video_id}.optimized.mp4"
            logger.info(f"[{video_id}] Stabilization done")
        except Exception as e:
            logger.error(f"[{video_id}] Stabilization error: {e}")
            processing_path = work_path if work_path.is_file() else temp_dir / f"proc_{video_id}.optimized.mp4"

        if processing_path != work_path and work_path.exists():
            work_path.unlink()

    # Step 4: SLAM analysis
    logger.info(f"[{video_id}] SLAM analysis...")
    # Use GPU providers for ONNX
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    logger.info(f"ONNX providers available: {ort.get_available_providers()}")

    detector_model = Path(__file__).resolve().parent / "models" / "detector.onnx"
    processor = FullFeatureProcessor(
        input_dir=str(temp_dir),
        output_dir=str(temp_dir),
        scale_factor=scale_factor,
        progress_callback=_make_progress_callback(status, 0, 1),
        use_homography=True,
        use_kalman=True,
        use_akaze=True,
        frame_skip=frame_skip,
        target_width=900,
        use_optical_flow=True,
        detect_interval=max(1, int(detect_interval)),
        turn_vote_threshold=max(1, min(5, int(turn_vote_threshold))),
        use_ml_roi=use_ml_roi,
        ml_model_path=str(detector_model),
    )

    result = await run_in_threadpool(processor.process_video, processing_path)

    # Cleanup temp
    if processing_path.exists():
        processing_path.unlink()

    if not result:
        raise Exception(
            "Процессор не вернул результат (не удалось обработать кадры). "
            "Попробуйте отключить стабилизацию или загрузить видео в формате MP4 (H.264)."
        )

    traj = result.get("trajectory") or []
    if not traj:
        raise Exception(
            "Траектория пуста: алгоритм не смог построить путь по кадрам. "
            "Попробуйте другое видео, отключите стабилизацию или улучшите освещение/текстуру сцены."
        )

    # Normalise result
    normalized = result.copy()
    for key in ("turn_points", "raw_turn_points", "trajectory_turn_points"):
        if key in normalized and normalized[key]:
            normalized[key] = _normalize_turn_list(normalized[key])

    if map_context:
        try:
            normalized = apply_map_postprocessing(normalized, map_context)
            if normalized.get("map_turn_points"):
                normalized["map_turn_points"] = _normalize_turn_list(normalized["map_turn_points"])
        except Exception as e:
            logger.warning(f"[{video_id}] Map post-processing error: {e}")

    normalized["final_turn_points"] = (
        normalized.get("map_turn_points")
        or normalized.get("turn_points")
        or []
    )

    normalized = _to_json_serializable(normalized)
    return normalized


# ──────────────────────────────────────────────
# API endpoints
# ──────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "TrackAI GPU Worker"}

@app.post("/api/process-video")
async def process_video(
    video_file: UploadFile = File(...),
    video_id: str = Form(...),
    original_filename: str = Form(...),
    scale_factor: float = Form(12.306),
    stabilize: bool = Form(True),
    detect_interval: int = Form(3),
    turn_vote_threshold: int = Form(3),
    use_ml_roi: bool = Form(True),
    map_context: Optional[str] = Form(None),
):
    """Receive video + params via multipart, run GPU-accelerated analysis."""
    start_ts = time.time()
    logger.info(f"[{video_id}] Received video via multipart: {original_filename}")

    return await _handle_process_request(
        video_id=video_id,
        original_filename=original_filename,
        video_file_obj=video_file,
        scale_factor=scale_factor,
        stabilize=stabilize,
        detect_interval=detect_interval,
        turn_vote_threshold=turn_vote_threshold,
        use_ml_roi=use_ml_roi,
        map_context=map_context,
    )


@app.post("/api/process-video-raw/{video_id}")
async def process_video_raw(
    video_id: str,
    request: Request,
    original_filename: str = "",
    scale_factor: float = 12.306,
    stabilize: str = "true",
    detect_interval: int = 3,
    turn_vote_threshold: int = 3,
    use_ml_roi: str = "true",
    map_context: Optional[str] = None,
    use_uploaded: bool = False,
):
    """Receive raw video bytes via streaming POST body + query params.
    This endpoint streams directly to disk without buffering the whole file in memory.
    
    If use_uploaded=true, reads from the pre-uploaded file in uploaded/ directory
    instead of reading from request body.
    """
    start_ts = time.time()
    logger.info(f"[{video_id}] Receiving raw video stream: {original_filename} (use_uploaded={use_uploaded})")

    # Convert string params to proper types
    stabilize_bool = stabilize.lower() in ('true', '1', 'yes')
    use_ml_roi_bool = use_ml_roi.lower() in ('true', '1', 'yes')

    if use_uploaded:
        # ─── Видео уже на сервере — используем предзагруженный файл ───
        UPLOAD_DIR = WORK_DIR / "uploaded"
        ext = Path(original_filename).suffix if original_filename and '.' in original_filename else '.mp4'
        uploaded_path = UPLOAD_DIR / f"{video_id}{ext}"
        if not uploaded_path.exists():
            raise HTTPException(status_code=404, detail=f"Uploaded video {video_id} not found")
        local_path = uploaded_path
        bytes_written = uploaded_path.stat().st_size
        logger.info(f"[{video_id}] Using pre-uploaded file: {local_path} ({bytes_written} bytes)")
        # Drain request body silently
        async for _ in request.stream():
            pass
    else:
        # ─── Получаем видео из тела запроса ───
        ext = Path(original_filename).suffix if original_filename and '.' in original_filename else '.mp4'
        local_path = WORK_DIR / f"in_{video_id}{ext}"

        bytes_written = 0
        with open(local_path, 'wb') as f:
            async for chunk in request.stream():
                f.write(chunk)
                bytes_written += len(chunk)

    logger.info(f"[{video_id}] Streamed {bytes_written} bytes to {local_path}")

    # Parse map_context
    mc = None
    if map_context:
        try:
            mc = json.loads(map_context)
        except json.JSONDecodeError:
            logger.warning(f"[{video_id}] Invalid map_context JSON, ignoring")

    try:
        result = await run_analysis_pipeline(
            video_id=video_id,
            video_path=local_path,
            original_filename=original_filename or f"video{ext}",
            scale_factor=scale_factor,
            stabilize=stabilize_bool,
            detect_interval=detect_interval,
            turn_vote_threshold=turn_vote_threshold,
            use_ml_roi=use_ml_roi_bool,
            map_context=mc,
        )

        elapsed = time.time() - start_ts
        logger.info(f"[{video_id}] Analysis completed in {elapsed:.1f}s")

        return {
            "success": True,
            "video_id": video_id,
            "processing_time": round(elapsed, 1),
            "result": result,
        }

    except Exception as e:
        elapsed = time.time() - start_ts
        logger.error(f"[{video_id}] Analysis failed after {elapsed:.1f}s: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "video_id": video_id, "error": str(e)},
        )

    finally:
        if local_path.exists():
            local_path.unlink()


@app.post("/api/process-existing-file/{video_id}")
async def process_existing_file(
    video_id: str,
    file_path: str = Query(...),
    original_filename: str = "",
    scale_factor: float = 12.306,
    stabilize: str = "true",
    detect_interval: int = 3,
    turn_vote_threshold: int = 3,
    use_ml_roi: str = "true",
    map_context: Optional[str] = None,
):
    """Process a video file that already exists on the GPU server disk."""
    start_ts = time.time()

    stabilize_bool = stabilize.lower() in ('true', '1', 'yes')
    use_ml_roi_bool = use_ml_roi.lower() in ('true', '1', 'yes')

    local_path = Path(file_path)
    if not local_path.exists():
        return JSONResponse(status_code=404, content={"error": f"File not found: {file_path}"})

    file_size = local_path.stat().st_size
    logger.info(f"[{video_id}] Processing existing file: {file_path} ({file_size} bytes)")

    # Parse map_context
    mc = None
    if map_context:
        try:
            mc = json.loads(map_context)
        except json.JSONDecodeError:
            logger.warning(f"[{video_id}] Invalid map_context JSON, ignoring")

    try:
        result = await run_analysis_pipeline(
            video_id=video_id,
            video_path=local_path,
            original_filename=original_filename or local_path.name,
            scale_factor=scale_factor,
            stabilize=stabilize_bool,
            detect_interval=detect_interval,
            turn_vote_threshold=turn_vote_threshold,
            use_ml_roi=use_ml_roi_bool,
            map_context=mc,
        )

        elapsed = time.time() - start_ts
        logger.info(f"[{video_id}] Analysis completed in {elapsed:.1f}s")

        return {
            "success": True,
            "video_id": video_id,
            "file_path": str(local_path),
            "processing_time": round(elapsed, 1),
            "result": result,
        }

    except Exception as e:
        elapsed = time.time() - start_ts
        logger.error(f"[{video_id}] Analysis failed after {elapsed:.1f}s: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "video_id": video_id, "error": str(e)},
        )


async def _handle_process_request(
    video_id: str,
    original_filename: str,
    video_file_obj,
    scale_factor: float,
    stabilize: bool,
    detect_interval: int,
    turn_vote_threshold: int,
    use_ml_roi: bool,
    map_context: Optional[str],
):
    """Shared handler: save file to disk, run pipeline, return JSON result."""
    start_ts = time.time()

    # Determine extension
    fn = getattr(video_file_obj, 'filename', None) or original_filename or 'video.mp4'
    if not fn.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        fn += '.mp4'

    ext = Path(fn).suffix
    local_path = WORK_DIR / f"in_{video_id}{ext}"

    # Read the uploaded file
    content = await video_file_obj.read()
    with open(local_path, "wb") as f:
        f.write(content)
    logger.info(f"[{video_id}] Saved {len(content)} bytes to {local_path}")

    # Parse map_context
    mc = None
    if map_context:
        try:
            mc = json.loads(map_context)
        except json.JSONDecodeError:
            logger.warning(f"[{video_id}] Invalid map_context JSON, ignoring")

    try:
        result = await run_analysis_pipeline(
            video_id=video_id,
            video_path=local_path,
            original_filename=original_filename or fn,
            scale_factor=scale_factor,
            stabilize=stabilize,
            detect_interval=detect_interval,
            turn_vote_threshold=turn_vote_threshold,
            use_ml_roi=use_ml_roi,
            map_context=mc,
        )

        elapsed = time.time() - start_ts
        logger.info(f"[{video_id}] Analysis completed in {elapsed:.1f}s")

        return {
            "success": True,
            "video_id": video_id,
            "processing_time": round(elapsed, 1),
            "result": result,
        }

    except Exception as e:
        elapsed = time.time() - start_ts
        logger.error(f"[{video_id}] Analysis failed after {elapsed:.1f}s: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "video_id": video_id, "error": str(e)},
        )

    finally:
        if local_path.exists():
            local_path.unlink()


# ──────────────────────────────────────────────
# R³ (Depth Anything 3 Reconstruction) endpoint
# ──────────────────────────────────────────────

import math

R3_WRAPPER = Path(__file__).resolve().parent / "r3_worker_wrapper.py"
R3_OUTPUT_DIR = WORK_DIR / "r3_output"

CONF_THRESHOLDS = [0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

_r3_pointcloud_lock = threading.Lock()
_r3_pointcloud_jobs: Dict[str, threading.Event] = {}
_r3_pointcloud_statuses: Dict[str, dict] = {}


def _r3_pointcloud_status_path(output_dir: Path) -> Path:
    return output_dir / "pointcloud_status.json"


def _set_r3_pointcloud_status(video_id: str, output_dir: Path, **updates: Any) -> dict:
    with _r3_pointcloud_lock:
        current = dict(_r3_pointcloud_statuses.get(video_id, {}))
        current.update({
            "video_id": video_id,
            "updated_at": time.time(),
            **updates,
        })
        _r3_pointcloud_statuses[video_id] = current
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path = _r3_pointcloud_status_path(output_dir)
        temp_path = status_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(_to_json_serializable(current), indent=2), encoding="utf-8")
        os.replace(temp_path, status_path)
    except Exception:
        pass
    return current


def _get_r3_pointcloud_status(video_id: str, output_dir: Path) -> dict:
    with _r3_pointcloud_lock:
        current = _r3_pointcloud_statuses.get(video_id)
        if current:
            return dict(current)
    status_path = _r3_pointcloud_status_path(output_dir)
    if status_path.exists():
        try:
            return json.loads(status_path.read_text())
        except Exception:
            pass
    pointcloud_path = output_dir / "pointcloud.npz"
    if pointcloud_path.exists():
        return {
            "video_id": video_id,
            "status": "completed",
            "stage": "ready",
            "progress": 100,
            "message": "3D-облако готово",
        }
    return {
        "video_id": video_id,
        "status": "not_started",
        "stage": "waiting",
        "progress": 0,
        "message": "3D-облако ещё не запущено",
    }


def _cancel_r3_pointcloud_job(video_id: str) -> None:
    with _r3_pointcloud_lock:
        cancel_event = _r3_pointcloud_jobs.get(video_id)
    if cancel_event is not None:
        cancel_event.set()


def _schedule_r3_pointcloud_build(video_id: str, output_dir: Path) -> dict:
    """Start memory-bounded point-cloud export without blocking trajectory."""
    pointcloud_path = output_dir / "pointcloud.npz"
    if pointcloud_path.exists():
        return _set_r3_pointcloud_status(
            video_id,
            output_dir,
            status="completed",
            stage="ready",
            progress=100,
            message="3D-облако готово",
        )

    with _r3_pointcloud_lock:
        existing = _r3_pointcloud_jobs.get(video_id)
        if existing is not None and not existing.is_set():
            return dict(_r3_pointcloud_statuses.get(video_id, {}))
        cancel_event = threading.Event()
        _r3_pointcloud_jobs[video_id] = cancel_event

    queued = _set_r3_pointcloud_status(
        video_id,
        output_dir,
        status="queued",
        stage="queued",
        progress=0,
        message="Траектория готова; 3D-облако поставлено в фоновую очередь",
    )

    def run() -> None:
        try:
            def on_progress(payload: dict[str, Any]) -> None:
                _set_r3_pointcloud_status(video_id, output_dir, **payload)

            result = build_sampled_pointcloud(
                output_dir,
                stride=max(1, int(os.getenv("R3_POINTCLOUD_STRIDE", "4"))),
                max_points=max(1_000, int(os.getenv("R3_POINTCLOUD_MAX_POINTS", "200000"))),
                min_conf=float(os.getenv("R3_POINTCLOUD_MIN_CONF", "1.0")),
                progress_callback=on_progress,
                should_cancel=cancel_event.is_set,
                return_points=False,
            )
            _set_r3_pointcloud_status(
                video_id,
                output_dir,
                status="completed",
                stage="ready",
                progress=100,
                message=f"3D-облако готово: {result['num_points']:,} точек",
                points=result["num_points"],
                source_points=result["source_points"],
                frames_used=result["frames_used"],
                elapsed_seconds=result["elapsed_seconds"],
                full_debug_saved=result["full_debug_saved"],
            )
        except PointCloudBuildCancelled:
            _set_r3_pointcloud_status(
                video_id,
                output_dir,
                status="cancelled",
                stage="cancelled",
                progress=0,
                message="Построение 3D-облака отменено новым запуском",
            )
        except Exception as exc:
            logger.error(f"[{video_id}] Background R3 point cloud failed: {exc}", exc_info=True)
            _set_r3_pointcloud_status(
                video_id,
                output_dir,
                status="error",
                stage="error",
                progress=0,
                message=f"Ошибка построения 3D-облака: {exc}",
                error=str(exc),
            )
        finally:
            with _r3_pointcloud_lock:
                if _r3_pointcloud_jobs.get(video_id) is cancel_event:
                    _r3_pointcloud_jobs.pop(video_id, None)

    threading.Thread(
        target=run,
        name=f"r3-pointcloud-{video_id[:12]}",
        daemon=True,
    ).start()
    return queued


def _sanitize_for_json(obj):
    """Recursively convert NaN/Inf to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _r3_output_dir(video_id: str) -> Path:
    return R3_OUTPUT_DIR / video_id


def _r3_find_pointcloud_path(pc_dir: Path, prefer_debug: bool = False) -> Path:
    names = ["pointcloud_full_debug.npz", "pointcloud.npz"] if prefer_debug else ["pointcloud.npz", "pointcloud_full_debug.npz"]
    for name in names:
        cand = pc_dir / name
        if cand.exists():
            return cand
    subdirs = sorted(
        [d for d in pc_dir.iterdir() if d.is_dir() and d.name != "frames"],
        key=lambda p: p.stat().st_mtime,
    ) if pc_dir.exists() else []
    for sd in reversed(subdirs):
        for name in names:
            cand = sd / name
            if cand.exists():
                return cand
    return pc_dir / names[0]


def _load_r3_pointcloud(video_id: str, prefer_debug: bool = False):
    import numpy as np

    pc_dir = _r3_output_dir(video_id)
    if not pc_dir.exists():
        raise HTTPException(status_code=404, detail=f"No R³ output for {video_id}")
    npz_path = _r3_find_pointcloud_path(pc_dir, prefer_debug=prefer_debug)
    if not npz_path.exists():
        raise HTTPException(status_code=404, detail="Point cloud not found (R³ may still be processing)")
    data = np.load(str(npz_path))
    if "points" not in data:
        raise HTTPException(status_code=500, detail=f"Point cloud {npz_path.name} has no 'points' array")
    return data["points"], npz_path


def _sample_points(points, max_points: int, strategy: str = "random"):
    import numpy as np

    if max_points <= 0 or len(points) <= max_points:
        return points
    strategy = strategy or "random"
    if strategy == "per_frame_uniform":
        if points.ndim != 2 or points.shape[1] <= 7:
            raise HTTPException(
                status_code=409,
                detail="per_frame_uniform requires a point cloud with frame_idx",
            )
        frames = points[:, 7]
        valid_frames = frames[np.isfinite(frames)]
        unique_frames = np.unique(valid_frames.astype(np.int64))
        if unique_frames.size == 0:
            return points[:0]
        per_frame = max(1, int(math.ceil(max_points / unique_frames.size)))
        rng = np.random.RandomState(42)
        chunks = []
        for frame in unique_frames:
            frame_points = points[frames == frame]
            if len(frame_points) <= per_frame:
                chunks.append(frame_points)
                continue
            idx = rng.choice(len(frame_points), per_frame, replace=False)
            idx = np.sort(idx)
            chunks.append(frame_points[idx])
        sampled = np.concatenate(chunks, axis=0) if chunks else points[:0]
        if len(sampled) > max_points:
            idx = np.linspace(0, len(sampled) - 1, max_points).astype(np.int64)
            sampled = sampled[idx]
        return sampled
    if strategy == "random":
        idx = np.random.RandomState(42).choice(len(points), max_points, replace=False)
        return points[idx]
    if strategy == "voxel":
        xyz = points[:, :3]
        mins = np.nanmin(xyz, axis=0)
        span = np.maximum(np.nanmax(xyz, axis=0) - mins, 1e-6)
        grid = max(8, int(round(max_points ** (1 / 3))))
        keys = np.floor((xyz - mins) / span * grid).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        idx = np.sort(idx)
        if len(idx) > max_points:
            idx = idx[np.linspace(0, len(idx) - 1, max_points).astype(np.int64)]
        return points[idx]
    if points.ndim == 2 and points.shape[1] > 6:
        idx = np.argsort(points[:, 6])[-max_points:]
        idx = idx[np.argsort(idx)]
        return points[idx]
    step = max(1, len(points) // max_points)
    return points[::step][:max_points]


def _filter_r3_points(
    points,
    min_conf: float = 1.0,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
):
    import numpy as np

    if points.ndim != 2:
        raise HTTPException(status_code=500, detail=f"Point cloud must be 2D, got shape {points.shape}")
    if points.shape[1] < 7:
        raise HTTPException(
            status_code=409,
            detail=f"Point cloud has {points.shape[1]} columns; regenerate it to include confidence",
        )
    mask = np.isfinite(points[:, :7]).all(axis=1) & (points[:, 6] >= min_conf)
    if frame_start is not None or frame_end is not None:
        if points.shape[1] <= 7:
            raise HTTPException(
                status_code=409,
                detail="Frame filtering requires a point cloud with frame_idx; regenerate R³ point cloud",
            )
        frames = points[:, 7]
        mask = mask & np.isfinite(frames)
        if frame_start is not None:
            mask = mask & (frames >= frame_start)
        if frame_end is not None:
            mask = mask & (frames <= frame_end)
    return points[mask]


def _r3_counts_by_threshold(conf_values):
    import numpy as np

    return {str(t): int((conf_values >= t).sum()) for t in CONF_THRESHOLDS}


def _r3_conf_stats(conf_values):
    import numpy as np

    if conf_values.size == 0:
        return {"percentiles": {}, "counts_by_threshold": {}}
    p = np.percentile(conf_values, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    keys = ["p0", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p100"]
    return {
        "percentiles": {k: float(v) for k, v in zip(keys, p)},
        "counts_by_threshold": _r3_counts_by_threshold(conf_values),
    }


def _load_r3_run_params(base: Path, point_count: int = 0) -> tuple[dict, dict]:
    """Load R3 params and attach accepted fallback boundaries from its edge log."""
    run_params: dict = {}
    run_params_path = base / "run_params.json"
    if run_params_path.exists():
        try:
            loaded = json.loads(run_params_path.read_text())
            if isinstance(loaded, dict):
                run_params = loaded
        except Exception as exc:
            run_params = {"error": str(exc)}

    edges = []
    pose_edges_path = base / "pose_edge_log.json"
    if pose_edges_path.exists():
        try:
            loaded_edges = json.loads(pose_edges_path.read_text())
            if isinstance(loaded_edges, list):
                edges = loaded_edges
        except Exception:
            edges = []

    try:
        bridge_window = int(run_params.get("fallback_num_bridge_frames") or 10)
    except (TypeError, ValueError):
        bridge_window = 10
    fallback_summary = summarize_fallback_edges(
        edges,
        point_count=point_count,
        bridge_window=bridge_window,
    )
    run_params = dict(run_params)
    run_params["fallback_boundaries"] = fallback_summary["boundaries"]
    run_params["fallback_boundary_source"] = fallback_summary["source"]
    run_params["fallback_events"] = fallback_summary["events"]
    pose_graph_summary = load_pose_graph_summary(
        base / "pose_graph_edges.npz",
        point_count=point_count,
    )
    run_params["pose_graph_optimizer_ready"] = pose_graph_summary.get("optimizer_ready", False)
    run_params["pose_graph_edge_count"] = pose_graph_summary.get("edge_count", 0)
    candidate_summary = load_pose_graph_candidate_summary(base)
    run_params["pose_graph_candidate_accepted"] = candidate_summary.get("accepted", False)
    run_params["pose_graph_optimizer_seconds"] = candidate_summary.get("runtime_seconds")
    return run_params, fallback_summary


def _load_r3_trajectory_bundle(
    base: Path,
    trajectory_source: str = "raw",
) -> tuple[dict, list[dict]]:
    """Load R3 pose artifacts once and keep 3D/plan-space products separate."""
    import numpy as np

    camera_poses: list[dict] = []
    for camera_file in sorted((base / "camera").glob("*.npz")):
        try:
            camera = np.load(str(camera_file))
            pose = camera["pose"]
            camera_poses.append({
                "frame": int(camera_file.stem),
                "pose": pose.tolist(),
                "intrinsics": camera["intrinsics"].tolist() if "intrinsics" in camera else None,
            })
        except Exception:
            continue

    pose_confidence = None
    pose_confidence_path = base / "pose_conf.npy"
    if pose_confidence_path.exists():
        try:
            pose_confidence = np.load(str(pose_confidence_path)).tolist()
        except Exception:
            pose_confidence = None

    frame_selection = {}
    selection_path = base / "frame_selection.json"
    if selection_path.exists():
        try:
            frame_selection = json.loads(selection_path.read_text())
        except Exception:
            frame_selection = {}

    run_params, _ = _load_r3_run_params(base, point_count=len(camera_poses))

    selected_camera_poses, source_selection = select_r3_trajectory_camera_poses(
        base,
        camera_poses,
        trajectory_source,
    )
    bundle = build_r3_trajectory(
        selected_camera_poses,
        pose_confidence,
        frame_selection,
        run_params,
    )
    bundle["trajectory_source_requested"] = source_selection["requested"]
    bundle["trajectory_source"] = source_selection["selected"]
    bundle["trajectory_source_fallback_reason"] = source_selection["fallback_reason"]
    bundle["trajectory_source_selection"] = source_selection
    return bundle, selected_camera_poses


def _clean_r3_trajectory_points(raw_points):
    """Return a display-safe R³ camera path with pose-jump clipping and smoothing."""
    import numpy as np

    if not raw_points or len(raw_points) < 5:
        return raw_points or [], {"quality": "too_short", "raw_points": len(raw_points or []), "cleaned_points": len(raw_points or [])}

    pts = np.array(raw_points, dtype=np.float64)
    finite = np.isfinite(pts).all(axis=1)
    if finite.sum() < 5:
        return raw_points, {"quality": "invalid", "raw_points": len(raw_points), "cleaned_points": int(finite.sum())}

    for i in range(len(pts)):
        if not finite[i]:
            pts[i] = pts[i - 1] if i > 0 else pts[finite][0]

    raw_steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    positive = raw_steps[raw_steps > 1e-9]
    if positive.size == 0:
        return pts.tolist(), {"quality": "static", "raw_points": len(raw_points), "cleaned_points": len(raw_points)}

    med = float(np.median(positive))
    p75 = float(np.percentile(positive, 75))
    p90 = float(np.percentile(positive, 90))
    p99 = float(np.percentile(positive, 99))
    step_limit = max(med * 3.0, p75 * 2.0, 1e-6)

    clipped = [pts[0].copy()]
    clipped_steps = 0
    for i in range(1, len(pts)):
        delta = pts[i] - pts[i - 1]
        dist = float(np.linalg.norm(delta))
        if dist > step_limit:
            delta = delta / max(dist, 1e-9) * step_limit
            clipped_steps += 1
        clipped.append(clipped[-1] + delta)
    pts = np.array(clipped, dtype=np.float64)

    window = min(31, len(pts) if len(pts) % 2 == 1 else len(pts) - 1)
    if window >= 7:
        pad = window // 2
        padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(window, dtype=np.float64) / window
        smoothed = np.empty_like(pts)
        for dim in range(pts.shape[1]):
            smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
        smoothed -= smoothed[0] - pts[0]
        pts = smoothed

    cleaned_steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    quality = "ok"
    if p99 > max(med * 20.0, 1e-6) or clipped_steps > len(raw_steps) * 0.15:
        quality = "unstable_pose"

    return (
        [[round(float(x), 6), round(float(y), 6), round(float(z), 6)] for x, y, z in pts],
        {
            "quality": quality,
            "raw_points": len(raw_points),
            "cleaned_points": len(pts),
            "raw_step_median": med,
            "raw_step_p90": p90,
            "raw_step_p99": p99,
            "step_limit": step_limit,
            "clipped_steps": clipped_steps,
            "cleaned_distance": float(cleaned_steps.sum()) if cleaned_steps.size else 0.0,
        },
    )


def _r3_frame_stats(points, max_histogram_items: int = 320) -> dict:
    import numpy as np

    if points.ndim != 2 or points.shape[1] <= 7 or len(points) == 0:
        return {
            "has_frame_idx": False,
            "frames": 0,
            "frame_min": None,
            "frame_max": None,
            "frame_histogram": {},
        }
    frames = points[:, 7]
    frames = frames[np.isfinite(frames)].astype(np.int64)
    if frames.size == 0:
        return {
            "has_frame_idx": True,
            "frames": 0,
            "frame_min": None,
            "frame_max": None,
            "frame_histogram": {},
        }
    unique, counts = np.unique(frames, return_counts=True)
    if len(unique) <= max_histogram_items:
        hist = {str(int(f)): int(c) for f, c in zip(unique, counts)}
    else:
        first = list(zip(unique[:20], counts[:20]))
        last = list(zip(unique[-20:], counts[-20:]))
        hist = {str(int(f)): int(c) for f, c in first + last}
    return {
        "has_frame_idx": True,
        "frames": int(len(unique)),
        "frame_min": int(unique.min()),
        "frame_max": int(unique.max()),
        "points_min_per_frame": int(counts.min()),
        "points_max_per_frame": int(counts.max()),
        "points_mean_per_frame": float(counts.mean()),
        "frame_histogram": hist,
    }


def _r3_cloud_mean_by_frame(points, frame_ids=(0, 1, 2, 10, 50, 100, 200, 299)) -> dict:
    import numpy as np

    out = {}
    if points.ndim != 2 or points.shape[1] <= 7 or len(points) == 0:
        return out
    frames = points[:, 7]
    for frame_id in frame_ids:
        mask = np.isfinite(frames) & (frames.astype(np.int64) == int(frame_id))
        pts = points[mask]
        if len(pts) == 0:
            out[str(frame_id)] = {"points": 0}
            continue
        out[str(frame_id)] = {
            "points": int(len(pts)),
            "xyz_mean": np.round(np.nanmean(pts[:, :3], axis=0), 5).tolist(),
            "xyz_std": np.round(np.nanstd(pts[:, :3], axis=0), 5).tolist(),
            "xyz_min": np.round(np.nanmin(pts[:, :3], axis=0), 5).tolist(),
            "xyz_max": np.round(np.nanmax(pts[:, :3], axis=0), 5).tolist(),
        }
    return out


def _r3_camera_translation_samples(base: Path, frame_ids=(0, 1, 2, 10, 50, 100, 200, 299)) -> dict:
    import numpy as np

    out = {}
    camera_dir = base / "camera"
    for frame_id in frame_ids:
        cf = camera_dir / f"{int(frame_id):06d}.npz"
        if not cf.exists():
            out[str(frame_id)] = {"exists": False}
            continue
        try:
            cam = np.load(str(cf))
            pose = cam["pose"]
            out[str(frame_id)] = {
                "exists": True,
                "pose_shape": list(pose.shape),
                "translation": np.round(pose[:3, 3], 5).tolist(),
                "det_R": float(np.linalg.det(pose[:3, :3])),
            }
        except Exception as e:
            out[str(frame_id)] = {"exists": True, "error": str(e)}
    return out


def _render_r3_projection_png(points, out_path: Path, axes=(0, 1), max_points: int = 250000) -> Optional[str]:
    import numpy as np
    import cv2

    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        return None

    sample = points
    if len(sample) > max_points:
        if sample.shape[1] > 6:
            idx = np.argsort(sample[:, 6])[-max_points:]
        else:
            idx = np.linspace(0, len(sample) - 1, max_points).astype(np.int64)
        sample = sample[idx]

    xy = sample[:, list(axes)]
    finite = np.isfinite(xy).all(axis=1)
    sample = sample[finite]
    xy = xy[finite]
    if len(sample) == 0:
        return None

    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    norm = np.clip((xy - lo) / span, 0, 1)

    width, height, pad = 1400, 1000, 36
    px = (pad + norm[:, 0] * (width - pad * 2)).astype(np.int32)
    py = (height - pad - norm[:, 1] * (height - pad * 2)).astype(np.int32)

    img = np.full((height, width, 3), 255, dtype=np.uint8)
    if sample.shape[1] >= 6:
        rgb = sample[:, 3:6]
        if np.nanmax(rgb) <= 1.0:
            rgb = rgb * 255.0
        colors = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        colors = np.zeros((len(sample), 3), dtype=np.uint8)
    img[py, px] = colors
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return str(out_path)


def _r3_projection_debug(
    video_id: str,
    max_points: int = 250000,
    min_conf: float = 1.4,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    sampling_strategy: str = "per_frame_uniform",
) -> dict:
    import numpy as np

    points, npz_path = _load_r3_pointcloud(video_id, prefer_debug=True)
    filtered = _filter_r3_points(points, min_conf=min_conf, frame_start=frame_start, frame_end=frame_end)
    sampled = _sample_points(filtered, max_points=max_points, strategy=sampling_strategy)
    base = _r3_output_dir(video_id)
    diag_dir = base / "diagnostics"
    suffix = f"conf{str(min_conf).replace('.', '_')}_f{frame_start if frame_start is not None else 'all'}_{frame_end if frame_end is not None else 'all'}_{sampling_strategy}_{max_points}"
    projections = {
        "top_x_y": _render_r3_projection_png(sampled, diag_dir / f"projection_top_x_y_{suffix}.png", axes=(0, 1), max_points=max_points),
        "front_x_z": _render_r3_projection_png(sampled, diag_dir / f"projection_front_x_z_{suffix}.png", axes=(0, 2), max_points=max_points),
        "right_y_z": _render_r3_projection_png(sampled, diag_dir / f"projection_right_y_z_{suffix}.png", axes=(1, 2), max_points=max_points),
    }
    return {
        "success": True,
        "video_id": video_id,
        "pointcloud_file": npz_path.name,
        "source_points": int(len(points)),
        "filtered_points": int(len(filtered)),
        "projected_points": int(len(sampled)),
        "min_conf": min_conf,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "sampling_strategy": sampling_strategy,
        "projections": projections,
    }


def _inspect_camera_npz(camera_file: Path) -> dict:
    import numpy as np

    d = np.load(str(camera_file))
    info = {"file": camera_file.name, "keys": list(d.files)}
    pose = d["pose"] if "pose" in d else None
    if pose is not None:
        R = pose[:3, :3]
        t = pose[:3, 3] if pose.shape[1] >= 4 else np.zeros(3)
        info.update({
            "pose_shape": list(pose.shape),
            "det_R": float(np.linalg.det(R)),
            "translation_norm": float(np.linalg.norm(t)),
            "candidate_convention": "c2w_saved_by_r3_infer",
            "pose": pose.tolist(),
        })
    if "intrinsics" in d:
        K = d["intrinsics"]
        info["intrinsics_shape"] = list(K.shape)
        info["intrinsics"] = K.tolist()
    return info


def _r3_run_diagnostics(video_id: str) -> dict:
    import numpy as np

    base = _r3_output_dir(video_id)
    if not base.exists():
        return {"success": False, "video_id": video_id, "output_exists": False}

    files = {}
    for name, pattern in {
        "depth_count": "depth/*.npy",
        "conf_count": "conf/*.npy",
        "color_count": "color/*.png",
        "camera_count": "camera/*.npz",
    }.items():
        files[name] = len(list(base.glob(pattern)))

    pointcloud = {"exists": False}
    conf_stats = {}
    npz_path = _r3_find_pointcloud_path(base, prefer_debug=True)
    if npz_path.exists():
        pts = np.load(str(npz_path))["points"]
        pointcloud = {
            "exists": True,
            "file": npz_path.name,
            "shape": list(pts.shape),
            "has_conf": bool(pts.ndim == 2 and pts.shape[1] > 6),
            "has_frame_idx": bool(pts.ndim == 2 and pts.shape[1] > 7),
        }
        pointcloud["frame_stats"] = _r3_frame_stats(pts)
        if pts.ndim == 2 and pts.shape[1] >= 3 and len(pts) > 0:
            pointcloud.update({
                "xyz_min": np.nanmin(pts[:, :3], axis=0).tolist(),
                "xyz_max": np.nanmax(pts[:, :3], axis=0).tolist(),
                "xyz_mean": np.nanmean(pts[:, :3], axis=0).tolist(),
                "xyz_std": np.nanstd(pts[:, :3], axis=0).tolist(),
            })
        if pts.ndim == 2 and pts.shape[1] >= 6 and len(pts) > 0:
            pointcloud.update({
                "rgb_min": np.nanmin(pts[:, 3:6], axis=0).tolist(),
                "rgb_max": np.nanmax(pts[:, 3:6], axis=0).tolist(),
            })
        if pts.ndim == 2 and pts.shape[1] > 6:
            conf_stats = _r3_conf_stats(pts[:, 6])
        if pts.ndim == 2 and pts.shape[1] > 7:
            pointcloud["cloud_mean_by_frame_sample"] = _r3_cloud_mean_by_frame(pts)

    camera_sample = []
    camera_files = sorted((base / "camera").glob("*.npz"))
    run_params, fallback_summary = _load_r3_run_params(base, point_count=len(camera_files))
    for cf in (camera_files[:3] + camera_files[-3:]):
        try:
            camera_sample.append(_inspect_camera_npz(cf))
        except Exception as e:
            camera_sample.append({"file": cf.name, "error": str(e)})

    pose_confidence = {"exists": False}
    pose_confidence_path = base / "pose_conf.npy"
    if pose_confidence_path.exists():
        try:
            values = np.load(str(pose_confidence_path))
            finite_values = values[np.isfinite(values)]
            pose_confidence = {
                "exists": True,
                "count": int(values.size),
                "finite_count": int(finite_values.size),
                "percentiles": _r3_conf_stats(finite_values).get("percentiles", {}),
            }
        except Exception as e:
            pose_confidence = {"exists": True, "error": str(e)}

    pose_edges = {"exists": False}
    pose_edges_path = base / "pose_edge_log.json"
    if pose_edges_path.exists():
        try:
            edges = json.loads(pose_edges_path.read_text())
            type_counts: Dict[str, int] = {}
            for edge in edges if isinstance(edges, list) else []:
                edge_type = str(edge.get("edge_type", "unknown")) if isinstance(edge, dict) else "unknown"
                type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
            pose_edges = {
                "exists": True,
                "count": len(edges) if isinstance(edges, list) else 0,
                "type_counts": type_counts,
                "fallback_summary": fallback_summary,
            }
        except Exception as e:
            pose_edges = {"exists": True, "error": str(e)}

    trajectory = {"quality": "unavailable"}
    try:
        trajectory_bundle, _ = _load_r3_trajectory_bundle(base)
        trajectory = {
            "points": len(trajectory_bundle.get("plan_trajectory", [])),
            "turns": len(trajectory_bundle.get("turn_points", [])),
            "quality": trajectory_bundle.get("trajectory_quality", {}),
            "source_frame_indices": trajectory_bundle.get("source_frame_indices", []),
            "source_timestamps_seconds": trajectory_bundle.get("source_timestamps_seconds", []),
        }
    except Exception as e:
        trajectory = {"quality": "error", "error": str(e)}

    pose_graph = load_pose_graph_summary(
        base / "pose_graph_edges.npz",
        point_count=len(list((base / "camera").glob("*.npz"))),
    )
    pose_graph_candidate = load_pose_graph_candidate_summary(base)

    return {
        "success": True,
        "video_id": video_id,
        "output_exists": True,
        "output_dir": str(base),
        "files": files,
        "pointcloud": pointcloud,
        "conf_stats": conf_stats,
        "run_params": run_params,
        "fallback_summary": fallback_summary,
        "camera_sample": camera_sample,
        "camera_translation_sample": _r3_camera_translation_samples(base),
        "pose_confidence": pose_confidence,
        "pose_edges": pose_edges,
        "pose_graph": pose_graph,
        "pose_graph_candidate": pose_graph_candidate,
        "trajectory": trajectory,
    }


def _backproject_depth_pointcloud(output_dir: Path, stride: int = 2, max_points: int = 100000) -> list:
    """Read depth maps and camera poses, back-project to 3D world points.

    Keep parity with R3/view.py:
      pts_cam = depth_to_cam_coords_points(depth, intrinsics)
      pts_world = pts_cam @ pose[:3, :3].T + pose[:3, 3]

    Returns a list of [x, y, z, r, g, b, conf] points (downsampled).
    """
    import numpy as np
    import cv2

    camera_dir = output_dir / "camera"
    depth_dir = output_dir / "depth"
    conf_dir = output_dir / "conf"
    frames_dir = output_dir / "frames"
    color_dir = output_dir / "color"

    if not camera_dir.exists() or not depth_dir.exists():
        return []

    camera_files = sorted(camera_dir.glob("*.npz"))
    if len(camera_files) < 2:
        return []

    all_pts = []

    for cf in camera_files:
        try:
            cam_data = np.load(str(cf), allow_pickle=True)
            pose = cam_data["pose"]
            intrinsics = cam_data.get("intrinsics", None)

            depth_path = depth_dir / f"{cf.stem}.npy"
            if not depth_path.exists():
                continue
            depth = np.load(str(depth_path), allow_pickle=True)
            H, W = depth.shape

            rgb = None
            frame_path = color_dir / f"{cf.stem}.png"
            if not frame_path.exists():
                frame_path = frames_dir / f"frame_{int(cf.stem):06d}.jpg"
            if frame_path.exists():
                bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if rgb.shape[:2] != (H, W):
                        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

            fx = float(intrinsics[0, 0]) if intrinsics is not None else float(max(H, W)) * 1.2
            fy = float(intrinsics[1, 1]) if intrinsics is not None else fx
            cx = float(intrinsics[0, 2]) if intrinsics is not None else W / 2.0
            cy = float(intrinsics[1, 2]) if intrinsics is not None else H / 2.0

            if pose.shape == (3, 4):
                pose_mat = np.eye(4)
                pose_mat[:3, :] = pose
            else:
                pose_mat = pose

            # Load confidence
            conf = None
            conf_path = conf_dir / f"{cf.stem}.npy"
            if conf_path.exists():
                conf = np.load(str(conf_path))

            # Pixel grid with stride
            ys, xs = np.meshgrid(
                np.arange(0, H, stride), np.arange(0, W, stride), indexing='ij',
            )
            ys = ys.ravel()
            xs = xs.ravel()
            depth_vals = depth[ys, xs]

            # Filter invalid depths
            valid = np.isfinite(depth_vals) & (depth_vals > 0.001)

            conf_vals = None
            if conf is not None:
                conf_vals = conf[ys, xs]
                valid = valid & np.isfinite(conf_vals) & (conf_vals > 1.0)

            if not valid.any():
                continue

            xs_v = xs[valid]
            ys_v = ys[valid]
            z_vals = depth_vals[valid]
            conf_out = conf_vals[valid].astype(np.float32) if conf_vals is not None else np.full(z_vals.shape, 2.0, dtype=np.float32)
            frame_out = np.full(z_vals.shape, int(cf.stem), dtype=np.float32)

            # Back-project: pixel → camera → world
            x_cam = (xs_v - cx) * z_vals / fx
            y_cam = (ys_v - cy) * z_vals / fy
            ones = np.ones_like(x_cam)
            cam_pts = np.stack([x_cam, y_cam, z_vals, ones], axis=-1)
            world_pts = (pose_mat @ cam_pts.T).T[:, :3]
            if rgb is not None:
                rgb_vals = rgb[ys_v, xs_v].astype(np.float32) / 255.0
            else:
                rgb_vals = np.full((world_pts.shape[0], 3), 0.75, dtype=np.float32)

            all_pts.append(np.concatenate([world_pts, rgb_vals, conf_out[:, None], frame_out[:, None]], axis=1))

        except Exception:
            continue

    if not all_pts:
        return []
    combined = np.concatenate(all_pts, axis=0)
    xyz = combined[:, :3]
    rgb = combined[:, 3:6]
    conf = combined[:, 6]
    valid_combined = np.isfinite(xyz).all(axis=1) & np.isfinite(rgb).all(axis=1) & np.isfinite(conf)
    combined = combined[valid_combined]

    if combined.shape[0] < 100:
        return []

    # Save for future use
    try:
        full_npz_path = output_dir / "pointcloud_full_debug.npz"
        np.savez_compressed(str(full_npz_path), points=combined)
        if combined.shape[1] > 6:
            diag_dir = output_dir / "diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            stats = _r3_conf_stats(combined[:, 6])
            stats["total_points_before_filter"] = int(combined.shape[0])
            (diag_dir / "conf_stats.json").write_text(json.dumps(_to_json_serializable(stats), indent=2))
    except Exception:
        pass

    # ── Downsample production cloud ────────────────────────────────────
    production = combined
    if production.shape[0] > max_points:
        idx = np.random.RandomState(42).choice(production.shape[0], max_points, replace=False)
        production = production[idx]

    # Save production cloud for future use
    try:
        npz_path = output_dir / "pointcloud.npz"
        np.savez_compressed(str(npz_path), points=production[:, :7])
    except Exception:
        pass

    return production[:, :7].tolist()


@app.post("/api/r3-process-video")
async def r3_process_video(
    video_file: UploadFile = File(...),
    video_id: str = Form(...),
    frame_stride: int = Form(5),
    max_frames: int = Form(1500),
    ckpt: str = Form("r3_long.safetensors"),
    size: int = Form(392),
    mode: str = Form("strided"),
):
    """Process video with R³ (Depth Anything 3 Reconstruction).
    
    Returns camera poses, depth maps metadata, and trajectories.
    R³ runs in its own conda environment (r3) via subprocess.
    """
    start_ts = time.time()
    logger.info(f"[{video_id}] R³ processing started: {video_file.filename}")

    # Save uploaded video
    ext = Path(video_file.filename or "video.mp4").suffix
    local_video = WORK_DIR / f"r3_in_{video_id}{ext}"
    content = await video_file.read()
    with open(local_video, "wb") as f:
        f.write(content)
    logger.info(f"[{video_id}] Saved {len(content)} bytes to {local_video}")

    # Create output dir
    video_output_dir = R3_OUTPUT_DIR / video_id
    _reset_r3_output_dir(video_output_dir)

    try:
        # Run R³ wrapper via subprocess
        logger.info(f"[{video_id}] Starting R³ inference (frame_stride={frame_stride}, ckpt={ckpt})")

        conda_cmd = [
            "/home/artem/miniconda3/bin/conda", "run", "-n", "r3", "--cwd", str(WORK_DIR.parent / "R3"),
            "python3", str(R3_WRAPPER),
            "--video_path", str(local_video),
            "--output_dir", str(video_output_dir),
            "--frame_stride", str(frame_stride),
            "--max_frames", str(max_frames),
            "--ckpt", ckpt,
            "--size", str(size),
            "--mode", mode,
        ]

        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        result = subprocess.run(
            conda_cmd,
            capture_output=True, text=True,
            timeout=14400,  # 4 hours max
            env=env,
        )

        if result.returncode != 0:
            error_src = (result.stderr or result.stdout or "").strip()
            error_msg = error_src[-6000:]
            logger.error(f"[{video_id}] R³ failed: {error_msg}")
            raise Exception(f"R³ error: {error_msg}")

        # Parse JSON output (last line with "complete" event)
        output_lines = result.stdout.strip().split("\n")
        final_result = None
        for line in reversed(output_lines):
            try:
                data = json.loads(line)
                if data.get("event") == "complete":
                    # New format: {"event":"complete","data":{"result":...}}
                    # Old format: {"event":"complete","result":...}
                    final_result = data.get("data", {}).get("result") or data.get("result")
                    break
            except json.JSONDecodeError:
                continue

        if not final_result:
            raise Exception("No R³ result found in output")

        final_result["pointcloud_status"] = _schedule_r3_pointcloud_build(
            video_id, video_output_dir,
        )
        elapsed = time.time() - start_ts
        logger.info(f"[{video_id}] R³ completed in {elapsed:.1f}s, {final_result.get('num_frames', 0)} frames")

        return _sanitize_for_json({
            "success": True,
            "video_id": video_id,
            "processing_time": round(elapsed, 1),
            "result": final_result,
        })

    except subprocess.TimeoutExpired:
        logger.error(f"[{video_id}] R³ timed out after 4 hours")
        return JSONResponse(status_code=504, content={
            "success": False, "video_id": video_id, "error": "R³ inference timed out"
        })
    except Exception as e:
        elapsed = time.time() - start_ts
        logger.error(f"[{video_id}] R³ failed after {elapsed:.1f}s: {e}")
        return JSONResponse(status_code=500, content={
            "success": False, "video_id": video_id, "error": str(e)
        })
    finally:
        if local_video.exists():
            local_video.unlink()


@app.post("/api/r3-process-video-raw/{video_id}")
async def r3_process_video_raw(
    video_id: str,
    request: Request,
    original_filename: str = "video.mp4",
    frame_stride: int = 5,
    max_frames: int = 1500,
    ckpt: str = "r3_long.safetensors",
    size: int = 392,
    mode: str = "strided",
    use_uploaded: bool = False,
):
    """Raw streaming R³ endpoint — принимает байты видео напрямую (как SLAM raw endpoint).

    If use_uploaded=true, reads from the pre-uploaded file in uploaded/ directory
    instead of reading from request body.
    """
    start_ts = time.time()
    logger.info(f"[{video_id}] R³ raw streaming started ({original_filename}, use_uploaded={use_uploaded})")
    if not _r3_try_mark_busy(video_id):
        return JSONResponse(status_code=409, content={
            "success": False,
            "video_id": video_id,
            "error": "R³ processing is already running for this video_id",
        })

    ext = Path(original_filename).suffix or ".mp4"

    if use_uploaded:
        # ─── Видео уже на сервере — используем предзагруженный файл ───
        UPLOAD_DIR = WORK_DIR / "uploaded"
        uploaded_path = UPLOAD_DIR / f"{video_id}{ext}"
        if not uploaded_path.exists():
            raise HTTPException(status_code=404, detail=f"Uploaded video {video_id} not found on GPU worker")
        local_video = uploaded_path
        logger.info(f"[{video_id}] Using pre-uploaded file: {local_video} ({uploaded_path.stat().st_size} bytes)")
        # Drain request body silently
        async for _ in request.stream():
            pass
    else:
        # ─── Получаем видео из тела запроса ───
        local_video = WORK_DIR / f"r3_in_{video_id}{ext}"
        with open(local_video, "wb") as f:
            async for chunk in request.stream():
                if not chunk:
                    break
                f.write(chunk)
        file_size = local_video.stat().st_size
        logger.info(f"[{video_id}] Saved {file_size} bytes from raw stream to {local_video}")

    video_output_dir = R3_OUTPUT_DIR / video_id
    _reset_r3_output_dir(video_output_dir)

    try:
        logger.info(f"[{video_id}] Starting R³ inference (frame_stride={frame_stride}, ckpt={ckpt})")

        conda_cmd = [
            "/home/artem/miniconda3/bin/conda", "run", "-n", "r3", "--cwd", str(Path(__file__).resolve().parent.parent / "R3"),
            "python3", str(R3_WRAPPER),
            "--video_path", str(local_video),
            "--output_dir", str(video_output_dir),
            "--frame_stride", str(frame_stride),
            "--max_frames", str(max_frames),
            "--ckpt", ckpt,
            "--size", str(size),
            "--mode", mode,
        ]

        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        result = subprocess.run(
            conda_cmd, capture_output=True, text=True,
            timeout=14400, env=env,
        )

        if result.returncode != 0:
            error_src = (result.stderr or result.stdout or "").strip()
            error_msg = error_src[-6000:]
            raise Exception(f"R³ error: {error_msg}")

        output_lines = result.stdout.strip().split("\n")
        final_result = None
        for line in reversed(output_lines):
            try:
                data = json.loads(line)
                if data.get("event") == "complete":
                    # New format: {"event":"complete","data":{"result":...}}
                    # Old format: {"event":"complete","result":...}
                    final_result = data.get("data", {}).get("result") or data.get("result")
                    break
            except json.JSONDecodeError:
                continue

        if not final_result:
            raise Exception("No R³ result found in output")

        final_result["pointcloud_status"] = _schedule_r3_pointcloud_build(
            video_id, video_output_dir,
        )
        elapsed = time.time() - start_ts
        logger.info(f"[{video_id}] R³ completed in {elapsed:.1f}s, {final_result.get('num_frames', 0)} frames")

        return _sanitize_for_json({
            "success": True,
            "video_id": video_id,
            "processing_time": round(elapsed, 1),
            "result": final_result,
        })

    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=504, content={
            "success": False, "video_id": video_id, "error": "R³ inference timed out"
        })
    except Exception as e:
        elapsed = time.time() - start_ts
        logger.error(f"[{video_id}] R³ raw failed after {elapsed:.1f}s: {e}")
        return JSONResponse(status_code=500, content={
            "success": False, "video_id": video_id, "error": str(e)
        })
    finally:
        _r3_clear_busy(video_id)
        if local_video.exists():
            local_video.unlink()


# ──────────────────────────────────────────────
# R³ SSE Streaming endpoint — real-time trajectory building
# ──────────────────────────────────────────────

# Global tracking of active R³ processes
_r3_active_processes: Dict[str, subprocess.Popen] = {}
_r3_active_lock = threading.Lock()
_r3_busy_video_ids: Set[str] = set()


def _r3_try_mark_busy(video_id: str) -> bool:
    with _r3_active_lock:
        if video_id in _r3_busy_video_ids:
            return False
        _r3_busy_video_ids.add(video_id)
        return True


def _r3_clear_busy(video_id: str) -> None:
    with _r3_active_lock:
        _r3_busy_video_ids.discard(video_id)


def _reset_r3_output_dir(output_dir: Path) -> None:
    """Remove stale R³ artifacts before a fresh inference for the same video_id."""
    _cancel_r3_pointcloud_job(output_dir.name)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _r3_run_matches(output_dir: Path, max_frames: int, ckpt: str, mode: str) -> bool:
    """Return True only when existing artifacts were generated with compatible R³ params."""
    params_path = output_dir / "run_params.json"
    if not params_path.exists():
        return False
    try:
        params = json.loads(params_path.read_text())
        selection_path = output_dir / "frame_selection.json"
        if not selection_path.exists():
            # Older runs may only cover the first minutes of long AVI files.
            return False
        selection = json.loads(selection_path.read_text())

        saved_ckpt = str(params.get("ckpt") or "")
        saved_mode = str(params.get("mode") or "").lower()
        requested_mode = str(mode or "").lower()
        mode_l = "long" if selection.get("continuous_long") else requested_mode
        release_preset = (os.getenv("R3_USE_RELEASE_PRESET") or "true").lower() in {
            "1", "true", "yes", "on",
        }
        expected_pose_method = (
            "greedy"
            if release_preset
            else (os.getenv("R3_REL_POSE_METHOD") or "greedy").strip().lower()
        )
        scale_policy = (os.getenv("R3_SCALE_POLICY") or "bridge_continuity").strip().lower()
        expected_metric_scale = scale_policy == "metric_reanchor"
        expected_bridge_ratio = (
            0.35
            if release_preset
            else float(os.getenv("R3_FALLBACK_MIN_BRIDGE_BASELINE_RATIO") or "0.35")
        )
        expected_bridge_lookback = (
            40
            if release_preset
            else int(float(os.getenv("R3_FALLBACK_MAX_BRIDGE_LOOKBACK") or "40"))
        )
        saved_pose_method = str(params.get("rel_pose_reconstruction_method") or "greedy").lower()
        saved_metric_scale = bool(params.get("metric_scale_enabled"))
        saved_bridge_ratio = float(params.get("fallback_min_bridge_baseline_ratio") or 0.0)
        saved_bridge_lookback = int(params.get("fallback_max_bridge_lookback") or 0)
        expected_max_frames = 0 if selection.get("long_video_sampling") else int(max_frames)
        basic_match = (
            int(params.get("max_frames") or 0) == expected_max_frames
            and saved_ckpt.endswith(str(ckpt))
            and saved_mode == mode_l
            and saved_pose_method == expected_pose_method
            and saved_metric_scale == expected_metric_scale
            and math.isclose(saved_bridge_ratio, expected_bridge_ratio, rel_tol=0.0, abs_tol=1e-9)
            and saved_bridge_lookback == expected_bridge_lookback
        )
        if not basic_match:
            return False

        total_frames = int(selection.get("total_frames") or 0)
        source_frame_max = selection.get("source_frame_max")
        saved_frames = int(selection.get("saved_frames") or 0)
        if saved_frames <= 0 or source_frame_max is None:
            return False
        if total_frames <= 0:
            return True

        coverage = float(source_frame_max) / max(float(total_frames - 1), 1.0)
        return coverage >= 0.95
    except Exception:
        return False


def _start_r3_wrapper(
    video_id: str,
    video_path: Path,
    output_dir: Path,
    frame_stride: int = 5,
    max_frames: int = 1500,
    ckpt: str = "r3_long.safetensors",
    size: int = 392,
    mode: str = "strided",
    live: bool = False,
) -> subprocess.Popen:
    """Start r3_worker_wrapper.py as subprocess and return Popen handle."""
    conda_cmd = [
        "/home/artem/miniconda3/bin/conda", "run", "-n", "r3", "--cwd", str(Path(__file__).resolve().parent.parent / "R3"),
        "python3", str(R3_WRAPPER),
        "--video_path", str(video_path),
        "--output_dir", str(output_dir),
        "--frame_stride", str(frame_stride),
        "--max_frames", str(max_frames),
        "--ckpt", ckpt,
        "--size", str(size),
        "--mode", mode,
    ]
    if live:
        conda_cmd.append("--live")

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    proc = subprocess.Popen(
        conda_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, text=True, bufsize=1,
    )

    with _r3_active_lock:
        _r3_active_processes[video_id] = proc

    logger.info(f"[{video_id}] R³ wrapper started (PID={proc.pid}, live={live})")
    return proc


@app.post("/api/r3-process-stream/{video_id}")
async def r3_process_stream(
    video_id: str,
    request: Request,
    original_filename: str = "video.mp4",
    frame_stride: int = 5,
    max_frames: int = 1500,
    ckpt: str = "r3_long.safetensors",
    size: int = 392,
    mode: str = "strided",
    replay: bool = Query(False),  # replay=1 → не загружать видео, вернуть существующие результаты
):
    """SSE endpoint — стримит события сразу, видео сохраняет внутри генератора.
    
    SSE события:
      - event: connected         (соединение установлено)
      - event: receiving         (идёт приём видео)
      - event: frame_processed   (новый кадр обработан)
      - event: complete          (вся обработка завершена)
      - event: error             (ошибка)
    """
    start_ts = time.time()
    logger.info(f"[{video_id}] R³ SSE streaming started ({original_filename})")

    video_output_dir = R3_OUTPUT_DIR / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)
    camera_dir = video_output_dir / "camera"

    # Проверяем, есть ли уже готовые .npz файлы (R³ уже выполнен)
    existing_npz = sorted(camera_dir.glob("*.npz")) if camera_dir.exists() else []
    compatible_existing_run = len(existing_npz) >= 2 and _r3_run_matches(video_output_dir, max_frames, ckpt, mode)
    REPLAY_MODE = replay or compatible_existing_run

    if replay and not REPLAY_MODE:
        # Явно запрошен replay, но данных нет → 404, прокси перезапросит с загрузкой
        logger.info(f"[{video_id}] Replay requested but no .npz found")
        return JSONResponse(status_code=404, content={
            "success": False, "error": "No R³ data available for replay"
        })

    if REPLAY_MODE:
        logger.info(f"[{video_id}] R³ output already exists ({len(existing_npz)} .npz files) — replaying")

    ext = Path(original_filename).suffix or ".mp4"
    SSE_VIDEO_DIR = WORK_DIR / "sse"
    SSE_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    local_video = SSE_VIDEO_DIR / f"sse_in_{video_id}{ext}"

    async def event_generator():
        nonlocal local_video, video_output_dir, camera_dir, existing_npz
        proc = None
        marked_busy = False
        try:
            # 1. Emit connected event immediately
            yield f"event: connected\ndata: {json.dumps({'video_id': video_id, 'status': 'receiving_video'})}\n\n"

            if REPLAY_MODE:
                # ─── Replay mode: пропускаем загрузку, эмитим точки из .npz ───
                yield f"event: replay\ndata: {json.dumps({'mode': 'replay', 'npz_files': len(existing_npz)})}\n\n"

                import numpy as _np
                num_total = len(existing_npz)
                num_processed = 0

                for cf in existing_npz:
                    if await request.is_disconnected():
                        break
                    try:
                        data = _np.load(str(cf))
                        pose = data["pose"].tolist() if "pose" in data else None
                        if pose and len(pose) >= 3 and len(pose[0]) >= 4:
                            traj_point = [pose[0][3], pose[1][3], pose[2][3]]
                            num_processed += 1
                            payload = _sanitize_for_json({
                                'num_processed': num_processed,
                                'num_total': num_total,
                                'new_trajectory_points': [traj_point],
                            })
                            yield "event: frame_processed\ndata: " + json.dumps(payload) + "\n\n"
                    except Exception:
                        pass

                # ─── Now generate point cloud from depth/*.npy ───────────
                elapsed = time.time() - start_ts
                complete_payload = {
                    'total_time_s': round(elapsed, 1),
                    'num_frames': num_total,
                    'processing_stats': {
                        'fps': round(num_total / max(elapsed, 0.1), 1),
                        'estimated_distance': 0,
                        'turns_detected': 0,
                    },
                }

                # Check for existing pointcloud.npz or depth/*.npy
                depth_dir = video_output_dir / "depth"
                pointcloud_npz = video_output_dir / "pointcloud.npz"

                if pointcloud_npz.exists():
                    # Already generated by previous run
                    try:
                        pc_data = _np.load(str(pointcloud_npz), allow_pickle=True)
                        pc_points = pc_data["points"]
                        # Only include a tiny sample in SSE (full cloud loaded via separate API)
                        sample = pc_points[_np.random.RandomState(42).choice(len(pc_points), min(2000, len(pc_points)), replace=False)].tolist() if len(pc_points) > 2000 else pc_points.tolist()
                        complete_payload["pointcloud_sample"] = sample
                        complete_payload["pointcloud_count"] = int(len(pc_points))
                        logger.info(f"[{video_id}] Loaded existing pointcloud.npz ({len(pc_points)} pts, sending 2000 sample)")
                    except Exception as e:
                        logger.warning(f"[{video_id}] Failed to load pointcloud.npz: {e}")
                elif depth_dir.exists() and list(depth_dir.glob("*.npy")):
                    cloud_status = _schedule_r3_pointcloud_build(video_id, video_output_dir)
                    complete_payload["pointcloud_status"] = cloud_status
                    yield f"event: pointcloud_status\ndata: {json.dumps(_sanitize_for_json(cloud_status))}\n\n"

                yield "event: complete\ndata: " + json.dumps(complete_payload) + "\n\n"
                # Keepalive — не закрываем соединение
                while True:
                    if await request.is_disconnected():
                        break
                    await asyncio.sleep(30)
                    yield ": keepalive\n\n"
                return
            # ─── End of replay mode ───

            if not _r3_try_mark_busy(video_id):
                yield "event: error\ndata: " + json.dumps({
                    "message": "R³ processing is already running for this video_id",
                    "video_id": video_id,
                }) + "\n\n"
                return
            marked_busy = True

            # 2. Save video while yielding progress
            total_bytes = 0
            last_progress = time.time()
            with open(local_video, "wb") as f:
                async for chunk in request.stream():
                    if not chunk:
                        break
                    f.write(chunk)
                    total_bytes += len(chunk)
                    # Yield progress every 2 seconds
                    now = time.time()
                    if now - last_progress >= 2.0:
                        last_progress = now
                        yield f"event: receiving\ndata: {json.dumps({'received_bytes': total_bytes})}\n\n"
                        # Check if client disconnected during upload
                        if await request.is_disconnected():
                            logger.info(f"[{video_id}] Client disconnected during upload")
                            return

            file_size = local_video.stat().st_size
            logger.info(f"[{video_id}] Saved {file_size} bytes for SSE streaming")

            yield f"event: video_received\ndata: {json.dumps({'bytes': file_size, 'file': local_video.name})}\n\n"

            # 3. Start R³ wrapper with --live
            _reset_r3_output_dir(video_output_dir)
            camera_dir = video_output_dir / "camera"
            proc = _start_r3_wrapper(
                video_id, local_video, video_output_dir,
                frame_stride, max_frames, ckpt, size, mode, live=True,
            )

            yield f"event: processing_started\ndata: {json.dumps({'video': str(local_video), 'live': True})}\n\n"

            # 4. Read stdout line by line and send as SSE
            for line in iter(proc.stdout.readline, ''):
                if not line.strip():
                    continue

                if await request.is_disconnected():
                    logger.info(f"[{video_id}] SSE client disconnected")
                    break

                try:
                    event_data = json.loads(line)
                    event_type = event_data.get("event", "progress")

                    if event_type == "frame_processed":
                        d = event_data.get("data", {})
                        d = _sanitize_for_json(d)
                        yield f"event: frame_processed\ndata: {json.dumps(d)}\n\n"
                    elif event_type == "video_info":
                        yield f"event: video_info\ndata: {json.dumps(event_data.get('data', {}))}\n\n"
                    elif event_type == "frames_extracted":
                        yield f"event: frames_extracted\ndata: {json.dumps(event_data.get('data', {}))}\n\n"
                    elif event_type == "r3_start":
                        yield f"event: r3_start\ndata: {json.dumps(event_data.get('data', {}))}\n\n"
                    elif event_type == "complete":
                        final_result = event_data.get("data", {}).get("result", {})
                        final_result = _sanitize_for_json(final_result)
                        elapsed = time.time() - start_ts
                        final_result["total_time_s"] = round(elapsed, 1)
                        cloud_status = _schedule_r3_pointcloud_build(video_id, video_output_dir)
                        final_result["pointcloud_status"] = cloud_status
                        # Strip full point cloud from SSE (too large) and keep only a tiny sample
                        pc_full = final_result.pop("pointcloud", None)
                        if pc_full and isinstance(pc_full, list) and len(pc_full) > 0:
                            import random as _rnd
                            pc_sample = pc_full[:2000] if len(pc_full) <= 2000 else _rnd.Random(42).sample(pc_full, 2000)
                            final_result["pointcloud_sample"] = pc_sample
                            final_result["pointcloud_count"] = len(pc_full)
                        yield f"event: complete\ndata: {json.dumps(final_result)}\n\n"
                        # Keepalive — не закрываем соединение
                        while True:
                            if await request.is_disconnected():
                                break
                            await asyncio.sleep(30)
                            yield ": keepalive\n\n"
                        return
                    elif event_type == "pointcloud_status":
                        # Forward point cloud generation progress to frontend
                        d = event_data.get("data", {})
                        d = _sanitize_for_json(d)
                        yield f"event: pointcloud_status\ndata: {json.dumps(d)}\n\n"
                    elif event_type == "start":
                        yield f"event: processing_started\ndata: {json.dumps(event_data.get('data', {}))}\n\n"
                    else:
                        yield f"event: {event_type}\ndata: {json.dumps(event_data.get('data', {}))}\n\n"

                except json.JSONDecodeError:
                    yield f"event: log\ndata: {json.dumps({'raw': line})}\n\n"

            # Check for errors
            if proc:
                ret = proc.poll()
                if ret is not None and ret != 0:
                    stderr_out = proc.stderr.read() if proc.stderr else ""
                    error_out = stderr_out[:1000] if stderr_out else f"exit code {ret}"
                    yield f"event: error\ndata: {json.dumps({'message': f'R³ failed: {error_out}'})}\n\n"

        except asyncio.CancelledError:
            logger.warning(f"[{video_id}] SSE client disconnected (cancelled)")
            yield f"event: disconnected\ndata: {json.dumps({'message': 'client disconnected'})}\n\n"
        except (ConnectionError, BrokenPipeError, OSError) as e:
            logger.warning(f"[{video_id}] SSE connection error ({type(e).__name__}): {e}")
            yield f"event: disconnected\ndata: {json.dumps({'message': f'connection error: {e}'})}\n\n"
        except Exception as e:
            logger.error(f"[{video_id}] SSE stream error ({type(e).__name__}): {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            if marked_busy:
                _r3_clear_busy(video_id)
            with _r3_active_lock:
                if video_id in _r3_active_processes:
                    del _r3_active_processes[video_id]
            # Clean up SSE video file
            if local_video and local_video.exists():
                try:
                    local_video.unlink()
                except Exception:
                    pass
            # Also clean any incomplete R³ output
            if video_output_dir and video_output_dir.exists() and not any(video_output_dir.iterdir()):
                try:
                    video_output_dir.rmdir()
                except Exception:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/r3-pointcloud-status/{video_id}")
async def r3_get_pointcloud_status(video_id: str):
    """Return background point-cloud build stage and real progress."""
    output_dir = _r3_output_dir(video_id)
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"No R³ output for {video_id}")
    status = _get_r3_pointcloud_status(video_id, output_dir)
    if status.get("status") == "not_started" and (output_dir / "depth").exists():
        status = _schedule_r3_pointcloud_build(video_id, output_dir)
    return _sanitize_for_json(status)


@app.get("/api/r3-pointcloud/{video_id}")
async def r3_get_pointcloud(
    video_id: str,
    max_points: int = Query(50000),
    min_conf: float = Query(1.0),
):
    """Get the fused point cloud from a completed R³ run."""
    import numpy as np

    try:
        points, npz_path = _load_r3_pointcloud(video_id, prefer_debug=False)
        if points.ndim != 2 or points.shape[1] < 7:
            raise HTTPException(
                status_code=409,
                detail=f"{npz_path.name} has {points.shape[1] if points.ndim == 2 else 'invalid'} columns; regenerate R³ point cloud to include confidence",
            )
        conf = points[:, 6]
        points = points[np.isfinite(conf) & (conf >= min_conf)]
        total = len(points)
        if len(points) > max_points:
            idx = np.random.RandomState(42).choice(len(points), max_points, replace=False)
            points = points[idx]
        return {
            "success": True,
            "video_id": video_id,
            "num_points": len(points),
            "num_points_total": total,
            "points": points.tolist(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load point cloud: {e}")


@app.get("/api/r3-pointcloud-filtered/{video_id}")
async def r3_get_pointcloud_filtered(
    video_id: str,
    max_points: int = Query(100000),
    min_conf: float = Query(1.4),
    frame_start: Optional[int] = Query(None),
    frame_end: Optional[int] = Query(None),
    sampling_strategy: str = Query("random"),
    include_trajectory: bool = Query(True),
    include_cameras: bool = Query(True),
):
    """Get server-side filtered R³ point cloud plus optional trajectory/camera diagnostics."""
    try:
        points, npz_path = _load_r3_pointcloud(video_id, prefer_debug=True)
        source_points = int(len(points))
        filtered = _filter_r3_points(points, min_conf=min_conf, frame_start=frame_start, frame_end=frame_end)
        filtered_points = int(len(filtered))
        sampled = _sample_points(filtered, max_points=max_points, strategy=sampling_strategy)
        source_frame_stats = _r3_frame_stats(points)
        filtered_frame_stats = _r3_frame_stats(filtered)
        returned_frame_stats = _r3_frame_stats(sampled)

        # ``trajectory`` is plan-space for backwards compatibility.  The raw
        # c2w translations are intentionally returned separately so a map can
        # never accidentally treat R3's vertical axis as its Y coordinate.
        trajectory = []
        raw_plan_trajectory = []
        raw_trajectory_3d = []
        turn_points = []
        source_frame_indices = []
        source_timestamps_seconds = []
        cameras = []
        trajectory_quality = None
        base = _r3_output_dir(video_id)
        camera_count = len(list((base / "camera").glob("*.npz")))
        run_params, fallback_summary = _load_r3_run_params(base, point_count=camera_count)
        pose_graph_summary = load_pose_graph_summary(
            base / "pose_graph_edges.npz",
            point_count=camera_count,
        )
        pose_graph_candidate = load_pose_graph_candidate_summary(base)
        run_mode = str(run_params.get("mode") or "").lower()
        # Missing mode means the output was produced by an older wrapper that
        # cannot be trusted for the new strided+fallback+metric R3 preset.
        stale_run = run_mode not in {"long", "strided"}
        if include_trajectory or include_cameras:
            trajectory_bundle, loaded_cameras = _load_r3_trajectory_bundle(base)
            if include_trajectory:
                trajectory = trajectory_bundle["plan_trajectory"]
                raw_plan_trajectory = trajectory_bundle.get("raw_plan_trajectory", trajectory)
                raw_trajectory_3d = trajectory_bundle["raw_trajectory_3d"]
                turn_points = trajectory_bundle["turn_points"]
                source_frame_indices = trajectory_bundle["source_frame_indices"]
                source_timestamps_seconds = trajectory_bundle.get("source_timestamps_seconds", [])
                trajectory_quality = trajectory_bundle["trajectory_quality"]
            if include_cameras:
                cameras = loaded_cameras

        return {
            "success": True,
            "video_id": video_id,
            "points": sampled[:, :8].tolist() if sampled.ndim == 2 and sampled.shape[1] > 7 else sampled[:, :7].tolist(),
            "trajectory": trajectory,
            "plan_trajectory": trajectory,
            "raw_plan_trajectory": raw_plan_trajectory,
            "raw_trajectory_3d": raw_trajectory_3d,
            "turn_points": turn_points,
            "source_frame_indices": source_frame_indices,
            "source_timestamps_seconds": source_timestamps_seconds,
            "cameras": cameras,
            "stats": {
                "source_points": source_points,
                "filtered_points": filtered_points,
                "returned_points": int(len(sampled)),
                "frames_in_source": source_frame_stats.get("frames"),
                "frames_after_filter": filtered_frame_stats.get("frames"),
                "frames_returned": returned_frame_stats.get("frames"),
                "frame_histogram_returned": returned_frame_stats.get("frame_histogram"),
                "min_conf": min_conf,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "sampling_strategy": sampling_strategy,
                "trajectory_quality": trajectory_quality,
            },
            "diagnostics": {
                "pointcloud_file": npz_path.name,
                "pointcloud_shape": list(points.shape),
                "has_conf": bool(points.ndim == 2 and points.shape[1] > 6),
                "has_frame_idx": bool(points.ndim == 2 and points.shape[1] > 7),
                "run_params": run_params,
                "fallback_summary": fallback_summary,
                "pose_graph": pose_graph_summary,
                "pose_graph_candidate": pose_graph_candidate,
                "stale_run": stale_run,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to filter point cloud: {e}")


@app.get("/api/r3-trajectory/{video_id}")
async def r3_get_trajectory(
    video_id: str,
    trajectory_source: str = Query("raw"),
):
    """Rebuild the lightweight plan trajectory from existing R3 pose artifacts."""
    try:
        base = _r3_output_dir(video_id)
        if not base.exists():
            raise HTTPException(status_code=404, detail="R3 output not found")
        trajectory_bundle, _ = _load_r3_trajectory_bundle(
            base,
            trajectory_source=trajectory_source,
        )
        run_params, fallback_summary = _load_r3_run_params(
            base,
            point_count=len(trajectory_bundle.get("plan_trajectory", [])),
        )
        pose_graph_summary = load_pose_graph_summary(
            base / "pose_graph_edges.npz",
            point_count=len(trajectory_bundle.get("plan_trajectory", [])),
        )
        pose_graph_candidate = load_pose_graph_candidate_summary(base)
        return _sanitize_for_json({
            "success": True,
            "video_id": video_id,
            "method": "r3_reconstruction",
            "trajectory": trajectory_bundle.get("plan_trajectory", []),
            "plan_trajectory": trajectory_bundle.get("plan_trajectory", []),
            "raw_plan_trajectory": trajectory_bundle.get("raw_plan_trajectory", []),
            "raw_trajectory_3d": trajectory_bundle.get("raw_trajectory_3d", []),
            "turn_points": trajectory_bundle.get("turn_points", []),
            "source_frame_indices": trajectory_bundle.get("source_frame_indices", []),
            "source_timestamps_seconds": trajectory_bundle.get("source_timestamps_seconds", []),
            "trajectory_quality": trajectory_bundle.get("trajectory_quality", {}),
            "trajectory_source_requested": trajectory_bundle.get(
                "trajectory_source_requested", "raw"
            ),
            "trajectory_source": trajectory_bundle.get("trajectory_source", "raw"),
            "trajectory_source_fallback_reason": trajectory_bundle.get(
                "trajectory_source_fallback_reason"
            ),
            "trajectory_source_selection": trajectory_bundle.get(
                "trajectory_source_selection", {}
            ),
            "run_params": run_params,
            "fallback_summary": fallback_summary,
            "pose_graph": pose_graph_summary,
            "pose_graph_candidate": pose_graph_candidate,
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load R3 trajectory: {exc}")


@app.get("/api/r3-diagnostics/{video_id}")
async def r3_get_diagnostics(video_id: str):
    """Return R³ output diagnostics for debugging reconstruction quality."""
    return _sanitize_for_json(_r3_run_diagnostics(video_id))


@app.get("/api/r3-projection-debug/{video_id}")
async def r3_get_projection_debug(
    video_id: str,
    max_points: int = Query(250000),
    min_conf: float = Query(1.4),
    frame_start: Optional[int] = Query(None),
    frame_end: Optional[int] = Query(None),
    sampling_strategy: str = Query("per_frame_uniform"),
):
    """Generate top/front/right PNG projections from the same filtered R³ cloud."""
    try:
        return _sanitize_for_json(_r3_projection_debug(
            video_id=video_id,
            max_points=max_points,
            min_conf=min_conf,
            frame_start=frame_start,
            frame_end=frame_end,
            sampling_strategy=sampling_strategy,
        ))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate R³ projections: {e}")


@app.post("/api/upload-video-stream/{video_id}")
async def upload_video_stream(video_id: str, request: Request, original_filename: str = "video.mp4"):
    """Endpoint для прямого стрима видео с клиента, без сохранения на локальном сервере.
    
    Сохраняет файл в gpu_worker_data/uploaded/ для последующей обработки.
    """
    UPLOAD_DIR = WORK_DIR / "uploaded"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(original_filename).suffix or ".mp4"
    dest = UPLOAD_DIR / f"{video_id}{ext}"
    
    total = 0
    with open(dest, "wb") as f:
        async for chunk in request.stream():
            if chunk:
                f.write(chunk)
                total += len(chunk)
    
    logger.info(f"[{video_id}] Direct upload saved: {dest.name} ({total} bytes)")
    return {"success": True, "file_size": total, "path": str(dest)}


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting TrackAI GPU Worker on port 8003 (CUDA/GPU mode)")
    uvicorn.run(app, host="0.0.0.0", port=8003)
