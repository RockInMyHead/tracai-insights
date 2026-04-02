from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from starlette.requests import Request
import os
import shutil
import tempfile
import time
import asyncio
import aiohttp
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
import sqlite3
import json
import uuid
import fitz  # PyMuPDF

# Import video tracker
from video_tracker.src.processor import FullFeatureProcessor
from video_tracker.src.map_postprocessing import apply_map_postprocessing
from video_tracker.src.stabilization import stabilize_video

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="TrackAI Video Analysis API", version="1.0.0")

# Global storage for processing status
processing_status = {}
convert_dwg_status: Dict[str, Dict] = {}

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
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_serializable(v) for v in obj]

    # Fallback: leave as-is (bool, int, float, str, None, etc.)
    return obj

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
        "http://95.174.93.76",
        "https://95.174.93.76",
        "http://176.123.167.109",
        "http://45.67.57.72",
        "https://trackai.eu.ngrok.io",
        "https://trackai-app.eu.ngrok.io",
        "https://trackai-backend.loca.lt",
        "https://trackai-frontend.loca.lt",
        "https://fa44db5269c86bf8-185-104-115-196.serveousercontent.com",
        "https://14e265884d57c1eb-185-104-115-196.serveousercontent.com",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|95\.174\.93\.76|45\.67\.57\.72)(:\d+)?$",
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
        await request.form(max_part_size=UPLOAD_MAX_PART_SIZE)
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
MANUAL_TRAJECTORIES_PATH = Path("backend/data/manual_trajectories.json")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
MANUAL_TRAJECTORIES_PATH.parent.mkdir(exist_ok=True, parents=True)

# Database initialization
DB_PATH = Path("backend/data/database.db")
DB_PATH.parent.mkdir(exist_ok=True, parents=True)

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
    map_context = {
        "floor_plan_data": payload.get("floor_plan_data"),
        "drawn_plan": _parse_json_field(payload.get("drawn_plan")),
        "reference_point": _parse_json_field(payload.get("reference_point")),
        "direction_point": _parse_json_field(payload.get("direction_point")),
    }
    if not map_context["floor_plan_data"] and not map_context["drawn_plan"]:
        return None
    return map_context


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


def _make_manual_result(trajectory: Any, turn_points: Optional[List[Any]] = None) -> Dict[str, Any]:
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
TELEGRAM_BOT_TOKEN = "8231968530:AAGh0gdqmcTYka2q-zEPaSurhHvibEooDQE"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

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

@app.get("/")
async def root():
    return {"message": "TrackAI Video Analysis API", "version": "1.0.0"}

@app.get("/api/status/{video_id}")
async def get_video_status(video_id: str):
    """
    Get the current status and progress of video analysis
    """
    if video_id not in processing_status:
        return {"id": video_id, "status": "unknown", "progress": 0}
    return processing_status[video_id]

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
        form = await request.form(max_part_size=UPLOAD_MAX_PART_SIZE)
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
    video_path: Path,
    original_filename: str,
    scale_factor: float,
    stabilize: bool,
    detect_interval: int = 3,
    turn_vote_threshold: int = 3,
    use_ml_roi: bool = True,
    map_context: Optional[Dict[str, Any]] = None,
):
    """Background task to process video without blocking the API"""
    _update_task_status(video_id, "processing", 10)
    try:

        from fastapi.concurrency import run_in_threadpool
        
        ffmpeg_path = '/usr/bin/ffmpeg' if os.path.exists('/usr/bin/ffmpeg') else 'ffmpeg'
        
        # Step 0: AVI → MP4 конвертация (для совместимости с OpenCV/ffmpeg)
        if video_path.suffix.lower() == '.avi':
            logger.info(f"[{video_id}] Конвертация AVI в MP4...")
            processing_status[video_id].update({
                "status": "converting",
                "progress": 10,
                "message": "Конвертация AVI в MP4 (может занять 15–30 мин для больших файлов)..."
            })
            avi_mp4_path = video_path.with_suffix('.from_avi.mp4')
            try:
                duration_sec = await run_in_threadpool(_get_video_duration_sec, video_path)
                def run_avi_convert():
                    cmd = [
                        ffmpeg_path, '-i', str(video_path),
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                        '-c:a', 'aac', '-y', str(avi_mp4_path)
                    ]
                    if duration_sec > 0:
                        _run_ffmpeg_with_progress(cmd, video_id, duration_sec, 10, 14, "Конвертация AVI в MP4")
                    else:
                        subprocess.run(cmd, check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
                await run_in_threadpool(run_avi_convert)
                if _validate_video_readable(avi_mp4_path):
                    video_path.unlink()
                    video_path = avi_mp4_path
                else:
                    raise Exception("AVI->MP4 output is not readable")
                logger.info(f"[{video_id}] AVI успешно сконвертирован в MP4")
            except subprocess.TimeoutExpired:
                error_msg = f"Таймаут конвертации AVI (>{FFMPEG_TIMEOUT//60} мин). Попробуйте файл поменьше."
                logger.error(f"[{video_id}] {error_msg}")
                await send_error_to_telegram(error_msg, f"AVI конвертация: {original_filename}")
                raise Exception(error_msg)
            except Exception as e:
                error_msg = f"Ошибка конвертации AVI: {str(e)}"
                logger.error(f"[{video_id}] {error_msg}")
                await send_error_to_telegram(error_msg, f"AVI конвертация: {original_filename}")
                raise Exception(error_msg)
        
        # Step 1: Optimization/Normalization
        logger.info(f"[{video_id}] Normalizing video format and resolution...")
        processing_status[video_id].update({
            "status": "converting",
            "progress": 15,
            "message": "Оптимизация видео (уменьшение до 720p для скорости)..."
        })
        
        normalized_path = video_path.with_suffix('.optimized.mp4')
        
        try:
            def run_ffmpeg_norm():
                return subprocess.run([
                    ffmpeg_path, '-i', str(video_path),
                    '-vf', 'scale=-2:min(ih\,720)',
                    '-c:v', 'libx264', '-preset', 'superfast', '-crf', '23',
                    '-c:a', 'aac', '-y', str(normalized_path)
                ], check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
            
            await run_in_threadpool(run_ffmpeg_norm)

            # Replace original with optimized file only if readable
            if _validate_video_readable(normalized_path):
                video_path.unlink()
                video_path = normalized_path
                logger.info(f"[{video_id}] Successfully optimized video")
            else:
                logger.warning(f"[{video_id}] Optimized video not readable, keeping original")
                if normalized_path.exists():
                    normalized_path.unlink()
        except Exception as e:
            logger.warning(f"[{video_id}] Optimization pass failed: {e}")
            logger.info(f"[{video_id}] Proceeding with original file...")

        # Create temporary copy in /tmp for processing
        temp_dir = Path("/tmp/trackai_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_path = temp_dir / f"proc_{video_id}{video_path.suffix}"
        
        def copy_to_temp():
            shutil.copy2(video_path, temp_path)
        await run_in_threadpool(copy_to_temp)

        processing_path = temp_path

        # Step 2: Stabilization
        if stabilize:
            logger.info(f"[{video_id}] Starting video stabilization...")
            processing_status[video_id].update({
                "status": "stabilizing",
                "progress": 20,
                "message": "Стабилизация видео (это может занять время)..."
            })
            
            stabilized_path = temp_dir / f"stab_{video_id}{video_path.suffix}"

            try:
                def stab_progress(p):
                    overall_p = 20 + (p * 0.4)
                    processing_status[video_id]["progress"] = int(overall_p)
                
                processing_path = await run_in_threadpool(
                    stabilize_video,
                    temp_path,
                    stabilized_path,
                    progress_callback=stab_progress,
                    dynamic_smoothing=True
                )
                if not _validate_video_readable(processing_path):
                    logger.warning(f"[{video_id}] Stabilized video not readable, falling back to original temp")
                    processing_path = temp_path
                logger.info(f"[{video_id}] Stabilization completed")
            except Exception as e:
                error_msg = f"Ошибка стабилизации: {str(e)}"
                logger.error(f"[{video_id}] {error_msg}")
                await send_error_to_telegram(error_msg, f"Стабилизация: {original_filename}")
                processing_path = temp_path

            if temp_path.exists() and processing_path != temp_path:
                temp_path.unlink()

        # Step 3: SLAM analysis
        logger.info(f"[{video_id}] Starting SLAM trajectory analysis...")
        processing_status[video_id].update({
            "status": "analyzing",
            "progress": 60,
            "message": "SLAM анализ траектории..."
        })
        
        def slam_progress(p):
            overall_p = 60 + (p * 0.35)
            processing_status[video_id]["progress"] = int(overall_p)

        _detector_onnx = Path(__file__).resolve().parent / "models" / "detector.onnx"
        processor = FullFeatureProcessor(
            input_dir=str(UPLOAD_DIR),
            output_dir=str(OUTPUT_DIR),
            scale_factor=scale_factor,
            progress_callback=slam_progress,
            use_homography=True,
            use_kalman=True,
            use_akaze=True,
            frame_skip=1,
            target_width=900,
            use_optical_flow=True,
            detect_interval=max(1, int(detect_interval)),
            turn_vote_threshold=max(1, min(5, int(turn_vote_threshold))),
            use_ml_roi=use_ml_roi,
            ml_model_path=str(_detector_onnx),
        )

        result = await run_in_threadpool(processor.process_video, processing_path)

        # Cleanup temp
        if processing_path.exists():
                processing_path.unlink()

        if result:
            # Не считаем успехом пустую траекторию (VO мог не набрать точек)
            traj = result.get("trajectory") or []
            if not traj or len(traj) == 0:
                raise Exception(
                    "Траектория пуста: алгоритм не смог построить путь по кадрам. "
                    "Попробуйте другое видео, отключите стабилизацию или улучшите освещение/текстуру сцены."
                )
            # Normalize result
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

            normalized_result = result.copy()
            for key in ("turn_points", "raw_turn_points", "trajectory_turn_points"):
                if key in normalized_result and normalized_result[key]:
                    normalized_result[key] = _normalize_turn_list(normalized_result[key])

            if map_context:
                normalized_result = apply_map_postprocessing(normalized_result, map_context)
                if normalized_result.get("map_turn_points"):
                    normalized_result["map_turn_points"] = _normalize_turn_list(normalized_result["map_turn_points"])
            # Источник истины по приоритету: (1) IMU [пока нет], (2) map_turn_points, (3) turn_points (integrated yaw)
            normalized_result["final_turn_points"] = normalized_result.get("map_turn_points") or normalized_result.get("turn_points") or []

            # Глобально приводим результат к JSON-совместимым типам (numpy.bool_ и т.п.)
            normalized_result = _to_json_serializable(normalized_result)

            # Save results
            analysis_data = {
                "video_id": video_id,
                "video_filename": video_path.name,
                "original_filename": original_filename,
                "scale_factor": scale_factor,
                "stabilized": stabilize,
                "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(video_path.stat().st_mtime)),
                "analysis_result": normalized_result
            }

            analysis_file = OUTPUT_DIR / f"{video_id}_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=_to_json_serializable)

            processing_status[video_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Обработка завершена успешно",
                "result": normalized_result
            })
            _update_task_status(video_id, "completed", 100)
            logger.info(f"[{video_id}] Analysis completed successfully")

        else:
            raise Exception(
                "Процессор не вернул результат (не удалось обработать кадры). "
                "Попробуйте отключить стабилизацию или загрузить видео в формате MP4 (H.264)."
            )

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

@app.post("/api/upload-video")
async def upload_video(request: Request) -> Dict[str, Any]:
    """Загрузить видео на сервер (отдельно от анализа). Возвращает video_id для последующего анализа."""
    try:
        form = await request.form(max_part_size=UPLOAD_MAX_PART_SIZE)
        file = form.get("file")
        if not file or not hasattr(file, "filename") or not file.filename:
            logger.warning(f"upload-video: file missing. Form keys: {list(form.keys()) if form else 'empty'}")
            raise HTTPException(status_code=422, detail="Требуется файл видео. Убедитесь, что поле формы называется 'file'.")
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Неподдерживаемый формат видео")

        employee_name = form.get("employee_name") or None
        if isinstance(employee_name, str):
            employee_name = employee_name.strip() or None

        video_id = str(uuid.uuid4())
        video_filename = f"{video_id}_{file.filename}"
        video_path = VIDEOS_DIR / video_filename

        # Потоковая запись чанками (для больших файлов)
        chunk_size = 1024 * 1024  # 1 MB
        with open(video_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)

        UPLOADED_VIDEOS[video_id] = video_filename
        logger.info(f"Uploaded video: {video_filename} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")

        # Сразу создаём задачу в tracking_tasks для админки
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO tracking_tasks (id, employee_name, video_filename, original_filename, status) VALUES (?, ?, ?, ?, ?)",
                (video_id, employee_name, video_filename, file.filename, "uploaded")
            )
            conn.commit()
            conn.close()
            logger.info(f"Created tracking task {video_id} for employee '{employee_name}'")
        except Exception as db_err:
            logger.error(f"Failed to save tracking task on upload: {db_err}")

        return {
            "success": True,
            "video_id": video_id,
            "filename": video_filename,
            "original_filename": file.filename,
            "file_size": video_path.stat().st_size,
            "message": "Видео загружено на сервер"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload_video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/tasks/{task_id}/context")
async def update_task_context(task_id: str, request: Request) -> Dict[str, Any]:
    """Обновить контекст задачи (чертеж, план, имя сотрудника) — вызывается фронтом при каждом изменении."""
    try:
        body = await request.json()
        map_context = {}
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

        # Проверяем существует ли задача
        cursor.execute("SELECT id FROM tracking_tasks WHERE id = ?", (task_id,))
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Задача не найдена")

        updates = ["map_context = ?", "updated_at = CURRENT_TIMESTAMP"]
        params = [json.dumps(map_context)]

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

        # Если администратор задал ручную траекторию, возвращаем её сразу.
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
                "message": "Использована ручная траектория администратора",
                "result": manual_result,
                "start_time": time.time(),
            }
            return {
                "success": True,
                "video_id": video_id,
                "status": "completed",
                "message": "Использована ручная траектория",
                "data": manual_result,
            }

        # Ищем файл: UPLOADED_VIDEOS, потом glob; если файл не найден — пересканируем
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
                raise HTTPException(status_code=404, detail=f"Видео {video_id} не найдено на сервере")

        processing_status[video_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Поставлено в очередь на обработку",
            "start_time": time.time()
        }

        # Save task to DB
        try:
            employee_name = payload.get("employee_name")
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO tracking_tasks (id, employee_name, video_filename, original_filename, map_context, status) VALUES (?, ?, ?, ?, ?, ?)",
                (video_id, employee_name, video_path.name, original_filename, json.dumps(map_context), "queued")
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            logger.error(f"Failed to save task to DB: {db_err}")


        from fastapi.concurrency import run_in_threadpool
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
            map_context
        )

        return {
            "success": True,
            "video_id": video_id,
            "status": "queued",
            "message": "Анализ запущен"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_video_by_id: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-video")
async def analyze_video(background_tasks: BackgroundTasks, request: Request) -> Dict[str, Any]:
    """Analyze uploaded video file and return trajectory data."""
    form = await request.form(max_part_size=UPLOAD_MAX_PART_SIZE)
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

        # Start background processing
        background_tasks.add_task(
            process_video_background,

            video_id,
            video_path,
            file.filename,
            scale_factor,
            stabilize,
            detect_interval,
            turn_vote_threshold,
            use_ml_roi,
            map_context
        )

        return {
            "success": True,
            "video_id": video_id,
            "status": "queued",
            "message": "Видео загружено и поставлено в очередь на обработку"
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
                "map_context": map_ctx
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



@app.get("/api/uploaded-video/{video_id}/stream")
async def stream_uploaded_video(video_id: str):
    """Return uploaded video file by video_id (no analysis required)."""
    video_filename = UPLOADED_VIDEOS.get(video_id)
    video_path = (VIDEOS_DIR / video_filename) if video_filename else None

    if not video_path or not video_path.exists():
        matches = list(VIDEOS_DIR.glob(f"{video_id}_*"))
        for m in matches:
            if m.is_file():
                video_path = m
                UPLOADED_VIDEOS[video_id] = m.name
                break

    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded video not found")

    return FileResponse(str(video_path), filename=video_path.name)


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
        "message": "Ручная траектория сохранена администратором",
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
            try:
                import json
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    videos.append({
                        "video_id": data["video_id"],
                        "filename": data["original_filename"],
                        "uploaded_at": data["uploaded_at"],
                        "file_size": Path(VIDEOS_DIR / data["video_filename"]).stat().st_size if (VIDEOS_DIR / data["video_filename"]).exists() else 0,
                        "scale_factor": data["scale_factor"],
                        "stabilized": data["stabilized"],
                        "has_analysis": True
                    })
            except Exception as e:
                logger.warning(f"Error reading analysis file {json_file}: {e}")

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
        deepseek_api_key = "sk-af4d1592e8cc4bb8ba1881efb4bc8139"

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
                                'https://api.telegram.org/bot8231968530:AAGh0gdqmcTYka2q-zEPaSurhHvibEooDQE/sendMessage',
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
