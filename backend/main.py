from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# Import video tracker
from video_tracker.src.processor import FullFeatureProcessor
from video_tracker.src.stabilization import stabilize_video

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="TrackAI Video Analysis API", version="1.0.0")

# Global storage for processing status
processing_status = {}
convert_dwg_status: Dict[str, Dict] = {}

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:3000",
        "http://localhost:8081",
        "http://176.123.167.109",
        "https://trackai.eu.ngrok.io",  # Backend ngrok Pro domain
        "https://trackai-app.eu.ngrok.io",  # Frontend ngrok Pro domain
        "https://trackai-backend.loca.lt",  # LocalTunnel backend
        "https://trackai-frontend.loca.lt",  # LocalTunnel frontend
        "https://fa44db5269c86bf8-185-104-115-196.serveousercontent.com", # Serveo frontend
        "https://14e265884d57c1eb-185-104-115-196.serveousercontent.com", # Serveo backend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Увеличиваем лимит загрузки для DWG/PDF/видео (Starlette 0.40+)
UPLOAD_MAX_PART_SIZE = 1024 * 1024 * 500  # 500 MB

@app.middleware("http")
async def increase_upload_limit(request: Request, call_next):
    # Только convert-dwg: middleware парсит form с большим лимитом (upload-video и analyze-video парсят в своих хендлерах)
    if request.method == "POST" and request.url.path == "/api/convert-dwg":
        await request.form(max_part_size=UPLOAD_MAX_PART_SIZE)
    return await call_next(request)

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
VIDEOS_DIR = Path("videos")  # Для хранения оригинальных видео
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)

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
    conn.commit()
    conn.close()

init_db()

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
    form = await request.form(max_part_size=UPLOAD_MAX_PART_SIZE)
    file = form.get("file")
    if not file or not hasattr(file, "filename") or not file.filename:
        raise HTTPException(status_code=400, detail="Требуется файл .pdf")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Требуется файл .pdf")
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_in:
        tmp_in.write(content)
        tmp_path = Path(tmp_in.name)
    out_prefix = tmp_path.with_suffix('')
    try:
        result = subprocess.run(
            ['/usr/bin/pdftoppm', '-png', '-singlefile', '-scale-to', '2048', '-f', '1', '-l', '1', str(tmp_path), str(out_prefix)],
            capture_output=True, text=True, timeout=120
        )
        png_path = out_prefix.with_suffix('.png') if out_prefix.with_suffix('.png').exists() else next(out_prefix.parent.glob(out_prefix.name + '-*.png'), None)
        if result.returncode != 0 or not png_path:
            raise HTTPException(status_code=500, detail="Не удалось конвертировать PDF. Установите poppler-utils.")
        png_data = png_path.read_bytes()
        png_b64 = base64.b64encode(png_data).decode('ascii')
        return {"success": True, "png": png_b64, "filename": file.filename}
    finally:
        tmp_path.unlink(missing_ok=True)
        out_prefix.with_suffix('.png').unlink(missing_ok=True)
        for f in out_prefix.parent.glob(out_prefix.name + '-*.png'):
            f.unlink(missing_ok=True)

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

async def process_video_background(
    video_id: str,
    video_path: Path,
    original_filename: str,
    scale_factor: float,
    stabilize: bool
):
    """Background task to process video without blocking the API"""
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
                def run_avi_convert():
                    return subprocess.run([
                        ffmpeg_path, '-i', str(video_path),
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                        '-c:a', 'aac', '-y', str(avi_mp4_path)
                    ], check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
                await run_in_threadpool(run_avi_convert)
                video_path.unlink()
                video_path = avi_mp4_path
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
            
            # Replace original with optimized file
            video_path.unlink()
            video_path = normalized_path
            logger.info(f"[{video_id}] Successfully optimized video")
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
                
                processing_path = await run_in_threadpool(stabilize_video, temp_path, stabilized_path, progress_callback=stab_progress)
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

        processor = FullFeatureProcessor(
            input_dir=str(UPLOAD_DIR),
            output_dir=str(OUTPUT_DIR),
            scale_factor=scale_factor,
            progress_callback=slam_progress
        )

        result = await run_in_threadpool(processor.process_video, processing_path)

        # Cleanup temp
        if processing_path.exists():
            processing_path.unlink()

        if result:
            # Normalize result
            normalized_result = result.copy()
            if "turn_points" in normalized_result:
                normalized_turns = []
                for turn in normalized_result["turn_points"]:
                    nt = turn.copy()
                    if isinstance(turn.get("position"), dict):
                        pos = turn["position"]
                        nt["position"] = [pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)]
                    elif not isinstance(turn.get("position"), list):
                        nt["position"] = [0, 0, 0]
                    normalized_turns.append(nt)
                normalized_result["turn_points"] = normalized_turns

            # Save results
            analysis_data = {
                "video_id": video_id,
                "video_filename": video_path.name,
                "original_filename": original_filename,
                "scale_factor": scale_factor,
                "stabilized": stabilize,
                "analysis_result": normalized_result
            }

            analysis_file = OUTPUT_DIR / f"{video_id}_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)

            processing_status[video_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Обработка завершена успешно",
                "result": normalized_result
            })
            logger.info(f"[{video_id}] Analysis completed successfully")
        else:
            raise Exception("Processor returned no result")

    except Exception as e:
        error_msg = str(e)
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

        # Ищем файл: сначала в UPLOADED_VIDEOS, потом по маске в VIDEOS_DIR
        video_filename = UPLOADED_VIDEOS.get(video_id)
        if not video_filename:
            matches = list(VIDEOS_DIR.glob(f"{video_id}_*"))
            if matches:
                video_filename = matches[0].name
            else:
                raise HTTPException(status_code=404, detail=f"Видео {video_id} не найдено на сервере")

        video_path = VIDEOS_DIR / video_filename
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Файл видео не найден: {video_filename}")

        processing_status[video_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Поставлено в очередь на обработку",
            "start_time": time.time()
        }

        from fastapi.concurrency import run_in_threadpool
        background_tasks.add_task(
            process_video_background,
            video_id,
            video_path,
            original_filename,
            scale_factor,
            stabilize
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
        
        # Start background processing
        background_tasks.add_task(
            process_video_background,
            video_id,
            video_path,
            file.filename,
            scale_factor,
            stabilize
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
                import json
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data["video_id"] == video_id:
                        video_path = VIDEOS_DIR / data["video_filename"]
                        if video_path.exists():
                            return JSONResponse(content={"download_url": f"/videos/{data['video_filename']}"})
                        else:
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