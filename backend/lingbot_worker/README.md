# TrackAI LingBot-Map GPU Worker

MVP FastAPI service for running LingBot-Map reconstruction on the RTX 3090 host.
The worker uses upstream `demo_render/batch_demo.py` in offline mode, not the
interactive viewer demo.

## Files

- `main.py` - FastAPI app.
- `schemas.py` - request/response schemas.
- `service.py` - session lifecycle and background execution.
- `lingbot_adapter.py` - wrapper around upstream LingBot-Map inference/demo code.
- `storage.py` - session folders and JSON/file IO.
- `config.py` - paths and runtime defaults.

Session outputs are stored under `data/lingbot_sessions/{session_id}/` by default:

- `input.mp4`
- `status.json`
- `trajectory.json`
- `pointcloud.ply` or `pointcloud.npz`
- `metadata.json`
- `logs.txt`

## Install On RTX 3090 Host

```bash
cd /home/artem/trackai/backend
conda create -n lingbot-map python=3.11
conda activate lingbot-map
pip install -r lingbot_worker/requirements.txt

mkdir -p third_party checkpoints data/lingbot_sessions
git clone https://github.com/Robbyant/lingbot-map third_party/lingbot-map
cd third_party/lingbot-map
pip install -e '.[vis]'
```

Download the model from Hugging Face and put it at:

```bash
/home/artem/trackai/backend/checkpoints/lingbot-map.pt
```

or override the path:

```bash
export LINGBOT_MODEL_PATH=/absolute/path/to/lingbot-map.pt
```

## Run

```bash
cd backend
uvicorn lingbot_worker.main:app --host 0.0.0.0 --port 8004
```

Port `8003` is reserved for the existing TrackAI `gpu_worker.py`; LingBot should
run on `8004`.

## Environment

- `LINGBOT_REPO_PATH` default: `backend/third_party/lingbot-map`
- `LINGBOT_MODEL_PATH` default: `backend/checkpoints/lingbot-map.pt`
- `LINGBOT_SESSIONS_DIR` default: `backend/data/lingbot_sessions`
- `LINGBOT_DEFAULT_FPS` default: `10`
- `LINGBOT_TARGET_FRAMES` default: `1500`
- `LINGBOT_DEFAULT_KEYFRAME_INTERVAL` default: `6`
- `LINGBOT_IMAGE_SIZE` default: `518`
- `LINGBOT_USE_SDPA` default: `true`
- `LINGBOT_MASK_SKY` default: `false`
- `LINGBOT_PYTHON` optional Python executable for LingBot-Map subprocess

Note: the current RTX 3090 deployment forces `--use_sdpa` even if an old client
posts `use_sdpa=false`, because FlashInfer is not installed in the LingBot
environment.

## API

```bash
curl http://127.0.0.1:8004/health
```

```bash
curl -X POST http://127.0.0.1:8004/sessions \
  -H 'Content-Type: application/json' \
  -d '{
    "video_path": "/absolute/path/to/video.mp4",
    "fps": 10,
    "keyframe_interval": 6,
    "use_sdpa": true,
    "mask_sky": false
  }'
```

For the VPS/frontend path, use multipart upload so the video is copied onto the
3090 host before reconstruction:

```bash
curl -X POST http://127.0.0.1:8004/sessions/upload \
  -F file=@/absolute/path/to/video.mp4 \
  -F fps=10 \
  -F target_frames=1500 \
  -F keyframe_interval=6 \
  -F use_sdpa=true \
  -F mask_sky=false
```

```bash
curl http://127.0.0.1:8004/sessions/{session_id}/status
curl http://127.0.0.1:8004/sessions/{session_id}/trajectory
curl -OJ http://127.0.0.1:8004/sessions/{session_id}/pointcloud
curl http://127.0.0.1:8004/sessions/{session_id}/metadata
```

The public TrackAI backend proxies these as:

- `GET /api/lingbot-health`
- `GET /api/lingbot-sessions/{session_id}/status`
- `GET /api/lingbot-sessions/{session_id}/trajectory`
- `GET /api/lingbot-sessions/{session_id}/pointcloud`
- `GET /api/lingbot-sessions/{session_id}/metadata`
