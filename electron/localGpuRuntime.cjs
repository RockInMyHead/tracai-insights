const fs = require('fs');
const path = require('path');
const { spawn, execFile } = require('child_process');
const { app } = require('electron');

const PORT = 18765;
let worker = null;
let starting = null;

function runtimeDir() {
  return process.env.TRACKAI_GPU_RUNTIME_DIR
    || path.join(process.resourcesPath || '', 'local-gpu');
}

function execFileText(file, args, timeout = 8000) {
  return new Promise((resolve, reject) => {
    execFile(file, args, { windowsHide: true, timeout }, (error, stdout, stderr) => {
      if (error) reject(new Error((stderr || error.message).trim()));
      else resolve(stdout.trim());
    });
  });
}

function versionAtLeast(actual, minimum) {
  const a = String(actual).split('.').map(Number);
  const b = String(minimum).split('.').map(Number);
  for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
    if ((a[index] || 0) !== (b[index] || 0)) return (a[index] || 0) > (b[index] || 0);
  }
  return true;
}

async function inspect() {
  if (process.platform !== 'win32') return { ready: false, reason: 'windows_only' };
  const root = runtimeDir();
  const manifestPath = path.join(root, 'runtime-manifest.json');
  const python = path.join(root, 'python', 'python.exe');
  if (!fs.existsSync(manifestPath) || !fs.existsSync(python)) {
    return { ready: false, reason: 'runtime_missing' };
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  if (!manifest.complete) return { ready: false, reason: 'runtime_incomplete' };
  let rows;
  try {
    rows = await execFileText('nvidia-smi', [
      '--query-gpu=name,memory.total,driver_version',
      '--format=csv,noheader,nounits',
    ]);
  } catch {
    return { ready: false, reason: 'nvidia_driver_missing' };
  }
  const gpus = rows.split(/\r?\n/).filter(Boolean).map((line) => {
    const [name, memory, driver] = line.split(',').map((value) => value.trim());
    return { name, memoryMb: Number(memory), driver };
  });
  const compatible = gpus.find((gpu) =>
    gpu.memoryMb >= Number(manifest.minimum_vram_mb || 12000)
    && versionAtLeast(gpu.driver, manifest.minimum_nvidia_driver_windows));
  return compatible
    ? { ready: true, root, python, manifest, gpu: compatible }
    : { ready: false, reason: 'gpu_incompatible' };
}

async function health() {
  const response = await fetch(`http://127.0.0.1:${PORT}/health`);
  if (!response.ok) throw new Error(`Local worker health failed: ${response.status}`);
  return response.json();
}

async function start(log = () => {}) {
  if (starting) return starting;
  starting = (async () => {
    const status = await inspect();
    if (!status.ready) return status;
    try {
      const current = await health();
      if (current.ok) return { ...status, health: current };
    } catch {}
    const dataDir = path.join(app.getPath('userData'), 'local-gpu');
    fs.mkdirSync(dataDir, { recursive: true });
    worker = spawn(status.python, [
      '-m', 'uvicorn', 'worker:app',
      '--app-dir', status.root,
      '--host', '127.0.0.1', '--port', String(PORT), '--no-access-log',
    ], {
      cwd: status.root,
      windowsHide: true,
      env: {
        ...process.env,
        TRACKAI_RUNTIME_DIR: status.root,
        TRACKAI_LOCAL_DATA_DIR: dataDir,
        PYTHONNOUSERSITE: '1',
        PATH: `${path.join(process.resourcesPath, 'ffmpeg')};${process.env.PATH || ''}`,
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    worker.stdout.on('data', (value) => log('info', 'local-gpu:stdout', { message: value.toString().trim() }));
    worker.stderr.on('data', (value) => log('warn', 'local-gpu:stderr', { message: value.toString().trim() }));
    worker.on('exit', (code) => { log('warn', 'local-gpu:exit', { code }); worker = null; });
    for (let attempt = 0; attempt < 60; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 500));
      try {
        const current = await health();
        if (current.ok) return { ...status, health: current };
      } catch {}
    }
    stop();
    return { ready: false, reason: 'worker_start_failed' };
  })().finally(() => { starting = null; });
  return starting;
}

function stop() {
  if (worker && !worker.killed) worker.kill();
  worker = null;
}

async function processVideo(video) {
  const sourcePath = video.localPath;
  if (!sourcePath || !fs.existsSync(sourcePath)) throw new Error('Локальная копия видео не найдена');
  const stream = fs.createReadStream(sourcePath);
  const response = await fetch(`http://127.0.0.1:${PORT}/process/${encodeURIComponent(video.video_id)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: stream,
    duplex: 'half',
  });
  if (!response.ok) throw new Error((await response.text()).slice(-2000));
  const result = await response.json();
  const items = readHistory();
  const record = {
    video_id: video.video_id,
    filename: video.filename,
    original_filename: video.original_filename || video.filename,
    file_size: video.file_size || 0,
    uploaded_at: new Date().toISOString(),
    has_analysis: true,
    localPath: sourcePath,
    data: result.data,
  };
  fs.writeFileSync(historyPath(), JSON.stringify(
    [record, ...items.filter((item) => item.video_id !== video.video_id)].slice(0, 100),
    null, 2,
  ));
  return result;
}

function historyPath() {
  return path.join(app.getPath('userData'), 'local-gpu-history.json');
}

function readHistory() {
  try {
    const parsed = JSON.parse(fs.readFileSync(historyPath(), 'utf8'));
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function getHistory() {
  return readHistory().map(({ localPath, data, ...item }) => item);
}

function getAnalysis(videoId) {
  const item = readHistory().find((entry) => entry.video_id === videoId);
  if (!item?.data) throw new Error('Результат анализа не найден');
  return { success: true, video_id: videoId, data: item.data };
}

module.exports = { inspect, start, stop, processVideo, getHistory, getAnalysis };
