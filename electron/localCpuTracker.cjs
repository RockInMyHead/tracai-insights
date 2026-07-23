const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { app } = require('electron');

const PLAN_WIDTH = 800;
const PLAN_HEIGHT = 600;
const SAMPLE_WIDTH = 160;
const SAMPLE_HEIGHT = 90;
const FRAME_BYTES = SAMPLE_WIDTH * SAMPLE_HEIGHT;
const MAX_SAMPLES = 900;

function dataPath() {
  return path.join(app.getPath('userData'), 'offline-history.json');
}

function videosPath() {
  return path.join(app.getPath('userData'), 'offline-videos');
}

function readHistory() {
  try {
    const items = JSON.parse(fs.readFileSync(dataPath(), 'utf8'));
    return Array.isArray(items) ? items : [];
  } catch {
    return [];
  }
}

function writeHistory(items) {
  fs.mkdirSync(path.dirname(dataPath()), { recursive: true });
  fs.writeFileSync(dataPath(), JSON.stringify(items.slice(0, 100), null, 2));
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function patchVariance(frame, x, y, radius) {
  const values = [];
  for (let py = -radius; py <= radius; py += 1) {
    for (let px = -radius; px <= radius; px += 1) values.push(frame[(y + py) * SAMPLE_WIDTH + x + px]);
  }
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  return values.reduce((sum, value) => sum + (value - average) ** 2, 0) / values.length;
}

function patchSad(previous, current, x, y, dx, dy, radius) {
  let sad = 0;
  for (let py = -radius; py <= radius; py += 1) {
    for (let px = -radius; px <= radius; px += 1) {
      sad += Math.abs(previous[(y + py) * SAMPLE_WIDTH + x + px] - current[(y + dy + py) * SAMPLE_WIDTH + x + dx + px]);
    }
  }
  return sad;
}

function estimateShift(previous, current) {
  const shifts = [];
  const radius = 4;
  for (let y = 18; y <= 68; y += 12) {
    for (let x = 28; x <= 132; x += 13) {
      if (patchVariance(previous, x, y, radius) < 130) continue;
      let best = { dx: 0, dy: 0, sad: Infinity };
      for (let dy = -4; dy <= 4; dy += 1) {
        for (let dx = -4; dx <= 4; dx += 1) {
          const sad = patchSad(previous, current, x, y, dx, dy, radius);
          if (sad < best.sad) best = { dx, dy, sad };
        }
      }
      if (best.sad / 81 < 42) shifts.push(best);
    }
  }
  if (shifts.length < 5) return { dx: 0, dy: 0, support: shifts.length };
  return { dx: median(shifts.map((shift) => shift.dx)), dy: median(shifts.map((shift) => shift.dy)), support: shifts.length };
}

function decodeFrames(buffer) {
  const frameCount = Math.min(Math.floor(buffer.length / FRAME_BYTES), MAX_SAMPLES);
  const frames = [];
  for (let index = 0; index < frameCount; index += 1) {
    frames.push(buffer.subarray(index * FRAME_BYTES, (index + 1) * FRAME_BYTES));
  }
  return frames;
}

function buildTrajectory(frames) {
  if (frames.length < 3) throw new Error('В видео недостаточно кадров для локального трекинга');
  let x = 336;
  let y = 108;
  const points = [[x, y, 0]];
  let movement = 0;
  for (let index = 1; index < frames.length; index += 1) {
    const shift = estimateShift(frames[index - 1], frames[index]);
    if (shift.support < 5) continue;
    const step = Math.hypot(shift.dx, shift.dy);
    if (step > 5.5) continue;
    x = Math.max(18, Math.min(PLAN_WIDTH - 18, x - shift.dx * 2.1));
    y = Math.max(18, Math.min(PLAN_HEIGHT - 18, y - shift.dy * 2.1));
    movement += step;
    points.push([Math.round(x * 10) / 10, Math.round(y * 10) / 10, 0]);
  }
  if (points.length < 3 || movement < 3) {
    throw new Error('Не удалось увидеть устойчивое движение в видео. Проверьте запись и попробуйте ещё раз.');
  }
  const stride = Math.max(1, Math.ceil(points.length / 400));
  return points.filter((_point, index) => index % stride === 0 || index === points.length - 1);
}

function extractFrames(videoPath, onProgress) {
  const bundledWindowsFfmpeg = path.join(process.resourcesPath || '', 'ffmpeg', 'ffmpeg.exe');
  let ffmpegPath = process.platform === 'win32' && fs.existsSync(bundledWindowsFfmpeg) ? bundledWindowsFfmpeg : null;
  if (!ffmpegPath) {
    try {
      ffmpegPath = require('ffmpeg-static');
    } catch {
      throw new Error('Модуль обработки видео не установлен. Переустановите TrackAI.');
    }
  }
  return new Promise((resolve, reject) => {
    const child = spawn(ffmpegPath, ['-hide_banner', '-loglevel', 'error', '-i', videoPath, '-vf', 'fps=1,scale=160:90:flags=fast_bilinear,format=gray', '-frames:v', String(MAX_SAMPLES), '-f', 'rawvideo', 'pipe:1'], { windowsHide: true });
    const chunks = [];
    let stderr = '';
    child.stdout.on('data', (chunk) => chunks.push(chunk));
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code !== 0) return reject(new Error(`Не удалось прочитать видео локально: ${stderr.slice(0, 240)}`));
      if (typeof onProgress === 'function') onProgress(75);
      resolve(Buffer.concat(chunks));
    });
  });
}

async function copyToLocal({ filePath, fileName, onProgress }) {
  const id = crypto.randomUUID();
  const targetDir = videosPath();
  const targetPath = path.join(targetDir, `${id}_${fileName}`);
  const stat = await fs.promises.stat(filePath);
  await fs.promises.mkdir(targetDir, { recursive: true });
  await new Promise((resolve, reject) => {
    let copied = 0;
    const input = fs.createReadStream(filePath);
    const output = fs.createWriteStream(targetPath, { flags: 'wx' });
    input.on('data', (chunk) => {
      copied += chunk.length;
      if (typeof onProgress === 'function' && stat.size) onProgress(Math.min(100, (copied / stat.size) * 100));
    });
    input.on('error', reject);
    output.on('error', reject);
    output.on('finish', resolve);
    input.pipe(output);
  });
  return { video_id: id, filename: fileName, original_filename: fileName, file_size: stat.size, localPath: targetPath };
}

async function processLocalVideo(video) {
  const history = readHistory();
  const item = history.find((entry) => entry.video_id === video.video_id);
  const sourcePath = video.localPath || item?.localPath;
  if (!sourcePath || !fs.existsSync(sourcePath)) throw new Error('Копия видео не найдена');
  const raw = await extractFrames(sourcePath);
  const trajectory = buildTrajectory(decodeFrames(raw));
  const result = {
    method: 'local_cpu_optical_flow',
    trajectory,
    map_trajectory: trajectory,
    turn_points: [],
    map_turn_points: [],
    processing_stats: { estimated_distance: 0, fps: 1, trajectory_points: trajectory.length, map_matching_applied: false, local_cpu: true },
    video_info: { width: SAMPLE_WIDTH, height: SAMPLE_HEIGHT, fps: 1, frame_count: trajectory.length, duration: trajectory.length },
  };
  const record = { video_id: video.video_id, filename: video.filename, original_filename: video.original_filename || video.filename, file_size: video.file_size || 0, uploaded_at: new Date().toISOString(), has_analysis: true, localPath: sourcePath, data: result };
  writeHistory([record, ...history.filter((entry) => entry.video_id !== video.video_id)]);
  return { success: true, status: 'completed', video_id: video.video_id, data: result };
}

function getHistory() {
  return readHistory().map(({ localPath, data, ...item }) => ({ ...item, scale_factor: 1, stabilized: false }));
}

function getAnalysis(videoId) {
  const item = readHistory().find((entry) => entry.video_id === videoId);
  if (!item?.data) throw new Error('Результат анализа не найден');
  return { success: true, video_id: videoId, data: item.data };
}

module.exports = { copyToLocal, processLocalVideo, getHistory, getAnalysis };
