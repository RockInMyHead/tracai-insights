const fs = require('fs');
const path = require('path');
const { Readable } = require('stream');
const { app } = require('electron');

function queuePath() {
  return path.join(app.getPath('userData'), 'admin-mirror-queue.json');
}
function readQueue() {
  try { return JSON.parse(fs.readFileSync(queuePath(), 'utf8')) || {}; } catch { return {}; }
}
function writeQueue(queue) {
  fs.writeFileSync(queuePath(), JSON.stringify(queue, null, 2));
}
function enqueueVideo(video) {
  const queue = readQueue();
  queue[video.video_id] = { ...queue[video.video_id], video, videoUploaded: false };
  writeQueue(queue);
}
function enqueueResult(videoId, result) {
  const queue = readQueue();
  queue[videoId] = { ...queue[videoId], result, resultUploaded: false };
  writeQueue(queue);
}
async function flush(serverUrl, log = () => {}) {
  const queue = readQueue();
  for (const [videoId, item] of Object.entries(queue)) {
    try {
      if (!item.videoUploaded && item.video?.localPath && fs.existsSync(item.video.localPath)) {
        const stat = fs.statSync(item.video.localPath);
        const response = await fetch(`${serverUrl}/api/desktop/archive-video/${encodeURIComponent(videoId)}?filename=${encodeURIComponent(item.video.original_filename || item.video.filename)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(stat.size), 'X-TrackAI-Client': 'desktop-local' },
          body: Readable.toWeb(fs.createReadStream(item.video.localPath)),
          duplex: 'half',
        });
        if (!response.ok) throw new Error(`archive ${response.status}: ${(await response.text()).slice(0, 300)}`);
        item.videoUploaded = true;
      }
      if (item.videoUploaded && item.result && !item.resultUploaded) {
        const response = await fetch(`${serverUrl}/api/desktop/local-analysis/${encodeURIComponent(videoId)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-TrackAI-Client': 'desktop-local' },
          body: JSON.stringify({ data: item.result }),
        });
        if (!response.ok) throw new Error(`result ${response.status}: ${(await response.text()).slice(0, 300)}`);
        item.resultUploaded = true;
      }
      if (item.videoUploaded && (!item.result || item.resultUploaded)) delete queue[videoId];
    } catch (error) {
      item.lastError = error.message;
      item.lastAttemptAt = new Date().toISOString();
      log('warn', 'admin-mirror:retry-pending', { videoId, message: error.message });
    }
    writeQueue(queue);
  }
}
module.exports = { enqueueVideo, enqueueResult, flush };
