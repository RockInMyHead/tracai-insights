const fs = require('fs');
const path = require('path');
const { Readable } = require('stream');

async function initUpload(serverUrl, filename, employeeName) {
  const response = await fetch(`${serverUrl}/api/init-upload`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-TrackAI-Client': 'desktop',
    },
    body: JSON.stringify({
      filename,
      employee_name: employeeName || null,
      client_source: 'camera_auto',
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Init upload failed (${response.status}): ${text.slice(0, 200)}`);
  }

  const payload = await response.json();
  if (!payload?.video_id) {
    throw new Error('Init upload did not return video_id');
  }
  return payload;
}

async function uploadVideoStream(serverUrl, videoId, filePath, onProgress) {
  const stat = await fs.promises.stat(filePath);
  const totalBytes = stat.size;
  let uploadedBytes = 0;

  const nodeStream = fs.createReadStream(filePath);
  nodeStream.on('data', (chunk) => {
    uploadedBytes += chunk.length;
    if (typeof onProgress === 'function' && totalBytes > 0) {
      onProgress(Math.min(100, (uploadedBytes / totalBytes) * 100));
    }
  });

  const webStream = Readable.toWeb(nodeStream);
  const response = await fetch(`${serverUrl}/api/upload-video/${videoId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/octet-stream',
      'Content-Length': String(totalBytes),
      'X-TrackAI-Client': 'desktop',
    },
    body: webStream,
    duplex: 'half',
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Upload failed (${response.status}): ${text.slice(0, 200)}`);
  }

  return response.json();
}

async function uploadFileFromPath({
  serverUrl,
  filePath,
  employeeName,
  onProgress,
}) {
  const filename = path.basename(filePath);
  const init = await initUpload(serverUrl, filename, employeeName);
  await uploadVideoStream(serverUrl, init.video_id, filePath, onProgress);
  return {
    video_id: init.video_id,
    filename,
    original_filename: init.original_filename || filename,
    file_size: (await fs.promises.stat(filePath)).size,
  };
}

module.exports = {
  uploadFileFromPath,
};
