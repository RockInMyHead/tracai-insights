const fs = require('fs');
const path = require('path');
const { Readable } = require('stream');

const RETRYABLE_STATUSES = new Set([408, 429, 500, 502, 503, 504]);

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    const abort = () => {
      clearTimeout(timer);
      reject(signal?.reason || new Error('Upload cancelled'));
    };
    if (signal?.aborted) {
      abort();
      return;
    }
    signal?.addEventListener('abort', abort, { once: true });
  });
}

function isRetryableUploadError(error) {
  if (!error || error.name === 'AbortError') {
    return false;
  }
  const message = String(error.message || error);
  const statusMatch = message.match(/\((\d{3})\)/);
  if (statusMatch && RETRYABLE_STATUSES.has(Number(statusMatch[1]))) {
    return true;
  }
  return /fetch failed|network|socket|ECONNRESET|ETIMEDOUT|EPIPE|UND_ERR/i.test(message);
}

async function withUploadRetry(label, operation, { signal, attempts = 4 } = {}) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await operation(attempt);
    } catch (error) {
      lastError = error;
      if (signal?.aborted || attempt >= attempts || !isRetryableUploadError(error)) {
        throw error;
      }
      const delayMs = Math.min(15000, 1000 * 2 ** (attempt - 1));
      console.warn(`${label} failed, retrying in ${delayMs}ms (${attempt}/${attempts})`, error?.message || error);
      await sleep(delayMs, signal);
    }
  }
  throw lastError;
}

async function initUpload(serverUrl, filename, employeeName, signal, analysisContext = null) {
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
      map_context: analysisContext || null,
    }),
    signal,
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

async function uploadVideoStream(serverUrl, videoId, filePath, onProgress, signal) {
  const stat = await fs.promises.stat(filePath);
  const totalBytes = stat.size;
  let uploadedBytes = 0;

  const nodeStream = fs.createReadStream(filePath);
  const abort = () => nodeStream.destroy(signal?.reason || new Error('Upload cancelled'));
  signal?.addEventListener('abort', abort, { once: true });
  nodeStream.on('data', (chunk) => {
    uploadedBytes += chunk.length;
    if (typeof onProgress === 'function' && totalBytes > 0) {
      onProgress(Math.min(100, (uploadedBytes / totalBytes) * 100));
    }
  });

  const webStream = Readable.toWeb(nodeStream);
  let response;
  try {
    response = await fetch(`${serverUrl}/api/upload-video/${videoId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Length': String(totalBytes),
        'X-TrackAI-Client': 'desktop',
      },
      body: webStream,
      duplex: 'half',
      signal,
    });
  } finally {
    signal?.removeEventListener('abort', abort);
  }

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
  signal,
  analysisContext,
}) {
  const filename = path.basename(filePath);
  const init = await withUploadRetry(
    'Init upload',
    () => initUpload(serverUrl, filename, employeeName, signal, analysisContext),
    { signal, attempts: 3 },
  );
  const uploaded = await withUploadRetry(
    'Video upload',
    () => uploadVideoStream(serverUrl, init.video_id, filePath, onProgress, signal),
    { signal, attempts: 5 },
  );
  return {
    ...uploaded,
    video_id: init.video_id,
    filename,
    original_filename: init.original_filename || filename,
    file_size: (await fs.promises.stat(filePath)).size,
  };
}

module.exports = {
  uploadFileFromPath,
};
