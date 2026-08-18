const fs = require('fs');
const http = require('http');
const https = require('https');
const path = require('path');
const { URL } = require('url');

const RETRYABLE_STATUSES = new Set([408, 429, 500, 502, 503, 504]);
// Мелкие чанки + native http pipe: без Web Streams bridge, иначе большие AVI → OOM.
const UPLOAD_CHUNK_BYTES = 64 * 1024;
const UPLOAD_TIMEOUT_MIN_MS = 3 * 60 * 1000;
const UPLOAD_TIMEOUT_MAX_MS = 45 * 60 * 1000;

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

function uploadTimeoutMs(totalBytes) {
  // ~256 KB/s worst-case + 90s slack for server ACK; clamp 3–45 min.
  const estimated = Math.ceil(Math.max(1, totalBytes) / UPLOAD_CHUNK_BYTES) * 1000 + 90_000;
  return Math.min(UPLOAD_TIMEOUT_MAX_MS, Math.max(UPLOAD_TIMEOUT_MIN_MS, estimated));
}

function mergeAbortSignals(userSignal, timeoutMs) {
  const timeoutSignal = typeof AbortSignal.timeout === 'function'
    ? AbortSignal.timeout(timeoutMs)
    : (() => {
      const controller = new AbortController();
      const timer = setTimeout(() => {
        controller.abort(new Error(`Upload timed out after ${Math.round(timeoutMs / 1000)}s`));
      }, timeoutMs);
      controller.signal.addEventListener('abort', () => clearTimeout(timer), { once: true });
      return controller.signal;
    })();

  if (!userSignal) return timeoutSignal;
  if (typeof AbortSignal.any === 'function') {
    return AbortSignal.any([userSignal, timeoutSignal]);
  }

  const merged = new AbortController();
  const forward = (signal) => {
    if (signal.aborted) {
      merged.abort(signal.reason || new Error('Upload cancelled'));
      return;
    }
    signal.addEventListener('abort', () => {
      merged.abort(signal.reason || new Error('Upload cancelled'));
    }, { once: true });
  };
  forward(userSignal);
  forward(timeoutSignal);
  return merged.signal;
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
  const initSignal = mergeAbortSignals(signal, 60_000);
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
      auto_analysis: false,
      map_context: {
        ...(analysisContext && typeof analysisContext === 'object' ? analysisContext : {}),
        auto_analysis: false,
      },
    }),
    signal: initSignal,
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

function postStream(urlString, headers, bodyStream, signal) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlString);
    const transport = url.protocol === 'https:' ? https : http;
    const req = transport.request(
      {
        protocol: url.protocol,
        hostname: url.hostname,
        port: url.port || undefined,
        path: `${url.pathname}${url.search}`,
        method: 'POST',
        headers,
        signal,
      },
      (res) => {
        const chunks = [];
        res.on('data', (chunk) => chunks.push(chunk));
        res.on('end', () => {
          resolve({
            statusCode: res.statusCode || 0,
            body: Buffer.concat(chunks),
          });
        });
        res.on('error', reject);
      },
    );

    req.on('error', reject);
    bodyStream.on('error', (error) => {
      req.destroy(error);
      reject(error);
    });
    bodyStream.pipe(req);
  });
}

async function uploadVideoStream(serverUrl, videoId, filePath, onProgress, signal) {
  const stat = await fs.promises.stat(filePath);
  const totalBytes = stat.size;
  let uploadedBytes = 0;
  let lastReportedPercent = -1;
  let lastReportAt = 0;
  let closed = false;
  const combinedSignal = mergeAbortSignals(signal, uploadTimeoutMs(totalBytes));

  const closeStream = () => {
    if (closed) return;
    closed = true;
    nodeStream.removeAllListeners('data');
    nodeStream.removeAllListeners('error');
    if (!nodeStream.destroyed) {
      nodeStream.destroy();
    }
  };

  const reportProgress = (force = false, { finalize = false } = {}) => {
    if ((closed && !finalize) || typeof onProgress !== 'function' || totalBytes <= 0) return;
    // До ответа сервера максимум 99% — иначе UI «висит» на 100% пока сервер не ACK.
    const raw = (uploadedBytes / totalBytes) * 100;
    const percent = finalize ? 100 : Math.min(99, raw);
    const now = Date.now();
    const jumped = percent - lastReportedPercent;
    if (
      !force
      && lastReportedPercent >= 0
      && jumped < 0.4
      && now - lastReportAt < 120
      && percent < 99
    ) {
      return;
    }
    lastReportedPercent = percent;
    lastReportAt = now;
    onProgress(percent);
  };

  const nodeStream = fs.createReadStream(filePath, { highWaterMark: UPLOAD_CHUNK_BYTES });
  nodeStream.on('data', (chunk) => {
    if (closed) return;
    uploadedBytes += chunk.length;
    reportProgress(false);
  });

  const abort = () => {
    const reason = combinedSignal.reason || signal?.reason || new Error('Upload cancelled');
    closeStream();
  };
  combinedSignal.addEventListener('abort', abort, { once: true });
  reportProgress(true);

  let response;
  try {
    response = await postStream(
      `${serverUrl}/api/upload-video/${videoId}`,
      {
        'Content-Type': 'application/octet-stream',
        'Content-Length': String(totalBytes),
        'X-TrackAI-Client': 'desktop',
      },
      nodeStream,
      combinedSignal,
    );
  } catch (error) {
    closeStream();
    if (combinedSignal.aborted && !signal?.aborted) {
      const timeoutError = new Error(
        `Upload timed out after ${Math.round(uploadTimeoutMs(totalBytes) / 1000)}s (network/server did not finish)`,
      );
      timeoutError.cause = error;
      throw timeoutError;
    }
    throw error;
  } finally {
    combinedSignal.removeEventListener('abort', abort);
    closeStream();
  }

  if (response.statusCode < 200 || response.statusCode >= 300) {
    const text = response.body.toString('utf8');
    throw new Error(`Upload failed (${response.statusCode}): ${text.slice(0, 200)}`);
  }

  uploadedBytes = totalBytes;
  reportProgress(true, { finalize: true });
  try {
    return JSON.parse(response.body.toString('utf8'));
  } catch {
    throw new Error('Upload response was not valid JSON');
  }
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
