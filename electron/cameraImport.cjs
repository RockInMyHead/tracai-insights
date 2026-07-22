const fs = require('fs');
const path = require('path');
const { app } = require('electron');
const { uploadFileFromPath } = require('./uploadFromPath.cjs');

const VIDEO_EXTENSIONS = new Set([
  '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.3gp', '.mts', '.m2ts',
]);

const CAMERA_SCAN_DIRS = ['DCIM', 'PRIVATE', 'MISC'];
const SYSTEM_VOLUME_NAMES = new Set([
  'Macintosh HD',
  'Recovery',
  'Preboot',
  'VM',
  'Data',
  'Time Machine Backups',
]);

const CAMERA_VOLUME_HINTS = [
  'general - audio',
  'no name',
  'gopro',
  'dji',
  'insta',
  'akaso',
  'sjcam',
  'action',
  'camera',
  'dcim',
];

function getStatePath() {
  return path.join(app.getPath('userData'), 'camera-import-state.json');
}

function loadState() {
  try {
    const raw = fs.readFileSync(getStatePath(), 'utf8');
    const parsed = JSON.parse(raw);
    return {
      imported: Array.isArray(parsed.imported) ? parsed.imported : [],
      lastVolumes: Array.isArray(parsed.lastVolumes) ? parsed.lastVolumes : [],
    };
  } catch {
    return { imported: [], lastVolumes: [] };
  }
}

function saveState(state) {
  fs.mkdirSync(path.dirname(getStatePath()), { recursive: true });
  fs.writeFileSync(getStatePath(), JSON.stringify(state, null, 2));
}

function fileFingerprint(filePath, stat) {
  return `${filePath}|${stat.size}|${Math.floor(stat.mtimeMs)}`;
}

function isVideoFileName(name) {
  const lower = name.toLowerCase();
  for (const ext of VIDEO_EXTENSIONS) {
    if (lower.endsWith(ext)) return true;
  }
  return false;
}

function getVolumeRoots() {
  if (process.platform === 'darwin') {
    try {
      return fs.readdirSync('/Volumes', { withFileTypes: true })
        .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
        .filter((entry) => !SYSTEM_VOLUME_NAMES.has(entry.name))
        .map((entry) => ({
          name: entry.name,
          rootPath: path.join('/Volumes', entry.name),
        }));
    } catch {
      return [];
    }
  }

  if (process.platform === 'win32') {
    const roots = [];
    for (let code = 68; code <= 90; code += 1) {
      const letter = String.fromCharCode(code);
      const rootPath = `${letter}:\\`;
      try {
        fs.accessSync(rootPath, fs.constants.F_OK);
        roots.push({ name: `${letter}:`, rootPath });
      } catch {
        /* drive not present */
      }
    }
    return roots;
  }

  const candidates = ['/media', '/run/media', '/mnt'];
  const roots = [];
  for (const base of candidates) {
    try {
      for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
        if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
        const rootPath = path.join(base, entry.name);
        if (process.platform === 'linux') {
          try {
            for (const sub of fs.readdirSync(rootPath, { withFileTypes: true })) {
              if (sub.isDirectory() && !sub.name.startsWith('.')) {
                roots.push({
                  name: sub.name,
                  rootPath: path.join(rootPath, sub.name),
                });
              }
            }
          } catch {
            roots.push({ name: entry.name, rootPath });
          }
        } else {
          roots.push({ name: entry.name, rootPath });
        }
      }
    } catch {
      /* path unavailable */
    }
  }
  return roots;
}

function looksLikeCameraVolume(volume) {
  const lowerName = volume.name.toLowerCase();
  if (CAMERA_VOLUME_HINTS.some((hint) => lowerName.includes(hint))) {
    return true;
  }

  for (const dirName of CAMERA_SCAN_DIRS) {
    try {
      if (fs.existsSync(path.join(volume.rootPath, dirName))) {
        return true;
      }
    } catch {
      /* ignore */
    }
  }

  return false;
}

function scanDirectory(dirPath, results, depth = 0) {
  if (depth > 6) return;

  let entries;
  try {
    entries = fs.readdirSync(dirPath, { withFileTypes: true });
  } catch {
    return;
  }

  for (const entry of entries) {
    if (entry.name.startsWith('.') || entry.name === 'System Volume Information') continue;

    const fullPath = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      scanDirectory(fullPath, results, depth + 1);
      continue;
    }

    if (!entry.isFile() || !isVideoFileName(entry.name)) continue;

    try {
      const stat = fs.statSync(fullPath);
      if (!stat.isFile() || stat.size <= 0) continue;
      results.push({
        path: fullPath,
        name: entry.name,
        size: stat.size,
        mtimeMs: stat.mtimeMs,
        fingerprint: fileFingerprint(fullPath, stat),
      });
    } catch {
      /* ignore unreadable file */
    }
  }
}

function scanCameraVolume(volume) {
  const results = [];
  const seen = new Set();

  for (const dirName of CAMERA_SCAN_DIRS) {
    const dirPath = path.join(volume.rootPath, dirName);
    if (!fs.existsSync(dirPath)) continue;
    const before = results.length;
    scanDirectory(dirPath, results);
    for (let i = before; i < results.length; i += 1) {
      seen.add(results[i].path);
    }
  }

  if (results.length === 0) {
    scanDirectory(volume.rootPath, results);
  } else {
    const extra = [];
    scanDirectory(volume.rootPath, extra);
    for (const file of extra) {
      if (!seen.has(file.path)) {
        results.push(file);
        seen.add(file.path);
      }
    }
  }

  results.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  return results;
}

function getVolumeStats(rootPath) {
  try {
    const statfs = fs.statfsSync ? fs.statfsSync(rootPath) : null;
    if (!statfs) return null;
    const blockSize = statfs.bsize || statfs.frsize || 0;
    const total = (statfs.blocks || 0) * blockSize;
    const free = (statfs.bavail || 0) * blockSize;
    if (!total) return null;
    return { totalBytes: total, freeBytes: free };
  } catch {
    return null;
  }
}

function scanConnectedCameraVolumes() {
  return getVolumeRoots()
    .filter(looksLikeCameraVolume)
    .map((volume) => {
      const files = scanCameraVolume(volume);
      const stats = getVolumeStats(volume.rootPath);
      return {
        name: volume.name,
        rootPath: volume.rootPath,
        fileCount: files.length,
        totalBytes: files.reduce((sum, file) => sum + file.size, 0),
        freeBytes: stats?.freeBytes ?? null,
        totalVolumeBytes: stats?.totalBytes ?? null,
        files,
      };
    });
}

function createCameraImportService(options) {
  const {
    serverUrl,
    getOwnerName,
    isEnabled,
    onStatus,
    onProgress,
    onBatchComplete,
    onError,
  } = options;

  let pollTimer = null;
  let importInProgress = false;
  let currentStatus = {
    enabled: false,
    scanning: false,
    importing: false,
    volumes: [],
    pendingFiles: [],
    lastError: null,
  };

  const emitStatus = () => {
    if (typeof onStatus === 'function') {
      onStatus({ ...currentStatus, volumes: [...currentStatus.volumes], pendingFiles: [...currentStatus.pendingFiles] });
    }
  };

  const getPendingFiles = (volumes, state) => {
    const imported = new Set(state.imported);
    const pending = [];
    for (const volume of volumes) {
      for (const file of volume.files) {
        if (!imported.has(file.fingerprint)) {
          pending.push({ ...file, volumeName: volume.name });
        }
      }
    }
    return pending;
  };

  const importFiles = async (files, ownerName) => {
    if (!files.length || importInProgress) return [];
    if (!ownerName?.trim()) {
      const error = new Error('Укажите имя сотрудника для автоимпорта с камеры');
      currentStatus.lastError = error.message;
      emitStatus();
      if (typeof onError === 'function') onError(error);
      return [];
    }

    importInProgress = true;
    currentStatus.importing = true;
    currentStatus.lastError = null;
    emitStatus();

    const state = loadState();
    const uploaded = [];

    try {
      for (let index = 0; index < files.length; index += 1) {
        const file = files[index];
        if (typeof onProgress === 'function') {
          onProgress({
            index,
            total: files.length,
            fileName: file.name,
            filePath: file.path,
            percent: 0,
            phase: 'uploading',
          });
        }

        const result = await uploadFileFromPath({
          serverUrl,
          filePath: file.path,
          employeeName: ownerName.trim(),
          onProgress: (percent) => {
            if (typeof onProgress === 'function') {
              onProgress({
                index,
                total: files.length,
                fileName: file.name,
                filePath: file.path,
                percent,
                phase: 'uploading',
              });
            }
          },
        });

        state.imported.push(file.fingerprint);
        saveState(state);

        uploaded.push({
          ...result,
          ownerName: ownerName.trim(),
          sourcePath: file.path,
          volumeName: file.volumeName || null,
        });
      }

      if (typeof onBatchComplete === 'function' && uploaded.length > 0) {
        onBatchComplete(uploaded);
      }
      return uploaded;
    } catch (error) {
      currentStatus.lastError = error instanceof Error ? error.message : String(error);
      emitStatus();
      if (typeof onError === 'function') onError(error);
      throw error;
    } finally {
      importInProgress = false;
      currentStatus.importing = false;
      emitStatus();
    }
  };

  const scanNow = async ({ forceImport = false } = {}) => {
    if (currentStatus.scanning) return currentStatus;
    currentStatus.scanning = true;
    currentStatus.enabled = isEnabled();
    emitStatus();

    try {
      const volumes = scanConnectedCameraVolumes();
      const state = loadState();
      const pendingFiles = getPendingFiles(volumes, state);

      currentStatus.volumes = volumes.map(({ files, ...rest }) => rest);
      currentStatus.pendingFiles = pendingFiles;
      state.lastVolumes = volumes.map((volume) => volume.rootPath);
      saveState(state);

      if (isEnabled() && pendingFiles.length > 0 && (forceImport || !importInProgress)) {
        const ownerName = getOwnerName();
        if (ownerName?.trim()) {
          await importFiles(pendingFiles, ownerName);
        }
      }

      return currentStatus;
    } finally {
      currentStatus.scanning = false;
      emitStatus();
    }
  };

  const start = () => {
    if (pollTimer) return;
    pollTimer = setInterval(() => {
      scanNow().catch((error) => {
        currentStatus.lastError = error instanceof Error ? error.message : String(error);
        emitStatus();
      });
    }, 3000);
    scanNow().catch(() => {});
  };

  const stop = () => {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  };

  const setEnabled = (enabled) => {
    currentStatus.enabled = enabled;
    emitStatus();
    if (enabled) {
      scanNow({ forceImport: true }).catch(() => {});
    }
  };

  return {
    start,
    stop,
    scanNow,
    importFiles,
    setEnabled,
    getStatus: () => ({ ...currentStatus }),
  };
}

module.exports = {
  createCameraImportService,
  scanConnectedCameraVolumes,
  scanCameraVolume,
  looksLikeCameraVolume,
};
