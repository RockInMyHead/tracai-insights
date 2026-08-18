const { app, BrowserWindow, Menu, dialog, shell, ipcMain } = require('electron');
const path = require('path');
const { createCameraImportService } = require('./cameraImport.cjs');
const { createDesktopLogService } = require('./desktopLogs.cjs');
const localCpuTracker = require('./localCpuTracker.cjs');
const localGpuRuntime = require('./localGpuRuntime.cjs');
const adminMirror = require('./adminMirror.cjs');
const { resolveServerUrl } = require('./serverConfig.cjs');

const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

function getServerUrl() {
  return resolveServerUrl({
    resourcesPath: process.resourcesPath,
    userDataPath: app.getPath('userData'),
  });
}

let mainWindow = null;
let cameraImportService = null;
let desktopLogs = null;
let processingMode = 'online';
let processingModeDetails = {};

const cameraImportSettings = {
  enabled: false,
  ownerName: 'Экшен-камера',
};

function broadcastToRenderer(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

const RETRYABLE_DESKTOP_STATUSES = new Set([408, 429, 500, 502, 503, 504]);

async function fetchDesktopJson(pathname, options = {}) {
  const attempts = Number(options.attempts) || 5;
  let lastError = null;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120_000);
    try {
      const response = await fetch(`${getServerUrl()}${pathname}`, {
        headers: {
          'X-TrackAI-Client': 'desktop',
        },
        signal: controller.signal,
      });
      if (!response.ok) {
        if (RETRYABLE_DESKTOP_STATUSES.has(response.status) && attempt < attempts - 1) {
          await new Promise((resolve) => setTimeout(resolve, 1500 * (attempt + 1)));
          continue;
        }
        const text = await response.text().catch(() => '');
        throw new Error(`Desktop API failed (${response.status}): ${text.slice(0, 200)}`);
      }
      return response.json();
    } catch (error) {
      lastError = error;
      const message = error instanceof Error ? error.message : String(error);
      const transient = error instanceof Error && error.name === 'AbortError'
        || /Desktop API failed \((408|429|500|502|503|504)\)/.test(message)
        || /fetch failed|network|timed out|aborted|ECONNRESET|ETIMEDOUT/i.test(message);
      if (transient && attempt < attempts - 1) {
        await new Promise((resolve) => setTimeout(resolve, 1500 * (attempt + 1)));
        continue;
      }
      throw error instanceof Error ? error : new Error(String(error));
    } finally {
      clearTimeout(timer);
    }
  }
  throw lastError instanceof Error ? lastError : new Error('Desktop API request failed');
}

function setupCameraImport() {
  /** video_id -> { resolve, timer } — next upload waits until renderer finishes analysis */
  const analysisGates = new Map();

  const clearAnalysisGate = (videoId, reason = 'done') => {
    const id = String(videoId || '');
    const gate = analysisGates.get(id);
    if (!gate) return false;
    clearTimeout(gate.timer);
    analysisGates.delete(id);
    gate.resolve({ videoId: id, reason });
    return true;
  };

  const waitForAnalysisGate = (videoId, timeoutMs = 45 * 60 * 1000) => {
    const id = String(videoId || '');
    if (!id) return Promise.resolve({ videoId: id, reason: 'missing-id' });
    clearAnalysisGate(id, 'superseded');
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        analysisGates.delete(id);
        desktopLogs?.log('warn', 'camera-import:analysis-gate-timeout', { video_id: id, timeoutMs });
        resolve({ videoId: id, reason: 'timeout' });
      }, timeoutMs);
      analysisGates.set(id, { resolve, timer });
    });
  };

  cameraImportService = createCameraImportService({
    serverUrl: getServerUrl(),
    importFile: async (input) => {
      if (processingMode === 'online') {
        try {
          const serverUrl = getServerUrl();
          const modulePath = require.resolve('./uploadFromPath.cjs');
          delete require.cache[modulePath];
          let uploadModule = require(modulePath);
          if (typeof uploadModule.uploadFileFromPath !== 'function') {
            delete require.cache[modulePath];
            uploadModule = require(modulePath);
          }
          if (typeof uploadModule.uploadFileFromPath !== 'function') {
            throw new Error(
              'uploadFileFromPath недоступен — перезапустите TrackAI Desktop',
            );
          }
          return await uploadModule.uploadFileFromPath({ serverUrl, ...input });
        } catch (error) {
          desktopLogs?.log('error', 'camera-import:online-upload-failed', {
            fileName: input.fileName,
            error: error instanceof Error ? error.message : String(error),
          });
          throw error;
        }
      }
      const copied = await localCpuTracker.copyToLocal(input);
      adminMirror.enqueueVideo(copied);
      void adminMirror.flush(getServerUrl(), (level, event, data) => desktopLogs?.log(level, event, data));
      return copied;
    },
    getOwnerName: () => cameraImportSettings.ownerName,
    isEnabled: () => cameraImportSettings.enabled,
    onStatus: (status) => broadcastToRenderer('camera-import:status', status),
    onProgress: (progress) => broadcastToRenderer('camera-import:progress', progress),
    onFileImported: async (uploaded) => {
      desktopLogs?.log('info', 'camera-import:file-imported', {
        video_id: uploaded.video_id,
        filename: uploaded.original_filename || uploaded.filename,
      });
      // Register gate before broadcast so a fast renderer notify cannot miss it.
      const gatePromise = waitForAnalysisGate(uploaded.video_id);
      broadcastToRenderer('camera-import:file-imported', uploaded);
      // Progressive pipeline: upload next camera file only after this video is analyzed.
      const gate = await gatePromise;
      desktopLogs?.log('info', 'camera-import:analysis-gate-released', {
        video_id: uploaded.video_id,
        reason: gate?.reason,
      });
    },
    onBatchComplete: (uploaded) => {
      desktopLogs?.log('info', 'camera-import:batch-complete', { count: uploaded.length });
      broadcastToRenderer('camera-import:complete', uploaded);
    },
    onError: (error) => {
      desktopLogs?.log('error', 'camera-import:error', error);
      broadcastToRenderer('camera-import:error', {
        message: error instanceof Error ? error.message : String(error),
      });
    },
    manualOnly: true,
  });

  ipcMain.handle('camera-import:get-settings', () => ({
    enabled: cameraImportSettings.enabled,
    ownerName: cameraImportSettings.ownerName,
  }));

  ipcMain.handle('camera-import:set-settings', (_event, settings = {}) => {
    if (typeof settings.enabled === 'boolean') {
      cameraImportSettings.enabled = settings.enabled;
      cameraImportService.setEnabled(settings.enabled);
    }
    if (typeof settings.ownerName === 'string') {
      cameraImportSettings.ownerName = settings.ownerName;
    }
    return {
      enabled: cameraImportSettings.enabled,
      ownerName: cameraImportSettings.ownerName,
    };
  });

  ipcMain.handle('camera-import:scan-now', async (_event, options = {}) => {
    const maxFiles = Number(options.maxFiles);
    return cameraImportService.scanNow({
      forceImport: Boolean(options.forceImport),
      ignoreImported: Boolean(options.ignoreImported),
      maxFiles: Number.isFinite(maxFiles) && maxFiles > 0 ? maxFiles : 0,
      fileNameIncludes: typeof options.fileNameIncludes === 'string' ? options.fileNameIncludes : null,
      analysisContext: options.analysisContext && typeof options.analysisContext === 'object'
        ? options.analysisContext
        : null,
    });
  });

  ipcMain.handle('camera-import:get-status', () => cameraImportService.getStatus());
  ipcMain.handle('camera-import:cancel', () => {
    cameraImportSettings.enabled = false;
    cameraImportService.setEnabled(false);
    for (const videoId of [...analysisGates.keys()]) {
      clearAnalysisGate(videoId, 'cancelled');
    }
    return cameraImportService.cancelImport();
  });
  ipcMain.handle('camera-import:reset-imported-state', () => cameraImportService.resetImportedState());
  ipcMain.handle('camera-import:notify-processed', (_event, videoId) => {
    const released = clearAnalysisGate(videoId, 'processed');
    return { ok: true, released };
  });
  ipcMain.handle('desktop-api:get-uploaded-videos', async () => {
    return fetchDesktopJson('/api/uploaded-videos');
  });
  ipcMain.handle('desktop-api:get-video-analysis', async (_event, videoId) => {
    return fetchDesktopJson(`/api/video/${encodeURIComponent(String(videoId))}`);
  });
  ipcMain.handle('desktop-api:get-processing-status', async (_event, videoId) => {
    return fetchDesktopJson(`/api/processing-status/${encodeURIComponent(String(videoId))}`);
  });

  ipcMain.handle('processing:resolve-mode', async () => {
    const gpu = await localGpuRuntime.start((level, event, data) => desktopLogs?.log(level, event, data));
    if (gpu.ready) {
      processingMode = 'local_gpu';
      processingModeDetails = {
        mode: processingMode,
        label: 'Локально',
        reason: null,
        gpu: gpu.gpu,
        minimumDriver: gpu.manifest?.minimum_nvidia_driver_windows,
      };
      desktopLogs?.log('info', 'processing:mode-resolved', {
        mode: processingMode,
        gpu_memory_mb: gpu.gpu?.memoryMb,
        cuda_runtime: gpu.health?.cuda_runtime,
      });
      return processingModeDetails;
    }
    processingMode = 'online';
    processingModeDetails = {
      mode: processingMode,
      label: 'Серверная GPU',
      reason: 'server_gpu_fallback',
    };
    desktopLogs?.log('info', 'processing:mode-resolved', {
      ...processingModeDetails,
      localGpuReason: gpu.reason,
      localGpuRoot: gpu.root,
    });
    return processingModeDetails;
  });
  ipcMain.handle('local-gpu:process', async (_event, video) => {
    const result = await localGpuRuntime.processVideo(video);
    adminMirror.enqueueResult(video.video_id, result.data);
    void adminMirror.flush(getServerUrl(), (level, event, data) => desktopLogs?.log(level, event, data));
    return result;
  });
  ipcMain.handle('local-gpu:history', () => localGpuRuntime.getHistory());
  ipcMain.handle('local-gpu:analysis', (_event, videoId) => localGpuRuntime.getAnalysis(videoId));
  ipcMain.handle('local-cpu:process', async (_event, video) => {
    const result = await localCpuTracker.processLocalVideo(video, (progress) => {
      broadcastToRenderer('local-cpu:progress', {
        video_id: video.video_id,
        filename: video.original_filename || video.filename,
        ...progress,
      });
    });
    adminMirror.enqueueResult(video.video_id, result.data);
    void adminMirror.flush(getServerUrl(), (level, event, data) => desktopLogs?.log(level, event, data));
    return result;
  });
  ipcMain.handle('local-cpu:history', () => localCpuTracker.getHistory());
  ipcMain.handle('local-cpu:analysis', (_event, videoId) => localCpuTracker.getAnalysis(videoId));

  cameraImportService.start();
}

function createMenu() {
  const template = [
    {
      label: 'TrackAI',
      submenu: [
        {
          label: 'О программе',
          click: () => dialog.showMessageBox(mainWindow, {
            type: 'info',
            title: 'TrackAI',
            message: `TrackAI Desktop v${app.getVersion()}`,
            detail: 'Production-анализ: R³ → robust graph → scale-aware → LingBot → план Kerama Marazzi.',
          }),
        },
        { type: 'separator' },
        { role: 'quit', label: 'Выход' },
      ],
    },
    {
      label: 'Вид',
      submenu: [
        { role: 'reload', label: 'Обновить' },
        { role: 'forceReload', label: 'Жёсткое обновление' },
        { type: 'separator' },
        { role: 'resetZoom', label: 'Сбросить масштаб' },
        { role: 'zoomIn', label: 'Увеличить' },
        { role: 'zoomOut', label: 'Уменьшить' },
        { role: 'togglefullscreen', label: 'Полный экран' },
      ],
    },
    {
      label: 'Помощь',
      submenu: [
        { label: 'Открыть TrackAI в браузере', click: () => shell.openExternal(getServerUrl()) },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
      preload: path.join(__dirname, 'preload.cjs'),
    },
    icon: path.join(__dirname, '../public/favicon.ico'),
    title: 'TrackAI - Анализ траектории движения',
    show: false,
    backgroundColor: '#f8fafc',
  });
  desktopLogs?.attachWindow(mainWindow);

  mainWindow.once('ready-to-show', () => {
    desktopLogs?.log('info', 'window:ready-to-show');
    mainWindow.show();
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (!url.startsWith(getServerUrl())) {
      shell.openExternal(url);
      return { action: 'deny' };
    }
    return { action: 'allow' };
  });

  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith(getServerUrl()) && !url.startsWith('http://localhost:8081') && !url.startsWith('file:')) {
      event.preventDefault();
      shell.openExternal(url);
    }
  });

  mainWindow.webContents.on('did-fail-load', (_event, errorCode, errorDescription, url, isMainFrame) => {
    if (!isMainFrame || errorCode === -3) return;
    desktopLogs?.log('error', 'window:did-fail-load', {
      errorCode,
      errorDescription,
      url,
    });
    dialog.showMessageBox(mainWindow, {
      type: 'error',
      title: 'TrackAI недоступен',
      message: 'Не удалось подключиться к серверу TrackAI.',
      detail: `${errorDescription}\n${url}`,
      buttons: ['Повторить', 'Закрыть'],
    }).then(({ response }) => {
      if (response === 0) mainWindow.loadFile(path.join(__dirname, '../dist/index.html'), { hash: '/trajectory?desktop=1' });
      else mainWindow.close();
    });
  });

  if (isDev) {
    mainWindow.loadURL('http://localhost:8081/trajectory?desktop=1');
  } else {
    // Интерфейс входит в дистрибутив. Поэтому он открывается и без сети,
    // а сетевой backend используется только когда проверка выбрала RTX 3090.
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'), { hash: '/trajectory?desktop=1' });
  }

  mainWindow.on('closed', () => { mainWindow = null; });
  createMenu();
}

app.whenReady().then(() => {
  desktopLogs = createDesktopLogService({
    app,
    dialog,
    getProcessingMode: () => processingMode,
  });
  ipcMain.handle('logs:download', () => desktopLogs.download(mainWindow));
  setupCameraImport();
  void adminMirror.flush(getServerUrl(), (level, event, data) => desktopLogs?.log(level, event, data));
  createWindow();
});

app.on('window-all-closed', () => {
  if (cameraImportService) {
    cameraImportService.stop();
  }
  localGpuRuntime.stop();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
