const { app, BrowserWindow, Menu, dialog, shell, ipcMain } = require('electron');
const path = require('path');
const { createCameraImportService } = require('./cameraImport.cjs');
const { createDesktopLogService } = require('./desktopLogs.cjs');
const localCpuTracker = require('./localCpuTracker.cjs');
const localGpuRuntime = require('./localGpuRuntime.cjs');
const adminMirror = require('./adminMirror.cjs');

const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
const APP_URL = 'http://93.189.231.189';
const DESKTOP_APP_URL = `${APP_URL}/trajectory?desktop=1`;

let mainWindow = null;
let cameraImportService = null;
let desktopLogs = null;
let processingMode = 'local_cpu';
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

function setupCameraImport() {
  cameraImportService = createCameraImportService({
    serverUrl: APP_URL,
    importFile: async (input) => {
      if (processingMode === 'online') {
        try {
          return await require('./uploadFromPath.cjs').uploadFileFromPath({ serverUrl: APP_URL, ...input });
        } catch (error) {
          processingMode = 'local_cpu';
          processingModeDetails = {
            mode: processingMode,
            label: 'Локально',
            onlineUploadError: error instanceof Error ? error.message : String(error),
          };
          desktopLogs?.log('warn', 'camera-import:online-upload-fallback-local', {
            fileName: input.fileName,
            error: processingModeDetails.onlineUploadError,
          });
        }
      }
      const copied = await localCpuTracker.copyToLocal(input);
      adminMirror.enqueueVideo(copied);
      void adminMirror.flush(APP_URL, (level, event, data) => desktopLogs?.log(level, event, data));
      return copied;
    },
    getOwnerName: () => cameraImportSettings.ownerName,
    isEnabled: () => cameraImportSettings.enabled,
    onStatus: (status) => broadcastToRenderer('camera-import:status', status),
    onProgress: (progress) => broadcastToRenderer('camera-import:progress', progress),
    onFileImported: (uploaded) => {
      desktopLogs?.log('info', 'camera-import:file-imported', {
        video_id: uploaded.video_id,
        filename: uploaded.original_filename || uploaded.filename,
      });
      broadcastToRenderer('camera-import:file-imported', uploaded);
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
    return cameraImportService.scanNow({
      forceImport: Boolean(options.forceImport),
    });
  });

  ipcMain.handle('camera-import:get-status', () => cameraImportService.getStatus());
  ipcMain.handle('camera-import:cancel', () => cameraImportService.cancelImport());
  ipcMain.handle('camera-import:reset-imported-state', () => cameraImportService.resetImportedState());

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
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 3500);
    try {
      const response = await fetch(`${APP_URL}/api/health`, { signal: controller.signal });
      processingMode = response.ok ? 'online' : 'local_cpu';
    } catch {
      processingMode = 'local_cpu';
    } finally {
      clearTimeout(timer);
    }
    processingModeDetails = {
      mode: processingMode,
      label: 'Локально',
      localGpuReason: gpu.reason,
      localGpuDetails: {
        reason: gpu.reason,
        gpus: gpu.gpus,
        minimumDriver: gpu.minimumDriver,
        driverUrl: gpu.driverUrl,
      },
    };
    desktopLogs?.log('info', 'processing:mode-resolved', processingModeDetails);
    return processingModeDetails;
  });
  ipcMain.handle('local-gpu:process', async (_event, video) => {
    const result = await localGpuRuntime.processVideo(video);
    adminMirror.enqueueResult(video.video_id, result.data);
    void adminMirror.flush(APP_URL, (level, event, data) => desktopLogs?.log(level, event, data));
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
    void adminMirror.flush(APP_URL, (level, event, data) => desktopLogs?.log(level, event, data));
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
        { label: 'Открыть TrackAI в браузере', click: () => shell.openExternal(APP_URL) },
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
    backgroundColor: '#07111f',
  });
  desktopLogs?.attachWindow(mainWindow);

  mainWindow.once('ready-to-show', () => {
    desktopLogs?.log('info', 'window:ready-to-show');
    mainWindow.show();
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (!url.startsWith(APP_URL)) {
      shell.openExternal(url);
      return { action: 'deny' };
    }
    return { action: 'allow' };
  });

  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith(APP_URL) && !url.startsWith('http://localhost:8081') && !url.startsWith('file:')) {
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
  void adminMirror.flush(APP_URL, (level, event, data) => desktopLogs?.log(level, event, data));
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
