const { contextBridge, shell, clipboard, ipcRenderer } = require('electron');

// Sandboxed Electron preload scripts cannot require local CommonJS modules.
// Keep this value aligned with electron/server.json and serverConfig.cjs.
const serverUrl = 'http://159.194.202.216';

contextBridge.exposeInMainWorld('electronAPI', {
  platform: process.platform,
  versions: process.versions,
});

contextBridge.exposeInMainWorld('trackai', {
  isDesktop: true,
  version: process.env.npm_package_version || '1.18.19',
  serverUrl,
  processing: {
    resolveMode: () => ipcRenderer.invoke('processing:resolve-mode'),
  },
  logs: {
    download: () => ipcRenderer.invoke('logs:download'),
  },
  desktopApi: {
    getUploadedVideos: () => ipcRenderer.invoke('desktop-api:get-uploaded-videos'),
    getVideoAnalysis: (videoId) => ipcRenderer.invoke('desktop-api:get-video-analysis', videoId),
    getProcessingStatus: (videoId) => ipcRenderer.invoke('desktop-api:get-processing-status', videoId),
  },
  localCpu: {
    process: (video) => ipcRenderer.invoke('local-cpu:process', video),
    history: () => ipcRenderer.invoke('local-cpu:history'),
    analysis: (videoId) => ipcRenderer.invoke('local-cpu:analysis', videoId),
    onProgress: (callback) => {
      const listener = (_event, payload) => callback(payload);
      ipcRenderer.on('local-cpu:progress', listener);
      return () => ipcRenderer.removeListener('local-cpu:progress', listener);
    },
  },
  localGpu: {
    process: (video) => ipcRenderer.invoke('local-gpu:process', video),
    history: () => ipcRenderer.invoke('local-gpu:history'),
    analysis: (videoId) => ipcRenderer.invoke('local-gpu:analysis', videoId),
  },
  openExternal: (url) => shell.openExternal(url),
  copyToClipboard: (text) => clipboard.writeText(String(text)),
  readFromClipboard: () => clipboard.readText(),
  cameraImport: {
    getSettings: () => ipcRenderer.invoke('camera-import:get-settings'),
    setSettings: (settings) => ipcRenderer.invoke('camera-import:set-settings', settings),
    scanNow: (options) => ipcRenderer.invoke('camera-import:scan-now', options),
    cancel: () => ipcRenderer.invoke('camera-import:cancel'),
    resetImportedState: () => ipcRenderer.invoke('camera-import:reset-imported-state'),
    notifyProcessed: (videoId) => ipcRenderer.invoke('camera-import:notify-processed', videoId),
    getStatus: () => ipcRenderer.invoke('camera-import:get-status'),
    onStatus: (callback) => {
      const listener = (_event, payload) => callback(payload);
      ipcRenderer.on('camera-import:status', listener);
      return () => ipcRenderer.removeListener('camera-import:status', listener);
    },
    onProgress: (callback) => {
      const listener = (_event, payload) => callback(payload);
      ipcRenderer.on('camera-import:progress', listener);
      return () => ipcRenderer.removeListener('camera-import:progress', listener);
    },
    onFileImported: (callback) => {
      const listener = (_event, payload) => callback(payload);
      ipcRenderer.on('camera-import:file-imported', listener);
      return () => ipcRenderer.removeListener('camera-import:file-imported', listener);
    },
    onComplete: (callback) => {
      const listener = (_event, payload) => callback(payload);
      ipcRenderer.on('camera-import:complete', listener);
      return () => ipcRenderer.removeListener('camera-import:complete', listener);
    },
    onError: (callback) => {
      const listener = (_event, payload) => callback(payload);
      ipcRenderer.on('camera-import:error', listener);
      return () => ipcRenderer.removeListener('camera-import:error', listener);
    },
  },
});
