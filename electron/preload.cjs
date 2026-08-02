const { contextBridge, shell, clipboard, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  platform: process.platform,
  versions: process.versions,
});

contextBridge.exposeInMainWorld('trackai', {
  isDesktop: true,
  version: process.env.npm_package_version || '1.18.2',
  serverUrl: 'http://93.189.231.189',
  processing: {
    resolveMode: () => ipcRenderer.invoke('processing:resolve-mode'),
  },
  logs: {
    download: () => ipcRenderer.invoke('logs:download'),
  },
  localCpu: {
    process: (video) => ipcRenderer.invoke('local-cpu:process', video),
    history: () => ipcRenderer.invoke('local-cpu:history'),
    analysis: (videoId) => ipcRenderer.invoke('local-cpu:analysis', videoId),
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
