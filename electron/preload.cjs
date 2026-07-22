const { contextBridge, shell, clipboard } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  platform: process.platform,
  versions: process.versions,
});

contextBridge.exposeInMainWorld('trackai', {
  isDesktop: true,
  version: process.env.npm_package_version || '1.18.1',
  serverUrl: 'http://93.189.231.189',
  openExternal: (url) => shell.openExternal(url),
  copyToClipboard: (text) => clipboard.writeText(String(text)),
  readFromClipboard: () => clipboard.readText(),
});
