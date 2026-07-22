const { app, BrowserWindow, Menu, dialog, shell } = require('electron');
const path = require('path');

const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
const APP_URL = 'http://93.189.231.189';
const DESKTOP_APP_URL = `${APP_URL}/trajectory?desktop=1`;

let mainWindow = null;

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

  mainWindow.once('ready-to-show', () => mainWindow.show());

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (!url.startsWith(APP_URL)) {
      shell.openExternal(url);
      return { action: 'deny' };
    }
    return { action: 'allow' };
  });

  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith(APP_URL) && !url.startsWith('http://localhost:8081')) {
      event.preventDefault();
      shell.openExternal(url);
    }
  });

  mainWindow.webContents.on('did-fail-load', (_event, errorCode, errorDescription, url, isMainFrame) => {
    if (!isMainFrame || errorCode === -3) return;
    dialog.showMessageBox(mainWindow, {
      type: 'error',
      title: 'TrackAI недоступен',
      message: 'Не удалось подключиться к серверу TrackAI.',
      detail: `${errorDescription}\n${url}`,
      buttons: ['Повторить', 'Закрыть'],
    }).then(({ response }) => {
      if (response === 0) mainWindow.loadURL(DESKTOP_APP_URL);
      else mainWindow.close();
    });
  });

  if (isDev) {
    mainWindow.loadURL('http://localhost:8081/trajectory?desktop=1');
  } else {
    mainWindow.loadURL(DESKTOP_APP_URL);
  }

  mainWindow.on('closed', () => { mainWindow = null; });
  createMenu();
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
