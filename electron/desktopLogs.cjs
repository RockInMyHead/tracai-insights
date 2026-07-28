const fs = require('fs');
const path = require('path');

const MAX_LOG_FILE_BYTES = 128 * 1024 * 1024;
const ROTATE_LOG_AT_BYTES = 10 * 1024 * 1024;
const ROTATED_LOG_COUNT = 5;

function makeCrcTable() {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value & 1) ? (0xedb88320 ^ (value >>> 1)) : (value >>> 1);
    }
    table[index] = value >>> 0;
  }
  return table;
}

const CRC_TABLE = makeCrcTable();

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function dosDateTime(date = new Date()) {
  const year = Math.max(1980, date.getFullYear());
  return {
    time: (
      (date.getHours() << 11)
      | (date.getMinutes() << 5)
      | Math.floor(date.getSeconds() / 2)
    ),
    date: (
      ((year - 1980) << 9)
      | ((date.getMonth() + 1) << 5)
      | date.getDate()
    ),
  };
}

function safeArchivePath(value) {
  return String(value)
    .replaceAll('\\', '/')
    .split('/')
    .filter((part) => part && part !== '.' && part !== '..')
    .join('/');
}

function createStoredZip(entries) {
  const localChunks = [];
  const centralChunks = [];
  let offset = 0;

  for (const entry of entries) {
    const name = Buffer.from(safeArchivePath(entry.name), 'utf8');
    const data = Buffer.isBuffer(entry.data)
      ? entry.data
      : Buffer.from(String(entry.data), 'utf8');
    const checksum = crc32(data);
    const timestamp = dosDateTime(entry.mtime || new Date());

    const localHeader = Buffer.alloc(30);
    localHeader.writeUInt32LE(0x04034b50, 0);
    localHeader.writeUInt16LE(20, 4);
    localHeader.writeUInt16LE(0x0800, 6);
    localHeader.writeUInt16LE(0, 8);
    localHeader.writeUInt16LE(timestamp.time, 10);
    localHeader.writeUInt16LE(timestamp.date, 12);
    localHeader.writeUInt32LE(checksum, 14);
    localHeader.writeUInt32LE(data.length, 18);
    localHeader.writeUInt32LE(data.length, 22);
    localHeader.writeUInt16LE(name.length, 26);
    localHeader.writeUInt16LE(0, 28);
    localChunks.push(localHeader, name, data);

    const centralHeader = Buffer.alloc(46);
    centralHeader.writeUInt32LE(0x02014b50, 0);
    centralHeader.writeUInt16LE(20, 4);
    centralHeader.writeUInt16LE(20, 6);
    centralHeader.writeUInt16LE(0x0800, 8);
    centralHeader.writeUInt16LE(0, 10);
    centralHeader.writeUInt16LE(timestamp.time, 12);
    centralHeader.writeUInt16LE(timestamp.date, 14);
    centralHeader.writeUInt32LE(checksum, 16);
    centralHeader.writeUInt32LE(data.length, 20);
    centralHeader.writeUInt32LE(data.length, 24);
    centralHeader.writeUInt16LE(name.length, 28);
    centralHeader.writeUInt16LE(0, 30);
    centralHeader.writeUInt16LE(0, 32);
    centralHeader.writeUInt16LE(0, 34);
    centralHeader.writeUInt16LE(0, 36);
    centralHeader.writeUInt32LE(0, 38);
    centralHeader.writeUInt32LE(offset, 42);
    centralChunks.push(centralHeader, name);
    offset += localHeader.length + name.length + data.length;
  }

  const centralDirectory = Buffer.concat(centralChunks);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(0, 4);
  end.writeUInt16LE(0, 6);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(centralDirectory.length, 12);
  end.writeUInt32LE(offset, 16);
  end.writeUInt16LE(0, 20);
  return Buffer.concat([...localChunks, centralDirectory, end]);
}

async function collectFiles(rootPath, archiveRoot) {
  const files = [];
  let entries;
  try {
    entries = await fs.promises.readdir(rootPath, { withFileTypes: true });
  } catch {
    return files;
  }

  for (const entry of entries) {
    const absolutePath = path.join(rootPath, entry.name);
    const archivePath = safeArchivePath(path.posix.join(archiveRoot, entry.name));
    if (entry.isDirectory()) {
      files.push(...await collectFiles(absolutePath, archivePath));
      continue;
    }
    if (!entry.isFile()) continue;
    try {
      const stat = await fs.promises.stat(absolutePath);
      if (stat.size > MAX_LOG_FILE_BYTES) {
        files.push({
          name: `${archivePath}.skipped.txt`,
          data: `Файл пропущен: размер ${stat.size} байт превышает лимит ${MAX_LOG_FILE_BYTES} байт.`,
          mtime: stat.mtime,
        });
        continue;
      }
      files.push({
        name: archivePath,
        data: await fs.promises.readFile(absolutePath),
        mtime: stat.mtime,
      });
    } catch {
      // A log may rotate between readdir and readFile.
    }
  }
  return files;
}

function serializeLogValue(value) {
  if (value instanceof Error) {
    return { name: value.name, message: value.message, stack: value.stack };
  }
  if (typeof value === 'string') return value;
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return String(value);
  }
}

function timestampForFile(date = new Date()) {
  return date.toISOString().replaceAll(':', '-').replaceAll('.', '-');
}

function createDesktopLogService({ app, dialog, getProcessingMode }) {
  const logDirectory = path.join(app.getPath('userData'), 'logs');
  const logFile = path.join(logDirectory, 'trackai-desktop.log');
  let handlersInstalled = false;

  fs.mkdirSync(logDirectory, { recursive: true });

  function rotateLogsIfNeeded() {
    let size = 0;
    try {
      size = fs.statSync(logFile).size;
    } catch {
      return;
    }
    if (size < ROTATE_LOG_AT_BYTES) return;
    try {
      fs.rmSync(`${logFile}.${ROTATED_LOG_COUNT}`, { force: true });
      for (let index = ROTATED_LOG_COUNT - 1; index >= 1; index -= 1) {
        const source = `${logFile}.${index}`;
        const target = `${logFile}.${index + 1}`;
        if (fs.existsSync(source)) fs.renameSync(source, target);
      }
      fs.renameSync(logFile, `${logFile}.1`);
    } catch {
      // Rotation failure must not interrupt the application.
    }
  }

  function log(level, event, details = undefined) {
    const record = {
      timestamp: new Date().toISOString(),
      level,
      event,
      ...(details === undefined ? {} : { details: serializeLogValue(details) }),
    };
    try {
      rotateLogsIfNeeded();
      fs.appendFileSync(logFile, `${JSON.stringify(record)}\n`, 'utf8');
    } catch {
      // Logging must never crash the desktop application.
    }
  }

  function installProcessHandlers() {
    if (handlersInstalled) return;
    handlersInstalled = true;
    process.on('uncaughtExceptionMonitor', (error) => {
      log('error', 'process:uncaught-exception', error);
    });
    process.on('unhandledRejection', (reason) => {
      log('error', 'process:unhandled-rejection', reason);
    });
  }

  function attachWindow(window) {
    window.webContents.on('console-message', (_event, level, message, line, sourceId) => {
      log(level >= 3 ? 'error' : level === 2 ? 'warning' : 'info', 'renderer:console', {
        message,
        line,
        source: sourceId ? path.basename(sourceId) : null,
      });
    });
    window.webContents.on('render-process-gone', (_event, details) => {
      log('error', 'renderer:process-gone', details);
    });
    window.on('unresponsive', () => log('warning', 'renderer:unresponsive'));
    window.on('responsive', () => log('info', 'renderer:responsive'));
  }

  async function buildArchiveEntries() {
    log('info', 'logs:export-started');
    const roots = new Map();
    const configuredRoots = [
      ['logs', logDirectory],
      ['crash-dumps', app.getPath('crashDumps')],
    ];
    for (const [name, root] of configuredRoots) {
      const resolved = path.resolve(root);
      if (!roots.has(resolved)) roots.set(resolved, name);
    }

    const entries = [];
    for (const [root, archiveRoot] of roots) {
      entries.push(...await collectFiles(root, archiveRoot));
    }
    entries.push({
      name: 'diagnostics.json',
      data: JSON.stringify({
        exported_at: new Date().toISOString(),
        application: {
          name: app.getName(),
          version: app.getVersion(),
          packaged: app.isPackaged,
        },
        runtime: {
          platform: process.platform,
          architecture: process.arch,
          node: process.versions.node,
          electron: process.versions.electron,
          chrome: process.versions.chrome,
          processing_mode: typeof getProcessingMode === 'function'
            ? getProcessingMode()
            : null,
        },
        included_files: entries.map((entry) => entry.name),
      }, null, 2),
    });
    return entries;
  }

  async function download(parentWindow) {
    const result = await dialog.showSaveDialog(parentWindow, {
      title: 'Скачать логи TrackAI',
      defaultPath: path.join(
        app.getPath('downloads'),
        `TrackAI-logs-${timestampForFile()}.zip`,
      ),
      filters: [{ name: 'ZIP-архив', extensions: ['zip'] }],
      properties: ['createDirectory', 'showOverwriteConfirmation'],
    });
    if (result.canceled || !result.filePath) {
      log('info', 'logs:export-canceled');
      return { ok: false, canceled: true };
    }

    const entries = await buildArchiveEntries();
    const archive = createStoredZip(entries);
    const temporaryPath = `${result.filePath}.tmp-${process.pid}`;
    await fs.promises.mkdir(path.dirname(result.filePath), { recursive: true });
    await fs.promises.writeFile(temporaryPath, archive);
    await fs.promises.rename(temporaryPath, result.filePath);
    log('info', 'logs:export-completed', {
      fileCount: entries.length,
      archiveBytes: archive.length,
    });
    return {
      ok: true,
      canceled: false,
      filePath: result.filePath,
      fileName: path.basename(result.filePath),
      fileCount: entries.length,
      archiveBytes: archive.length,
    };
  }

  installProcessHandlers();
  log('info', 'app:log-service-ready', {
    version: app.getVersion(),
    platform: process.platform,
    architecture: process.arch,
  });

  return {
    log,
    attachWindow,
    download,
    buildArchiveEntries,
    logFile,
  };
}

module.exports = {
  crc32,
  createStoredZip,
  safeArchivePath,
  collectFiles,
  createDesktopLogService,
};
