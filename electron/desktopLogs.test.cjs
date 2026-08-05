const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const {
  collectFiles,
  crc32,
  createStoredZip,
  safeArchivePath,
} = require('./desktopLogs.cjs');

test('crc32 matches the standard check value', () => {
  assert.equal(crc32(Buffer.from('123456789')), 0xcbf43926);
});

test('archive paths cannot escape the ZIP root', () => {
  assert.equal(safeArchivePath('../logs\\nested/../../app.log'), 'logs/nested/app.log');
});

test('stored ZIP contains valid UTF-8 files', async (context) => {
  const temporaryDirectory = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'trackai-logs-test-'));
  context.after(() => fs.promises.rm(temporaryDirectory, { recursive: true, force: true }));
  const archivePath = path.join(temporaryDirectory, 'logs.zip');
  const archive = createStoredZip([
    { name: 'logs/trackai.log', data: 'hello' },
    { name: 'диагностика.json', data: '{"ok":true}' },
  ]);
  await fs.promises.writeFile(archivePath, archive);

  assert.equal(archive.readUInt32LE(0), 0x04034b50);
  assert.equal(archive.readUInt32LE(archive.length - 22), 0x06054b50);
  assert.ok(archive.includes(Buffer.from('logs/trackai.log')));
  assert.ok(archive.includes(Buffer.from('диагностика.json')));
});

test('collectFiles recursively includes desktop logs', async (context) => {
  const temporaryDirectory = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'trackai-collect-test-'));
  context.after(() => fs.promises.rm(temporaryDirectory, { recursive: true, force: true }));
  await fs.promises.mkdir(path.join(temporaryDirectory, 'nested'));
  await fs.promises.writeFile(path.join(temporaryDirectory, 'desktop.log'), 'main');
  await fs.promises.writeFile(path.join(temporaryDirectory, 'nested', 'renderer.log'), 'renderer');

  const entries = await collectFiles(temporaryDirectory, 'logs');
  assert.deepEqual(
    entries.map((entry) => entry.name).sort(),
    ['logs/desktop.log', 'logs/nested/renderer.log'],
  );
});
