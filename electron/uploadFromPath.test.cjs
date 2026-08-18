const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { uploadFileFromPath } = require('./uploadFromPath.cjs');

test('desktop upload registers and streams the complete video to the server', async (context) => {
  const temporaryDirectory = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'trackai-upload-test-'));
  const videoPath = path.join(temporaryDirectory, 'desktop-e2e-source.AVI');
  const videoBytes = Buffer.from('trackai-desktop-video-smoke-test');
  await fs.promises.writeFile(videoPath, videoBytes);
  context.after(() => fs.promises.rm(temporaryDirectory, { recursive: true, force: true }));

  const requests = [];
  const server = http.createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    requests.push({
      method: request.method,
      url: request.url,
      headers: request.headers,
      body: Buffer.concat(chunks),
    });

    response.setHeader('Content-Type', 'application/json');
    if (request.url === '/api/init-upload') {
      response.end(JSON.stringify({
        success: true,
        video_id: 'desktop-e2e-video-id',
        original_filename: 'desktop-e2e-source.AVI',
      }));
      return;
    }
    if (request.url === '/api/upload-video/desktop-e2e-video-id') {
      response.end(JSON.stringify({ success: true }));
      return;
    }
    response.statusCode = 404;
    response.end(JSON.stringify({ detail: 'not found' }));
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise((resolve) => server.close(resolve)));

  const progress = [];
  const result = await uploadFileFromPath({
    serverUrl: `http://127.0.0.1:${server.address().port}`,
    filePath: videoPath,
    employeeName: 'Desktop test',
    onProgress: (value) => progress.push(value),
  });

  assert.equal(result.video_id, 'desktop-e2e-video-id');
  assert.equal(result.filename, 'desktop-e2e-source.AVI');
  assert.equal(result.file_size, videoBytes.length);
  assert.equal(requests.length, 2);

  const registration = requests[0];
  assert.equal(registration.method, 'POST');
  assert.equal(registration.headers['x-trackai-client'], 'desktop');
  assert.deepEqual(JSON.parse(registration.body.toString()), {
    filename: 'desktop-e2e-source.AVI',
    employee_name: 'Desktop test',
    client_source: 'camera_auto',
    auto_analysis: false,
    map_context: { auto_analysis: false },
  });

  const upload = requests[1];
  assert.equal(upload.method, 'POST');
  assert.equal(upload.headers['x-trackai-client'], 'desktop');
  assert.equal(Number(upload.headers['content-length']), videoBytes.length);
  assert.deepEqual(upload.body, videoBytes);
  assert.equal(progress.at(-1), 100);
});
