const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..', 'windows-runtime', 'dist');
const required = ['runtime-manifest.json', 'worker.py', 'python/python.exe', 'python/Lib/site-packages/torch/__init__.py', 'R3/infer.py', 'R3/ckpt/r3_long.safetensors', 'backend/r3_worker_wrapper.py', 'backend/floorplan_constraints.py'];
const missing = required.filter((relative) => !fs.existsSync(path.join(root, relative)));
if (missing.length) {
  console.error('Windows GPU runtime is incomplete. Run scripts/build-windows-gpu-runtime.ps1 first.');
  for (const item of missing) console.error(` - ${item}`);
  process.exit(1);
}
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'runtime-manifest.json'), 'utf8'));
if (!manifest.complete) process.exitCode = 1;
const weightBytes = fs.statSync(path.join(root, 'R3/ckpt/r3_long.safetensors')).size;
if (weightBytes < 1_000_000_000) process.exitCode = 1;
if (process.exitCode) console.error('Windows GPU runtime manifest/weight validation failed.');
else console.log(`Windows GPU runtime validated (${(weightBytes / 1024 ** 3).toFixed(2)} GiB weight).`);
