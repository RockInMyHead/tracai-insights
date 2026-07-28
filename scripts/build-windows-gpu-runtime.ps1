param(
  [Parameter(Mandatory=$true)][string]$R3Source,
  [Parameter(Mandatory=$true)][string]$R3LongWeight,
  [string]$Output = "windows-runtime/dist",
  [string]$PythonVersion = "3.11.9"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputPath = Join-Path $root $Output
$pythonZip = Join-Path $env:TEMP "python-$PythonVersion-embed-amd64.zip"
$getPip = Join-Path $env:TEMP "get-pip.py"
if (Test-Path $outputPath) { Remove-Item -Recurse -Force $outputPath }
New-Item -ItemType Directory -Force $outputPath, (Join-Path $outputPath "python"), (Join-Path $outputPath "R3\ckpt"), (Join-Path $outputPath "backend") | Out-Null
Invoke-WebRequest "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" -OutFile $pythonZip
Expand-Archive $pythonZip -DestinationPath (Join-Path $outputPath "python")
$pth = Get-ChildItem (Join-Path $outputPath "python") -Filter "python*._pth" | Select-Object -First 1
(Get-Content $pth.FullName) -replace '^#import site$', 'import site' | Set-Content $pth.FullName
Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip
$python = Join-Path $outputPath "python\python.exe"
& $python $getPip --no-warn-script-location
& $python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
& $python -m pip install --no-cache-dir xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu126
& $python -m pip install --no-cache-dir -r (Join-Path $root "windows-runtime\requirements-windows.txt") --extra-index-url https://download.pytorch.org/whl/cu126
robocopy $R3Source (Join-Path $outputPath "R3") /E /XD .git ckpt outputs __pycache__ | Out-Null
if ($LASTEXITCODE -gt 7) { throw "R3 copy failed: $LASTEXITCODE" }
Copy-Item $R3LongWeight (Join-Path $outputPath "R3\ckpt\r3_long.safetensors")
Copy-Item (Join-Path $root "windows-runtime\worker.py") (Join-Path $outputPath "worker.py")
robocopy (Join-Path $root "backend") (Join-Path $outputPath "backend") /E /XD tests tools __pycache__ gpu_worker_data outputs videos | Out-Null
if ($LASTEXITCODE -gt 7) { throw "Backend copy failed: $LASTEXITCODE" }
$manifest = Get-Content (Join-Path $root "windows-runtime\runtime-manifest.template.json") | ConvertFrom-Json
$manifest.complete = $true
$manifest | Add-Member -NotePropertyName built_at -NotePropertyValue (Get-Date).ToUniversalTime().ToString("o")
$manifest | Add-Member -NotePropertyName weight_sha256 -NotePropertyValue (Get-FileHash (Join-Path $outputPath "R3\ckpt\r3_long.safetensors") -Algorithm SHA256).Hash.ToLower()
$manifest | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $outputPath "runtime-manifest.json") -Encoding UTF8
& $python -c "import torch, fastapi, cv2, xformers; assert torch.__version__.startswith('2.10.0'); assert torch.version.cuda; print(torch.__version__, torch.version.cuda, xformers.__version__)"
Write-Host "Windows GPU runtime ready: $outputPath"
