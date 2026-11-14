# Starts the whole FetalAI demo (FastAPI backend + Next.js frontend) on a Windows laptop, GPU or CPU-only.
# Run it through start_fetalai.bat in the repo root. The first run installs everything (needs internet,
# about 1.5 GB download, 10-20 min); later runs start in under a minute and work offline.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Fail($msg) {
    Write-Host "`nERROR: $msg" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

# ---------------------------------------------------------------- 1. model files
Step "Checking model files"
$models = @("outputs\hc18_unet_best.pth", "outputs\plane_classifier_best.pth", "outputs\ctg_xgb.json")
foreach ($m in $models) {
    if (-not (Test-Path $m)) { Fail "Missing $m. Copy the outputs folder from the original laptop (the .pth files are not in git)." }
}
Write-Host "  all three models present"

# ---------------------------------------------------------------- 2. Python environment
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Step "Creating Python environment (first run only)"
    # PyTorch 2.5 supports Python 3.9-3.12, so look for one of those.
    $base = $null
    foreach ($v in @("3.11", "3.12", "3.10")) {
        try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $base = @("py", "-$v"); break } } catch {}
    }
    if (-not $base) {
        try {
            $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($ver -in @("3.10", "3.11", "3.12")) { $base = @("python") }
        } catch {}
    }
    if (-not $base) { Fail "Python 3.10, 3.11 or 3.12 not found. Install Python 3.11 from python.org (tick 'Add to PATH') and run again." }
    $exe = $base[0]; $args0 = @($base | Select-Object -Skip 1)
    & $exe @args0 -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create .venv" }

    Step "Installing Python packages (CPU PyTorch; works with or without a GPU)"
    & $VenvPy -m pip install --upgrade pip
    & $VenvPy -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch install failed (check the internet connection)" }
    & $VenvPy -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Fail "pip install -r requirements.txt failed" }
}

# ---------------------------------------------------------------- 3. Frontend
Step "Checking Node.js"
try { $nodeVer = (& node -v) } catch { Fail "Node.js not found. Install Node.js 20 LTS or newer from nodejs.org and run again." }
$major = [int]($nodeVer.TrimStart("v").Split(".")[0])
if ($major -lt 20) { Fail "Node.js $nodeVer is too old; install Node.js 20 LTS or newer." }
Write-Host "  node $nodeVer"

Push-Location frontend
if (-not (Test-Path "node_modules")) {
    Step "Installing frontend packages (first run only)"
    & npm ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "npm ci failed" }
}
if (-not (Test-Path ".next\BUILD_ID")) {
    Step "Building the frontend (first run only, 1-3 min)"
    & npm run build
    if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "npm run build failed" }
}
Pop-Location

# ---------------------------------------------------------------- 4. Ports
foreach ($port in 8000, 3000) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        Fail "Port $port is already in use. Close the old FetalAI windows (or restart the laptop) and run again."
    }
}

# ---------------------------------------------------------------- 5. Start both servers
Step "Starting backend (http://127.0.0.1:8000) and frontend (http://localhost:3000)"
Start-Process -FilePath "cmd.exe" -WorkingDirectory $Root -ArgumentList @(
    "/k", "title FetalAI backend && `"$VenvPy`" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --no-access-log"
)
Start-Process -FilePath "cmd.exe" -WorkingDirectory (Join-Path $Root "frontend") -ArgumentList @(
    "/k", "title FetalAI frontend && npm start -- -p 3000"
)

Step "Waiting for the models to load"
$ok = $false
for ($i = 0; $i -lt 90; $i++) {
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2
        if ($h.model_loaded) { $ok = $true; break }
    } catch {}
    Start-Sleep -Seconds 2
}
if (-not $ok) { Fail "Backend did not become healthy in 3 minutes; look at the 'FetalAI backend' window for the error." }
Write-Host ("  status: {0}   plane gate: {1}   CTG model: {2}" -f $h.status, $h.plane_model_loaded, $h.ctg_model_loaded)

for ($i = 0; $i -lt 30; $i++) {
    try { Invoke-WebRequest -Uri "http://127.0.0.1:3000" -UseBasicParsing -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Seconds 1 }
}

Start-Process "http://localhost:3000"
Write-Host "`nFetalAI is running at http://localhost:3000  (API docs: http://127.0.0.1:8000/docs)" -ForegroundColor Green
Write-Host "Demo images are in the demo_samples folder (see demo_samples\README.txt)."
Write-Host "To stop: close the two 'FetalAI backend' / 'FetalAI frontend' windows."
Read-Host "`nPress Enter to close this window (the servers keep running)"
