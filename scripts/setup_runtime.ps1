param(
    [switch]$SkipVenv,
    [switch]$WithTTS,
    [switch]$SkipAssets
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

function Write-Step([string]$Message) {
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

Write-Step 'ISL translation runtime setup'
Write-Host 'Project: sign-language-translate'
Write-Host 'This script sets up the runtime only. It does NOT download datasets,'
Write-Host 'does NOT download gated resources, and does NOT train any model.'

Write-Step '1/5 Checking Python'
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error 'Python was not found on PATH. Install Python 3.10-3.12 from https://www.python.org/downloads/ and re-run.'
    exit 1
}
$pyVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$pyParsed = [version]::Parse($pyVersion)
if ($pyParsed -lt [version]'3.10' -or $pyParsed -gt [version]'3.12') {
    Write-Error "Python $pyVersion detected. This project targets Python 3.10-3.12 (3.12 recommended)."
    exit 1
}
Write-Host "Python $pyVersion OK"

Write-Step '2/5 Virtual environment'
$venvDir = Join-Path $ProjectDir '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'
if ($SkipVenv) {
    $venvPython = 'python'
    Write-Host 'Skipping venv creation (use system Python).'
}
elseif (-not (Test-Path $venvPython)) {
    Write-Host "Creating $venvDir"
    & python -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { Write-Error 'python -m venv failed.'; exit 1 }
}
else {
    Write-Host "Reusing existing environment at $venvDir"
}

Write-Step '3/5 Core dependencies'
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $ProjectDir 'requirements.txt')
if ($LASTEXITCODE -ne 0) { Write-Error 'Core dependency installation failed.'; exit 1 }
if ($WithTTS) {
    & $venvPython -m pip install -r (Join-Path $ProjectDir 'requirements-tts.txt')
    if ($LASTEXITCODE -ne 0) { Write-Error 'TTS dependency installation failed.'; exit 1 }
    Write-Host 'Kokoro TTS enabled. Weights download on first synthesis (Apache-2.0).'
}
else {
    Write-Host 'TTS dependencies not installed. The app runs text-only; re-run with -WithTTS for speech.'
}

if (-not $SkipAssets) {
    Write-Step '4/5 MediaPipe runtime models'
    $assets = @(
        @{
            Url  = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
            Path = 'assets\hand_landmarker.task'
            Sha  = 'FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1'
        },
        @{
            Url  = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
            Path = 'assets\pose_landmarker.lite.task'
            Sha  = '59929E1D1EE95287735DDD833B19CF4AC46D29BC7AFDDBBF6753C459690D574A'
        }
    )
    foreach ($asset in $assets) {
        $dest = Join-Path $ProjectDir $asset.Path
        if (-not (Test-Path (Split-Path -Parent $dest))) {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
        }
        if (Test-Path $dest) {
            $actual = (Get-FileHash -Algorithm SHA256 -Path $dest).Hash
            if ($actual -eq $asset.Sha) {
                Write-Host "Verified (unchanged): $($asset.Path)"
                continue
            }
            Write-Error "Checksum mismatch for $($asset.Path). Expected $($asset.Sha), got $actual. Remove the file manually and re-run, or verify the source URL."
            exit 1
        }
        Write-Host "Downloading $($asset.Path) ..."
        Invoke-WebRequest -Uri $asset.Url -OutFile $dest
        $actual = (Get-FileHash -Algorithm SHA256 -Path $dest).Hash
        if ($actual -ne $asset.Sha) {
            Write-Error "Downloaded $($asset.Path) failed SHA-256 verification. Expected $($asset.Sha), got $actual. Remove the file and re-run."
            exit 1
        }
        Write-Host "Verified: $($asset.Path)"
    }
}
else {
    Write-Step '4/5 MediaPipe runtime models'
    Write-Host 'Skipped (-SkipAssets).'
}

Write-Step '5/5 Checkpoint check'
$ckpt = 'models\multihead_v2\best.pt'
$alt = 'models\baseline.pt'
if (Test-Path (Join-Path $ProjectDir $ckpt)) {
    Write-Host "Found default checkpoint: $ckpt"
}
elseif (Test-Path (Join-Path $ProjectDir $alt)) {
    Write-Host "Found bundled checkpoint: $alt"
    Write-Host 'Start the live demo with: '
    Write-Host '  .\.venv\Scripts\python.exe scripts\demo.py --config configs\default.yaml --override configs\isl.yaml --checkpoint models\baseline.pt'
}
else {
    Write-Warning 'No inference checkpoint found. See docs/MODEL_AND_DATA_AVAILABILITY.md for the distribution decision and acquisition options.'
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host 'Smoke test (no TTS):'
Write-Host '  .\.venv\Scripts\python.exe scripts\demo.py --config configs\default.yaml --override configs\isl.yaml --checkpoint models\baseline.pt --no-tts'
Write-Host 'With speech (after -WithTTS):'
Write-Host '  .\.venv\Scripts\python.exe scripts\demo.py --config configs\default.yaml --override configs\isl.yaml --checkpoint models\baseline.pt'
Write-Host 'Research evidence: docs/research/README.md'
