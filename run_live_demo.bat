@echo off
REM run_live_demo.bat - Live webcam ISL demo on Windows (uses .venv and the
REM approved INCLUDE-only baseline checkpoint).
setlocal
set "PROJECT_DIR=%~dp0"
if not exist "%PROJECT_DIR%.venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Run: powershell -ExecutionPolicy Bypass -File scripts\setup_runtime.ps1
    exit /b 1
)
"%PROJECT_DIR%.venv\Scripts\python.exe" "%PROJECT_DIR%scripts\demo.py" --config "%PROJECT_DIR%configs\default.yaml" --override "%PROJECT_DIR%configs\isl.yaml" --checkpoint "%PROJECT_DIR%models\baseline.pt" %*
