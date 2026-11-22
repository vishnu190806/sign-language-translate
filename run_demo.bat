@echo off
REM ============================================================
REM run_demo.bat - One-command launcher for ISL demo on Windows
REM Checks Python >=3.10, creates venv, installs deps, runs demo
REM Prints latency badge (FPS) on exit
REM Works offline after first run (deps cached in .venv)
REM ============================================================

setlocal enabledelayedexpansion

REM --- Configuration ---
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%.venv"
set "REQUIREMENTS=%PROJECT_DIR%requirements.txt"
set "DEMO_SCRIPT=%PROJECT_DIR%scripts\real_world_test.py"
set "DEMO_PREDICTIONS=%PROJECT_DIR%demo\cached_predictions_full.json"

echo.
echo ============================================================
echo  ISL Sign Language Translation - Demo Launcher (Windows)
echo ============================================================
echo.

REM --- Step 1: Check Python version >= 3.10 ---
echo [1/5] Checking Python version...
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found in PATH.
    echo Please install Python 3.10 or higher from https://python.org
    exit /b 1
)

REM Parse version using Python for accuracy
for /f "delims=" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VERSION=%%i
for /f "tokens=1 delims=." %%a in ("%PY_VERSION%") do set PY_MAJOR=%%a
for /f "tokens=2 delims=." %%b in ("%PY_VERSION%") do set PY_MINOR=%%b

if %PY_MAJOR% LSS 3 (
    echo ERROR: Python 3.10+ required, found %PY_VERSION%
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
    echo ERROR: Python 3.10+ required, found %PY_VERSION%
    exit /b 1
)
echo       OK: Python %PY_VERSION% (>= 3.10)

REM --- Step 2: Create virtual environment if missing ---
echo.
echo [2/5] Checking virtual environment...
if not exist "%VENV_DIR%" (
    echo       Creating virtual environment in .venv...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        exit /b 1
    )
) else (
    echo       Virtual environment exists
)

REM --- Step 3: Activate venv and upgrade pip ---
echo.
echo [3/5] Activating virtual environment and upgrading pip...
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)

REM --- Step 4: Install pinned dependencies ---
echo.
echo [4/5] Installing pinned dependencies from requirements.txt...
if not exist "%REQUIREMENTS%" (
    echo ERROR: requirements.txt not found at %REQUIREMENTS%
    exit /b 1
)
"%VENV_DIR%\Scripts\pip" install -r "%REQUIREMENTS%" --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)
echo       Dependencies installed (cached for offline use)

REM --- Step 5: Run demo and measure latency ---
echo.
echo [5/5] Running demo mode...
echo ============================================================
echo.

REM Use Python to measure time and calculate FPS - single line command
"%VENV_DIR%\Scripts\python" -c "import time, subprocess, json, sys; demo_script=r'%DEMO_SCRIPT%'; demo_predictions=r'%DEMO_PREDICTIONS%'; clip_count=len(json.load(open(demo_predictions))); print(f'Clips to process: {clip_count}'); print(); start=time.perf_counter(); result=subprocess.run([sys.executable, demo_script, '--demo', '--demo-predictions', demo_predictions, '--no-tts']); elapsed=time.perf_counter()-start; fps=clip_count/elapsed if elapsed>0 else 0; print(); print('='*56); print('  LATENCY BADGE'); print('='*56); print(f'  Clips processed: {clip_count}'); print(f'  Elapsed time:    {elapsed:.2f} seconds'); print(f'  Throughput:      {fps:.2f} clips/sec (FPS)'); print('='*56); sys.exit(result.returncode)"
set DEMO_EXIT_CODE=%ERRORLEVEL%

echo.
if %DEMO_EXIT_CODE% EQU 0 (
    echo Demo completed successfully!
) else (
    echo Demo exited with code %DEMO_EXIT_CODE%
)

exit /b %DEMO_EXIT_CODE%