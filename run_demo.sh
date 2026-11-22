#!/usr/bin/env bash
# ============================================================
# run_demo.sh - One-command launcher for ISL demo on Linux/macOS
# Checks Python >=3.10, creates venv, installs deps, runs demo
# Prints latency badge (FPS) on exit
# Works offline after first run (deps cached in .venv)
# ============================================================

set -euo pipefail

# --- Configuration ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
REQUIREMENTS="${PROJECT_DIR}/requirements.txt"
DEMO_SCRIPT="${PROJECT_DIR}/scripts/real_world_test.py"
DEMO_PREDICTIONS="${PROJECT_DIR}/demo/cached_predictions_full.json"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

echo
echo "============================================================"
echo "  ISL Sign Language Translation - Demo Launcher (Linux/macOS)"
echo "============================================================"
echo

# --- Step 1: Check Python version >= 3.10 ---
log_info "Checking Python version..."

if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        log_error "Python not found in PATH."
        echo "Please install Python 3.10 or higher."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Get version using Python itself for accuracy
PY_VERSION=$(${PYTHON_CMD} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(${PYTHON_CMD} -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(${PYTHON_CMD} -c "import sys; print(sys.version_info.minor)")

if (( PY_MAJOR < 3 )) || (( PY_MAJOR == 3 && PY_MINOR < 10 )); then
    log_error "Python 3.10+ required, found ${PY_VERSION}"
    exit 1
fi
log_ok "Python ${PY_VERSION} (>= 3.10)"

# --- Step 2: Create virtual environment if missing ---
echo
log_info "Checking virtual environment..."

if [[ ! -d "${VENV_DIR}" ]]; then
    log_info "Creating virtual environment in .venv..."
    ${PYTHON_CMD} -m venv "${VENV_DIR}"
    if [[ $? -ne 0 ]]; then
        log_error "Failed to create virtual environment"
        exit 1
    fi
else
    log_ok "Virtual environment exists"
fi

# --- Step 3: Activate venv and upgrade pip ---
echo
log_info "Activating virtual environment and upgrading pip..."
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip --quiet
if [[ $? -ne 0 ]]; then
    log_warn "Failed to upgrade pip, continuing anyway..."
fi

# --- Step 4: Install pinned dependencies ---
echo
log_info "Installing pinned dependencies from requirements.txt..."

if [[ ! -f "${REQUIREMENTS}" ]]; then
    log_error "requirements.txt not found at ${REQUIREMENTS}"
    exit 1
fi

"${VENV_DIR}/bin/pip" install -r "${REQUIREMENTS}" --quiet
if [[ $? -ne 0 ]]; then
    log_error "Failed to install dependencies"
    exit 1
fi
log_ok "Dependencies installed (cached for offline use)"

# --- Step 5: Run demo and measure latency ---
echo
log_info "Running demo mode..."
echo "============================================================"
echo

START_TIME=$(date +%s.%N)

"${VENV_DIR}/bin/python" "${DEMO_SCRIPT}" --demo --demo-predictions "${DEMO_PREDICTIONS}" --no-tts
DEMO_EXIT_CODE=$?

END_TIME=$(date +%s.%N)
ELAPSED_SEC=$(echo "${END_TIME} - ${START_TIME}" | bc -l)

# Get clip count from demo predictions for FPS calculation
CLIP_COUNT=$("${VENV_DIR}/bin/python" -c "import json; print(len(json.load(open('${DEMO_PREDICTIONS}'))))" 2>/dev/null || echo "0")

if [[ "${CLIP_COUNT}" -gt 0 ]]; then
    FPS=$(echo "scale=2; ${CLIP_COUNT} / ${ELAPSED_SEC}" | bc -l)
    echo
    echo "============================================================"
    echo "  LATENCY BADGE"
    echo "============================================================"
    printf "  Clips processed: %s\n" "${CLIP_COUNT}"
    printf "  Elapsed time:    %.2f seconds\n" "${ELAPSED_SEC}"
    printf "  Throughput:      %.2f clips/sec (FPS)\n" "${FPS}"
    echo "============================================================"
fi

echo
if [[ ${DEMO_EXIT_CODE} -eq 0 ]]; then
    log_ok "Demo completed successfully!"
else
    log_error "Demo exited with code ${DEMO_EXIT_CODE}"
fi

exit ${DEMO_EXIT_CODE}