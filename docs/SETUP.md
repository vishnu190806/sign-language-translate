# Setup Guide

Step-by-step guide to set up the ISL translation system from scratch on Windows.

## Prerequisites

- Windows 10/11
- Python 3.12 (recommended)
- NVIDIA GPU with CUDA support (RTX 4070 or better recommended)
- Webcam (USB or built-in)
- 16GB RAM minimum

## 1. Clone Repository

```bash
git clone https://github.com/vishnu190806/sign-language-translate.git
cd sign-language-translate
```

## 2. Create Virtual Environment

```bash
python -m venv .venv
.\venv\Scripts\Activate.ps1
```

> All demo launchers (`run_demo.bat`, `run_demo.sh`, `run_live_demo.bat`) target `.venv`.

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements (requirements.txt)

Install the pinned runtime dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins: torch, torchvision, opencv-python, mediapipe, piper-tts, numpy, pandas, pyyaml, tensorboard, pytest, sounddevice, soundfile, plus ruff, mypy, matplotlib for development.

Optional workflows need extra packages not in `requirements.txt` (install as needed):
- Dataset download/conversion: `h5py`, `huggingface_hub`, `scikit-learn` → `pip install -r requirements-datasets.txt`
- LLM sentence polish (optional Qwen fallback): `transformers`, `llama-cpp-python` → `pip install -r requirements-llm.txt`

## 4. Download MediaPipe Models

Download the required MediaPipe task files:

```bash
# Hand landmark detection
curl -L -o assets/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# Pose landmark detection (lite version)
curl -L -o assets/pose_landmarker.lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

Or download manually from [MediaPipe Models](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).

## 5. Speech output (Kokoro default, Piper fallback)
`configs/default.yaml` sets `tts.engine: kokoro` with voice `af_heart`. Kokoro weights are Apache-2.0 and download from HuggingFace on first synthesis; they are not in Git. Install them with:
```bash
pip install -r requirements-tts.txt
```
Without this optional file the app still runs: `src/app/demo.py:174-175` disables speech with a warning, or use `--no-tts` explicitly. Kokoro steady-state synthesis ≈ Piper after warmup (~0.13s per 4.2s audio); first utterance loads weights on the playback thread, so video never stalls.

Piper `en_US-lessac-medium` is a **fallback** engine only. Its ONNX is not committed: the voice derives from the Blizzard 2013 Lessac corpus, whose licence is research-only and forbids redistribution without separate permission (the Piper project confirms no redistribution grant). If you have independently arranged rights, place `<voice>.onnx[.json]` under `assets/voices/en/en_US/lessac/medium/` and set `tts.engine: piper`. Do not assume the `piper-tts` package licence covers the voice.

For an automated Windows setup covering venv, pip, and SHA-256-verified MediaPipe downloads, run:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_runtime.ps1 -WithTTS
```

## 6. Download Dataset

### INCLUDE (core set, CC-BY-4.0)
Videos: Zenodo record 4010759 (https://zenodo.org/records/4010759, ~57GB). Landmarks for the full set already live in `data/processed/landmarks/isl/` (4,284 `.npz`) with `data/processed/landmarks/isl_label_map.json` (264 keys) — re-download only if missing.

> There is no `scripts/download_datasets.py` in this repo (older docs referenced it). Use the Zenodo link above or the HuggingFace mirror `ai4bharat/INCLUDE`.

### ISL500/ISL-DATA (diversity set, research/academic-only, gated)
1. Free HuggingFace account + token, and accept terms at https://huggingface.co/datasets/ISL500/ISL-DATA.
2. Login: `.\venv\Scripts\python.exe -c "from huggingface_hub import login; login('<token>')"`
3. Selective download (180 INCLUDE-overlap words × 15 users, resumable):
```bash
.\venv\Scripts\python.exe scripts/download_isl500.py --dest data/raw/isl500
```
4. Convert to training `.npz`:
```bash
.\venv\Scripts\python.exe scripts/convert_isl500.py --src data/raw/isl500 --dst data/processed/landmarks/isl500
```
Inspect an unknown `.h5` layout first with `--inspect`. See `docs/DATASETS.md` for the full dataset strategy and citations.

### Class Count Note (resolved)
The original `models/label_map.json` has **264 keys** with max index **263** (two keys normalize to `extra`). Joint/multi-head training uses `data/processed/landmarks/joint_label_map.json`: 263 normalized words, contiguous indices 0–262.

## 7. Extract Landmarks

Extract MediaPipe landmarks from videos:

```bash
python scripts/extract_landmarks.py --dataset isl --input-dir data/raw/isl --output-dir data/processed/landmarks/isl
```

This creates `.npz` files in `data/processed/landmarks/isl/` containing pre-extracted features.

## 8. Train Model

### Train on INCLUDE-50 (Quick)
```bash
python scripts/train.py --config configs/default.yaml --override configs/isl_include50.yaml
```

### Train on Full INCLUDE (264 classes)
```bash
python scripts/train.py --config configs/default.yaml --override configs/isl.yaml
```

### Joint INCLUDE+ISL500 Training (263 normalized classes, union head)
```bash
python scripts/train.py --config configs/default.yaml --override configs/joint_v2.yaml --npz-dir data/processed/landmarks/joint --label-map data/processed/landmarks/joint_label_map.json
```

### Multi-Head Co-Training (deployment model, shared encoder + per-source heads)
```bash
python scripts/train_multihead.py --epochs 100 --batch-size 128 --patience 15
```

Model checkpoints are saved to `models/` (`models/joint*/`, `models/multihead/` respectively).

## 9. Run Demo

### Live Demo (Camera)
```bash
# Windows
run_demo.bat

# Linux/macOS
./run_demo.sh

# Or directly (default: multi-head generalist checkpoint):
```bash
python scripts/demo.py --config configs/default.yaml --checkpoint models/multihead/best.pt
# INCLUDE specialist fallback:
python scripts/demo.py --config configs/default.yaml --checkpoint models/baseline.pt
```

### Demo Controls
| Key | Action |
|-----|--------|
| Space | Speak current sentence |
| A | Toggle auto-speak on/off |
| C | Clear sentence |
| Backspace | Remove last word |
| Q | Quit |

### Replay Demo (Historical/Not Currently Available)
The `replay_demo.py` script referenced in older documentation does not exist in the current codebase. Use the live demo above for testing.

## 10. Evaluate Model

```bash
python scripts/evaluate.py --checkpoint models/multihead/best.pt --npz-dir data/processed/landmarks/joint --label-map data/processed/landmarks/joint_label_map.json --split test --output-dir results/
```

Per-source breakdown (INCLUDE vs ISL500 heads) and real-world routing eval:

```bash
python scripts/eval_joint_by_source.py --checkpoint models/multihead/best.pt --output results/multihead_by_source.json
python scripts/real_world_test.py --model-path models/multihead/best.pt --dataset-dir isl_40_test --output-dir results
```

Outputs:
- `results/metrics.json` — Overall metrics
- `results/per_class_metrics.csv` — Per-class precision/recall/F1
- `results/confusion_matrix.png` — Confusion matrix heatmap
- `results/accuracy_by_class.png` — Per-class accuracy chart

## 11. Benchmark Latency

```bash
python scripts/benchmark_latency.py --checkpoint models/multihead/best.pt --output docs/benchmarks/metrics.json
```

Measured reference (`docs/benchmarks/v2_baseline_metrics.json`, `models/best.pt` on RTX 4070 Laptop GPU): 55 FPS, p50 18.14ms, p95 19.79ms.

## Troubleshooting

### Camera Not Working
- Try different backend: `--backend msmf` or `--backend ""`
- Check camera permissions in Windows Settings
- Test with: `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`

### TTS Not Working
- Install PortAudio: `pip install sounddevice`
- Check audio device: `python -c "import sounddevice; print(sounddevice.query_devices())"`
- Ensure Piper voice is cached (step 5 above)
- Try different TTS engine: Piper voice ID

### CUDA Out of Memory
- Reduce batch size: `--batch-size 32`
- Use CPU: Device will auto-detect, or set `CUDA_VISIBLE_DEVICES=""`

### Import Errors
- Ensure virtual environment is activated
- Reinstall: `pip install -r requirements.txt --force-reinstall`

## Project Structure

```
sign-language-translate/
├── configs/              # YAML configuration files
│   ├── default.yaml      # Base (batch 256, 6 workers, model, processing, paths)
│   ├── isl.yaml          # Full INCLUDE (264 classes, lr 5e-4)
│   ├── isl_include50.yaml # INCLUDE-50 subset config
│   ├── joint.yaml / joint_v2.yaml  # INCLUDE+ISL500 union training
│   └── v2_tuned.yaml / m8_*.yaml   # Augmentation / adapter profiles
├── src/                  # Source code
│   ├── config.py         # Configuration system
│   ├── data/             # Dataset handling
│   ├── features/         # Feature extraction
│   ├── models/           # HTMA model
│   ├── train/            # Training loop
│   ├── infer/            # Inference pipeline
│   ├── tts/              # Text-to-speech
│   └── app/              # Demo application
├── scripts/              # CLI entry points
├── tests/                # Unit tests
├── docs/                 # Documentation
├── assets/               # MediaPipe models
├── data/                 # Dataset (gitignored)
└── models/               # Checkpoints (gitignored)
```
