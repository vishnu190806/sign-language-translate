# AGENTS.md — Instructions for Coding Agents

## Project Overview
Real-time Indian Sign Language (ISL) translation: webcam → MediaPipe landmarks → HTMA model → word prediction → sentence assembly → Piper TTS speech.

## Python Environment
- Python 3.12 in `venv/` directory
- Activate: `.\venv\Scripts\Activate.ps1`
- Install deps: `.\venv\Scripts\pip install -r requirements.txt`

## Running Tests
```bash
.\venv\Scripts\python -m pytest tests/ -v --tb=short
```
All 257 tests must pass before any commit.

## Project Structure
```
src/
├── config.py           # YAML config loading with dataclasses
├── data/
│   ├── registry.py     # Dataset registry (add new languages here)
│   ├── download.py     # ISL INCLUDE dataset downloader
│   ├── metadata.py     # Metadata table builders, label maps
│   └── dataset.py      # PyTorch Dataset/DataLoader for .npz landmarks
├── features/
│   ├── spec.py         # FeatureSpec (168-dim layout)
│   ├── normalize.py    # Wrist/shoulder normalization
│   ├── extractor.py    # LandmarkExtractor (live) + OfflineExtractor
│   └── augment.py      # Landmark-space augmentations
├── models/
│   ├── common.py       # SinusoidalPE, RoPE, AttentionPooling
│   ├── htma.py         # HTMAClassifier (core model)
│   └── builder.py      # build_model() factory
├── train/
│   ├── trainer.py      # Training loop with AMP
│   └── metrics.py      # Top-1/5, macro accuracy, confusion matrix
├── infer/
│   ├── buffer.py       # RingBuffer for frame features
│   ├── gate.py         # StabilityGate for word emission
│   └── sentence.py     # SentenceAssembler
├── tts/
│   ├── base.py         # TTSEngine ABC
│   ├── piper_engine.py # Piper offline TTS
│   ├── factory.py      # create_tts_engine()
│   └── playback.py     # AudioPlaybackManager (non-blocking)
└── app/
    ├── camera.py       # ThreadedCamera
    ├── overlay.py      # OpenCV drawing helpers
    └── demo.py         # Main real-time demo loop
```

## Key Conventions
- All configs flow through `src/config.py` → `configs/default.yaml` + dataset overrides
- Feature vector is always 168 dims: 2×(21×3+1) hands + (13×3+1) pose
- Model input: `(B, T, 168)` with optional `(B, T)` attention mask
- No comments in code unless explicitly requested
- Use type hints everywhere
- Tests live in `tests/` with `test_` prefix

## Commands to Run After Changes
```bash
.\venv\Scripts\python -m pytest tests/ -v    # Always run tests
.\venv\Scripts\python -c "from src.models.builder import build_model; print('OK')"  # Import check
```

## Master Plan & Docs Discipline
- Master plan: the author's local implementation plan (not distributed with this repository). Read it before starting any task where available; append an addendum there when a milestone/task completes.
- After completing ANY task: update the relevant files in `docs/` (`project_status.md`, `RESULTS.md`, `QA_REPORT.md`, etc.) with measured results and the current date — docs must never lag the code.

## Dataset
- INCLUDE (AI4Bharat): 4,287 ISL videos, 263 word classes
- Download: `python scripts/download_datasets.py --dataset isl`
- Extract landmarks: `python scripts/extract_landmarks.py --dataset isl`
- Train: `python scripts/train.py --config configs/default.yaml --override configs/isl.yaml`
- Demo: `python scripts/demo.py`
