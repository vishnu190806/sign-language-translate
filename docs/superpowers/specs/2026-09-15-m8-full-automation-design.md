# M8 Full End-to-End Automation — Design Spec

**Date:** 2026-09-15
**Status:** Approved
**Author:** Alpha Orchestrator

---

## Goal

Fully automate M8 (Real-World Accuracy Improvement) — from dataset extraction through training, evaluation, and documentation updates. No physical action required from the user. All 4,284 INCLUDE videos processed, all 4 model variants trained, all evaluations run, all docs updated with real metrics.

## Hardware

| Component | Spec |
|-----------|------|
| CPU | i9-14900HX (8P+16E cores, 24 threads) |
| GPU | RTX 4070 8GB (Ada Lovelace, CC 8.9) |
| RAM | 16GB 5600MT/s |
| Storage | 1TB NVMe SSD |
| OS | Windows 11 + WSL2 |
| CUDA | Installed on both Windows and WSL2 |

## Problem

PyTorch is currently CPU-only (`torch 2.14.0+cpu`). Training 200 epochs on CPU takes ~100 hours. With CUDA bfloat16 on RTX 4070, training takes ~10-25 minutes total.

---

## Architecture

### Data Pipeline
```
archive.zip (56.7 GB, 4,284 videos)
  → Extract to data/raw/isl/
  → scripts/extract_landmarks.py
  → data/processed/landmarks/isl/ (.npz files)
  → metadata.csv + label_map.json
```

### Training Pipeline
```
configs/default.yaml + configs/isl.yaml
  → src/data/dataset.py (with AugmentationPipeline)
  → src/train/trainer.py (AMP bfloat16, AdamW, cosine scheduler)
  → models/{baseline,augmented,adapter,student}.pt
```

### Evaluation Pipeline
```
models/*.pt + data/processed/landmarks/isl/
  → scripts/ablation_study.py (6 conditions)
  → scripts/create_signer_splits.py (LOSO)
  → scripts/evaluate_real_world.py (cross-dataset)
  → scripts/real_world_test.py (per-signer)
  → results/*.json + results/*.md
```

---

## Tasks (Execution Order)

### Phase 0: CUDA Setup
**Task 0.1: Install CUDA-enabled PyTorch**
- Uninstall CPU-only PyTorch
- Install with `--index-url https://download.pytorch.org/whl/cu126`
- Force Python to use RTX 4070 (not Intel iGPU) via Windows Graphics Settings
- Verify: `torch.cuda.is_available()` returns True, GPU name = "RTX 4070 Laptop GPU"

### Phase 1: Data Pipeline
**Task 1.1: Extract dataset**
- Extract `archive.zip` → `data/raw/isl/`
- Verify: 4,284 video files across 44 categories

**Task 1.2: Extract landmarks**
- Run `scripts/extract_landmarks.py --dataset isl`
- Output: `data/processed/landmarks/isl/*.npz`
- Verify: ~4,284 .npz files with 168-dim features

### Phase 2: Integration
**Task 2.1: Wire AugmentationPipeline into training**
- Modify `src/data/dataset.py`: add `augmentation_mode` parameter ("none"/"basic"/"full")
- When mode="full", use `AugmentationPipeline` instead of basic `LandmarkAugmentor`
- Add `augmentation:` section to `configs/default.yaml`

**Task 2.2: Create adapter training script**
- New file: `scripts/train_adapter.py`
- Loads base model, wraps with DSTAdapter, freezes base, trains adapter
- Args: `--base-checkpoint`, `--dataset-dir`, `--output-dir`, `--epochs`, `--adapter-dim`

**Task 2.3: Create distillation script**
- New file: `scripts/distill.py`
- Loads teacher model, creates smaller student, runs KD training
- Uses `src/train/distiller.py` Distiller class
- Args: `--teacher-checkpoint`, `--dataset-dir`, `--output-dir`, `--temperature`, `--alpha`

**Task 2.4: Add M8 configs**
- New file: `configs/m8_augmentation.yaml` — augmentation probabilities
- New file: `configs/m8_adapter.yaml` — adapter dimensions
- New file: `configs/m8_distill.yaml` — student model config

### Phase 3: Training
**Task 3.1: Train baseline model**
- `python scripts/train.py --config configs/default.yaml --override configs/isl.yaml`
- 100 epochs, lr=3e-4, batch_size=64, AMP bfloat16
- Output: `models/baseline.pt`
- Early stopping patience=10

**Task 3.2: Train augmented model**
- Same as baseline but with `augmentation_mode=full`
- Output: `models/augmented.pt`

**Task 3.3: Train DSTAdapter**
- `python scripts/train_adapter.py --base-checkpoint models/best.pt --epochs 50`
- Output: `models/adapter.pt`

**Task 3.4: Distill student**
- `python scripts/distill.py --teacher-checkpoint models/best.pt --epochs 100`
- Output: `models/student.pt`

### Phase 4: Evaluation
**Task 4.1: Generate calibration data**
- Extract clips from INCLUDE videos for 40 test words
- Output: `data/calibration/` with metadata CSV

**Task 4.2: Run ablation study**
- `python scripts/ablation_study.py --model-path models/baseline.pt --conditions baseline,augmentation,adapters,distillation,all`
- Output: `results/ablation/ablation_results.json`

**Task 4.3: Create LOSO splits**
- `python scripts/create_signer_splits.py --metadata data/processed/landmarks/isl/metadata.csv`
- Output: `data/splits/loso/{train,val,test}.json`

**Task 4.4: Run cross-dataset evaluation**
- `python scripts/evaluate_real_world.py`
- Output: `results/real_world_accuracy.json`

**Task 4.5: Run per-signer analysis**
- `python scripts/real_world_test.py --model-path models/best.pt`
- Output: `results/per_signer_accuracy.json`

### Phase 5: Documentation
**Task 5.1: Update RESULTS.md**
- Real ablation table with actual metrics
- Cross-dataset evaluation results
- Per-signer accuracy variance

**Task 5.2: Update LIMITATIONS.md**
- Real domain gap numbers
- LOSO protocol results
- Per-signer variance

**Task 5.3: Update M8_REAL_WORLD_IMPROVEMENT.md**
- Phase 4 tasks marked complete
- Real metrics in all tables
- Citations verified

**Task 5.4: Update project_status.md**
- M8 fully complete
- All metrics updated
- Progress to ~98%

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| PyTorch | 2.x+cu126 | CUDA 12.6 for RTX 4070 |
| AMP dtype | bfloat16 | Native Ada Lovelace Tensor Core support |
| Batch size | 64 | Fits in 8GB VRAM (~5% utilization) |
| Learning rate | 3e-4 | Standard for small Transformers on SLR |
| Optimizer | AdamW | Weight decay regularization |
| Weight decay | 1e-2 | Regularization for 4K samples |
| LR schedule | Cosine annealing | Smooth convergence |
| Warmup | 5-10 epochs | Prevents early instability |
| Epochs | 100 max | Converges at 40-60 for 4K samples |
| Early stopping | patience=10 | Stops when val loss plateaus |
| Gradient clipping | max_norm=1.0 | Prevents explosion |
| num_workers | 4 | Optimal for landmark data |
| pin_memory | True | Faster CPU→GPU transfer |
| persistent_workers | True | Avoids restart overhead |
| torch.compile | Yes | +10-30% speedup |
| cudnn.benchmark | True | Autotune conv algorithms |

## Expected Performance

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| INCLUDE test top-1 | 93.02% | ≥93% | ≥95% |
| Cross-dataset top-1 | 20.9% | ≥50% | ≥60% |
| ISL500 top-1 | 2.5% | ≥40% | ≥50% |
| Training time | ~100h (CPU) | ~15min (GPU) | ~10min |
| VRAM usage | N/A | ~500MB | <1GB |

## Time Budget

| Phase | Estimated Time |
|-------|---------------|
| CUDA setup | 5 min |
| Dataset extraction | 30-60 min |
| Landmark extraction | 1-2 hours |
| Integration (scripts + configs) | 30 min |
| Training (4 models) | ~1 hour |
| Evaluation | 30-60 min |
| Documentation | 15 min |
| **Total** | **~3-4 hours** |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| CUDA not detected | Force GPU preference in Windows Graphics Settings |
| Windows Update downgrades driver | Use wushowhide.diagcab to hide NVIDIA updates |
| Intel iGPU intercepts Python | Set python.exe to "High performance" in Graphics Settings |
| Extraction timeout | Run in background, track with progress file |
| Training converges early | Early stopping saves time, still gets good model |
| Adapter training fails | Standalone script, can debug individually |
| Memory issues | 2M params on 8GB VRAM = massive headroom |

## Rollback

- **CUDA setup:** Reinstall CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- **Data extraction:** Delete `data/` directory
- **Integration:** `git checkout -- src/data/dataset.py`
- **New scripts:** Delete `scripts/train_adapter.py`, `scripts/distill.py`
- **Training:** Delete `models/baseline.pt`, `models/augmented.pt`, `models/adapter.pt`, `models/student.pt`
- **Documentation:** `git checkout -- docs/*.md`
