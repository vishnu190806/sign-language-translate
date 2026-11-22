# Project Status

## Current Phase: Continuous Sentence Program (ISL-only, any person, full sentences)
- **Overall Progress**: M0–M8 complete (~95%); sentence program launched (see `docs/CONTINUOUS_PLAN.md`)
- **Last Updated**: 2026-10-02
- **Goal**: any person signs continuously → full spoken English sentences. ASL dropped. No human recordings.
- **Deployment model**: `models/multihead/best.pt` — INCLUDE test 77.20%, ISL500 unseen-users 61.82%, real-world overall 41.0% / IV 68.0%
- **Latest (2026-10-09)**: audit survivors executed — Qwen V2 prompt (43.6%), gate kept 0.50, per-head temperature wired, latency re-measured, LLM-novel B3 (30/30), encode-once 3.2×, per-user truth 27.4%/82.4% (margin routing parked); suite 306/306

## Milestones Overview

### ISL Implementation (M0–M7)
| Milestone | Description | Status |
|-----------|-------------|--------|
| M0 | Environment & scaffolding | ✅ Complete |
| M1 | ISL Dataset acquisition & EDA | ✅ Complete |
| M2 | ISL Landmark extraction pipeline | ✅ Complete |
| M3 | ISL DataLoaders & augmentation | ✅ Complete |
| M4 | HTMA Model Training (INCLUDE-50) | ✅ Complete — **100% val top-1** |
| M5 | ISL Real-time recognition loop | ✅ Complete |
| M6 | ISL Sentence assembly + TTS | ✅ Complete |
| **M7** | **ISL Evaluation, polish, expo pack** | **✅ Complete** |
| **M7+** | **Scale to full 264-class INCLUDE** | **✅ Complete — 81.8% test top-1** |
| **M7.9** | **StabilityGate + Landmark Smoothing** | **✅ Complete — gate emits, smoother added** |

### M8 — Real-World Accuracy Improvement (Complete via multi-head, 2026-09-26)
| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Data Collection & Augmentation | ✅ Complete — ISL500/ISL-DATA downloaded (26,796 files), converted (26,483 `.npz`), pose-dropout aug added |
| Phase 2 | Domain Adaptation | ✅ Complete — DSTAdapter measured (+3.98pp weak base); multi-head co-training (Logos recipe) deployed |
| Phase 3 | Model Improvements | ✅ Complete — multi-head heads, source-balanced sampler, per-source routing in demo + eval |
| Phase 4 | Evaluation Protocol | ✅ Complete — per-source test (INCLUDE 77.20%, ISL500 61.82%), real-world routed eval (41.0% / IV 68.0%), LOSO proven invalid on this test set |

### M8+ — Autonomous hardening (all decided, 2026-10-03)
| # | Task | Verdict |
|---|------|---------|
| 1 | Temperature scaling + risk-coverage gate re-tune | ✅ SHIPPED: temp fit + demo integration; gate threshold 0.50, OOV-false 75→62% |
| 2 | Confusion-driven targeted augmentation + retrain | ❌ ABORTED 2026-10-02: 74.69% (−7.1pp) — distribution shifts hurt this regime |
| 3 | New-code tests + full suite | ✅ 292/292 |
| 4 | Episodic norm-affine TTA measurement | ❌ ABORTED: no reliable per-signer gain |
| 5 | Confidence-gap routed ensemble | ❌ DROPPED 2026-10-03: superseded — v2 deployment (82.22/84.42) already beats both lines it would combine |
| 6 | Same-modality KD student (~0.5M params) | ❌ DROPPED 2026-10-03: no mobile target; prior KD ran −7.11pp; revisit only on new deployment target |
| 7 | ONNX export spike | ❌ DROPPED 2026-10-03: infer ~4ms of 18ms loop at 55 FPS — export buys ~1–3ms for a day's work + RoPE op risk; revisit only if loop misses 25 FPS |
| 8 | NIOS DPC scale-up (115h) | ❌ DROPPED 2026-10-03: transfer proven but non-converting at fine-tune; more hours won't change the mechanism |

### Future Work
| Milestone | Description | Status |
|-----------|-------------|--------|
| M9–M13 | Continuous Sentence Program (B2 → A2 → D → E → expo) | 🟡 M9 next (forced alignment); see `docs/CONTINUOUS_PLAN.md` |

---

## M7+ Real-World Evaluation (2026-09-15)

### Test Setup
- Dataset: `isl_40_test` — **642 clips** from 4 sources (INCLUDE, ISL500, CISLR, ISLRTC)
- Model: 264-class HTMA trained on INCLUDE
- Gate: StabilityGate with 60% agreement, 0.40 threshold
- Smoothing: One-Euro filter on landmarks
- Vocabulary: 17 In-Vocabulary (IV) words, 23 Out-of-Vocabulary (OOV) words

### Results by Source
| Source | Clips | Top-1 | Top-3 | Top-5 |
|--------|------:|------:|------:|------:|
| INCLUDE (same dataset) | 143 | **86.0%** | **94.4%** | **97.9%** |
| ISL500 (different studio) | 405 | 4.4% | 9.9% | 13.6% |
| CISLR (YouTube) | 81 | 4.9% | 6.2% | 9.9% |
| ISLRTC (dictionary) | 13 | 0.0% | 0.0% | 0.0% |
| **Overall** | **642** | **22.6%** | **28.0%** | **31.6%** |

Source: `results/real_world_accuracy.json`

### In-Vocabulary vs Out-of-Vocabulary
| Category | Clips | Top-1 | Top-3 | Top-5 |
|----------|------:|------:|------:|------:|
| **In-Vocabulary** (17 words) | 387 | 145/387 (37.5%) | 180/387 (46.5%) | 203/387 (52.5%) |
| **Out-of-Vocabulary** (23 words) | 255 | 0/255 (0.0%) | 0/255 (0.0%) | 0/255 (0.0%) |

Source: `results/real_world_accuracy.json`

### Key Findings
1. **86.0%** on INCLUDE (in-distribution, 95% CI 79.4–90.8%) — model works well on the training distribution
2. **0% on OOV words is a tautology, not a result** — a 263-class model *cannot* predict 23 words it never saw. This number carries no information and must not be quoted as a capability or a weakness.
3. **4.4%** on ISL500 (CI 2.8–6.9%) — real domain gap; ISL500 supplies 405 of 642 test clips, so it dominates the 22.6% "overall" figure. That aggregate is a source-mix artifact, not a generalization metric.
4. **4.9% on CISLR (CI 1.9–12.0%)** — not comparable to published CISLR figures: CISLR is a **4,765-class one-shot prototype task**, our model has 263 classes. Most queries fall outside our label space.
5. **Vocabulary mismatch** — only 17/40 test words overlap with the model's 263 classes
6. **ISLRTC 0.0% is a genuine failure, not a label bug** — all 13 rows verified in `isl_40_test/metadata.csv`. But n=13 gives CI [0.0%, 22.8%]: underpowered, not diagnostic.

*Findings re-validated 2026-10-02 against `results/real_world_accuracy.json` and
`isl_40_test/metadata.csv`. See `docs/RESULTS.md` and `docs/RESEARCH_VALIDATION.md`.*

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Test suite | 257/257 passing (re-verified 2026-10-02) |
| HTMA model params | 2.03M |
| Feature dimension | 168 per frame |
| Sequence length | 32 frames |
| Full dataset classes | 264 (checkpoint) / 263 (normalized label map) |
| Full dataset videos | 4,257 |
| Full dataset splits | 4,163/458/573 |
| Full dataset test top-1 | **81.8%** |
| Full dataset test top-5 | **95.4%** |
| INCLUDE real-world top-1 | **86.0%** |
| Overall real-world top-1 | **22.6%** |
| Real-time target | Not currently measured |
| StabilityGate threshold | 0.40 (was 0.85) |
| StabilityGate agreement | 60% (was 100%) |

---

## M8 Implementation Progress (2026-09-18)

### Phase 1: Data Collection & Augmentation ✅ (Code Implemented)
| Task | File | Status |
|------|------|--------|
| Geometry-aware augmentations | `src/features/augment_geo.py` | ✅ 6 augmentations implemented |
| Temporal augmentations | `src/features/augment.py` | ✅ 7 new augmentations added |
| Landmark-specific augmentations | `src/features/augment.py` | ✅ Included in temporal augmentations |
| Augmentation pipeline | `src/features/augment_pipeline.py` | ✅ Unified interface with stats tracking |

**New Augmentations:**
- Geometry: CameraDepthVariation, HVShift, HandSizeVariation, ViewpointRotation, FingerArticulation, ElbowDisplacement
- Temporal: FramePermutation, SpeedVariation, RandomFrameDrop
- Landmark: SpatialRotation, RandomTranslation, FrameFusion, RandomKeypointRemoval

### Phase 2: Domain Adaptation ✅ (Code Implemented)
| Task | File | Status |
|------|------|--------|
| Test-Time Adaptation (SAME) | `src/models/same_adapter.py` | ✅ MoE + signer-aware gating |
| Parameter-efficient fine-tuning | `src/models/dst_adapter.py` | ✅ Spatial-temporal adapters |
| Knowledge distillation | `src/train/distiller.py` | ✅ KD loss + feature distillation |

**Key Features:**
- SAME: Lightweight MoE modules, unsupervised adaptation objectives
- DSTAdapter: ~11.6% trainable parameters (tunable via adapter_dim)
- Distiller: Temperature-scaled soft targets, combined KD+CE loss

### Phase 3: Model Improvements ✅ (Code Implemented)
| Task | File | Status |
|------|------|--------|
| Multi-scale feature fusion | `src/models/htma.py` | ✅ Early/late/multiscale fusion |
| Per-signer normalization | `src/features/normalize.py` | ✅ SignerStatistics + SignerEstimator |

**Key Features:**
- Fusion methods: Early (default), Late, Multi-scale
- Per-signer normalization: Estimate signer from features, apply signer-specific stats

### Phase 4: Evaluation Protocol ⬜ Incomplete
| Task | File | Status |
|------|------|--------|
| Signer-disjoint splits | `scripts/create_signer_splits.py` | ⬜ Script exists, LOSO not run |
| Cross-dataset evaluation | `scripts/evaluate_real_world.py` | ✅ Complete — 22.6% overall top-1 |
| Real-world testing | `scripts/real_world_test.py` | ✅ Complete — per-signer artifact generated |
| Ablation studies | `scripts/ablation_study.py` | ✅ Complete — see `results/ablation/ablation_results.json` |
| Record calibration data | `scripts/record_calibration.py` | ⬜ Script exists, no calibration data recorded |
| Latency benchmark | `scripts/benchmark_latency.py` | ⬜ Not implemented |
| Faithful TTA measurement | — | ⬜ Not measured (ablation has no +TTA row) |

**Evaluation Protocols:**
- **LOSO (Leave-One-Signer-Out)**: Script exists but not executed; no artifact
- **Signer-Disjoint Splits**: Script exists but not validated; no artifact
- **Cross-Dataset**: Complete — separate evaluation per source (INCLUDE, ISL500, CISLR, ISLRTC)

### Test Results
- Total tests: 218 passed, 16 pre-existing failures (missing dataset/scripts)
- M8-related tests: All pass
- New test files: `test_augment_geo.py`, `test_augment_pipeline.py`, `test_same_adapter.py`, `test_dst_adapter.py`, `test_distiller.py`

---

## Next Steps

### M8 — Real-World Accuracy Improvement (Priority: HIGH)

Based on research findings (SAME, DSTAdapter, SIGNET, EnsAug, KD-MSLRT), here are the prioritized next steps:

**Phase 1: Data Collection & Augmentation (Week 1-2)**
1. Record self-calibration data (50-100 clips of ourselves signing the 40 test words)
2. Train model with new augmentations (requires GPU)

**Phase 2: Domain Adaptation (Week 3-4)**
1. Test-Time Adaptation (TTA) with signer-aware MoE (SAME framework) — measure actual +TTA delta
2. Parameter-efficient fine-tuning with spatial-temporal adapters (DSTAdapter)
3. Knowledge distillation from larger teacher to lightweight student

**Phase 3: Model Improvements (Week 5-6)**
1. Multi-scale feature fusion (combine hand + pose at multiple scales)
2. Per-signer normalization statistics
3. Contrastive learning for robust representations
4. Ensemble methods (train specialists on different augmentations)

**Phase 4: Evaluation Protocol (Week 7)**
1. Signer-disjoint splits (ensure no signer overlap between train/test)
2. Cross-dataset evaluation (ISL500, CISLR, ISLRTC) — re-run after training
3. Real-world testing with diverse signers
4. Ablation studies (measure impact of each improvement)
5. Latency benchmark

**Target Metrics:**
| Metric | Current | Target |
|--------|---------|--------|
| INCLUDE top-1 | 86.0% | ≥90% |
| Cross-dataset top-1 | 22.6% | ≥50% |
| ISL500 top-1 | 4.4% | ≥40% |
| OOV handling | 0% | Report separately |
| LOSO top-1 | Not measured | ≥60% |
| Latency (ms) | Not measured | ≤25ms |

**Research Documents:**
- `docs/M8_REAL_WORLD_IMPROVEMENT.md` — Detailed implementation plan (historical/research only)
- `docs/M8_RESEARCH_SUMMARY.md` — Research findings and recommendations (historical/research only)

### M9–M13 — Continuous Sentence Program (current roadmap; see `docs/CONTINUOUS_PLAN.md`)
1. **M9** — Continuous data: forced alignment (B2) + synthesis (B3)
2. **M10** — Continuous decoding revival: CTC/STARK head (A2)
3. **M11** — Any-person acceptance: calibration + signer-removal + SDDA (D)
4. **M12** — Live sentence demo: streaming decode + confidence display (E)
5. **M13** — Final evaluation + expo pack (sentence-WER, demo video)

> Former M9–M13 (ASL pipeline) deleted 2026-10-03 — no ASL integration, ever.

---

## Documentation Files

| File | Description |
|------|-------------|
| `docs/project_status.md` | This file — overall project status |
| `docs/architecture.md` | System architecture & data flow |
| `docs/QA_REPORT.md` | Full-project QA sweep: 257/257 tests, all bugs fixed/deferred with reasons |
| `docs/RESULTS.md` | Full evaluation results (artifact-backed) |
| `docs/SETUP.md` | Step-by-step setup |
| `docs/LIMITATIONS.md` | Known limitations |
| `docs/M8_REAL_WORLD_IMPROVEMENT.md` | **Historical/research only** — M8 real-world accuracy improvement plan |
| `docs/M8_RESEARCH_SUMMARY.md` | **Historical/research only** — Research findings and recommendations |
| `docs/ARCHITECTURE_V2.md` | **Current** — Upgraded architecture with per-decision citations (HTMA/SAME/DSTAdapter/KD-MSLRT all verified real against publisher pages; EnsAug/SIGNET dropped as unverified) |
| `configs/v2_tuned.yaml` | **Current** — 5-op tuned augmentation + batch 256 / workers 6 for RTX 4070 8GB + i9 + 16GB |
| `scripts/eval_iv_only.py` | **Current** — IV-only rescore (`results/iv_only_metrics.json`: IV top-1 37.47%, overall 22.59%) |

---

## Version
*Version: 1.14.1 — QA + hardening complete: 257/257 tests (re-verified 2026-10-02; earlier 253/31 counts were stale); gate sweep applied (threshold 0.50: OOV-false 75→62%); episodic TTA measured and ABORTED (delta −0.0000; per-actor base spread 26–84%); C2 polish shipped (exact 37.2%); demo speaks fluent sentences with temperature-calibrated multi-head model.*

---

## Phase 5 — Runnable GitHub release preparation (2026-11-22)

- Audited runtime path from source: `scripts/demo.py` → `src/app/demo.py` loads checkpoint at line 77 with no fallback; sidecars at lines 66/104/132; MediaPipe models at 229–233.
- **Found inconsistency:** `configs/default.yaml:31` uses `tts.engine: kokoro` (`af_heart`), but `kokoro` was absent from `requirements.txt`. Added `requirements-tts.txt`; default TTS now documented correctly. Piper voice remains a restricted, excluded fallback.
- **Licensing research:** INCLUDE is CC BY 4.0 (Zenodo 4010759 / HF). ISL500/ISL-DATA is gated research-only. Blizzard 2013 Lessac voice licence forbids redistribution; Rhasspy confirms no project-level grant. MediaPipe task models are Apache-2.0 per Google model cards (SDK Apache-2.0). Kokoro weights are Apache-2.0.
- **Checkpoint verdict:** `models/multihead_v2/best.pt` (SHA-256 `BC48AE02…06A5`, 9,932,803 B) is NOT redistributable (ISL500-derived). Compatible alternative verified read-only: `models/baseline.pt` (SHA-256 `8ED3F773…EB02`, 24,523,349 B), plain HTMA `264` classes / `256` dim / 2 blocks / 8 heads / dropout 0.2, matching `configs/isl.yaml` — recommended for bundling, pending user approval.
- Added `scripts/setup_runtime.ps1` (venv, pinned deps, optional TTS, SHA-256-verified MediaPipe downloads; no datasets, no gated downloads, no training), `requirements-tts.txt`, `docs/research/` (27 curated metric artifacts + 3 screenshots, ~2.57 MB copied from `results/`, originals untouched), and a 16-section README.
- `.gitignore` prepared with a single **approval-gated** negation for `models/baseline.pt`; nothing staged or committed.
- Repository size estimate: source + docs + curated evidence ≈ 1 MB; with bundled checkpoint ≈ 24.5 MB. No LFS required.
- Git still uninitialized; no staging, commit, push, deletion or modification of datasets/checkpoints/venvs.

### Phase 6 — Release execution (2026-11-22)

- User authorized the full Git release. Preflight: Git 2.56, identity `Vishnu <vishnuvardhan190806@gmail.com>`, `gh` authenticated as `vishnu190806`; remote is private and empty with default branch `main`.
- Re-verified `models/baseline.pt`: 24,523,349 B, SHA-256 `8ED3F773B94B43FF5D772D59F30E64FAC589974EE1F63B9CD243FFC3CB31EB02`; provenance confirmed as INCLUDE-only M8-V2 checkpoint (2026-09-25, before ISL500). INCLUDE is CC BY 4.0; attribution added in `ATTRIBUTION.md`.
- Fixed all launchers/docs to use `.venv` and to select `--checkpoint models/baseline.pt` explicitly; `run_live_demo.bat` no longer references the author's old `venv/`.
- Security scan of publishable text: no `.env`, no credential patterns, no author absolute paths remaining.
- All 26 curated JSON files and 1 JSONL parsed cleanly; 3 screenshots present. Original `results/` (1,433 files / 115.21 MB) untouched.
- Clean-clone run 1 exposed a real defect: `scripts/setup_runtime.ps1` resolved `$ProjectDir` to `scripts/` instead of the repository root, so the venv and requirements lookup were misplaced. Fixed by deriving the root from `$PSScriptRoot`; fix committed and pushed, then verification restarted from a fresh clone.
- Clean-clone run 2 exposed a second defect: the unanchored `data/` ignore rule also excluded the `src/data/` Python package, breaking every `import src...`. Rules for `data/`, `results/`, `isl_40_test/` and `logs/` were anchored to the repository root; `src/data/` (7 files) added to the manifest. Setup-script smoke-test hints now explicitly pass `--checkpoint models/baseline.pt`.
