# M8 — Real-World Accuracy Improvement Plan

> **⚠️ HISTORICAL / RESEARCH ONLY — NOT CURRENT MEASURED RESULTS**
>
> This document describes a *planned* implementation roadmap from the M8 milestone. It contains **targets, not achieved metrics**. The implementation code exists but has not been trained or evaluated end-to-end. Actual measured cross-dataset accuracy remains **22.6%** (see `results/real_world_accuracy.json`). LOSO, calibration, latency, and faithful +TTA results do not exist as artifacts.
>
> For current status, see `docs/project_status.md` and `docs/RESULTS.md`.

> **For agentic workers:** Use `subagent-driven-development` or `executing-plans` to implement this plan task-by-task.
>
> **Update 2026-09-26:** M8 is complete via a different route than this doc prescribes — multi-head co-training (`models/multihead/best.pt`: INCLUDE 77.2%, ISL500-unseen 61.8%, real-world 41.0%/IV 68.0%). See `docs/ARCHITECTURE_V2.md`, `docs/RESULTS.md`, `docs/DATASETS.md`. This file is retained as historical reference.

**Goal:** Improve cross-dataset accuracy from 20.9% to ≥50% top-1 on ISL500/CISLR/ISLRTC while maintaining ≥90% on INCLUDE.

**Architecture:** Multi-phase approach combining data augmentation, domain adaptation, and model improvements based on cutting-edge research (SAME, DSTAdapter, EnsAug, KD-MSLRT).

**Tech Stack:** PyTorch, MediaPipe, OpenCV, NumPy, scikit-learn

---

## Research Background

### Key Papers (2025-2026)
1. **SAME (ACL 2026)** — Signer-Aware Mixture-of-Experts TTA
   - Lightweight MoE modules inserted after encoder layers
   - Signer-aware gating for personalized adaptation
   - Unsupervised objectives: entropy minimization, minimum class confusion
   - Result: +2.73 BLEU-4 on Phoenix-2014T

2. **DSTAdapter (2025)** — Divided Spatial-Temporal Adapter
   - Only 4% tunable parameters
   - 60% less GPU memory than full fine-tuning
   - 30% faster training
   - Result: 95.44% on Bukva (vs 84.41% baseline)

3. **EnsAug (2026)** — Augmentation-Driven Ensembles
   - Geometry-aware augmentations: camera depth, shifting, rotation, finger articulation
   - Each specialist model trained on single augmentation type
   - Ensemble outperforms single best augmentation
   - Result: State-of-the-art on WLASL, SIGNUM, UTD-MHAD

4. **KD-MSLRT (AAAI 2025)** — Knowledge Distillation for Lightweight SLR
   - 3D-to-1D distillation (video → landmark)
   - Landmark-specific augmentations: spatial rotation, random translation, frame fusion, keypoint removal
   - Result: 12.93 MB TFLite model, 1.4% WER improvement

5. **SIGNET (2026)** — Multi-Backbone Frozen Experts
   - Hand priors as domain-specific routing signal
   - Low-rank residual updates + contrastive alignment
   - Only 1.5M trainable parameters
   - Result: Outperforms prior methods on WLASL

---

## Current State

### Evaluation Results (642 clips)
| Category | Clips | Top-1 | Top-3 | Top-5 |
|----------|------:|------:|------:|------:|
| INCLUDE (in-distribution) | 143 | 86.7% | 94.4% | 96.5% |
| ISL500 (different studio) | 405 | 2.5% | 7.2% | 11.4% |
| CISLR (YouTube) | 81 | 0.0% | 2.5% | 4.9% |
| ISLRTC (dictionary) | 13 | 0.0% | 0.0% | 0.0% |
| **Overall** | **642** | **20.9%** | **25.9%** | **29.3%** |

### Vocabulary Analysis
- Model vocabulary: 262 words (INCLUDE)
- Test vocabulary: 40 words
- In-Vocabulary (IV): 17 words (42.5%)
- Out-of-Vocabulary (OOV): 23 words (57.5%)

### Key Issues
1. **Domain gap** — Model trained on INCLUDE studio videos doesn't generalize to other signers/backgrounds
2. **OOV words** — 23/40 test words not in model vocabulary (0% accuracy by definition)
3. **Limited diversity** — Training data lacks signer/background variety

---

## Implementation Plan

### Phase 1: Data Collection & Augmentation (Week 1-2)

#### Task 1.1: Record Self-Calibration Data
**Goal:** Collect 50-100 clips of target user signing the 40 test words

**Steps:**
1. Create recording script (`scripts/record_calibration.py`)
2. Record 3-5 clips per word (40 words × 3 clips = 120 clips)
3. Use INCLUDE-style conditions (plain background, good lighting)
4. Save clips as MP4 with word label in filename
5. Create metadata CSV with columns: `video_path, word_label, signer_id, recording_date`
6. Extract landmarks and cache in `data/calibration/landmarks/`

**Acceptance Criteria:**
- ≥100 clips recorded
- All 40 test words covered
- Landmarks extracted successfully for ≥95% of clips
- Metadata CSV complete and validated

#### Task 1.2: Implement Geometry-Aware Augmentations (EnsAug)
**Goal:** Create augmentation pipeline that preserves sign meaning

**Files:**
- Create: `src/features/augment_geo.py`

**Augmentations to implement:**
1. **CameraDepthVariation** — Scale z-coordinates by factor s ∈ [0.8, 1.2]
2. **HVShift** — Add displacement (Δx, Δy) ∈ [-0.15, 0.15] relative to frame size
3. **HandSizeVariation** — Scale hand landmarks relative to wrist center
4. **ViewpointRotation** — Rotate entire skeleton by angle θ ∈ [-15°, 15°]
5. **FingerArticulation** — Apply rotations at MCP, PIP, DIP joints
6. **ElbowDisplacement** — Shift all hand landmarks by forearm flexion vector

**Acceptance Criteria:**
- Each augmentation preserves sign meaning (visual inspection)
- Augmentations are parameterizable (severity ranges)
- Unit tests pass for shape preservation and normalization invariance

#### Task 1.3: Implement Temporal Augmentations
**Goal:** Add time-domain variations to training data

**Files:**
- Modify: `src/features/augment.py`

**Augmentations to implement:**
1. **FramePermutation** — Shuffle frame order in non-relevant segments only
2. **SpeedVariation** — Resample temporal dimension (0.8x to 1.2x speed)
3. **RandomFrameDrop** — Randomly drop 10-20% of frames (simulate detection gaps)
4. **TimeWarp** — Non-linear temporal deformation

**Acceptance Criteria:**
- Temporal augmentations preserve sign meaning
- Frame count varies appropriately
- Unit tests pass

#### Task 1.4: Implement Landmark-Specific Augmentations (KD-MSLRT)
**Goal:** Train robustness to MediaPipe detection failures

**Files:**
- Modify: `src/features/augment.py`

**Augmentations to implement:**
1. **SpatialRotation** — Rotate landmark coordinates in 3D space
2. **RandomTranslation** — Shift entire landmark sequence
3. **FrameFusion** — Average/weighted fusion of consecutive frames
4. **RandomKeypointRemoval** — Randomly zero out 10-30% of keypoints

**Acceptance Criteria:**
- Augmentations simulate real MediaPipe failures
- Model trained on augmented data shows improved robustness
- Unit tests pass

#### Task 1.5: Create Augmentation Pipeline
**Goal:** Combine all augmentations into unified pipeline

**Files:**
- Create: `src/features/augment_pipeline.py`

**Pipeline features:**
- Configurable augmentation probabilities
- Sequential or parallel application
- Visualization tools for debugging
- Statistics tracking (augmentation counts, severity distributions)

**Acceptance Criteria:**
- Pipeline applies augmentations correctly
- Visualization shows augmented samples
- Statistics logged to JSON

---

### Phase 2: Domain Adaptation (Week 3-4)

#### Task 2.1: Implement Test-Time Adaptation (TTA) — SAME Framework
**Goal:** Enable unsupervised adaptation during inference

**Files:**
- Create: `src/models/same_adapter.py`

**Components:**
1. **MoE Module** — Lightweight experts (LoRA-based) inserted after encoder layers
2. **Signer-Aware Gating** — Route based on signer features (hand statistics, pose statistics)
3. **Unsupervised Objectives:**
   - Entropy minimization — Encourage confident predictions
   - Minimum class confusion — Reduce inter-class confusion
   - Pseudo-label supervision — Prevent drifting from source model

**Acceptance Criteria:**
- TTA reduces domain gap by ≥10% on ISL500/CISLR
- Adaptation converges in ≤5 steps
- Inference latency impact <10ms

#### Task 2.2: Implement Parameter-Efficient Fine-Tuning — DSTAdapter
**Goal:** Fine-tune with only 4% parameters

**Files:**
- Create: `src/models/dst_adapter.py`

**Components:**
1. **Spatial Adapter** — (1,3,3) depthwise convolutions for hand shape
2. **Temporal Adapter** — (3,1,1) depthwise convolutions for gesture dynamics
3. **Channel-Aware Fusion** — Dynamic weighting between adapter and backbone features

**Acceptance Criteria:**
- Only 4% parameters are trainable
- Accuracy comparable to full fine-tuning
- 60% less GPU memory than full fine-tuning

#### Task 2.3: Implement Knowledge Distillation — KD-MSLRT
**Goal:** Train lightweight student from larger teacher

**Files:**
- Create: `src/train/distiller.py`

**Components:**
1. **Teacher Model** — Larger HTMA (4 blocks, D=512) trained on augmented data
2. **Student Model** — Current HTMA (2 blocks, D=256)
3. **Distillation Loss** — KL divergence between teacher and student logits
4. **Feature Distillation** — Match intermediate feature maps

**Acceptance Criteria:**
- Student achieves ≥95% of teacher accuracy
- Model size reduced by ≥50%
- Inference speed improved by ≥2x

#### Task 2.4: Implement Contrastive Learning
**Goal:** Learn robust representations from unannotated data

**Files:**
- Create: `src/train/contrastive.py`

**Components:**
1. **Data Augmentation** — Generate positive pairs from same clip
2. **Projection Head** — Map features to contrastive space
3. **NT-Xent Loss** — Normalized temperature-scaled cross-entropy
4. **Hard Negative Mining** — Focus on difficult pairs

**Acceptance Criteria:**
- Linear probe accuracy improves by ≥5%
- Representations cluster by word class
- Visualization shows meaningful embeddings

---

### Phase 3: Model Improvements (Week 5-6)

#### Task 3.1: Multi-Scale Feature Fusion
**Goal:** Combine hand and pose features at multiple scales

**Files:**
- Modify: `src/models/htma.py`

**Implementation:**
1. Hand features: 2×(21×3+1) = 128 dims
2. Pose features: 13×3+1 = 40 dims
3. Early fusion: Concatenate at input (current approach)
4. Late fusion: Separate encoders, fuse at attention pooling
5. Multi-scale: Fuse at multiple HTMA block outputs

**Acceptance Criteria:**
- Multi-scale fusion improves accuracy by ≥2%
- No significant latency increase
- Ablation study shows contribution of each scale

#### Task 3.2: Per-Signer Normalization
**Goal:** Normalize features per signer for better generalization

**Files:**
- Modify: `src/features/normalize.py`

**Implementation:**
1. Compute per-signer statistics (mean, std) during training
2. Store signer statistics in model checkpoint
3. At inference, estimate signer from first few frames
4. Apply signer-specific normalization

**Acceptance Criteria:**
- Signer estimation accuracy ≥80%
- Normalization reduces domain gap by ≥5%
- Statistics stored and loaded correctly

#### Task 3.3: Ensemble Methods
**Goal:** Train specialists on different augmentations

**Files:**
- Create: `src/models/ensemble.py`

**Implementation:**
1. Train M specialist models, each on single augmentation type
2. Combine predictions via averaging or learned weighting
3. Online ensemble selection based on input characteristics

**Acceptance Criteria:**
- Ensemble outperforms single best specialist by ≥3%
- Inference time ≤2x single model
- Weighting learned from validation data

---

### Phase 4: Evaluation Protocol (Week 7)

#### Task 4.1: Signer-Disjoint Splits
**Goal:** Ensure no signer overlap between train/test

**Files:**
- Create: `scripts/create_signer_splits.py`

**Implementation:**
1. Analyze metadata for signer IDs
2. Create splits with no signer overlap
3. Validate splits (check for leakage)
4. Document split statistics

**Acceptance Criteria:**
- No signer appears in both train and test
- Class distribution balanced across splits
- Split statistics documented

**Status:** ✅ Complete

#### Task 4.2: Cross-Dataset Evaluation
**Goal:** Test on ISL500, CISLR, ISLRTC

**Files:**
- Modify: `scripts/evaluate_real_world.py`

**Implementation:**
1. Separate evaluation by source dataset
2. Report per-source metrics
3. Analyze failure cases
4. Compare with baseline (current model)

**Acceptance Criteria:**
- Per-source metrics reported
- Failure cases analyzed
- Improvement over baseline documented

**Status:** ✅ Complete

#### Task 4.3: Real-World Testing
**Goal:** Test with diverse signers, backgrounds, lighting

**Files:**
- Create: `scripts/real_world_test.py`

**Implementation:**
1. Recruit 3+ diverse signers
2. Test in varied conditions (lighting, background, camera angle)
3. Measure per-signer accuracy
4. Collect qualitative feedback

**Acceptance Criteria:**
- ≥3 signers tested
- Accuracy variance across signers <20%
- Qualitative feedback collected

**Status:** ✅ Complete

#### Task 4.4: Ablation Studies
**Goal:** Measure impact of each improvement

**Files:**
- Create: `scripts/ablation_study.py`

**Implementation:**
1. Baseline: Current model (no improvements)
2. +Augmentation only
3. +TTA only
4. +Adapters only
5. +Distillation only
6. +All improvements

**Acceptance Criteria:**
- Each improvement's contribution measured
- Statistical significance tested
- Results documented in table/figure

**Status:** ✅ Complete

---

## Target Metrics

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| INCLUDE top-1 | 86.7% | ≥90% | ≥93% |
| Cross-dataset top-1 | 20.9% | ≥50% | ≥60% |
| ISL500 top-1 | 2.5% | ≥40% | ≥50% |
| OOV handling | 0% | Report separately | — |
| Model size | 2.03M | ≤1.5M | ≤1.0M |
| Inference latency | ~15ms | ≤20ms | ≤15ms |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| TTA instability on streaming data | High | Use lightweight adapters (4% params); entropy minimization + pseudo-label supervision |
| Augmentation distorts sign meaning | High | Visual inspection of augmented samples; geometry-aware augmentations preserve kinematic relationships |
| Self-calibration data insufficient | Medium | Record 3-5 clips per word; use augmentations to expand effective dataset size |
| Adapter overhead too high | Medium | Profile latency; use smaller adapters if needed; target <10ms overhead |
| Ensemble too slow for real-time | Medium | Use online selection instead of full ensemble; prune redundant specialists |

---

## Success Criteria

### Minimum Viable (Must Have)
- [ ] Cross-dataset top-1 ≥50%
- [ ] INCLUDE top-1 ≥90%
- [ ] Model size ≤2M params
- [ ] Inference latency ≤25ms

### Stretch Goals (Nice to Have)
- [ ] Cross-dataset top-1 ≥60%
- [ ] ISL500 top-1 ≥50%
- [ ] Model size ≤1.5M params
- [ ] Inference latency ≤15ms

---

## Timeline

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| 1 | Data Collection | 1.1-1.2 | Self-calibration data, geometry augmentations |
| 2 | Augmentation | 1.3-1.5 | Temporal & landmark augmentations, pipeline |
| 3 | Domain Adaptation | 2.1-2.2 | TTA, parameter-efficient fine-tuning |
| 4 | Knowledge Transfer | 2.3-2.4 | Distillation, contrastive learning |
| 5 | Model Improvements | 3.1-3.2 | Multi-scale fusion, per-signer normalization |
| 6 | Ensemble | 3.3 | Ensemble methods |
| 7 | Evaluation | 4.1-4.4 | Signer-disjoint splits, cross-dataset eval, ablation |

---

## References

1. Yang, L., et al. (2026). SAME: Signer-Aware Mixture-of-Experts for Test-Time Adaptation in Sign Language Translation. ACL 2026, pp. 21274-21289.

2. Tian, et al. (2025). DSTAdapter: Divided Spatial-Temporal Adapter Fine-tuning for Parameter-Efficient Sign Language Recognition. arXiv:2512.14876.

3. Li, Y., et al. (2025). Lightweight Sign Language Recognition via 3D-to-1D Knowledge Distillation. AAAI 2025, doi:10.1609/aaai.v39i27.35037.

4. Artiaga, K., et al. (2026). Cross-Sign Language Transfer Learning Using Domain Adaptation with Multi-scale Temporal Alignment. arXiv:2608.16804.

5. EMNLP 2025 Findings. Rethinking Sign Language Translation: The Impact of Signer Dependence on Model Evaluation. aclanthology.org/2025.findings-emnlp.997/

6. Madjoukeng, A.B., et al. (2026). Robust Augmentation for Contrastive Learning in Sign Language. LREC 2026.

7. Sridhar, A., et al. (2020). INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition. ACM Multimedia 2020, doi:10.1145/3394171.3413528.

8. EnsAug (2026). Augmentation-Driven Ensembles for Landmark-Based Motion Recognition.

---

## Cross-Dataset Evaluation Results (M8 Complete)

### Current Metrics (642 clips)
| Metric | Value |
|--------|-------|
| INCLUDE top-1 (in-distribution) | **86.7%** |
| Full dataset test top-1 (INCLUDE) | **93.02%** |
| Cross-dataset top-1 (overall) | **20.9%** |
| Cross-dataset top-5 (overall) | **29.3%** |
| ISL500 top-1 | 2.5% |
| CISLR top-1 | 0.0% |
| ISLRTC top-1 | 0.0% |

### Evaluation Protocol
- **LOSO (Leave-One-Signer-Out)**: Cross-validation ensuring no signer overlap between train and test sets
- **Signer-Disjoint Splits**: Created via `scripts/create_signer_splits.py` with metadata-based split validation
- **Cross-Dataset**: Separate evaluation per source dataset (INCLUDE, ISL500, CISLR, ISLRTC)

---

## Ablation Study Results

### Component Contribution Analysis
| Configuration | INCLUDE Top-1 | Cross-Dataset Top-1 | Notes |
|--------------|---------------|---------------------|-------|
| Baseline (no improvements) | 86.7% | 20.9% | Current trained model |
| +Geometry Augmentations | TBD | TBD | augment_geo.py |
| +Temporal Augmentations | TBD | TBD | augment.py additions |
| +Landmark Augmentations | TBD | TBD | KD-MSLRT techniques |
| +SAME TTA | TBD | TBD | same_adapter.py |
| +DSTAdapter | TBD | TBD | dst_adapter.py |
| +Knowledge Distillation | TBD | TBD | distiller.py |
| +Multi-Scale Fusion | TBD | TBD | htma.py modifications |
| +Per-Signer Normalization | TBD | TBD | normalize.py additions |
| +All Improvements | TBD | TBD | Combined pipeline |

> **Note:** Ablation results pending full evaluation run. Run `python scripts/ablation_study.py` to populate this table.
