# Limitations

## Current Limitations

### 1. Word-Level Recognition Only
The system recognizes isolated word signs, not continuous sign language sentences. Each sign must be performed separately with a brief pause between signs.

**Impact**: Users cannot sign natural, flowing sentences.
**Future**: Continuous recognition using CTC/seq2seq models (post-M13).

### 2. Domain Gap (Major Limitation)
The model is trained on INCLUDE dataset videos (studio quality) and performs significantly worse on other sources:

| Source | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | Clips | Description |
|--------|---------------:|---------------:|---------------:|------:|-------------|
| INCLUDE (in-distribution) | 86.0% | 94.4% | 97.9% | 143 | Same dataset as training |
| ISL500 (different studio) | 4.4% | 9.9% | 13.6% | 405 | Different signers, setup |
| CISLR (YouTube) | 4.9% | 6.2% | 9.9% | 81 | Wild videos, different cameras |
| ISLRTC (dictionary) | 0.0% | 0.0% | 0.0% | 13 | Re-encoded dictionary videos |
| **Overall** | **22.6%** | **28.0%** | **31.6%** | **642** | All sources blended |

Source: `results/real_world_accuracy.json`

**Impact**: Model only works reliably with INCLUDE-style conditions (studio lighting, clean background, specific signers). Cross-dataset accuracy drops from 86.0% to 4.4% on ISL500 and 0% on YouTube/dictionary sources.
**Update 2026-09-26 (multi-head, `models/multihead/best.pt`)**: INCLUDE test 77.20%, ISL500 unseen-users 61.82%, real-world overall 41.0% / IV 68.0% (`results/multihead_by_source.json`, `results/real_world_test_multihead.json`). The gap is narrowed but wild/YouTube sources (CISLR ~0–5%, ISLRTC 0%, mostly OOV) remain unsolved.
**Mitigation**: Use good lighting, plain background, face camera directly.
**Research**: SAME (ACL 2026) — Test-Time Adaptation with signer-aware MoE; DSTAdapter (IJMLC 2026) — 4% parameters for cross-dataset transfer; EnsAug (2026) — geometry-aware augmentation ensemble.
**Citation**: Performance drops of 3–6× on unseen signers are consistent with EMNLP 2025 findings: Rethinking Sign Language Translation: The Impact of Signer Dependence on Model Evaluation (EMNLP 2025 Findings). Dataset: Sridhar, A., et al. (2020). INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition. ACM Multimedia 2020.
**M8 Status**: Incomplete. Augmentation pipeline, TTA, DSTAdapter, knowledge distillation, multi-scale fusion, per-signer normalization implemented. Awaiting training with new augmentations and evaluation. Missing artifacts: LOSO evaluation, calibration data, latency benchmarks, faithful TTA measurement.

### 3. Limited Vocabulary (263 Normalized Classes)
The system recognizes 263 ISL word signs (normalized label space, `data/processed/landmarks/joint_label_map.json`, contiguous indices 0–262).
**Resolved 2026-09-26:** the original `models/label_map.json` has 264 keys with max index 263 (two keys normalize to `extra`) — any 263-output model crashes on class #263 (observed CUDA assertion). The joint map fixes this; historical 264-class artifacts keep their original numbering.
**Research Finding**: Only 17/40 test words overlap with model vocabulary (42.5%). 23 words (57.5%) are OOV and cannot be predicted. ISL500/ISL-DATA shares 180/263 words (68%, `results/vocab_overlap.json`) — the joint vocabulary ceiling without a second head.

**Impact**: Cannot recognize signs outside the 264-class vocabulary.
**Research Finding**: Only 17/40 test words overlap with model vocabulary (42.5%). 23 words (57.5%) are OOV and cannot be predicted.
**Future**: Expand vocabulary with additional datasets; report OOV accuracy separately.

### 4. Lighting Sensitivity
MediaPipe hand detection requires adequate lighting:
- Hands must be clearly visible
- No strong backlighting
- No harsh shadows on hands

**Impact**: Detection fails in low light or backlit conditions.
**Mitigation**: Ring light or window-facing setup recommended.

### 5. No Facial Expression Features
The system only uses hand and upper-body pose landmarks. Facial expressions (important for ISL grammar) are not captured.

**Impact**: Cannot distinguish signs that rely on facial expressions.
**Future**: Add FaceLandmarker for non-manual features (M11 any-person track).

### 6. English TTS Output Only
Speech output is in English using Piper TTS. Indian Sign Language has its own grammar and syntax that differs from English.

**Impact**: Translated sentences may not follow ISL grammar.
**Future**: ISL-to-English translation layer (M9–M13 sentence program).

### 7. GPU Recommended (Not Required)
Training requires NVIDIA GPU with CUDA support. Inference can run on CPU but at reduced speed.

**Impact**: Training slow without GPU; inference may drop below 25 FPS on CPU.
**Recommendation**: RTX 4070 or better for training; CPU viable for inference.

### 8. Single Language Support
Indian Sign Language (ISL) only, by decision (2026-10-03) — no ASL or other sign languages planned.

**Impact**: Cannot recognize ASL or other sign languages.

### 9. English Word Output
Sign language signs map to English words, not ISL glosses. Some ISL signs have no direct English equivalent.

**Impact**: Some concepts may be lost in translation.
**Future**: ISL gloss dictionary (M9–M13 sentence program).

### 10. No Real-Time Error Feedback
The system provides no confidence feedback when predictions are uncertain. Users cannot tell when the system is unsure.

**Impact**: Users may not know when to repeat or clarify a sign.
**Future**: Add confidence indicators and uncertainty feedback.

---

## Signer-Dependence Evaluation

### LOSO Protocol Results (Not Yet Measured)
Leave-One-Signer-Out (LOSO) cross-validation has not yet been run. Placeholder for when actual LOSO evaluation is performed:

| Fold | Signer Left Out | Top-1 Accuracy | Top-3 Accuracy | Notes |
|------|-----------------|---------------:|---------------:|-------|
| 1 | Signer A | TBD | TBD | — |
| 2 | Signer B | TBD | TBD | — |
| 3 | Signer C | TBD | TBD | — |
| ... | ... | TBD | TBD | — |
| **Mean** | — | **TBD** | **TBD** | Expected: significant drop from 81.8% (signer-dependent) |

**Expected outcome**: Per EMNLP 2025 findings, signer-independent evaluation typically shows 3–6× performance drops compared to signer-dependent splits (e.g., GFSLT-VLP BLEU-4 drops from 21.44 to 3.59 on PHOENIX14T).

**Citation**: Rethinking Sign Language Translation: The Impact of Signer Dependence on Model Evaluation. EMNLP 2025 Findings. aclanthology.org/2025.findings-emnlp.997/

### Per-Signer Accuracy Variance
Accuracy variance across signers within the INCLUDE test set has not been individually analyzed. Key considerations:
- INCLUDE contains videos from multiple signers under studio conditions
- Signer-dependent models may memorize signer-specific motion patterns
- Per-signer accuracy variance is expected to be high given cross-dataset results (86.0% INCLUDE vs. 4.4% ISL500)
- Recommended analysis: compute accuracy per signer ID in INCLUDE test set, report mean ± std

**Mitigation implemented (M8)**: Per-signer normalization statistics (`src/features/normalize.py`) and signer-aware gating in TTA (`src/models/same_adapter.py`).

---

## M8 Mitigation Status

| Component | Status | Location | Impact |
|-----------|--------|----------|--------|
| Geometry-aware augmentations | ✅ Implemented | `src/features/augment_geo.py` | 6 augmentation types |
| Temporal augmentations | ✅ Implemented | `src/features/augment.py` | 7 augmentation types |
| Augmentation pipeline | ✅ Implemented | `src/features/augment_pipeline.py` | Unified interface, stats tracking |
| Test-Time Adaptation (TTA) | ✅ Implemented | `src/models/same_adapter.py` | MoE + signer-aware gating |
| DSTAdapter (PEFT) | ✅ Implemented | `src/models/dst_adapter.py` | ~11.6% trainable params |
| Knowledge distillation | ✅ Implemented | `src/train/distiller.py` | KD loss + feature distillation |
| Multi-scale feature fusion | ✅ Implemented | `src/models/htma.py` | Early/late/multiscale |
| Per-signer normalization | ✅ Implemented | `src/features/normalize.py` | SignerStatistics + SignerEstimator |
| Training with new augmentations | ⬜ Pending | — | Requires GPU training run |
| LOSO evaluation | ⬜ Pending | — | Signer-disjoint splits needed |
| Cross-dataset evaluation (full) | ✅ Complete | `results/real_world_accuracy.json` | 22.6% overall top-1 |
| Ablation studies | ✅ Complete | `results/ablation/ablation_results.json` | Measured component impact |
| Record own signing clips | ⬜ Pending | — | Requires physical recording (50–100 clips) |
| Contrastive learning | ⬜ Future | — | Robust representations |
| Ensemble methods | ⬜ Future | — | Specialist models per augmentation |
| **Joint union-softmax (INCLUDE+ISL500)** | ❌ **Rejected (measured)** | `models/joint*/`, `results/joint_by_source.json` | INCLUDE collapsed to ~0.7% (pose-manifold + 10:1 in-class skew; hand-swap test negative). See `docs/RESULTS.md`. |
| **Multi-head co-training (Logos recipe)** | ✅ **Deployed** | `models/multihead/best.pt`, `results/multihead_by_source.json`, `results/real_world_test_multihead.json` | Shared encoder + per-source heads: INCLUDE 77.20%, ISL500-unseen 61.82%, real-world 41.0% overall / 68.0% IV |

**Citation**: Augmentation techniques based on KD-MSLRT (Li et al., 2025, AAAI 2025) and robust augmentation findings (Madjoukeng et al., 2026, LREC 2026).

---

## Known Issues

### MediaPipe Detection
- Fast hand movements may cause detection loss
- Occluded hands (behind body) not detected
- Multiple hands in frame may cause handedness confusion

### Model Performance
- 81.8% accuracy on full INCLUDE test set (478 samples, 264 classes) — `results/full_dataset/metrics.json` (specialist `models/baseline.pt`, retained as fallback)
- **77.20% INCLUDE / 61.82% ISL500-unseen / 68.0% real-world IV — `models/multihead/best.pt` (deployment model)**
- 22.6% accuracy on mixed real-world data (cross-dataset, 642 clips) — `results/real_world_accuracy.json` (superseded by 41.0% multi-head, `results/real_world_test_multihead.json`)
- 0% accuracy on YouTube-sourced videos (CISLR) — mostly OOV words, unsolved
- 0% accuracy on OOV words (23/40 test words)

### System Requirements
- Windows-only camera backend (dshow)
- PortAudio dependency for audio playback
- Piper TTS requires internet for first voice download (cached thereafter)

---

## Failure Modes
1. **Fast signing**: Detection drops below 25 FPS
2. **Low light**: Hand detection fails
3. **Occlusion**: Hands behind body not visible
4. **Similar signs**: Confusion between visually similar signs
5. **Unseen signers**: Accuracy drops significantly (86.0% → 4.4% on ISL500)
6. **OOV words**: 0% accuracy on words not in training vocabulary (23/40 test words)

---

## Recommendations

### For Demo Presentations
1. Use INCLUDE-style conditions (studio lighting, plain background)
2. Set up good lighting (ring light)
3. Use plain background
4. Practice signing at moderate speed
5. Have fallback ready (pre-recorded video)

### For Development
1. Use INCLUDE data for quick iteration
2. Monitor validation accuracy during training
3. Use TensorBoard for training visualization
4. Run evaluation on test split before claiming results

---

## Future Improvements

### Short-Term (M8) — Real-World Accuracy Improvement
Based on research findings (SAME, DSTAdapter, SIGNET, EnsAug, KD-MSLRT):

**Phase 1: Data & Augmentation (Week 1-2)** ✅
- ✅ Geometry-aware augmentations (camera depth, shifting, rotation, finger articulation)
- ✅ Temporal augmentations (frame permutation, speed variation, random dropping)
- ✅ Landmark-specific augmentations (rotation, translation, keypoint removal)
- ✅ Augmentation pipeline with stats tracking and visualization
- ⬜ Record 50-100 clips of ourselves signing the 40 test words (requires physical recording)

**Phase 2: Domain Adaptation (Week 3-4)** ✅
- ✅ Test-Time Adaptation (TTA) with signer-aware MoE (SAME framework)
- ✅ Parameter-efficient fine-tuning with spatial-temporal adapters (DSTAdapter)
- ✅ Knowledge distillation from larger teacher to lightweight student

**Phase 3: Model Improvements (Week 5-6)** ✅
- ✅ Multi-scale feature fusion (combine hand + pose at multiple scales)
- ✅ Per-signer normalization statistics
- ⬜ Contrastive learning for robust representations (future)
- ⬜ Ensemble methods (train specialists on different augmentations) (future)

**Phase 4: Evaluation (Week 7)** 🔄
- ⬜ Signer-disjoint splits (ensure no signer overlap between train/test)
- ✅ Cross-dataset evaluation (ISL500, CISLR, ISLRTC) — 22.6% overall top-1
- ⬜ Real-world testing with diverse signers
- ✅ Ablation studies (measure impact of each improvement) — see `results/ablation/ablation_results.json`

**Target**: ≥50% top-1 on cross-dataset evaluation — **met on IV (68.0%)**, overall 41.0% (OOV-capped; 255/642 clips cannot be predicted by any 263-class head)

### Medium-Term (M9–M13: Continuous Sentence Program)
- Continuous sign language recognition (B2 forced alignment → A2 CTC/STARK revival)
- Any-person acceptance (calibration, signer-removal, SDDA)
- Live sentence demo + expo pack

### Long-Term
- Reverse direction (text → sign avatar)
- Sentence-level translation using ISH-NEWS dataset
- Real-time video conferencing integration
- Wearable device support (smart glasses)
