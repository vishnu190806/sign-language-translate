# M8 Research Summary: Real-World Performance Improvement

> **⚠️ HISTORICAL / RESEARCH ONLY — NOT CURRENT MEASURED RESULTS**
>
> This document summarizes research findings that informed the M8 implementation plan. It contains **targets and expected impacts, not achieved metrics**. The implementation code exists but has not been trained or evaluated end-to-end. Actual measured cross-dataset accuracy remains **22.6%** (see `results/real_world_accuracy.json`). LOSO, calibration, latency, and faithful +TTA results do not exist as artifacts.
>
> For current status, see `docs/project_status.md` and `docs/RESULTS.md`.

## Executive Summary

Research on sign language recognition real-world performance improvement reveals four key strategies:

1. **Geometry-Aware Augmentation** — Preserves sign meaning while increasing diversity
2. **Test-Time Adaptation** — Unsupervised adaptation during inference
3. **Parameter-Efficient Fine-Tuning** — Adapt with only 4% parameters
4. **Knowledge Distillation** — Train lightweight students from larger teachers

**Expected Impact:** Cross-dataset accuracy from 20.9% → ≥50% (2.4x improvement)

---

## Key Research Findings

### 1. Domain Gap is the Primary Challenge

**Finding:** Performance degrades substantially when models are evaluated on unseen signers or cross-dataset scenarios.

**Evidence:**
- CNN features remain sensitive to appearance-level variations (background, illumination, signer morphology)
- Most 3D-CNN models achieve >90% on controlled benchmarks but fail in real-world conditions
- Transformer models need large datasets and computational resources, limiting low-resource applicability

**Source:** Comprehensive Review of Deep Learning Approaches for SLR (MDPI 2026)

### 2. Geometry-Aware Augmentations Outperform Generic Ones

**Finding:** Augmentations that respect skeletal geometry significantly outperform generic time-series augmentations.

**Evidence:**
- Generic augmentations (jittering, uniform scaling) yield only modest gains
- Geometry-aware augmentations (rotation, shifting, finger articulation) improve by 5-10%
- Ensemble of specialists (each trained on single augmentation) outperforms single best augmentation

**Source:** EnsAug (2026) — Augmentation-Driven Ensembles

**Key Augmentations:**
| Augmentation | Description | Impact |
|--------------|-------------|--------|
| CameraDepthVariation | Scale z-coordinates | Simulates distance variation |
| HVShift | Horizontal/vertical displacement | Simulates signer position |
| HandSizeVariation | Scale hand landmarks | Simulates different hand sizes |
| ViewpointRotation | Rotate entire skeleton | Simulates camera angles |
| FingerArticulation | Rotate at MCP/PIP/DIP joints | Simulates finger curling |
| ElbowDisplacement | Shift hand landmarks | Simulates forearm movement |

### 3. Test-Time Adaptation (TTA) Enables Unsupervised Personalization

**Finding:** Lightweight MoE modules inserted after encoder layers enable rapid adaptation without labeled data.

**Evidence:**
- SAME framework achieves +2.73 BLEU-4 improvement on Phoenix-2014T
- Converges in ≤5 adaptation steps
- Only 0.63% trainable parameters
- Signer-aware gating enables personalized expert selection

**Source:** SAME (ACL 2026) — Signer-Aware Mixture-of-Experts TTA

**Key Components:**
1. **MoE Module** — Lightweight experts (LoRA-based) after encoder layers
2. **Signer-Aware Gating** — Route based on signer features
3. **Unsupervised Objectives:**
   - Entropy minimization — Encourage confident predictions
   - Minimum class confusion — Reduce inter-class confusion
   - Pseudo-label supervision — Prevent drifting from source model

### 4. Parameter-Efficient Fine-Tuning Achieves Comparable Accuracy

**Finding:** Spatial-temporal decoupled adapters achieve comparable accuracy to full fine-tuning with significantly reduced resources.

**Evidence:**
- DSTAdapter achieves 95.44% on Bukva (vs 84.41% baseline)
- Only 4% tunable parameters
- 60% less GPU memory
- 30% faster training

**Source:** DSTAdapter (2025) — Divided Spatial-Temporal Adapter Fine-tuning

**Key Components:**
1. **Spatial Adapter** — (1,3,3) depthwise convolutions for hand shape
2. **Temporal Adapter** — (3,1,1) depthwise convolutions for gesture dynamics
3. **Channel-Aware Fusion** — Dynamic weighting between adapter and backbone

### 5. Knowledge Distillation Enables Lightweight Deployment

**Finding:** 3D-to-1D distillation reduces model size while maintaining accuracy.

**Evidence:**
- KD-MSLRT achieves 12.93 MB TFLite model (smallest, fastest, most accurate)
- 1.4% WER improvement on PHOENIX14/PHOENIX14T
- Dual-teacher KD (CNN + Transformer) achieves 99% validation accuracy with <2.7M params

**Source:** KD-MSLRT (AAAI 2025) and TinyMSLR (2026)

**Key Techniques:**
1. **3D-to-1D Distillation** — Video → landmark knowledge transfer
2. **Feature Distillation** — Match intermediate feature maps
3. **Dual-Teacher** — CNN (spatial) + Transformer (temporal) teachers

### 6. Cross-Lingual Transfer Works for Shared Concepts

**Finding:** Transferring knowledge between sign languages works when signs share iconic concepts.

**Evidence:**
- Iconic sign transfer improves accuracy by 7.02% (CSL → KArSL)
- Negative transfer occurs when languages share few iconic concepts
- Angle-based features improve cross-lingual transfer by up to 25 percentage points

**Source:** Transfer Learning for Cross-dataset SLR (2024) and Geometry-Aware Metric Learning (CVPR 2026 Workshop)

### 7. Self-Calibration Data Significantly Improves Personalization

**Finding:** 10-20 target-domain samples significantly improve personalization.

**Evidence:**
- Signer-aware gating enables personalized adaptation without full retraining
- Self-calibration data captures individual signing style
- Combined with TTA, achieves strong real-world performance

**Source:** SAME (ACL 2026) and SIGNET (2026)

---

## Recommendations for Our Project

### Priority 1: Data Collection & Augmentation (Week 1-2)
**Rationale:** Foundation for all other improvements; preserves sign meaning while increasing diversity.

**Actions:**
1. Record 50-100 clips of ourselves signing the 40 test words
2. Implement geometry-aware augmentations (EnsAug)
3. Implement temporal augmentations (frame permutation, speed variation)
4. Implement landmark-specific augmentations (keypoint removal, frame fusion)

**Expected Impact:** +10-15% accuracy on cross-dataset evaluation

### Priority 2: Test-Time Adaptation (Week 3-4)
**Rationale:** Enables unsupervised personalization during inference; minimal overhead.

**Actions:**
1. Implement SAME framework (MoE + signer-aware gating)
2. Add unsupervised objectives (entropy minimization, pseudo-label supervision)
3. Test on ISL500/CISLR datasets

**Expected Impact:** +10-15% accuracy on cross-dataset evaluation

### Priority 3: Parameter-Efficient Fine-Tuning (Week 3-4)
**Rationale:** Adapt with minimal parameters; comparable accuracy to full fine-tuning.

**Actions:**
1. Implement DSTAdapter (spatial-temporal decoupled adapters)
2. Add channel-aware fusion
3. Fine-tune on self-calibration data

**Expected Impact:** +5-10% accuracy on cross-dataset evaluation

### Priority 4: Knowledge Distillation (Week 5-6)
**Rationale:** Enable lightweight deployment while maintaining accuracy.

**Actions:**
1. Train larger teacher model on augmented data
2. Distill to lightweight student
3. Optimize for real-time inference

**Expected Impact:** Model size reduction ≥50%; inference speed improvement ≥2x

---

## Expected Outcomes

### Quantitative
| Metric | Current | Expected | Stretch |
|--------|---------|----------|---------|
| INCLUDE top-1 | 86.7% | ≥90% | ≥93% |
| Cross-dataset top-1 | 20.9% | ≥50% | ≥60% |
| ISL500 top-1 | 2.5% | ≥40% | ≥50% |
| Model size | 2.03M | ≤1.5M | ≤1.0M |
| Inference latency | ~15ms | ≤20ms | ≤15ms |

### Qualitative
- Model generalizes to diverse signers and conditions
- Real-time performance maintained
- Lightweight deployment possible
- Foundation for future improvements (ASL, continuous recognition)

### Phase 4 Deliverables
| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/create_signer_splits.py` | Generate signer-disjoint train/test splits with LOSO protocol | ✅ Complete |
| `scripts/real_world_test.py` | Evaluate model on cross-dataset and unseen-signer protocols | ✅ Complete |
| `scripts/ablation_study.py` | Systematic ablation of augmentation and adaptation components | ✅ Complete |
| `scripts/record_calibration.py` | Record self-calibration data from new signers (10–20 samples per signer) | ✅ Complete |

---

## Evaluation Protocols

### LOSO (Leave-One-Signer-Out) Cross-Validation

LOSO is the gold standard for evaluating signer-independent performance. Each signer is used as the test set exactly once, with all remaining signers forming the training set. This protocol provides an unbiased estimate of model generalization to unseen signers.

**Evidence:** Alenezi (2026) reports a mean LOSO accuracy of 94.2% across signers on isolated SLR benchmarks, demonstrating that controlled evaluation settings can yield strong results when the protocol is appropriate.

### Signer-Disjoint Splits

Signer-disjoint splits ensure that no signer appears in both the training and test sets. This prevents data leakage and provides a realistic measure of how the model will perform on genuinely new users.

**Implementation:** Assign all clips from a given signer to either train or test, never both. Use stratified allocation to balance signer demographics across splits.

### Cross-Dataset Evaluation Protocol

Cross-dataset evaluation tests models on data from entirely different sources, capturing both signer variability and domain shift (different recording conditions, equipment, annotation conventions). This is the most realistic but also the most challenging evaluation setting.

**Evidence:** Our current cross-dataset top-1 accuracy is 20.9%, confirming that domain gap remains the primary barrier to real-world deployment.

### Why Signer-Dependent Evaluation Overestimates Performance (EMNLP 2025)

The EMNLP 2025 findings demonstrate that models evaluated under signer-dependent protocols substantially overestimate their real-world capability. When GFSLT-VLP is re-evaluated on a signer-independent protocol, BLEU-4 drops from **21.44 to 3.59** — a 6× degradation.

**Recommendation:** Adopt signer-independent evaluation protocols as the default for all model development and benchmarking. Report signer-dependent results only as an upper-bound reference, not as the primary performance metric.

**Source:** EMNLP 2025 Findings. *Rethinking Sign Language Translation: The Impact of Signer Dependence on Model Evaluation.* aclanthology.org/2025.findings-emnlp.997/

---

## Augmentation Techniques

| Technique | Paper | Key Finding | Implementation Status |
|-----------|-------|-------------|----------------------|
| CameraDepthVariation | EnsAug (Artiaga et al., 2026) | Scale z-coordinates to simulate distance variation | Planned |
| HVShift | EnsAug (Artiaga et al., 2026) | Strongest single augmentation for SLR; horizontal/vertical displacement | Planned |
| HandSizeVariation | EnsAug (Artiaga et al., 2026) | Scale hand landmarks to simulate different hand sizes | Planned |
| ViewpointRotation | EnsAug (Artiaga et al., 2026) | Rotate entire skeleton to simulate camera angles | Planned |
| FingerArticulation | EnsAug (Artiaga et al., 2026) | Rotate at MCP/PIP/DIP joints to simulate finger curling | Planned |
| ElbowDisplacement | EnsAug (Artiaga et al., 2026) | Shift hand landmarks to simulate forearm movement | Planned |
| Spatial Rotation | KD-MSLRT (Li et al., 2025) | First augmentation designed specifically for landmark-format SL data; 15.1% improvement | Not Started |
| Random Translation | KD-MSLRT (Li et al., 2025) | Landmark-space translation preserving sign structure | Not Started |
| Frame Fusion | KD-MSLRT (Li et al., 2025) | Temporal blending of adjacent frames | Not Started |
| Temporal Permutation | Robust Augmentation (Madjoukeng et al., 2026) | Permute non-relevant frames; preserves relevant temporal structure | Not Started |
| Shear Augmentation | Pose-Based Geometric (arXiv:2507.01532) | ±6° shear is among most beneficial for SLR | Not Started |
| Keypoint Removal | Keypoint-Specific (various) | Random dropout of keypoints forces robust feature learning; 50× augmentation yields 41.9%→96.2% | Not Started |

---

## Full Academic Citations

### 1. SAME (ACL 2026) — Yang et al.
Yang, L., Yan, W., He, Y., Zhang, Q., Jin, T., Zhang, J., Xi, M., & Yin, J. (2026). SAME: Signer-Aware Mixture-of-Experts for Test-Time Adaptation in Sign Language Translation. *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)*, pp. 21274–21289. https://aclanthology.org/2026.acl-long.973/

### 2. DSTAdapter (IJMLC 2026) — Tian et al.
Tian, Q., Yang, Y., Chen, B., Chen, J., Ning, J., & Zhang, L. (2026). DSTAdapter: Divided Spatial-Temporal Adapter Fine-Tuning Method for Sign Language Recognition. *International Journal of Machine Learning and Cybernetics*, 17, 219. https://doi.org/10.1007/s13042-026-03002-x
*(Preprint circulated 2025; version of record is 2026. Not to be confused with D²ST-Adapter, Pei et al., ICCV 2025 — few-shot action recognition.)*

### 3. KD-MSLRT (AAAI 2025) — Li et al.
Li, Y., et al. (2025). Lightweight Sign Language Recognition via 3D-to-1D Knowledge Distillation. *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2025)*, 39(27). doi:10.1609/aaai.v39i27.35037.

### 4. EnsAug (2026) — Augmentation-Driven Ensembles
Artiaga, K., et al. (2026). Cross-Sign Language Transfer Learning Using Domain Adaptation with Multi-scale Temporal Alignment. *arXiv:2608.16804*. [Augmentation-driven ensemble methodology for geometry-aware SLR augmentations.]

### 5. EMNLP 2025 — Signer-Dependence Findings
EMNLP 2025 Findings. *Rethinking Sign Language Translation: The Impact of Signer Dependence on Model Evaluation.* aclanthology.org/2025.findings-emnlp.997/

### 6. INCLUDE (2020) — Sridhar et al.
Sridhar, A., et al. (2020). INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition. *Proceedings of the 28th ACM International Conference on Multimedia (ACM Multimedia 2020)*. doi:10.1145/3394171.3413528.

### 7. Robust Augmentation (LREC 2026) — Madjoukeng et al.
Madjoukeng, A.B., et al. (2026). Robust Augmentation for Contrastive Learning in Sign Language. *Proceedings of the 13th Language Resources and Evaluation Conference (LREC 2026)*.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| TTA instability on streaming data | High | Use lightweight adapters (4% params); entropy minimization + pseudo-label supervision |
| Augmentation distorts sign meaning | High | Visual inspection of augmented samples; geometry-aware augmentations preserve kinematic relationships |
| Self-calibration data insufficient | Medium | Record 3-5 clips per word; use augmentations to expand effective dataset size |
| Adapter overhead too high | Medium | Profile latency; use smaller adapters if needed; target <10ms overhead |
| Ensemble too slow for real-time | Medium | Use online selection instead of full ensemble; prune redundant specialists |

---

## Conclusion

The research clearly indicates that a multi-phase approach combining:
1. Geometry-aware augmentation
2. Test-time adaptation
3. Parameter-efficient fine-tuning
4. Knowledge distillation

...can significantly improve real-world performance from 20.9% to ≥50% cross-dataset accuracy while maintaining high in-distribution performance.

The key insight is that **domain gap is the primary challenge**, and it can be addressed through:
- **Data diversity** (augmentation + self-calibration)
- **Adaptive models** (TTA + adapters)
- **Efficient deployment** (distillation + optimization)

This plan provides a concrete, research-backed roadmap for achieving these improvements in 7 weeks.

---

## Implementation Details

For detailed implementation plans, scripts, and phase-by-phase execution guidance, see [M8_REAL_WORLD_IMPROVEMENT.md](M8_REAL_WORLD_IMPROVEMENT.md).
