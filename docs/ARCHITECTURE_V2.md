# Architecture V2 — Upgraded ISL System

## Basis
V2 keeps the proven M0–M7+ pipeline (webcam → MediaPipe Tasks 168-d → HTMA → StabilityGate → SentenceAssembler → Piper) and fixes measurement, data pipeline, and adaptation based on verified research.

## Decisions and citations

1. **Backbone: HTMA early-fusion, D=256, 2 blocks, 8 heads.**
   Varanasi et al., CVPRW 2026 MSLR: 95.88% original INCLUDE split, 83.82% signer-independent split, 19.20% CISLR, <10M params. Same paper: signer overlap inflates accuracy up to 12%; pseudo-signer grouping Alg.1 for signer-independent eval.
   `src/models/htma.py`, `src/models/builder.py`.

2. **Report IV-only, never blended overall.**
   `results/real_world_accuracy.json`: 642 clips = 387 IV + 255 OOV; OOV top-1 0/255 by definition. IV top-1 145/387 = 37.47%. Blended 22.6% understates model and conflates vocab coverage with generalization.

3. **Signer-disjoint eval before claiming gains.**
   EMNLP 2025 Findings (Artiaga et al.): GFSLT-VLP BLEU-4 21.44→3.59, GASLT 15.74→8.26 signer-independent. `scripts/create_signer_splits.py` LOSO exists; V2 wires it into eval protocol. `scripts/eval_iv_only.py` is step 1; pseudo-signer splits per HTMA Alg.1 is step 2.

4. **Data pipeline for RTX 4070 Laptop 8GB + i9-14900HX + 16GB + NVMe.**
   PyTorch DataLoader docs: `num_workers 4→6`, `prefetch_factor=2`, `persistent_workers=True`, `pin_memory=True` + `non_blocking=True` transfers. Batch `64→256`: 2M-param model uses <500MB at B=64; B=256 saturates Ada Tensor Cores. `torch.backends.cudnn.benchmark=True`, `set_float32_matmul_precision("high")`, bf16 AMP without GradScaler (bf16 needs no loss scaling; Per PyTorch AMP docs GradScaler is for fp16). `LandmarkDataset` manifest `_manifest.json` removes O(N) `np.load` scan; `set_epoch()` removes fixed-seed-42 repetition.
   `src/data/dataset.py`, `src/train/trainer.py`, `configs/default.yaml`.

5. **Augmentation: 5 proven ops, not 19.**
   Laines et al., CVPRW 2023 (TSSI/WLASL): scale, shift, noise, subsample, time-interp improve skeleton SLR. Bohacek & Hruz 2022: separate hand/body spaces + novel aug. SSLR 2025: in-plane rotation ≤13°, arm rotation ±4°, head-height + hand-bbox normalization (+10% over raw). KD-MSLRT (AAAI 2025, Li et al.): landmark-specific aug for MediaPipe robustness. V2 `configs/v2_tuned.yaml` enables only: `gaussian_noise 0.5, spatial_scale 0.3, random_translation 0.3, speed_variation 0.3, random_keypoint_removal 0.3`. Rationale: ablation showed 19-op full stack −6.07% ID (`results/ablation/ablation_results.json`).

6. **PEFT adapters on frozen backbone for calibration.**
   DSTAdapter, Springer 2026: divided spatial-temporal adapters for frozen-backbone video SLR. CLIP-SLA, CVPRW 2025 (Alyami/Luqman): SLA-Adapter/SLA-LoRA PEFT beats partial and full fine-tuning (full FT causes catastrophic forgetting). V2: freeze HTMA, train `src/models/dst_adapter.py` (~11% params) on calibration clips only.
   `configs/m8_adapter.yaml`.

7. **TTA: episodic entropy-min with confidence filter, faithfully measured.**
   SAME, ACL 2026 (Yang et al.): signer-aware MoE TTA after encoder layers, entropy + class-confusion + pseudo-label guards. EATA, ICML 2022 (Niu et al.): filter high-entropy/unreliable samples, Fisher anti-forgetting. SAR, ICLR 2023: sharpness-aware entropy. DualTTA, CVPR 2026: entropy-min likely-correct + entropy-max likely-incorrect. V2: `SAMEAdapter.adapt()` ≤5 steps, report delta on ISL500-IV only + latency overhead. Prior `+TTA` in `scripts/ablation_study.py` was augmentation-averaging, not adaptation — not reported as TTA.

8. **Distillation: KD-MSLRT recipe.**
   Li et al., AAAI 2025: 3D-to-1D cross-modal KD + text-correction, 12.93MB TFLite, −1.4% WER PHOENIX. V2 keeps `src/train/distiller.py` (T=4.0, alpha=0.7) for mobile export, not for accuracy gains (ablation showed −7.11% with current teacher).

9. **Dropped paper claims.**
   `EnsAug 2026`, `SIGNET 2026`, `TinyMSLR 2026` have no ACL/CVPR/AAAI/arXiv match. Corresponding code (`augment_geo.py` 6 ops, adapters) retained as custom heuristics, not paper replications.

## Targets (IV-only)
- INCLUDE-IV ≥85%, ISL500-IV ≥30–35% (not 50% blended — unrealistic with 57% OOV).
- Signer-disjoint INCLUDE ~70% expected (81.8% random-split minus leakage).
- Latency ≤25ms/frame, FPS ≥25 during TTS (`scripts/benchmark_latency.py`).

## Autonomous execution order
1. `scripts/eval_iv_only.py` (CPU, minutes).
2. Pseudo-signer split eval with frozen `models/best.pt` (no retrain).
3. Calibration recording (human, 120 clips) — only manual step.
4. One retrain with `configs/v2_tuned.yaml`.
5. PEFT adapter fine-tune on calibration.
6. Faithful TTA delta + latency table.

## Final verdict (2026-09-26, measured)
- Deployment model is `models/multihead/best.pt`: **INCLUDE test 77.20%, ISL500 unseen-users 61.82%, real-world overall 41.0%, real-world IV 68.0%** — one checkpoint covering both regimes. Demo default points at it (auto-detects multi-head/adapter/plain checkpoints; label map resolved from checkpoint dir).
- Joint-softmax path rejected with mechanism identified: pose-absent manifold + 10:1 in-class source skew collapsed INCLUDE to ~0.7% (proven by zero-pose 0.63%→18.62% experiment and slot-energy probe; hand-swap test negative).
- DSTAdapter PEFT verified useful: +3.98% on weak base (`best.pt` 75.73%→79.71%) with 11.6% trainable params; parity (−0.84%) on the champion. Adapter support built into live demo for future calibration personalization.
- LOSO on `isl_40_test` invalid (degenerate signer column); per-source IV eval stands as OOD evidence.
- Negative results kept as artifacts (`results/v2/`, `results/joint/`, `results/ablation_v2/`, `results/ablation_champion/`) — no silent discarding.
