# Evaluation Results

## Evidence and Scope

This document reports the measured result artifacts below:

- Full-dataset test metrics: `results/full_dataset/metrics.json`
- Preserved cross-dataset aggregate: `results/real_world_accuracy.json`
- Validated per-signer evaluation: `results/real_world_test.json`
- Ablation results: `results/ablation/ablation_results.json`

The artifacts do not establish signer-disjoint/LOSO performance, calibration, latency, or a measured +TTA result. No such claim is made here.

## Dataset and Vocabulary

- **Dataset**: INCLUDE (AI4Bharat)
- **Full-dataset test evaluation**: 478 samples across 264 classes
- **Cross-dataset artifact**: 642 clips from four sources
- **Test vocabulary**: 40 words
- **In-vocabulary (IV)**: 17 words and 387 clips
- **Out-of-vocabulary (OOV)**: 23 words and 255 clips

**Vocabulary/class-count ambiguity (resolved 2026-09-26):** the original `models/label_map.json` has **264 keys** with max index **263**, because two keys normalize to the same word (`extra` collision). Any 263-output model crashes on class #263 (CUDA assertion, observed). All joint/multi-head training uses `data/processed/landmarks/joint_label_map.json`: 263 normalized words, contiguous indices 0–262, all 180 ISL500-overlap words resolving. The 264-vs-263 wording below is preserved for the historical artifacts that used the original map.

No separately validated IV/OOV accuracy result is reported.

## Model

- **Architecture**: HTMA (Head-Token Mixing Attention)
- **Checkpoint classes**: 264
- **Normalized model labels**: 263 (see ambiguity above)
- **Parameters**: 2,026,056 total; 2,026,056 trainable
- **Embedding dimension**: 256
- **HTMA blocks**: 2
- **Attention heads**: 8
- **Input**: `(B, T, 168)`, with sequence length 32

## Full-Dataset Test Results

| Metric | Value |
|--------|------:|
| Top-1 accuracy | **81.80%** |
| Top-3 accuracy | **93.51%** |
| Top-5 accuracy | **95.40%** |
| Macro accuracy | **83.33%** |
| Samples | **478** |
| Classes | **264** |
| Parameters | **2,026,056** |

Source: `results/full_dataset/metrics.json`

## Cross-Dataset Aggregate Artifact

Source: `results/real_world_accuracy.json`.

- **Clips**: 642
- **Overall Top-1**: **22.6%**
- **Overall Top-3**: **28.0%**
- **Overall Top-5**: **31.6%**

### Source Top-1 Results

| Source | Clips | Top-1 | 95% CI (Wilson) |
|--------|------:|------:|-----------------|
| INCLUDE | 143 | **86.0%** | 79.4% – 90.8% |
| ISL500 | 405 | **4.4%** | 2.8% – 6.9% |
| CISLR | 81 | **4.9%** | 1.9% – 12.0% |
| ISLRTC_dictionary | 13 | **0.0%** | 0.0% – 22.8% |
| **Overall** | **642** | **22.6%** | — |

**How to read the 22.6% overall figure (important).** The overall number is dominated by source
mix, not by model quality: ISL500 supplies 405 of 642 clips and scores 4.4%, so it alone drags the
aggregate down. It should not be quoted as a headline "generalization" number.

**The OOV split is 0/255 by construction** (`results/real_world_accuracy.json`:
`out_of_vocabulary: {total: 255, correct_top1: 0}`). A 263-class model is evaluated on 23 words it
was never trained on, so 0% is a **tautology, not a result** — it carries no information and
should never be presented as a measured capability or a weakness on its own.

**CISLR is not an apples-to-apples comparison target.** CISLR (Joshi et al., EMNLP 2022,
`2022.emnlp-main.707`) is a **4,765-class one-shot prototype task** over YouTube-scraped dictionary
videos with ~1.5 videos per word. Our model has 263 classes, so most CISLR queries are outside its
label space entirely. Published CISLR numbers (16.81% original authors; 19.20% Varanasi et al.
CVPRW 2026) are all **one-shot-protocol** numbers and are not directly comparable to our 4.9%.
Our 95% CI is [1.9%, 12.0%], so 19.20% does sit outside it, but n=81 is too thin to make a
strong claim either way. Report CISLR only with the protocol stated.

**ISLRTC 0.0% is NOT a label-mapping bug.** Checked directly against
`isl_40_test/metadata.csv` (2026-10-02): all 13 ISLRTC rows carry valid, correctly-normalized
labels (`goodbye`←`Bye`/`Bye_Goodbye`, `stop`←`Stop`, `me`←`Myself`, `when`←`When_(For_Days)`, etc.).
The 0.0% is a genuine model failure on 13 clips — but with n=13 the 95% CI is [0.0%, 22.8%], so this
is **underpowered, not diagnostic**. It should not be used as evidence of a domain gap.

## Validated Per-Signer Artifact

Source: `results/real_world_test.json`.

- **Clips**: 642
- **Signer grouping labels**: 172
- **Clip-weighted Top-1**: **21.0280%**
- **Clip-weighted Top-3**: **28.0374%**
- **Clip-weighted Top-5**: **30.5296%**

The signer values are grouping labels from the artifact, not verified identities.

### Source Top-1 Rates

| Source | Clips | Top-1 |
|--------|------:|------:|
| INCLUDE | 143 | **82.5175%** |
| CISLR | 81 | **1.2346%** |
| ISL500 | 405 | **3.9506%** |
| ISLRTC_dictionary | 13 | **0.0%** |

The aggregate artifact and the validated per-signer artifact are separate outputs with different reported rates. They are presented independently and are not merged or treated as interchangeable.

## IV/OOV Counts

| Category | Words | Clips |
|----------|------:|------:|
| In-vocabulary | 17 | 387 |
| Out-of-vocabulary | 23 | 255 |
| **Total** | **40** | **642** |

These are vocabulary and clip counts from the preserved cross-dataset artifact. No IV/OOV accuracy percentage is asserted.

## Ablation Study

Source: `results/ablation/ablation_results.json`. The measured conditions are evaluated on the full-dataset test split with 478 samples.

| Condition | Top-1 | Top-3 | Top-5 | Delta Top-1 vs. baseline |
|-----------|------:|------:|------:|--------------------------:|
| Baseline | **81.80%** | **93.51%** | **95.40%** | 0.00% |
| +augmentation | **75.73%** | **92.26%** | **95.61%** | -6.07% |
| +adapters | **81.80%** | **93.51%** | **96.03%** | 0.00% |
| +distillation | **74.69%** | **92.47%** | **95.61%** | -7.11% |
| +all | **77.20%** | **92.68%** | **95.19%** | -4.60% |

There is **no measured +TTA row** in the ablation artifact. The artifact's `tta_passes: 5` metadata is not itself a measured +TTA result, so no +TTA accuracy or completion claim is reported.

## Evaluation Command Recorded for the Validated Per-Signer Run

```powershell
.\venv\Scripts\python.exe scripts\real_world_test.py --model-path models\baseline.pt --dataset-dir isl_40_test --output-dir results
```

The full-dataset and ablation values above are reported from their JSON artifacts; this document does not infer additional protocol claims from those files.

## V2 Final Results (2026-09-25, RTX 4070 Laptop GPU)

All artifacts measured this run. Deployment model: `models/baseline.pt` (demo default).

| Artifact | Result |
|----------|--------|
| `results/iv_only_metrics.json` | IV top-1 **37.47%** (145/387); overall 22.59%; OOV 0/255 by definition |
| `results/v2/metrics.json` (tuned 5-op retrain, batch 256) | Test top-1 **71.13%**, top-5 92.26%, macro 72.99% — rejected, worse than baseline |
| `results/ablation_v2/ablation_results.json` (base `best.pt` 75.73%) | +adapters **79.71% (+3.98)** — PEFT helps weak base |
| `results/ablation_champion/ablation_results.json` (base `baseline.pt`) | Baseline **81.80%** / 93.51% / 95.40% reproduced; +champion-adapter 80.96% (−0.84), top-5 96.23% (+0.83) — adapter at parity, plain champion deployed |
| `docs/benchmarks/v2_baseline_metrics.json` | **55 FPS, p50 18.14ms, p95 19.79ms** on RTX 4070 Laptop — within ≤25ms target |
| LOSO dry-run on `isl_40_test/metadata.csv` | Invalid protocol: `signer` column mixes per-video IDs with category labels (~172 degenerate folds, leakage every fold). Per-source IV eval remains the valid OOD evidence. |
| `results/multihead_by_source.json` (shared encoder, per-source heads) | INCLUDE test **77.20%** / 94.98% top-5 (478 clips); ISL500 unseen-users **61.82%** / 70.68% (4,198 clips) — both regimes alive in one checkpoint |
| `results/real_world_test_multihead.json` (642 clips, per-source head routing) | Overall top-1 **41.0%** (263/642) vs 21.0% baseline; **IV top-1 68.0%** (263/387) vs 37.5%; ISL500 users ~34.8% vs 4.4% |
| `results/prototype_decode_v3.json` (139 cosine prototypes, frozen encoder) | WER **145.2%** (sub=116 del=26 ins=70) — frozen space not discriminative for untrained words |
| `results/ce_window_decode.json` (CE-pretrained head, sliding windows) | WER **100.0%**, all deletions — isolated-trained head emits nothing on continuous windows (isolated↔continuous mismatch) |
| `results/temperature.json` + `models/multihead/temperature.json` (Guo et al., ICML 2017) | Per-head T on val: include **0.823** (NLL 0.870→0.813, ECE 0.095→0.056), isl500 **0.884** (NLL 0.519→0.499, ECE ~0.026 unchanged). Wired into live demo (`src/app/demo.py`); argmax accuracy unchanged by construction |
| `results/temperature_v2.json` + `models/multihead_v2/temperature.json` (Guo et al., ICML 2017) | Per-head T on val: include **0.828** (NLL 0.706→0.655, ECE 0.103→0.052), isl500 **0.851** (NLL 0.467→0.436, ECE 0.037→0.031), extra **0.798** (NLL 0.750→0.683, ECE 0.088→0.024 — biggest win, was uncalibrated). Wired per-head into live demo (`unified_predict(temperatures=)`, `src/infer/routing.py`); argmax unchanged by construction |
| `docs/benchmarks/v2_shipped_metrics.json` | Unified 3-head forward (batch-1, seq-32): **14.70ms** on RTX 4070 — encoder runs 3× (once per head); loop budget holds (~45 FPS at stride 2) but headroom thinner than stale 18ms claim. Optimization noted (encode-once): not done |
| `results/gate_sweep/sweep.jsonl` (15 settings, gate emissions scored) | Default (0.40/12/0.60/0.08): IV top-1 63.3%, prec 73%, OOV-false 75.3%. Winner **threshold 0.50**: IV top-1 61.5% (−1.8pp), prec 76%, OOV-false **62.0%** (−13pp). Applied to `configs/default.yaml`. No setting meets ≤10% OOV-false (closed-set classifier emits confidently on OOV — documented limit) |
| `results/gate_sweep_v2/sweep.jsonl` (17 settings, demo-faithful unified 846-dim probs) | Status quo 0.50/12/0.60/0.08 **KEPT**: IV 52.2% (611 clips — vocab expansion moved 224 clips OOV→IV vs v1 sweep), OOV-false 64.5% (n=31, noisy). No setting beats it within the ≤2pp rule (thr 0.60: −2.9pp IV for −9.7 OOV; delta 0.0: +2.0 IV but +3.2 OOV). Extra head fires 129/518 emissions (25%) — new vocab live in gate path. No config change |
| `results/tta_episodic.json` (SAME-MoE, k=8, 5 steps, lr=3e-5, per-actor reset) | Unseen actors 014/015: base 26.5%/84.4% → adapted 26.3%/84.5%, mean delta **−0.0000**, win rate 50% — **ABORTED per protocol** (noise-level, do not ship). Notable: base per-actor spread 26–84% confirms signer variance dominates |
| `results/polish_eval.json` (C2, Qwen2.5-1.5B-Instruct fp16 + transformers, greedy) | Exact **23.4%→37.2%** (target ≥35% ✅), F1 83.4%→83.3% (no regression ✅), 69 LLM / 25 rules-fallback, 1.53s/sent. Wired into live demo speak path with lazy load + fallback |
| `tests/test_multihead.py` (11 tests) | Multi-head shapes/heads/mask/roundtrip, grouped-sampler coverage, pose-dropout identity/zeroing — all pass; full fast suite 54/54 |
| `results/polish_eval_v2.json` → shipped (prompt tuning, 2026-10-03) | 3-way showdown: V0 37.2%/83.3, V1 (one-shot) 34.0%/82.7 ❌, V2 (no-new-content + length cap) **43.6%/84.7** ✅ — exact +6.4pp, F1 +1.4, fallbacks 25→18. V2 prompt shipped in `src/text/polish.py`. GPU verified in use (dmon: mean 23.7% SM, max 100%, bursty generate + Python gaps) |
| `results/confusion_test/metrics.json` + `results/official_split_confusion.json` (confusion-driven retrain, 2026-10-02) | 25 low-recall classes oversampled 2× (`results/confusion_targets.json`, 40% of confusion mass), champion recipe otherwise identical, 95 epochs, best val 84.96% → **ABORTED**: custom test **74.69%** (−7.1pp vs 81.8%), official **84.91%** (−0.3pp vs 85.22%). Targets moved (mean recall 0→0.22, 7/25 improved) but errors redistributed to previously-correct classes — net negative. Champion `baseline.pt` retained. Infra kept: `oversample_labels` in dataset/trainer + `tests/test_confusion_oversample.py` (5 tests) |
| `results/dpc_finetune_test/metrics.json` + `results/official_split_dpc.json` (DPC pilot, 2026-10-03) | OpenHands Sign_Library 29h → 168-d memmap (3.15M frames, md5-verified) → DPC (7×10 windows, 4-ctx, 3-pred, InfoNCE, batch 128, 50k iters in 19 min after 8× loop speedup) → probe ladder scratch 16.5% → **30k 32.3%** (plateau after) → fine-tune step-30k encoder, 85 epochs, best val 82.96% → **ABORTED per ship rule**: custom **76.99%** (−4.8pp vs 81.8%), official **85.52%** (+0.3pp vs 85.22, within noise). Transfer is real (probe +15.8pp) but full fine-tune doesn't convert it to test gains — supervised-from-scratch already saturates this regime. Champion retained. Infra kept: bridge/sampler/DPC/probe scripts + 8 tests |
| `results/multihead_v2_gates.json` + `results/extra_eval.json` (vocab expansion, 2026-10-03) | 320 ISL500-only words (≥73 train clips each) → frozen extra head (extra-val 79.54%, unseen-user test **60.25%**/7,578 clips; 73 shared tensors verified bit-identical) → joint 3-head fine-tune (warm-started, include-val floor guard, best mean-val 87.05 @ ep27). **SHIPPED as deployment default**: custom **82.22%** (+0.4pp vs 81.8, +5.0pp vs deployed v1), official **84.42%** (+1.9pp vs deployed v1 82.54; −0.8pp vs non-deployed specialist on mismatched 974-vs-1008 protocol). Vocab 263 → 583. Drive-by fix: live demo had been calling multi-head with `source=None` = random-init head; now max-over-heads unified routing (`src/infer/routing.py`, per-head maps) — strictly improves old demo too |
| `results/per_user_v2.json` + `results/per_user_rows.json` (signer-grouped eval, 2026-10-09) | ISL500 unseen users, max-over-heads: **014: 27.41%** (n=5,794) vs **015: 82.36%** (n=5,982) — signer spread dominates, consistent with TTA finding. Head-win precision: extra 45–90%, isl500 21–76%, include 3–29% (include misroutes on ISL500 domain). Margin rule (include needs +δ over others): δ=0.30 → 014 +1.0pp, 015 +1.1pp — **PARKED**: gain tiny, INCLUDE-domain impact unmeasured, behavior risk unjustified |
| `data/processed/synthetic_llm/` (B3 LLM half, 2026-10-09) | 30/30 Qwen-novel gloss sentences, 0 real overlap (`scripts/synth_llm_novel.py`); quirky semantics irrelevant (boundaries exact, pretraining-grade) |
| `docs/benchmarks/encode_once.json` | `predict_all` (single encode + 3 heads): 10.97→3.45ms (**3.2×**); live demo path already uses it |

## Joint-softmax path (rejected with mechanism, 2026-09-26)

| Artifact | Result |
|----------|--------|
| `models/joint/best.pt` (union 263-class head, balanced sampler) | Val 87.94%; test: ISL500-unseen 60.39% but INCLUDE **0.63%** — single head serves one regime only |
| `models/joint_v2/best.pt` (+pose-dropout aug, source-balanced sampler) | Val 89.38%; test: ISL500 62.51%, INCLUDE **0.84%** — fixes insufficient |
| Mechanism probes (`scripts/probe_slots.py`, `scripts/eval_include_ablate.py`) | Slot conventions match (single-hand signs in slot 1 both sets); pose energy ISL500 = 0.0; zeroing pose on INCLUDE inputs jumps joint model 0.63%→18.62% (pose-absent manifold); hand-swap test negative (0.63%, mirroring ruled out); remaining gap = 10:1 in-class source skew |
| Conclusion | Flat union softmax structurally wrong for pose-present + pose-absent sources. Multi-head (Logos recipe, Ovodov et al. 2025) adopted instead — see multi-head rows above. |
