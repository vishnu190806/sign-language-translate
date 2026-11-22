# Research Evidence Index

This directory contains a **curated, publication-sized copy** of selected results from
`results/` (which remains in the original project but is not committed to Git). The
originals were not moved or modified. Every number quoted in the README and in
`docs/RESULTS.md` traces to one of these artifacts.

Curated total: ~2.57 MB across 27 metric files and 3 screenshots.

## Headline results

| Artifact | Measured result |
|----------|-----------------|
| `evidence/full_dataset_metrics.json` | INCLUDE full-dataset test: **81.80% top-1**, 93.51% top-3, 95.40% top-5 (478 samples, 264 classes) |
| `evidence/official_split_metrics.json` | Matched official-split evaluation: **85.22% top-1** |
| `evidence/multihead_v2_gates.json` | Shipped 3-head/vocab-expansion gates: **82.22% custom** / **84.42% official**, 583-word vocabulary |
| `evidence/real_world_accuracy.json` | Cross-dataset aggregate: 642 clips, 22.6% top-1 — **source-mix dominated**, not a generalization headline |
| `evidence/iv_only_metrics.json` | In-vocabulary subset: **37.47% top-1** (145/387); OOV 0/255 is a tautology |
| `evidence/real_world_test_multihead.json` | Per-source head routing: 41.0% overall, **68.0% IV** on 642 clips |
| `evidence/multihead_by_source.json` | Shared encoder, per-source heads: INCLUDE 77.20%, ISL500 unseen-users 61.82% |
| `evidence/ablation_results.json` | Augmentation −6.07pp, adapters 0.00pp, distillation −7.11pp, all −4.60pp vs baseline |
| `evidence/ablation_champion_results.json` | Champion reproduced at 81.80%; champion adapter at parity (−0.84pp) |
| `evidence/polish_eval_v2.json` | Sentence polish V2 prompt: exact **43.6%**, F1 **84.7%** (up from 37.2% / 83.3%) |
| `evidence/gloss2text_rules.json` | Rule-based gloss→text: 23.4% exact, 83.4% F1 |
| `evidence/bench_v2_baseline_metrics.json` | 55 FPS, p50 18.14ms, p95 19.79ms on RTX 4070 Laptop |
| `evidence/bench_v2_shipped_metrics.json` | Unified 3-head forward: 14.70ms at batch-1, seq-32 |
| `evidence/bench_encode_once.json` | Encode-once optimisation: 10.97→3.45ms (3.2×) |
| `evidence/tta_episodic.json` | Episodic SAME-MoE: mean delta −0.0000, 50% win — **no reliable per-signer gain; not shipped** |
| `evidence/dpc_probe.json` / `evidence/official_split_dpc.json` | DPC pretraining: probe +15.8pp but fine-tune −4.8pp custom — **aborted** |
| `evidence/official_split_confusion.json` | Confusion-driven retrain: −7.1pp custom — **aborted** |
| `evidence/gate_sweep_v2.jsonl` | Stability-gate sweep (17 settings): status quo retained |
| `evidence/per_user_v2.json` | ISL500 unseen users 014: 27.41% vs 015: 82.36% — signer spread dominates |
| `evidence/vocab_overlap.json` | 180/263 INCLUDE words overlap ISL500 (68%) |
| `evidence/temperature_v2.json` | Per-head temperature scaling (ECE reductions) |
| `evidence/kokoro_ab.json` | Kokoro vs Piper A/B (Kokoro won listen-test; steady-state ≈ Piper) |

## Screenshots

- `screenshots/ui_clean.png` — clean live HUD with confidence bar and sentence bar.
- `screenshots/ui_debug.png` — debug view: per-head tops, gate state, latency, FPS sparkline.
- `screenshots/ui_sentence.png` — live rules-draft preview and auto-speak countdown.

## Failed and parked work (kept deliberately)

These are documented as failures, not rewritten as successes: CTC decoding collapse
(`docs/CONTINUOUS_PLAN.md`), 19-op augmentation stack (−6.07% ID), joint softmax
(`docs/RESULTS.md`), DPC and confusion retrains (above), TTA (above), and the
unverified-then-verified citation history (`docs/RESEARCH_VALIDATION.md`).

## Full documentation

- `docs/RESULTS.md` — all metrics with scope and caveats.
- `docs/RESEARCH_VALIDATION.md` — citation and evaluation validation.
- `docs/architecture.md` and `docs/ARCHITECTURE_V2.md` — architecture and its evolution.
- `docs/PROJECT_TRAJECTORY.md` — milestone history.
- `docs/QA_REPORT.md` — verification and test history.
- `docs/LIMITATIONS.md` — known limitations.
