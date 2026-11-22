# Autonomous Audit — zero-human-input levers (2026-10-03, static analysis only)

Scope: shipped `models/multihead_v2/best.pt` (custom 82.22%, official 84.42%,
`results/multihead_v2_gates.json`); extra head 60.25% unseen-user (`results/extra_eval.json`);
polish exact 37.2% (`results/polish_eval.json`); no code/config/model/result modified,
nothing downloaded, no training or GPU runs for this audit.

## Survivors (ranked by expected-gain-per-hour)

### 1. Qwen polish prompt tuning — SURVIVE (highest)
- Gain: +2–5pp exact-match plausible (37.2%→~40%), F1 ≥83.3% guardrail; artifact:
  refreshed `results/polish_eval.json` + winning prompt in `src/text/polish.py:11-18`
  (`SYSTEM`). Rules baseline is weak (exact 23.4%, `results/gloss2text_rules.json:3`),
  so headroom is prompt-shape, not model-shape. [UNVERIFIED magnitude / MEASURED protocol]
- Cost: ~1 h; ~94 sents × 1.53 s × 3 variants ≈ 7–10 min GPU, Qwen weights already local
  (`models/qwen15b/model.safetensors`), protocol script exists (`scripts/eval_polish.py`).
  Zero human input, zero downloads.

### 2. Re-tune gate threshold on v2 routing — SURVIVE
- Gain: OOV-false cut of similar size to last time (75.3%→62.0% at thr 0.50,
  `results/gate_sweep/sweep.jsonl:1,4`; threshold now in `configs/default.yaml:24`).
  Stale by construction: sweep ran on old single-head probs, deployment is now 583-class
  max-over-heads (`src/infer/routing.py:7-33`), which dilutes per-class confidence, so 0.50
  is unlikely to still be optimal. No top-1 change (gate never changes argmax). [MEASURED here]
- Cost: ~1–2 h; GPU forward over 642 clips + CPU rescore (`scripts/sweep_gate.py` exists).

### 3. Signer-grouped (pseudo-LOSO) eval from existing predictions — SURVIVE
- Gain: no accuracy; artifact is the honest signer-variance number we never published:
  per-actor base already measured at 26.5% (user014, n=2059) vs 84.4% (user015, n=2123)
  (`results/tta_episodic.json`). ISL500 carries signer IDs
  (`scripts/convert_isl500.py:124`, `signer=f"isl500_user{user}"`; train 001–011 / val
  012–013 / test 014–015 per `:11`), so grouping is free. True retrain-per-fold LOSO is
  killed (15 GPU retrains; training banned). [MEASURED here]
- Cost: ~15 min CPU, JSON regroup only.

### 4. LLM-novel B3 half — SURVIVE
- Gain: no immediate accuracy; artifact: novel gloss-sentence list extending the DONE
  template half (200 novel sentences, bigram coverage 2.00× real, `docs/CONTINUOUS_PLAN.md:13`).
  Qwen2.5-1.5B local (`models/qwen15b/`), `transformers` in venv per context,
  generator pattern exists (`scripts/synth_continuous.py`, `scripts/build_pseudo_sentences.py`).
  Novel content required (Robotka 2026; SignBT Zhou 2021; Saunders CVPR 2022,
  cited `docs/CONTINUOUS_PLAN.md:13`). [PRIMARY sources checked via plan citations]
- Cost: ~2–3 h GPU/CPU text-only; no human input.

### 5. Refit temperature on v2/extra heads — SURVIVE (low)
- Gain: calibration only (NLL/ECE), argmax unchanged by construction. Known gap: current
  T=0.823/0.884 fit on two heads (`results/temperature.json:2-5`); extra head has NO T and
  demo applies a single T across concatenated heads (`src/app/demo.py:130-142`,
  `src/infer/routing.py:25-26`). [MEASURED here]
- Cost: ~1 h CPU (LBFGS on saved logits) if logits cached, else one GPU forward pass.

### 6. Demo FPS/latency re-measurement on shipped stack — SURVIVE (info only)
- Gain: no accuracy; artifact: refreshed latency JSON. Current numbers predate v2:
  18.14 ms p50 on 2.03M-param single-head (`docs/benchmarks/v2_baseline_metrics.json:2-9`);
  shipped model is 9.9 MB 3-head (`models/multihead_v2/best.pt`) + Kokoro + UI overlay.
  Headroom claim (18 vs 25 ms) is therefore stale. Offline-capable
  (`scripts/benchmark_latency.py`, `src/infer/replay.py`, `src/infer/profiler.py`). [MEASURED here]
- Cost: ~0.5 h GPU.

### 7. Extra-tail repeat-list config (no retrain) — SURVIVE (weak)
- Gain: zero until next retrain; artifact: oversample spec. Tail is thin: exactly ONE word
  at 0% (`eat`), 12 words <20% (`back`/`few` 4.2, `button` 8.3, `spoon` 11.5, `gloves` 12.5,
  `oil` 15.4 …; counted from `results/extra_eval.json`). Caution from MEASURED evidence:
  the same oversample trick on the old champion moved targets (0→0.22 recall) but net −7.1pp
  via error redistribution (`docs/RESULTS.md:170`). [MEASURED here]
- Cost: ~1 h static (train-pool counts only).

### 8. Dead-code / disabled-path audit — SURVIVE (hygiene, zero gain)
- Artifact: cleanup list. Found: `SAMEAdapter.adapt` (`src/models/same_adapter.py:125`)
  and `Distiller` (`src/train/distiller.py`) reachable only from scripts/tests, never the
  live demo; DSTAdapter load path in demo (`src/app/demo.py:114-124`) with no shipped
  adapter checkpoint; `LandmarkReplayer` (`src/infer/replay.py`) test-only. [MEASURED here]
- Cost: ~1 h static; no behavior change.

## Killed (negative expected value)

- **Second confusion round on v2.** First round MEASURED −7.1pp custom (81.8%→74.69%,
  `docs/RESULTS.md:170`) with mechanism (oversampling redistributes errors). No
  multihead_v2 confusion matrix even exists yet (only `results/confusion_test/` and
  `results/v2/` for dead models), and cashing the gain needs a full retrain (banned). Kill.
- **Mask-after-augment redesign.** Real bug (augment mutates frames, mask untouched,
  `src/data/dataset.py:121-140`; deferred QA item, `docs/QA_REPORT.md:39`) but payoff is
  unmeasurable without pipeline rework + retrain — both banned here. Keep deferred. Kill.
- **bf16/`torch.compile` inference.** Infer is ~4 ms of the 18 ms loop at 55 FPS
  (basis: same reasoning that killed ONNX, `docs/project_status.md:44`); saving 1–3 ms
  buys nothing against the 25 ms budget while risking RoPE graph-breaks. Zero accuracy
  upside. Kill.

## Notes
- Extra-head 320-word floor (≥73 train clips each) and frozen-head bit-identity per
  `docs/RESULTS.md:172`; not re-litigated.
- Varanasi HTMA Alg.1 LOSO (CVPRW 2026) cited from repo docs; full LOSO blocked as above.
