# Project Trajectory — Past / Present / Future

> Compiled 2026-10-03, read-only. Every claim cites `repo-relative-path:line`. UNVERIFIED = not confirmed in files read.

## 1. PAST — what was built, shipped, aborted, parked

### Milestones
- M0 env/scaffold: SHIPPED — venv, RTX 4070, MediaPipe tasks, ≥25 FPS live loop bar (`plan:77-81`).
- M1 INCLUDE acquisition: SHIPPED — 4,287 videos / 263 classes metadata+CSV (`plan:83-86`); splits later 4,163/458/573, 4,257 videos (`docs/project_status.md:102-104`).
- M2 landmark extraction: SHIPPED — ≥95% videos → `.npz` float16 + overlays (`plan:88-92`).
- M3 loaders/augmentation: SHIPPED — (B,T,F)+mask batches, overfit-one-batch check (`plan:93-96`).
- M4 HTMA training: SHIPPED — INCLUDE-50 val 100% (`docs/project_status.md:19`); full-263 test top-1 **81.80%**/top-5 95.40%, n=478 (`docs/RESULTS.md:40-48`). Note: `plan:184` also claims 93.02%/573-sample full eval — DISCREPANCY, UNVERIFIED which artifact that is.
- M5 realtime loop: SHIPPED — ≥25 FPS + per-stage latency log (`plan:103-105`).
- M6 sentence+TTS: SHIPPED — 72 tests, Piper `en_US-lessac-medium`, non-blocking playback, `--no-tts`, auto-speak (`plan:107-124`).
- M7 eval/expo: SHIPPED (code+metrics) — `scripts/evaluate.py`, `benchmark_latency.py`, `category_analysis.py`, replay, RESULTS/SETUP/LIMITATIONS (`plan:167-179`); BUT `docs/demo_video_guide.md` + `docs/EXPO_SCRIPT.md` listed at `plan:175,179` are ABSENT from `docs/` (verified listing) — expo pack incomplete.
- M7.9 gate+smoothing: SHIPPED — threshold 0.85→0.40, 60% majority vote, One-Euro smoother, `--override` fix (`plan:201-212`); threshold later re-tuned to **0.50** (`docs/RESULTS.md:166`).
- M8 real-world: SHIPPED via multi-head (Logos recipe) — INCLUDE 77.20%, ISL500 unseen-users 61.82%, real-world 41.0%, IV 68.0% (was 37.5%) (`plan:425-427`; `docs/RESULTS.md:161-162`).
- M8-V2 items: IV-only metric 37.47% (`results/iv_only_metrics.json` via `docs/RESULTS.md:155`); temp T=0.823/0.884 wired to demo (`docs/RESULTS.md:165`; `src/app/demo.py:130-142`); C1 rules 23.4% exact/83.4% F1 + C2 Qwen polish 37.2%→V2-prompt **43.6%/84.7** (`docs/RESULTS.md:168,170`; `results/polish_eval_v2.json:2-4`); vocab expansion 320 words @60.25% unseen-user, unified max-over-heads routing (`docs/RESULTS.md:173`; `src/infer/routing.py` per `plan:570-577`); UI overhaul + M12 gaps (rules preview, 3s countdown) closed, suite 303/303 (`plan:578-587`).
- M9 (B2 forced-align + B3 synthesis): HALF-OPEN — B1 CSLTR extraction done (290 sent + 395 word npz, `docs/CONTINUOUS_PLAN.md:10`); B2 **PARKED** (8th collapse, `docs/CONTINUOUS_PLAN.md:11`; `plan:651-660`); B3 template-half DONE (200 sents, bigram 2.00×, `docs/CONTINUOUS_PLAN.md:13`), LLM-novel half PENDING.
- M10 CTC revival: **RE-PARKED** 2026-10-03 (collapse #9: 200-synth pretrain, loss 449→4.7, val WER→100% blank attractor; `plan:289-292`).
- M11 any-person: OPEN — harness+techniques spec'd, needs 120-clip calibration (`plan:294-296`).
- M12 live sentence demo: SHIPPED-ISH — spotter streaming + pause boundary + C1/C2 + Kokoro + confidence UI in `src/app/demo.py:237-373`; no CTC streaming decode (parked dependency); latency on v2 stack UNREMEASURED (see §3).
- M13 final eval+expo: OPEN — sentence-WER tables, demo video, expo script all pending (`plan:302-304`).

### Checkpoint inventory (`models/`, numbers from `docs/RESULTS.md` + `results/*.json`)
- `models/baseline.pt`: SHIPPED specialist — custom 81.80% (`docs/RESULTS.md:42`), official-split 85.22% (`results/official_split_metrics.json:8-9`); 55 FPS p95 19.79ms (`docs/RESULTS.md:159`).
- `models/multihead/best.pt`: SHIPPED v1 — 77.20%/61.82%/41.0%/IV68.0% (`plan:425-427`).
- `models/multihead_v2/best.pt`: DEPLOYED — custom **82.22%**, official 84.42%, heads include/isl500/extra=263/263/320 (`results/multihead_v2_gates.json:2-7`; ckpt `sources`/`head_classes` verified by load).
- `models/multihead_extra/best.pt` + `label_map_extra.json` (320 words): SHIPPED stage (frozen-head phase; `models/multihead_v2/label_maps.json:2-5`).
- `models/v2/best.pt`: REJECTED — 71.13% test, OOD collapse (`docs/RESULTS.md:156`).
- `models/joint/best.pt`, `models/joint_v2/best.pt`: REJECTED — INCLUDE 0.63%/0.84% (pose-absent manifold + 10:1 skew; `docs/RESULTS.md:179-181`).
- `models/confusion/best.pt`: ABORTED — 74.69% (−7.1pp redistribution; `docs/RESULTS.md:171`).
- `models/dpc_finetune/best.pt` + `models/dpc/encoder_step{2k,10k,20k,30k,40k,50k}.pt`: ABORTED — probe scratch 16.5%→30k **32.3%** (`results/dpc_probe.json:2-14`), finetune custom 76.99% non-converting (`docs/RESULTS.md:172`).
- `models/ctc*/best_ctc.pt` (ctc, v2–v5, ce-head, seed1/2/3b): ALL PARKED — 96–100% WER blank/empty collapse; A1 baseline WER 119% (`results/continuous_baseline.json:2-7`; `docs/CONTINUOUS_PLAN.md:11`). `models/ctc_synth/` artifacts UNVERIFIED on disk.
- `models/adapter_champion/best_adapter.pt`: PARKED — 80.96% (−0.84, parity, kept for calibration future; `docs/RESULTS.md:158`).
- `models/best.pt` (root): role UNVERIFIED (not traced to an artifact).
- `models/qwen15b/`: LIVE — Qwen2.5-1.5B polish backend (`src/app/demo.py:203-214`).

### Abort/park mechanisms, one line each
- TTA (SAME-MoE k=8): ABORT — per-actor deltas −0.0015/+0.0014, mean −2.2e-05, win 50% = noise (`docs/RESULTS.md:167`; `plan:476-478`).
- Tuned-aug (5-op): ABORT — 19-op stack −6.07% ID; 5-op retrain 71.13% (`docs/RESULTS.md:134,156`).
- Confusion oversample 2×: ABORT — targets 0→0.22 recall but errors redistributed, net −7.1pp (`docs/RESULTS.md:171`).
- DPC pretrain: ABORT — transfer real (+15.8pp probe) but finetune washed out, ship-rule fail (`docs/RESULTS.md:172`).
- CTC×9: PARK — isolated features don't transfer to co-articulated sentences at 290-sent volume, any init/penalty (`plan:651-660`).
- ONNX: KILL — infer ~4ms of 18ms loop; 1–3ms gain for a day + RoPE risk (`plan:627`).
- KD student: KILL — no mobile target; prior KD −7.11pp (`plan:626`; `docs/RESULTS.md:136`).
- Routed ensemble: KILL — superseded, v2 beats both lines (`plan:625`).
- NIOS DPC scale-up (115h): KILL — mechanism non-converting, more hours won't fix (`plan:628`).
- FDMSE-ISL: KILL per user — no direct link anywhere; mail queue (`plan:541-547,620-621`).

## 2. PRESENT — deployed configuration today
- Checkpoint: `models/multihead_v2/best.pt` — default in `src/app/demo.py:57` and CLI `src/app/demo.py:412`; launcher `run_live_demo.bat` (live) vs `run_demo.bat` (batch) (`plan:592`).
- TTS: engine **kokoro**, voice **af_heart**, sr 24000/22050, queue 1 (`configs/default.yaml:30-38`; `src/tts/factory.py:10-16`); Piper `lessac` retained as fallback (`plan:538`). Listen-test DONE — Kokoro won after warmup correction (steady-state 0.13s/4.2s audio ≈ Piper; `results/kokoro_ab.json:2-17`).
- Gate: threshold **0.50** / 12 frames / cooldown 30 / agreement 0.60 / delta 0.08 (`configs/default.yaml:23-28`).
- Vocab: 3 heads (include 263 + isl500 263 + extra 320) with max-over-heads unified routing ≈ **583** effective classes (ckpt verified; `models/multihead_v2/label_map.json`=263, `label_map_extra.json`=320).
- Sentence layer: auto-speak 3.0s, max 20 words, rules-draft preview + countdown bar (`src/app/demo.py:187-191,329-342`).
- Suite: **303 passed** measured 2026-10-03 (`pytest tests/ -q`); docs figures 257/292 are STALE (`docs/QA_REPORT.md:5`; `docs/project_status.md:8`).

## 3. FUTURE — open tracks (unblock → order → cost → verdict)
- M9-B3 LLM-novel half: scripts exist (`scripts/synth_continuous.py:63-73`, `build_pseudo_sentences.py`, `align_viterbi.py`); unblock: write Qwen-local generator (weights local `models/qwen15b/`); order: after nothing (autonomous now); cost HOURS GPU/CPU text-only; **KEEP**.
- M9-B2 forced-align: PARKED; unblock: STARK code release or new co-articulated data; cost DAYS+; **PARK (watch STARK)**.
- M10 CTC: PARKED; unblock: B2-quality boundaries AND (STARK public code); order: after B2; cost DAYS GPU; **PARK**.
- M11 any-person: unblock HUMAN 120-clip webcam calibration (`scripts/record_calibration.py` exists; only 40 self clips in `data/calibration/metadata.csv`); then DSTAdapter fine-tune; cost HUMAN-hours + GPU-hours; **KEEP (only lever with positive expectation, `plan:633`)**.
- M12 remainder: sentence-WER harness + latency re-measure on v2 stack (numbers predate v2: `docs/AUTONOMOUS_AUDIT.md:52-58`); fully autonomous; cost <1h; **KEEP**.
- M13 expo: unblock HUMAN demo video + expo script (files absent, §1-M7); cost HUMAN-hours; **KEEP**.
- STARK: WATCH — arXiv-abs only, no public code (`docs/CONTINUOUS_PLAN.md:28`). Isharah: PARK — Saudi SL, methods-only (`docs/CONTINUOUS_PLAN.md:29`). Gov-dictionary HF mirror (75GB): PARK backup — likely single-signer (`plan:547`). SHuBERT pose-idea: PARK — RGB-bound, 8×A6000 (`plan:517-519`). ONNX/KD/ensemble/NIOS/FDMSE-email: KILL (evidence §1). Extra-tail oversample: PARK weak — same trick measured −7.1pp before (`docs/AUTONOMOUS_AUDIT.md:61-66`).
- Calibration-gate rule stands: no new training runs without ship/abort gate (`plan:636`).

## 4. DEPENDENCY MAP
- HUMAN-blocked (explicit): (a) record 120 calibration clips via `scripts/record_calibration.py` (M11); (b) HF gated-auth if more ISL500 needed (`docs/DATASETS.md:23` — auth reportedly cached, re-verify at run time); (c) record 30–60s demo video (M13); (d) write/approve expo script (M13); (e) FDMSE email queue — DROPPED, no action (`plan:541`). Kokoro listen-test already done by human (`plan:533-540`).
- FULLY AUTONOMOUS: B3 LLM-novel synthesis; gate re-tune on v2 routing (`scripts/sweep_gate.py:260-265` defaults already target v2); temperature refit incl. extra head (`scripts/fit_temperature.py:59-61`); latency re-benchmark (`scripts/benchmark_latency.py:64-74`); pseudo-LOSO regroup from saved predictions; sentence-WER harness on CSLTR npz; STARK/Isharah literature watch.

## 5. TOP-3 next autonomous actions (ranked by expected value)
1. Re-tune gate on v2 583-class routing — 0.50 threshold fit on dead single-head probs; OOV-false cut repeatable (~13pp last time, `results/gate_sweep/sweep.jsonl:1,4`), zero top-1 risk; ~1–2h.
   `.\venv\Scripts\python.exe scripts/sweep_gate.py --dump-probs`
2. Refit temperature incl. extra head — extra head has NO T, demo applies single T across heads (`src/app/demo.py:130-142`); calibration-only gain (NLL/ECE), argmax safe; ~1h.
   `.\venv\Scripts\python.exe scripts/fit_temperature.py --checkpoint models/multihead_v2/best.pt --output results/temperature_v2.json`
3. Latency re-measure on shipped stack (v2 3-head + Kokoro + overlay) — 18.14ms/p95-19.79 claim predates v2 and Kokoro; gates M12 ≥25 FPS acceptance; ~0.5h.
   `.\venv\Scripts\python.exe scripts/benchmark_latency.py --checkpoint models/multihead_v2/best.pt --label-map models/multihead_v2/label_map.json --output docs/benchmarks/v2_shipped_metrics.json`
