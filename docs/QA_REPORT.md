# QA Report — Full-project sweep (2026-09-26)

Method: 3 parallel audit agents (demo path, training/eval, scripts/TTS/text) + full test suite. Every fix verified by tests or real execution.

## Suite result: 303/303 pass (2026-10-04)
- Re-verified 2026-10-02: `pytest tests/ -q` → **257 passed, 12 warnings in 9.61s**. Earlier drafts of this doc claimed 253/253; the count had drifted as tests were added.
- Re-verified 2026-10-04: **303 passed** (added: oversample 5, DPC 3, extra-head 3, routing 3, overlay 6, gate-status 2, playback-level 2, live-draft 4, kokoro 6, ctc-unfreeze 3, pseudo-sentences 2, synth 3, warm-start 2, TTS polish variants).
- Created missing `scripts/category_analysis.py` (M7.3, spec'd exactly by `tests/test_m7.py`).
- Made `test_dataloader_batches` data-agnostic (`< len(label_map)` instead of hardcoded `< 50`).

## Critical fixes (demo-breaking or wrong-results)

1. **Auto-speak fired every frame** (`src/infer/sentence.py`): `update()` never cleared words/timer → `speak()` at 30Hz on the same sentence. Fixed: fire once, clear words, reset timer.
2. **Unbound `extractor` in `finally`** (`src/app/demo.py`): constructor failure → `NameError` masking the real error. Fixed with `None` init + guards.
3. **Wrong-results risk, `ce_window_decode.py`**: decoded from raw projection, skipping APE+blocks. Rewrote with full encoder path + argparse + guards.
4. **Temperature schema mismatch**: `fit_temperature.py` wrote flat schema, `continuous_baseline.py` read nested. Unified on nested; refit reproduces exactly (T 0.823/0.884).
5. **Destructive `resplit_csltr.py`**: top-level code rewrote all npz on import. Now `main()` + `--force` dry-run default.
6. **Piper API mismatch** (found earlier, re-verified): `synthesize_wav`/`synthesize()` chunk API; `length_scale` now passed via `SynthesisConfig`; stream returns real sample rate; bare-name load raises actionable error. Real synthesis re-verified after edits.
7. **`_get_rope` regression** (found yesterday via new tests): method slipped into subclass, breaking plain-HTMA forward everywhere. Fixed; both checkpoints load.

## Major fixes
- Gate repeat-after-pause (`_last_emitted` now expires with cooldown).
- `max_words` cap enforced in demo (speak + clear).
- `c` key resets gate + buffer (no stale re-emission).
- Smoother skips/resets on all-zero frames (no filter poisoning).
- Polish moved off the video thread (daemon worker; UI never freezes).
- Camera `stop()` race (`_cap=None` init, alive-check, guarded release).
- Playback double-start/`stop()` (`sd.stop()`, alive-check, `_worker=None`).
- Trainer empty-val guard, `T_max=0` crash, sampler empty-set error.
- `train_adapter.py` CPU autocast + `random` import.
- `train_ctc.py` CLI paths (was hardcoded) + empty-set errors + mask-aware blank penalty.
- `normalize.py` pose off-by-one (`>=` → `>`).
- `evaluate.py` empty-split guard.
- `prototype_decode.py` class count from checkpoint.
- `diag_ctc.py` auto device.
- ZeroDivision guards: `eval_joint_by_source`, `probe_slots`, `eval_gloss2text`, `eval_polish`.
- Demo checkpoint default unified to `models/multihead/best.pt`; label map resolved from checkpoint dir.

## Deliberately deferred (documented, low ROI or high risk)
- Mask-after-augment skew (temporal ops zero padded frames, mask stays True): real but requires pipeline redesign; tracked, mitigated by basic-mode default.
- Manifest staleness, hash-seed nondeterminism, per-worker augmentation seeding: robustness upgrades, no observed impact.
- `+all` ablation overwrite, `+augmentation` fallback reporting, distiller export/load wire mismatch: ablation scripts superseded by per-condition scripts; rewrite when ablation reruns.
- ONNX migration: launch-bound at this model size (research report).
- Continual/lifelong TTA without reset: collapse risk (EATA/SAR evidence); episodic only.
- ISL-CSLTR test sentences with unique glosses: 12-sentence covered set chosen instead (sentence-disjoint preserved).

## Independent clean-clone verification (2026-11-22)

Fresh private-repo clone (final HEAD `fd38c2d`), independent Python 3.12 `.venv`, `setup_runtime.ps1 -WithTTS`,
verified MediaPipe SHA-256 hashes, bundled checkpoint load (264 classes / 65 tensors), model inference
`(1, 264)`, MediaPipe HandLandmarker + PoseLandmarker initialization, and Kokoro synthesis
(40,800 samples at 24 kHz) all PASS.

`pytest tests/ -v` on that clean clone: **289 passed, 17 skipped, 0 failed**. Dataset-dependent tests skip
explicitly when their excluded resources are absent:

- `tests/test_ctc_unfreeze.py` (3) — requires excluded `models/multihead/best.pt`.
- `tests/test_dataset.py` (12) — requires excluded processed landmarks and `isl_label_map.json`.
- `tests/test_openhands_bridge.py` (2) — requires optional `h5py` from `requirements-datasets.txt`.

They still run in the author's full research environment; no test assertion was weakened.

An actual webcam run of `scripts/demo.py` (real camera, 480x640 capture, Kokoro playback started) exposed a
crash: with no hands visible the attention mask is entirely empty, `AttentionPooling` softmaxed an
all-`-inf` row, produced NaN logits and broke the HUD drawing code. Fixed by resetting fully-masked rows
to zero before softmax (uniform, finite pooling); regression test added in
`tests/test_attention_pooling.py`. Webcam capture, MediaPipe processing and inference were confirmed live;
word emission from an actual ISL gesture and the resulting spoken sentence require a human signer and are
not covered by an automated check.
