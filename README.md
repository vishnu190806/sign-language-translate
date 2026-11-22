# Real-Time Indian Sign Language Translation

Webcam → MediaPipe landmarks → HTMA temporal model → word prediction → sentence assembly → offline speech.
An ISL-only research and engineering project: a working live application plus the full experimental record behind it.

## 2. Representative screenshot

![Live demo HUD](docs/research/screenshots/ui_clean.png)

## 3. Problem

Indian Sign Language (ISL) is the primary language of millions of Deaf and hard-of-hearing people in India, but
there is no widely deployed real-time ISL-to-English assistant. This project builds a word-level ISL recogniser
that runs locally on a consumer laptop, assembles predicted glosses into English sentences, and speaks them
with offline text-to-speech. It deliberately targets **isolated signs + sentence assembly**, not continuous
sign-language translation, so the recogniser can be trained on freely available word-level data.

## 4. Key features

- 168-dimensional landmark features per frame (2 hands × 21 keypoints + presence flags, 13 upper-body pose keypoints + flag).
- Wrist/shoulder-centred normalisation for signer- and distance invariance; One-Euro landmark smoothing.
- **HTMA** (Head-Token Mixing Attention) encoder with APE + RoPE, local temporal convolution, attention pooling and temperature-calibrated outputs.
- Multi-head routing (`include`, `isl500`, `extra`) over a shared encoder, giving a 583-word vocabulary.
- Stability gate with majority voting and cooldown; ring-buffer inference at 32 frames.
- Sentence assembly with ISL grammar rules, optional Qwen2.5-1.5B polish (falls back to rules), and Kokoro/Piper offline TTS with barge-in playback.
- Full research record: ablations, cross-dataset evaluation, calibration studies, failures and citation audits.

## 5. Demonstration

Two different workflows ship with the repository:

| Workflow | Entry point | What it does | Needs webcam | Needs checkpoint |
|---|---|---|---|---|
| **Live recognition** | `scripts/demo.py`, `run_live_demo.bat` | Captures webcam frames, extracts landmarks, predicts words, builds sentences, speaks | Yes | Yes |
| **Cached replay/evaluation** | `run_demo.bat`, `run_demo.sh` | Scores 10 cached predictions against a metadata table; no model inference | No | No |

Additional views: [debug HUD](docs/research/screenshots/ui_debug.png) · [sentence preview](docs/research/screenshots/ui_sentence.png).

The live workflow is the actual application. The replay workflow is a lightweight latency/demo test and needs
`isl_40_test/metadata.csv`, which is part of an excluded mixed-license dataset (see §12).

## 6. Architecture and pipeline

```
Webcam (30 FPS)
  → ThreadedCamera (OpenCV, always-latest frame, DSHOW backend)
  → MediaPipe Tasks: HandLandmarker (2 hands) + PoseLandmarker (lite), async
  → 168-dim feature vector + per-frame normalisation
  → One-Euro smoother → RingBuffer (32 frames)
  → HTMA encoder (+ RoPE) → attention pooling → per-head softmax (temperature-scaled)
  → max-over-heads unified routing → StabilityGate (confidence + votes + cooldown)
  → SentenceAssembler (dedup, ISL grammar rules, optional LLM polish)
  → TTS (Kokoro default, Piper fallback) → non-blocking playback
```

Model input is `(B, T, 168)` with an optional `(B, T)` attention mask. Full design:
[`docs/architecture.md`](docs/architecture.md) and [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md).

## 7. Technology stack

Python 3.10–3.12 · PyTorch 2.x · MediaPipe Tasks · OpenCV · NumPy/pandas/PyYAML · Piper or Kokoro TTS ·
sounddevice/soundfile · pytest. Training used an RTX 4070 Laptop 8 GB with bf16 AMP.

## 8. Installation and quick start

```powershell
git clone https://github.com/vishnu190806/sign-language-translate.git
cd sign-language-translate

# Guided Windows setup (venv, dependencies, verified MediaPipe downloads):
powershell -ExecutionPolicy Bypass -File scripts\setup_runtime.ps1 -WithTTS

# Or manually:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-tts.txt      # optional: enables speech (Kokoro)
```

Smoke test (text only, after the runtime resources in §9 are in place):

```powershell
.\.venv\Scripts\python.exe scripts\demo.py --config configs\default.yaml --override configs\isl.yaml --checkpoint models\baseline.pt --no-tts
```

Remove `--no-tts` for speech. Press `q` to quit, `a` to toggle auto-speak, `d` for the debug panel, `c` to clear.

Environment note: all three launchers use `.venv`. `run_live_demo.bat` explicitly selects
`models/baseline.pt`. The replay launchers (`run_demo.bat`, `run_demo.sh`) call
`scripts/real_world_test.py` and additionally need the excluded `isl_40_test/metadata.csv`.

## 9. Runtime model and asset acquisition

| Resource | Local path | Size | In Git | How to obtain |
|---|---|---:|---|---|
| Bundled inference checkpoint | `models/baseline.pt` | 24,523,349 B | **Yes** — SHA-256 `8ED3F773…EB02` | INCLUDE-only model; attribution in `ATTRIBUTION.md` |
| Default (stronger) checkpoint | `models/multihead_v2/best.pt` | 9,932,803 B | **No** | Not redistributable: trained partly on gated ISL500 data |
| Label maps / temperature | `models/multihead_v2/label_map.json`, `label_maps.json`, `temperature.json`, `models/multihead_extra/label_map_extra.json`, `models/label_map.json` | < 20 KB | Yes | Committed |
| MediaPipe hand model | `assets/hand_landmarker.task` | 7,819,105 B | No | `scripts/setup_runtime.ps1` (Google storage, Apache-2.0 per model card) |
| MediaPipe pose model | `assets/pose_landmarker.lite.task` | 5,777,746 B | No | Same script |
| Kokoro TTS | pip package + HF weights | ~312 MB at first use | No | `pip install -r requirements-tts.txt`; weights auto-download from `hexgrad/Kokoro-82M` (Apache-2.0) |
| Piper Lessac voice (fallback) | `assets/voices/.../*.onnx` | 63,201,294 B | No | Manual download; restricted research licence — see §16 |
| Qwen2.5-1.5B sentence polish | `models/qwen15b/*.safetensors` | ~2.9 GB | No | Optional; absent → `RulesOnlyEngine` |

Verified checksums for locally audited copies:

```
models/baseline.pt              8ED3F773B94B43FF5D772D59F30E64FAC589974EE1F63B9CD243FFC3CB31EB02
models/multihead_v2/best.pt     BC48AE02E8EDE57D91465DB22F81A2F0B173675C72A8ACAAB0CA3FABE5FF06A5
assets/hand_landmarker.task     FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1
assets/pose_landmarker.lite.task 59929E1D1EE95287735DDD833B19CF4AC46D29BC7AFDDBBF6753C459690D574A
```

### What a fresh clone can and cannot do

**Can:** install dependencies with two commands; download the MediaPipe models with checksum verification;
load the bundled INCLUDE-trained checkpoint `models/baseline.pt`; run live webcam
recognition, sentence assembly and Kokoro speech; run all tests; inspect all research evidence.

**Cannot, by design:** train or evaluate without downloading datasets (§12); load `models/multihead_v2/best.pt`
from the repository (it is excluded); run the cached replay demo without the excluded `isl_40_test/metadata.csv`;
use the Piper Lessac voice without a separate, licensed download; reproduce gated ISL500-derived results from
the public clone alone.

## 10. Results and quantitative evaluation

| Result | Value | Evidence |
|---|---:|---|
| INCLUDE full-dataset test top-1 | **81.80%** (93.51% top-3, 95.40% top-5) | `docs/research/evidence/full_dataset_metrics.json` |
| Matched official-split top-1 | **85.22%** | `docs/research/evidence/official_split_metrics.json` |
| Shipped 3-head checkpoint (custom / official split) | **82.22% / 84.42%** | `docs/research/evidence/multihead_v2_gates.json` |
| In-vocabulary top-1 | 37.47% (145/387) | `docs/research/evidence/iv_only_metrics.json` |
| Multi-head real-world (overall / IV) | 41.0% / 68.0% | `docs/research/evidence/real_world_test_multihead.json` |
| Cross-dataset aggregate | 22.6% (source-mix dominated — not a headline) | `docs/research/evidence/real_world_accuracy.json` |
| Latency (unified 3-head forward) | 14.70 ms; earlier 55 FPS, p95 19.79 ms | `evidence/bench_*` |
| Sentence polish exact / F1 | 43.6% / 84.7% | `docs/research/evidence/polish_eval_v2.json` |

Full artifact-backed analysis, including confidence intervals and protocol caveats:
[`docs/RESULTS.md`](docs/RESULTS.md) and the [research evidence index](docs/research/README.md).

## 11. Research methodology

HTMA evolved from a single-head INCLUDE classifier (81.80% test) through a tuned-augmentation retrain that
**failed** (−6.07pp ID), a champion baseline, a shared-encoder multi-head model (41.0% real-world), and finally
a 583-word three-head expansion. Temperature scaling (Guo et al., ICML 2017) was fitted per head. Ablations
showed adapters and distillation did not help the champion, episodic TTA produced no reliable per-signer gain,
and CTC continuous decoding collapsed into a blank attractor. Negative results are retained deliberately:
[`docs/RESULTS.md`](docs/RESULTS.md), [`docs/RESEARCH_VALIDATION.md`](docs/RESEARCH_VALIDATION.md),
[`docs/PROJECT_TRAJECTORY.md`](docs/PROJECT_TRAJECTORY.md). Tests: 257 passing (see `docs/QA_REPORT.md`).

## 12. Dataset information

| Dataset | Scale | Access | Licence | Role |
|---|---|---|---|---|
| [INCLUDE](https://zenodo.org/records/4010759) | 4,292 videos / 263 words | Open | CC BY 4.0 | Core training set |
| [ISL500 / ISL-DATA](https://huggingface.co/datasets/ISL500/ISL-DATA) | ~7,500 videos / 500–1000 words / 15 signers | Gated (HF login + terms) | Research/academic only | Signer diversity, 583-word expansion |
| [ISL-CSLTR](https://data.mendeley.com/datasets/kcmpdxky7p/1) | 700 sentences / 7 signers | Open | CC BY 4.0 | Continuous-data experiments |
| [CISLR](https://huggingface.co/datasets/Exploration-Lab/CISLR) | ~7,050 pairs / ~4,700 words | Gated | AFL-3.0 | One-shot evaluation only |
| OpenHands ISL corpus | ~1,129 h | Undocumented local copy | Undocumented | DPC pretraining pilot |

Details: [`docs/DATASETS.md`](docs/DATASETS.md). Raw data is never committed.

## 13. Training and evaluation

```powershell
python scripts/download_datasets.py --dataset isl
python scripts/extract_landmarks.py --dataset isl
python scripts/train.py --config configs/default.yaml --override configs/isl.yaml
python scripts/real_world_test.py --model-path models\baseline.pt --dataset-dir isl_40_test --output-dir results
.\.venv\Scripts\python.exe -m pytest tests\ -v
```

These reproduce the *pipeline*, not the exact weights; determinism is limited by GPU kernels and data order.
Gated datasets require the user's own credentials and accepted terms. No training happens during setup.

## 14. Project development milestones

M0 environment → M1 INCLUDE EDA → M2 landmark extraction → M3 dataloaders/augmentation → M4 HTMA training →
M5 real-time loop → M6 sentence + TTS → M7 evaluation/expo → M8 real-world improvement and verified-research
upgrade → continuous-sentence program (CTC parked, spotter + rules + LLM shipped). Full history:
[`docs/PROJECT_TRAJECTORY.md`](docs/PROJECT_TRAJECTORY.md).

## 15. Limitations and future work

Strong in-domain performance does not transfer: cross-dataset and unseen-signer accuracy remain low
(per-user spread 27–84%). OOV words cannot be predicted by a closed-set classifier; the 0/255 OOV result is a
tautology. Continuous decoding was parked after measured collapse. Real-time performance is measured on an
RTX 4070 Laptop, not on low-end hardware. Next steps: user calibration data, signer-disjoint evaluation,
and a legally distributable checkpoint. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## 16. Third-party attribution and licensing

- **INCLUDE** — Sridhar et al., ACM MM 2020, CC BY 4.0. Models trained on it require attribution.
- **ISL500 / ISL-DATA** — gated, research/academic only. Redistribution of derived weights is **not established**.
- **MediaPipe** — Google AI Edge; task models are Apache-2.0 per their model cards (SDK Apache-2.0).
- **Piper** — engine MIT; the `en_US-lessac-medium` voice derives from the Blizzard 2013 Lessac corpus whose
  licence is research-only and forbids redistribution without separate permission. The Piper maintainer
  explicitly states there is no project-level redistribution grant. The voice is therefore **not bundled**.
- **Kokoro** — `hexgrad/Kokoro-82M`, Apache-2.0 weights; fetched at runtime, not bundled.
- **Qwen** — separate Alibaba licence; optional and not bundled.

**Release decision:** `models/baseline.pt` (INCLUDE-only, CC BY 4.0) is bundled as the runnable default;
its attribution is recorded in [`ATTRIBUTION.md`](ATTRIBUTION.md). `models/multihead_v2/best.pt` is excluded
because ISL500-derived redistribution rights are not established — a private repository does not change that.
See [`docs/MODEL_AND_DATA_AVAILABILITY.md`](docs/MODEL_AND_DATA_AVAILABILITY.md).
