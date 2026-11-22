# Model & Data Availability

Source of truth for which resources each workflow needs, their licences, and how to obtain them.
Verified 2026-11-22. This document supersedes the earlier source-only publication manifest.

## 1. Runtime dependency manifest (verified against source)

### Live webcam workflow — `scripts/demo.py` → `src/app/demo.py`

Read from the code, not inferred:

- `src/app/demo.py:77` — `torch.load(checkpoint, weights_only=False)`; no fallback, missing file raises.
- `src/app/demo.py:66` — label map at `Path(checkpoint).parent / "label_map.json"`, then `paths.checkpoints`.
- `src/app/demo.py:104-110` — `label_maps.json` sidecar resolves per-head maps by **relative path**; for the
  default checkpoint these are `models/multihead_v2/label_map.json` and `models/multihead_extra/label_map_extra.json`.
- `src/app/demo.py:132` — temperature sidecar at `Path(checkpoint).parent / "temperature.json"`.
- `src/app/demo.py:229-233` — MediaPipe models at `config.paths.assets` → `assets/hand_landmarker.task`,
  `assets/pose_landmarker.lite.task`.
- `configs/default.yaml:31` — `tts.engine: kokoro`, `voice: af_heart` (Piper is the fallback engine).
- `src/app/demo.py:209-215` — Qwen at `models/qwen15b/*.safetensors`; absent → `RulesOnlyEngine`.
- `scripts/demo.py:22` — default checkpoint `models/multihead_v2/best.pt`; flags `--config`, `--override`,
  `--checkpoint`, `--no-tts`, `--offline`.

Exact minimum resources:

| Resource | Path | Bytes | In Git? |
|---|---|---:|---|
| HTMA checkpoint | `models/baseline.pt` (bundled, documented default) or `models/multihead_v2/best.pt` (excluded original default) | 24,523,349 / 9,932,803 | Yes / No |
| Per-head label maps + temperature | 4 sidecars under `models/multihead_v2/`, `models/multihead_extra/` | 4,252 + 162 + 949 + 5,157 | Yes |
| Plain-head label map | `models/label_map.json` | 5,591 | Yes |
| Hand landmarker | `assets/hand_landmarker.task` | 7,819,105 | No |
| Pose landmarker | `assets/pose_landmarker.lite.task` | 5,777,746 | No |
| TTS engine | pip `kokoro` (default) or `piper-tts` (fallback) | — | No (pip) |
| LLM polish | `models/qwen15b/` | ~2.88 GB | No (optional) |

### Cached replay workflow — `run_demo.bat` / `.sh`

Requires `demo/cached_predictions_full.json` (6,196 B, **in Git**) and `isl_40_test/metadata.csv`
(**excluded**, mixed licence). `real_world_test.py:403-408` exits when the metadata file is absent. No webcam,
checkpoint, MediaPipe or TTS required.

## 2. Checkpoint redistribution assessment

### Default checkpoint — `models/multihead_v2/best.pt`

- SHA-256 `BC48AE02E8EDE57D91465DB22F81A2F0B173675C72A8ACAAB0CA3FABE5FF06A5`, 9,932,803 bytes.
- Verified architecture: multi-head checkpoint with `sources` and `head_classes`; demo rebuilds a
  `MultiHeadHTMAClassifier` from those keys (`src/app/demo.py:80-113`).
- Training provenance: INCLUDE (CC BY 4.0) **plus gated ISL500 data plus 320 ISL500-only expansion words**
  (master-plan addendum 2026-10-03; `results/multihead_v2_gates.json`).
- ISL500/ISL-DATA is gated, research/academic only; its terms do not grant redistribution of derived weights.
  A private repository does not change this.
- **Conclusion: redistribution rights cannot be established. Do not publish without explicit authorisation.**

### Legally cleaner alternative — `models/baseline.pt`

- SHA-256 `8ED3F773B94B43FF5D772D59F30E64FAC589974EE1F63B9CD243FFC3CB31EB02`, 24,523,349 bytes.
- Verified read-only metadata: plain HTMA, `embedding_dim=256`, `htma_blocks=2`, `num_heads=8`,
  `dropout=0.2`, `num_classes=264`, 65 state tensors, head `(264, 256)`.
- `configs/isl.yaml` sets `num_classes: 264` and `dropout: 0.2`, so
  `model.load_state_dict(ckpt["model_state_dict"])` at `src/app/demo.py:126` matches exactly when the demo is
  run with `--override configs/isl.yaml`. Compatible with the existing inference path — no code change needed.
- Its label map is `models/label_map.json` (264 keys; the known index-263 collision is documented in
  `docs/RESULTS.md`), also committed.
- Provenance: INCLUDE-only champion trained before ISL500 acquisition (2026-09-25 vs 2026-09-26); INCLUDE is
  CC BY 4.0, which permits redistribution with attribution.
- **Release decision (2026-11-22):** this checkpoint is bundled as the documented default runnable model.
  Its exact size (24,523,349 B) and SHA-256
  (`8ED3F773B94B43FF5D772D59F30E64FAC589974EE1F63B9CD243FFC3CB31EB02`) are recorded in the README and below.
  INCLUDE attribution is in `ATTRIBUTION.md`.

Other checkpoints (`v2/`, `confusion/`, `dpc_finetune/`, `joint*`, `ctc*`, `qwen15b/`) remain excluded: they are
either worse, rejected, derived from restricted corpora, oversized, or unrelated to the default workflow.

## 3. Third-party runtime assets

| Asset | Licence evidence | Decision |
|---|---|---|
| MediaPipe `.task` models | Google AI Edge developer docs (updated 2026-11-20) list task models as Apache-2.0 per their model cards; SDK is Apache-2.0. Exceptions (Gemma, two speech models) do not include Hand/Pose Landmarker. | Small enough to bundle, but kept **out** of Git; `scripts/setup_runtime.ps1` downloads and SHA-256-verifies them. This avoids vendoring third-party binaries while keeping clone-and-run after one command. |
| Piper `en_US-lessac-medium` | MODEL_CARD cites the Blizzard 2013 Lessac corpus licence: research/exploration only, no commercial use, no distribution without written consent. Rhasspy maintainer confirms no project-level redistribution grant. | **Excluded.** Preserve local copies; optional manual download with user-arranged licence. Piper is only a fallback engine. |
| Kokoro `af_heart` + base weights | `hexgrad/Kokoro-82M` declares Apache-2.0 weights; runtime auto-downloads them on first use; not stored in Git. | Install via `requirements-tts.txt`; no binary committed. |
| Qwen2.5-1.5B | Separate Alibaba licence; ~2.88 GB. | Excluded; optional; falls back to rules. |

## 4. Bundling decision for `.gitignore`

`models/**/*.pt` etc. stay ignored. The single approved exception is an explicit negation for
`models/baseline.pt` (see `.gitignore`). All other checkpoints, Qwen weights and Piper voices remain ignored.
Verified with `git check-ignore -v` after initialization.

## 5. Datasets (training/reproduction only — never fetched by setup)

| Dataset | Size locally | Access | Licence | Source |
|---|---:|---|---|---|
| INCLUDE | ~55.1 GB | Open | CC BY 4.0 | https://zenodo.org/records/4010759 · https://huggingface.co/datasets/ai4bharat/INCLUDE |
| ISL500 / ISL-DATA | ~2.6 GB | Gated | Research/academic only | https://huggingface.co/datasets/ISL500/ISL-DATA |
| ISL-CSLTR | ~25.6 GB | Open | CC BY 4.0 | DOI 10.17632/kcmpdxky7p.1 |
| CISLR | part of `isl_40_test/` | Gated | AFL-3.0 | https://huggingface.co/datasets/Exploration-Lab/CISLR |
| OpenHands | ~11.84 GB | Undocumented | Undocumented | No valid source documented — treat as non-recoverable |
| `isl_40_test/` | 289.57 MB | Local aggregate | Mixed per source | Evaluation only; excluded from Git |

## 6. Release decisions and remaining open items

1. **Resolved:** `models/baseline.pt` is bundled (INCLUDE-only, CC BY 4.0, attribution in `ATTRIBUTION.md`).
2. **Standing restriction:** `models/multihead_v2/best.pt` is not published unless ISL500-derived redistribution rights are obtained.
3. **Resolved:** the GitHub repository is private (`vishnu190806/sign-language-translate`).
4. **Open:** `isl_40_test/metadata.csv` remains excluded; the cached replay workflow is not available on a fresh clone (live webcam workflow is the supported path).
5. **Open:** no project `LICENSE` file; the repository is provided for review without an explicit project licence grant.
