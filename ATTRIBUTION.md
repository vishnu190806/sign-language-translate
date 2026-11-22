# Third-Party Attribution

## INCLUDE dataset (basis of the published checkpoint)

The inference checkpoint bundled with this repository, `models/baseline.pt`, was trained
**exclusively** on landmark features extracted from the INCLUDE dataset.

> Sridhar, A., Ganesan, R., Kumar, P., & Khapra, M. (2020).
> *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition.*
> Proceedings of the 28th ACM International Conference on Multimedia (MM '20), 3166–3175.
> https://dl.acm.org/doi/10.1145/3394171.3413528

- Dataset: https://zenodo.org/records/4010759 and https://huggingface.co/datasets/ai4bharat/INCLUDE
- Licence: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- Authors: IIT Madras / AI4Bharat; recordings made with deaf signers from St. Louis School for the Deaf, Adyar, Chennai.
- Derived work notice: `models/baseline.pt` is a models/output derived from INCLUDE. It is redistributed under the attribution terms of that dataset. The original videos are **not** redistributed here.

## Runtime dependencies

| Component | Licence | Distribution |
|---|---|---|
| MediaPipe Tasks SDK and Hand/Pose Landmarker task models | Apache-2.0 (per Google AI Edge model cards) | Downloaded by `scripts/setup_runtime.ps1`, not vendored |
| Kokoro (`hexgrad/Kokoro-82M`, voice `af_heart`) | Apache-2.0 weights | Downloaded on first use, not vendored |
| Piper TTS engine | MIT | pip dependency only |
| Piper `en_US-lessac-medium` voice | Blizzard 2013 Lessac corpus licence — research-only, redistribution not granted | **Not bundled** |
| Qwen2.5-1.5B (optional polish) | Separate model licence | **Not bundled** |
| PyTorch, OpenCV, NumPy, pandas, PyYAML, pytest | Their respective open-source licences | pip dependencies |

See `docs/MODEL_AND_DATA_AVAILABILITY.md` for the full per-resource assessment and
`docs/DATASETS.md` for dataset acquisition terms.
