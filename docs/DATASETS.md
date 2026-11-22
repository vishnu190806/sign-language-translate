# Dataset Strategy — ISL-only (no ASL integration, decided 2026-10-03)

## Verified ISL ranking (researched 2026-09-26, overlap measured locally)

| Rank | Dataset | Scale | Download | License | Access | Role |
|------|---------|-------|----------|---------|--------|------|
| 1 | **INCLUDE** (Sridhar et al., ACM MM 2020, https://dl.acm.org/doi/10.1145/3394171.3413528) | 4,292 vids / 263 words / ~7 signers | https://zenodo.org/records/4010759 (direct, ~57GB) | CC-BY-4.0 | Open ✅ (already have landmarks: 4,284 `.npz`) | Core training set, canonical 263-label space |
| 2 | **ISL500/ISL-DATA** (community, https://huggingface.co/datasets/ISL500/ISL-DATA) | ~7,500+ vids / 500–1000 words / **15 signers**, MP4 + MediaPipe `.h5` + MMPose `.npy`, 159GB | HF, selective via `scripts/download_isl500.py` | Research/academic only | **Gated (HF login + accept terms)** — listing verified, file download 401s without auth | Diversity source: **180/263 INCLUDE words overlap (68%, measured `results/vocab_overlap.json`)** across 15 signers |
| 3 | **ISL-CSLTR** (Elakkiya et al., Mendeley 2021, DOI 10.17632/kcmpdxky7p.1, https://data.mendeley.com/datasets/kcmpdxky7p/1) | 700 sentence vids / 7 signers, varied conditions | Mendeley / Kaggle mirror | CC-BY-4.0 | Open ✅ | Word-segment mining for signer diversity (needs time-boundary cutting) |
| 4 | **CISLR** (Joshi et al., EMNLP 2022, https://aclanthology.org/2022.emnlp-main.707/) | ~7,050 pairs / ~4,700 words / ~1 signer | https://huggingface.co/datasets/Exploration-Lab/CISLR | AFL-3.0 | Gated | One-shot eval / gloss inventory only (1 example/class breaks softmax training) |
| — | ISLTranslate / iSign (ACL 2023/2024) | 31k–118k sentence pairs | Exploration-Lab HF | CC-BY-NC(-SA)-4.0 | Gated + NC | Excluded from training (sentence-level + non-commercial) |
| — | ISH-NEWS (Sci. Reports 2026) | 4,222 sentence vids | via authors only | unknown | Unavailable | Excluded until accessible |
| — | "ISL500 paper" | — | — | — | Does not exist as a publication | Local `ISL500` test rows = the HF community dataset above (verified via metadata `repository` field) |

## Combination recipe (evidence)
- **Joint multi-head co-training beats flat union softmax** (Logos, Ovodov et al., arXiv:2505.10481, 2025: co-train 97.92/97.83/66.82 vs pretrain→fine-tune; §5.1–5.4, https://arxiv.org/html/2505.10481v1). **ASL-pretrain → ISL fine-tune validated by our own backbone paper** (HTMA, Varanasi et al., CVPRW 2026: WLASL2000+MS-ASL+ASL-Citizen pretrain, 83.82% INCLUDE).
- **Crowd/wild consented data helps; scraped YouTube alone hurts signer-independent accuracy** (ASL-Citizen, Desai et al., NeurIPS DB 2023: 63.10% vs WLASL-model 8.49% on unseen signers, https://arxiv.org/html/2304.05934).
- **Per-dataset geometric normalization, never global** (Roh, SignLang Workshop 2024, +6.05% WLASL-100: https://aclanthology.org/2024.signlang-1.36.pdf). Our wrist/shoulder normalization already complies (`src/features/normalize.py`).
- **Domain-balanced sampling** so 100k-scale sources don't drown INCLUDE (~4k): round-robin/`dataset_id` batches (Logos §4.3).

## Phased plan
- **Phase D1 (done):** vocab overlap artifact — 180 shared words (`scripts/vocab_overlap.py` → `results/vocab_overlap.json`).
- **Phase D2 (needs 2 min of you):** HF login + accept terms → `.\venv\Scripts\python.exe scripts/download_isl500.py` (overlap-words landmarks only, ~1–4GB est., resumable).
- **Phase D3:** inspect `.h5` layout → `.h5`→`.npz` (168-d) converter → joint INCLUDE+ISL500 training on 180 shared words (union head) + 83 INCLUDE-only retained; ISL500-only words → second head later (Logos multi-head).
- **Phase D4 (dropped 2026-10-03 with the ASL track):** was ASL-Citizen-first staging. No replacement — ISL vocab expansion shipped instead (multihead v2, 583 words).

## Artifacts this run
- `isl500_vocab.json` (1,000 ISL500 words from public file listing)
- `results/vocab_overlap.json` (180 overlap words)
- `scripts/vocab_overlap.py`, `scripts/download_isl500.py`
- **Downloaded 2026-09-26**: 26,796 landmark files → `data/raw/isl500/` (~2h, 0 failures)
- **Converted 2026-09-26**: 26,483 `.npz` → `data/processed/landmarks/isl500/` (`scripts/convert_isl500.py`), 310 OOV-skipped, 0 failed; `.h5` schema `(T, 2, 21, 3)` hands-only, 30fps
- **Joint set**: `data/processed/landmarks/joint/` — 30,767 hardlinked files, `joint_label_map.json` (263 words, indices 0–262)
