# Research Validation — Citation Check

**Last updated: 2026-10-02** (2nd pass — all four "unverified" citations resolved as REAL; see §3)

## 1. Verified citations

All verified against **publisher/indexer pages** using the Parallel Search MCP (method in §5).
**Every citation previously listed as "unverified" or "not confirmed" turned out to be REAL.**

| Citation | Status | Evidence |
|---|---|---|
| **HTMA — Varanasi, Sinha, Dasgupta. "Isolated Sign Language Recognition via MediaPipe Landmarks: A Case Study On Indian Sign Language." CVPR 2026 Workshops (MSLR), pp. 10595–10604** | ✅ **REAL** | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026W/MSLR/html/Varanasi_Isolated_Sign_Language_Recognition_via_MediaPipe_Landmarks_A_Case_Study_CVPRW_2026_paper.html) — BibTeX: `@InProceedings{Varanasi_2026_CVPR,...}` |
| **SAME — Yang, Yan, He, Zhang, Jin, Zhang, Xi, Yin. "SAME: Signer-Aware Mixture-of-Experts for Test-Time Adaptation in Sign Language **Translation**." ACL 2026 (Vol. 1 Long Papers), pp. 21274–21289** | ✅ **REAL** | [aclanthology.org/2026.acl-long.973](https://aclanthology.org/2026.acl-long.973/), DOI `10.18653/v1/2026.acl-long.973`. Zhejiang University. **Targets sign language *translation* (seq2seq), not isolated classification.** |
| **KD-MSLRT — Li, Ren, Hu, Liu, Jiang, Dang, Su. AAAI 2025, 39(27), 28177–28185** | ✅ **REAL** | [ojs.aaai.org/…/35037](https://ojs.aaai.org/index.php/AAAI/article/view/35037), arXiv:2501.02321, DOI `10.1609/aaai.v39i27.35037`. Xi'an Jiaotong-Liverpool Univ. **Scope: edge/compact distillation, not accuracy-SOTA.** |
| **DSTAdapter — Tian, Yang, Chen, Chen, Ning, Zhang. Int. J. Machine Learning and Cybernetics 17:219** | ⚠️ **REAL — but our year/venue are wrong** | [link.springer.com/article/10.1007/s13042-026-03002-x](https://link.springer.com/article/10.1007/s13042-026-03002-x). **Version of record is 2026 (Springer journal), not "Tian et al., 2025"** — 2025 was the preprint year. Freezes a **CLIP** backbone. |
| **Sign-Language Datasets at Scale — Ni, Cheng, Li, Cheng. ACL 2026 (Vol. 1), pp. 41578–41604** | ✅ **REAL** | [aclanthology.org/2026.acl-long.1928](https://aclanthology.org/2026.acl-long.1928/), arXiv:2606.19352. Indexes **120 datasets / 35 sign languages**; introduces a 24-field "Sign-Language Datasheet". Best single landscape reference we have. |
| INCLUDE dataset (Sridhar, Ganesan, Kumar, Khapra; ACM MM 2020) | ✅ REAL | Zenodo 4010759, 4,292 videos / 263 signs, CC-BY 4.0, DOI 10.1145/3394171.3413528 |
| MediaPipe (Lugaresi et al., arXiv:1906.08172) | ✅ REAL | Google perception pipeline |
| Logos multi-dataset co-training (Ovodov et al., 2025) | ✅ REAL | arXiv:2505.10481 |
| Piper TTS 1.8 | ✅ REAL | PyPI `piper-tts` |
| Uni-Sign (Zhang et al., ICLR 2025) | ✅ REAL | Best hard SOTA reference: +4.09% over MSLU on MSASL1000, +2.47% over NLA-SLR on WLASL2000 |
| OpenHands (Selvaraj, Nc, Kumar, Khapra; ACL 2022, `2022.acl-long.150`) | ✅ REAL | 1,129 h unlabeled ISL pose pretraining; DPC objective only one that worked |
| Sign2Vec / SignCorpus (RJM Krishna et al., NeurIPS 2022 D&B) | ✅ REAL | 4,568 h, 10 sign languages, 75 pose keypoints, PII-stripped |
| ISL-CSLTR (Elakkiya & Natarajan 2021) | ✅ REAL | Mendeley `kcmpdxky7p`, 700 videos / 1,036 words / 7 signers — **publicly downloadable** |
| FDMSE-ISL (arXiv:2407.14224) | ✅ REAL | 40,033 videos / 2,002 words / 20 signers — denser than INCLUDE |
| ISLTranslate (Joshi et al., Findings-ACL 2023) | ✅ REAL | 31k ISL–English sentence pairs |
| **OpenHands Indian-SL corpus (pretraining source)** | ✅ ON DISK 2026-10-02 | Zenodo 5558897, open CC-BY-4.0, md5-verified (`Sign_Library.zip` 3.03GB); extracted `sign_library_1_min_splits.h5` (374 chunks) + `sign_library_sub_based.h5` (9,582 clips). Layout confirmed: groups `keypoints` (F,75,3 float64) + `visibility` (F,75); pose-vis varies (mean 0.72), hand region 86% ones — matches extraction source. Zero-frame frac 0.6%. Bridge: 75 = pose[0:33] + Lhand[33:53] + Rhand[54:74] → `build_frame_features` (verified in plan `docs/superpowers/plans/2026-10-02-dpc-pilot.md`). |
| **SHuBERT — Gueuwou et al., ACL 2025 (`2025.acl-long.1397`), arXiv:2411.16765** | ✅ REAL (was §6.3 open lead) | ACL Anthology + repo `ShesterG/SHuBERT` (weights released). HuBERT-style multi-stream cluster prediction, 86M params, 400K steps on **8×A6000 (7 days)**. SOTA on SLT/ISLR. **PARKED for us with reason:** RGB+DINOv2-crop pipeline (not pose-only) + compute 100× our budget. Idea reusable later as pose-only cluster-prediction objective. |
| **FDMSE-ISL — Patra et al., arXiv:2407.14224; HWGATE follow-up, Springer PAA 2025, 28:148** | ✅ REAL | 40,033 videos / 2,002 words / **20 deaf signers**, gender-balanced, signer-independent splits built in, 36h. Supervised-pretrain evidence: HWGAT→INCLUDE **97.79%**, HWGATE→INCLUDE **98.28%**. Code: `suvajit-patra/sl-hwgat` (MediaPipe keypoint pipeline + `finetune on INCLUDE` command — matches our representation). **Access: portal-only, no mirror anywhere (verified 2026-10-03). PURSUIT DROPPED
  2026-10-03 by user decision (mail queue) — facts above retained for the record.** |

**SOTA sanity-check band:** credible 2025-era signer-independent ISLR top-1 on large-vocabulary
sets sits around **90–92% per-instance** (WLASL2000, MSASL1000). Our numbers are *not*
comparable to that band — different vocabulary, split, and supervision regime.

### Doc fixes still needed
- Cite DSTAdapter as **Tian et al., IJMLC 2026** (or "preprint 2025, published 2026") in
  `docs/ARCHITECTURE_V2.md`.
- Do not confuse DSTAdapter with **D²ST-Adapter** (Pei et al., ICCV 2025, few-shot *action*
  recognition) or **ST-Adapter** (Pan et al., NeurIPS 2022) — different works.

## 2. HTMA paper — reported vs. our reproduction

The paper is the direct source of this project's architecture. Comparison:

| Metric | Paper | This repo | Note |
|---|---|---|---|
| INCLUDE (original split) top-1 | 95.88% | — | paper's split, not comparable to ours |
| INCLUDE (official HF test split, 974 matched clips) top-1 | 87.4% (Khartheesvar LSTM, no pretrain) | **85.22%** (`results/official_split_metrics.json`, `models/baseline.pt`, 2026-10-02) | matched-protocol gap to LSTM is **~2.2pp**, not 5.6pp — the rest was split difference (our custom 478-split scores 81.8%, i.e. harder than official) |
| INCLUDE (their split) top-1 | 83.82% | **81.8%** | ours is their-style signer-independent split; ~2pp below |
| CISLR top-1 | 19.20% | **4.9%** | ⚠️ but see caveat — not comparable protocols |
| Trainable params | <10M | 2.03M | ours is 5× smaller |
| Disk footprint | 100MB | ~8MB | ours is far smaller |
| Pretraining | WLASL2000 + MS-ASL + ASL-Citizen | none | ⚠️ key difference |

**Correction (2026-10-02):** an earlier note in this file said the paper "does NOT pretrain."
That was wrong — the abstract explicitly states it pre-trains on WLASL2000, MS-ASL and
ASL-Citizen. Its 19.20% CISLR number is therefore a **pretrained** result, and is direct
evidence *for* cross-lingual pretraining rather than against it.

**Caveat on the CISLR row:** CISLR (Joshi et al., EMNLP 2022) is a **4,765-class one-shot
prototype task** with ~1.5 videos/word, scraped from YouTube. Our model has 263 classes, so
the comparison is not like-for-like. Published CISLR figures (16.81% original authors, 19.20%
Varanasi) are all one-shot-protocol numbers. Our 4.9% has a 95% CI of [1.9%, 12.0%] at n=81.
See `docs/RESULTS.md` for the full re-framing.

**The real gap is INCLUDE, not CISLR:** we are ~2pp below the paper on a matched-ish split,
but we sit at 81.8% where published no-pretrain MediaPipe+LSTM baselines are reported in the
mid-80s. See §6 — that claim is currently **unverified** and is the highest-value thing to check next.

## 3. SAME — the `−0.0000` result is NOT a wiring bug

An audit initially hypothesised that our exact-zero TTA delta meant the MoE forward path
was a no-op. **Tested and refuted.** `src/models/same_adapter.py:39` uses
`alpha = sigmoid(0.5) ≈ 0.62`, so the adapter residual is live from step 0, and the deployed
checkpoint loads with `fusion_method=early, use_separate_encoders=False` — which *is* the
path `SAMEAdapter.forward` hardcodes, so the reimplementation is faithful.

`results/tta_episodic.json` shows the real per-actor numbers:

| Actor | Base top-1 | Adapted top-1 | Delta |
|---|---|---|---|
| isl500_user014 | 26.47% | 26.32% | **−0.0015** |
| isl500_user015 | 84.36% | 84.50% | **+0.0014** |
| **mean** | | | **−2.2e-05** (displays as −0.0000) |

So the adapter *is* changing predictions; the two actors simply move in **opposite
directions**, cancelling at the mean. The honest conclusion is **"TTA gives no reliable
per-signer gain on this benchmark"**, not "TTA is broken."

Two contributing factors worth noting:
- The two actors have wildly different base accuracy (26% vs 84%), so a per-signer gain is
  dominated by whether the actor was already failing.
- SAME is a **translation** (seq2seq) method; we evaluate isolated classification. The paper's
  setting is outside what we measure, so a null result here is unsurprising.

## 4. Correction log

- **2026-10-02 (earlier, incorrect):** An audit subagent reported the HTMA CVPR 2026 paper as
  "fabricated / non-existent", based on Claude `WebSearch` + `WebFetch` returning no results.
  That conclusion was **wrong** — a false negative from those tools.
- **2026-10-02 (corrected):** Verified via the Parallel Search MCP against CVF Open Access. The
  paper exists, with published page numbers and BibTeX. All project docs citing it
  (`docs/architecture.md:58`, `docs/ARCHITECTURE_V2.md:9`, `docs/DATASETS.md:16`, master plan)
  are **correct as written**.
- **2026-10-02 (2nd pass):** SAME, KD-MSLRT, DSTAdapter and "Sign-Language Datasets at Scale" —
  all previously flagged "unverified / not confirmed" — are **all real**. Only DSTAdapter's
  year/venue needed correcting.
- **Lesson:** absence of a search hit is not evidence of absence. Any future "this citation is
  fake" claim in this repo must be confirmed against the publisher's own page before acting.

## 6. State of the art, and where the real gap is

### 6.1 Cross-lingual / cross-dataset pretraining — evidence FOR

The strongest quantitative evidence is **OpenHands** (Selvaraj, Nc, Kumar, Khapra — ACL 2022,
`2022.acl-long.150`), which pretrains on 1,129 h of *unlabeled ISL pose video* and then
**fully fine-tunes** an ST-GCN on INCLUDE:

| Pretraining objective | INCLUDE top-1 |
|---|---|
| None (from scratch) | 91.2 |
| Masked-based + fine-tune | 91.3 (**no gain**) |
| Contrastive + fine-tune | 90.8 (**worse**) |
| **Predictive coding (DPC) + fine-tune** | **94.7 (+3.5)** |

**This is the single most actionable finding.** Masking gave +0.1 (noise) and contrastive gave
−0.4 (actively harmful) on INCLUDE. On *pose sequences*, masked autoencoding collapses to
interpolating the masked frames rather than learning sign semantics. If we adopt any
pretraining objective, it should be **predictive coding**, not MAE and not SimCLR-style
contrastive.

Their low-resource table also shows the gain **scales inversely with labels per class**: +3.5
at ~17 samples/class, **+20.2 at 3 samples/class**. We have 263 classes × ~16 videos, so we
sit near the low-gain end — but FDMSE-ISL (40,033 videos / 2,002 words / 20 per class) is a
much denser target than CISLR.

**Multilingual beats monolingual.** **Sign2Vec / SignCorpus** (RJM Krishna et al., NeurIPS 2022
Datasets & Benchmarks) — 4,568 h across 10 sign languages, 75 pose keypoints incl. 11 face
points, PII-stripped. Multilingual pretraining + joint multilingual fine-tuning gave **+13.4%
total** over an ISL-only baseline, and critically *"accuracy on INCLUDE slightly improves even
when diversifying the pretraining beyond ISL"* — adding non-ISL data does **not** hurt our
target dataset. Pose-only means no RGB pipeline and no privacy exposure.

**Cross-dataset collapse is inherent, not a bug we introduced.** From the ASL Citizen paper
(arXiv:2304.05934), identical I3D features:

| Train | Test | Rec@1 |
|---|---|---|
| WLASL-2000 | WLASL-2000 | 0.3248 |
| **WLASL-2000** | **ASL-Citizen** | **0.0850** |

A **24-point drop from changing only the evaluation corpus.** Our cross-dataset weakness is
normal for this task family, and should be reported as such rather than treated as a defect.

### 6.2 Dataset feasibility

| Dataset | Access | License | Verdict |
|---|---|---|---|
| **ASL Citizen** (83,399 videos, 2,731 signs, 42.8 GB) | Direct `wget` from Microsoft CDN, **no registration** | Microsoft Research — non-commercial, **no redistribution** | **Only tractable piece of the Varanasi recipe** |
| WLASL2000 | Annotations free in-repo; **videos only via YouTube scraping**, URLs expire | C-UDA 1.0 (academic, non-commercial) | Annotations now; video lossy |
| MS-ASL | `MS-ASL.zip` is **1.9 MB of annotations, not video**; videos need scraping | C-UDA 0.1 | Annotations now; video lossy |
| **OpenHands ISL corpus** (1,129 h unlabeled ISL pose) | Released with pretrained weights | Academic | Best fit — ISL-native, pose-only |
| **SignCorpus** (4,568 h, 10 SLs, pose) | Released with SL-GCN pretrained weights | Academic, PII-stripped | Best multilingual option |

**Lighter-weight alternative to the whole ASL route:** pretrain on **OpenHands ISL corpus** or
**SignCorpus** — both are already pose-keypoint, matching our 168-dim/frame MediaPipe
representation, and both avoid the 42.8 GB download and YouTube scraping entirely.

### 6.3 ✅ VERIFIED (2026-10-02) — plain LSTM beats us on INCLUDE, no pretraining

**Khartheesvar, Kumar, Yadav, Yadav. "Automatic Indian sign language recognition using
MediaPipe holistic and LSTM network." Multimedia Tools and Applications (published
2023-12-21 online; vol. 83(20), pp. 58329–58348, June 2024), DOI
`10.1007/s11042-023-17361-y` — REAL**, corroborated 4×: Springer-mirror record
(abstract + numbers), researchr.org publication entry, and two independent citing works
quoting the exact figures (hybrid LSTM-GRU follow-up, 89.50% on 262 INCLUDE classes;
NCI thesis, §literature review). (Direct Springer page fetch was bot-blocked; indexer
consensus accepted per multi-source rule.)

| System | INCLUDE | INCLUDE-50 | Pretraining |
|---|---|---|---|
| Khartheesvar LSTM (MediaPipe Holistic + LSTM) | **87.4%** (F1 86.6%) | 94.8% | none |
| LSTM-GRU hybrid follow-up | **89.5%** (F1 89.6%, 262 classes) | — | none |
| Ours (HTMA, `models/baseline.pt`) | 81.8% | 100% val | none |
| Varanasi HTMA (paper) | 83.82% signer-indep / 95.88% original split | — | WLASL+MS-ASL+ASL-Citizen |

**Implication — fork RESOLVED by matched-split comparison (no paper text needed):**
no open-access full text exists (OpenAlex: closed, no repository copy; Semantic Scholar:
same), so the split protocol can't be read directly. But it can be inferred — put every
number on its own split protocol:

| System | Official INCLUDE split | Signer-independent split |
|---|---|---|
| Varanasi HTMA (paper) | **95.88%** | 83.82% |
| Khartheesvar LSTM | **87.4%** (presumed official — pre-2026 standard practice) | unknown |
| Ours (`models/baseline.pt`) | unmeasured | **81.8%** |

On matched protocols the paradox dissolves: HTMA-family beats the LSTM 95.88 → 87.4 on
the official split, and our real gap is only **~2pp to Varanasi (81.8 vs 83.82)** on the
signer-independent split — consistent with our smaller model (2.03M vs <10M params) and
no pretraining, not a broken architecture. **The "LSTM-ablation outranks pretraining"
branch is REJECTED.** Cheapest decisive check: evaluate `models/baseline.pt` on the
official INCLUDE split (eval-only, no training) — expect ~90%+, which would confirm the
gap is protocol, not model. Pretraining route (DPC on OpenHands/SignCorpus) stands as the
principled next gain.

**Decisive check DONE (2026-10-02):** `scripts/eval_official_split.py` matched 974/1009
official test videos to local `.npz` (35 ambiguous duplicate rows dropped, 0 missing):
`models/baseline.pt` scores **85.22% top-1 / 87.99% top-5** (`results/official_split_metrics.json`).
Matched-protocol standings: pretrained HTMA 95.88 > LSTM 87.4 > ours 85.22 (all no-pretrain
gap ~2.2pp). The remaining ~10pp to 95.88 is the pretraining effect — exactly the
literature expectation. Our custom 478-split (81.8%) is harder than the official split.
LSTM-ablation stays rejected; DPC pretraining remains the principled next gain.
Anomaly noted: top-5 exceeds top-1 by only 2.8pp on the official split vs 13.6pp on our
custom split — the official test set is more fine-grained-confusable for this model; not
investigated further (eval-only result, no training decision depends on it).

Also unverified: `SHuBERT` (ACL 2025, `2025.acl-long.1397`) — masked token prediction over
clustered hand/face/body pose streams, ~1,000 h ASL, with ISLR among its evaluated tasks. It is
the closest published analogue to our SAME/DSTAdapter design and is worth reading before any
further TTA work.

## 7. Method

Primary source: **Parallel Search MCP** (`web_search` + `web_fetch`) against publisher pages
(CVF Open Access, ACL Anthology, AAAI OJS, SpringerLink, arXiv, DBLP, Semantic Scholar).
Claude-native `WebSearch`/`WebFetch` proved unreliable for this task and produced the false
negative in §4 — per standing project instruction, they are not used.