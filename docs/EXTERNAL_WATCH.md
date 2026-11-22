# External Watch: STARK code + FDMSE-ISL direct download
Checked: 2026-10-09 ~06:30 UTC. Re-check monthly or on new STARK arXiv version.

## 1. STARK (arXiv:2603.16163, Mar 2026) — NO public code repo
- arXiv abs (https://arxiv.org/abs/2603.16163): v1 only (17 Mar 2026), no "Code" link;
  "Code, Data and Media" section shows only empty toggles (CatalyzeX/DagsHub/HF).
- RKMVERI ISL portal recognition page (https://cs.rkmvu.ac.in/~isl/recognition.html):
  STARK card has `[Code]()` = EMPTY href (dead button). Dataset button points to
  PHOENIX-2014-T (https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/), not STARK.
- GitHub org/user (https://github.com/suvajit-patra?tab=repositories): 31 public repos, none
  STARK-related (only sl-hwgat, sl-hwgat-demo, sl-vqvae-gen-demo, data-collection/annotation, viewers).
  API search `STARK sign language keypoints` (https://api.github.com/search/repositories?q=STARK+sign+language+keypoints)
  returns total_count=0. No forks possible (nothing to fork).
- Papers-with-Code / HF Papers: no paper+code entry found (web search returns only arXiv/HTML/PDF mirrors;
  https://huggingface.co/papers/2603.16163 exists as paper page with NO code link).
  paperswithcode.com/* now redirects to HF trending papers, so legacy PWC URLs are unverifiable live.
- Contrast: sibling HWGAT code IS public at https://github.com/suvajit-patra/sl-hwgat — STARK has no equivalent.
- Verdict: DEFINITIVE ABSENCE as of 2026-10-09. Action: watch portal Code button + new arXiv versions.

## 2. FDMSE-ISL dataset (40,033 vids / 2,002 words, arXiv:2407.14224) — NO direct download; portal-request-only
- Portal (https://cs.rkmvu.ac.in/~isl/ + /recognition.html): BOTH dataset buttons empty
  (`[Dataset]()` on HWGAT/HWGATE card). Only asset: per-word demo dictionary videos, no bulk download.
  Paper wording: "will be made publicly available (https://cs.rkmvu.ac.in/~isl) with permission for
  research purposes only" = gated, not open.
- HuggingFace datasets API (https://huggingface.co/api/datasets?search=fdmse): `[]` (zero hits).
  Broader `?search=indian sign language` returns only small/unrelated sets (alphabet images,
  govt-dictionary re-encodes) — no 40k-video FDMSE-ISL.
- Kaggle search `fdmse-isl` (https://www.kaggle.com/search?q=fdmse-isl): no match. Closest hits are
  unrelated: ISLVT 76-sentence set (https://www.kaggle.com/datasets/manojkumarcs28/indian-sign-language-video-and-text-dataset)
  and an old alphabet-image set — not FDMSE-ISL.
- Papers-with-Code: registry/benchmark entry existed (cited as https://paperswithcode.com/dataset/fdmse-isl;
  task page lists `FDMSE-ISL / HWGAT` benchmark) but PWC never hosts files — link-only, and the URL
  now redirects to HF. No downloadable artifact.
- Third-party mirror (https://www.selectdataset.com/dataset/ff9ae85df1bf70f4f50448af6e9d8c6b/fdmse-isl):
  metadata/description only, no files. Massey catalog (https://salp.massey.ac.nz/sign_lan_datasets.html)
  just links back to the RKMVERI portal.
- Verdict: CONFIRMED still portal-request-only as of 2026-10-09. Action: request via portal/author
  email (suvajit.patra.cs20@gm.rkmvu.ac.in); sl-hwgat README (`FDMSE-ISL_meta_gen.py`) is pipeline-only, needs local copy.
