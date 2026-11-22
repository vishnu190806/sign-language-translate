#!/usr/bin/env python
"""Sliding-window decode with a CE-pretrained head (full encoder path)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.continuous_baseline import wer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", default="models/ctc_ce/head_ce.pt")
    parser.add_argument("--base-checkpoint", default="models/multihead/best.pt")
    parser.add_argument("--vocab", default="models/ctc/gloss_vocab.json")
    parser.add_argument("--npz-dir", default="data/processed/isl-csltr/landmarks")
    parser.add_argument("--output", default="results/ce_window_decode.json")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--blank-conf", type=float, default=0.3)
    args = parser.parse_args()

    from src.models.ctc_head import HTMACTCModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = json.loads(Path(args.vocab).read_text())
    iv = {i: k for k, i in v.items()}
    m = HTMACTCModel(args.base_checkpoint, len(v) - 1, device=device).to(device).eval()
    m.head.load_state_dict(torch.load(args.head, map_location=device)["head_state_dict"])
    print("CE head loaded (trained on word clips)")

    files = []
    for p in sorted(Path(args.npz_dir).glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        if str(d.get("split", "")) == "test":
            files.append(p)
    print(f"test sentences: {len(files)}")
    if not files:
        raise ValueError("No test files found")

    S = D = I = N = 0
    rows = []
    with torch.no_grad():
        for p in files:
            d = np.load(p, allow_pickle=True)
            feats = np.asarray(d["features"], dtype=np.float32)
            ref = [str(g).upper() for g in d["glosses"]]
            seq = []
            for s in range(0, max(len(feats) - args.window + 1, 1), args.stride):
                w = feats[s:s + args.window]
                if len(w) < args.window:
                    w = np.concatenate([w, np.zeros((args.window - len(w), 168), np.float32)])
                t = torch.from_numpy(w[None]).to(device)
                mm = torch.ones(1, args.window, dtype=torch.bool).to(device)
                h = m.encoder.input_proj(t)
                h = m.encoder.input_norm(h)
                h = m.encoder.appe(h)
                cos, sin = m.encoder._get_rope(h)
                for block in m.encoder.blocks:
                    h = block(h, cos, sin, mm)
                logits = m.head(h)
                probs = torch.softmax(logits.mean(1), -1).cpu().numpy()[0]
                top = int(probs[1:].argmax()) + 1
                seq.append(iv[top] if probs[top] >= args.blank_conf else "<blank>")
            hyp = []
            for g in seq:
                if g == "<blank>":
                    continue
                if not hyp or hyp[-1] != g:
                    hyp.append(g)
            s, d_, i, n = wer(hyp, ref)
            S += s
            D += d_
            I += i
            N += n
            rows.append({"file": p.name, "ref": ref, "hyp": hyp})
    out = {"n": len(files), "wer": (S + D + I) / max(N, 1), "sub": S, "del": D, "ins": I,
           "ref_words": N, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(f"WER: {out['wer']*100:.1f}% (sub={S} del={D} ins={I} ref={N})")


if __name__ == "__main__":
    main()
