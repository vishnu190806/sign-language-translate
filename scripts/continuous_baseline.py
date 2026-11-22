#!/usr/bin/env python
"""A1 sliding-window continuous baseline over ISL-CSLTR sentence landmarks.

Window (32f, stride 8) -> multihead include head (+temperature) -> argmax ->
collapse repeats, drop low-conf 'blanks' -> gloss sequence. Reports WER
(insertions/deletions/substitutions) vs gold glosses.
"""
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


def wer(hyp: list[str], ref: list[str]) -> tuple[int, int, int, int]:
    R, H = len(ref), len(hyp)
    d = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        d[i][0] = i
    for j in range(H + 1):
        d[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, s, rem = R, H, 0, 0
    ins = del_ = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            del_ += 1
            i -= 1
        else:
            s += 1
            i -= 1
            j -= 1
            rem += 0
    return s, del_, ins, R


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/multihead/best.pt")
    parser.add_argument("--npz-dir", default="data/processed/isl-csltr/landmarks")
    parser.add_argument("--label-map", default="models/multihead/label_map.json")
    parser.add_argument("--output", default="results/continuous_baseline.json")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--blank-conf", type=float, default=0.3)
    args = parser.parse_args()

    from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx2word: dict[int, str] = {int(v): k for k, v in json.loads(Path(args.label_map).read_text()).items()}
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    htm = HTMAConfig(num_classes=len(idx2word))
    model = MultiHeadHTMAClassifier(htm, sources=tuple(ckpt["sources"])).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    temp = 1.0
    tp = Path(args.checkpoint).parent / "temperature.json"
    if tp.exists():
        temp = float(json.loads(tp.read_text())["temperatures"]["include"])

    files = []
    for p in sorted(Path(args.npz_dir).glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        if str(d.get("split", "")) == "test":
            files.append(p)
    print(f"test sentences: {len(files)}")

    S = D = I = N = OOV = 0
    rows = []
    with torch.no_grad():
        for p in files:
            d = np.load(p, allow_pickle=True)
            feats = np.asarray(d["features"], dtype=np.float32)
            ref = [str(g) for g in d["glosses"]]
            seq = []
            for s in range(0, max(len(feats) - args.window + 1, 1), args.stride):
                w = feats[s:s + args.window]
                if len(w) < args.window:
                    w = np.concatenate([w, np.zeros((args.window - len(w), 168), np.float32)])
                m = (np.abs(w).sum(axis=1) > 1e-6)
                logits = model(torch.from_numpy(w[None]).to(device),
                               torch.from_numpy(m[None]).to(device), source="include") / temp
                probs = torch.softmax(logits, 1).cpu().numpy()[0]
                top = int(probs.argmax())
                seq.append((idx2word[top], float(probs[top])) if probs[top] >= args.blank_conf else ("<blank>", 0.0))
            hyp = []
            for word, _ in seq:
                if word == "<blank>":
                    continue
                if not hyp or hyp[-1] != word:
                    hyp.append(word)
            s, d_, i, n = wer(hyp, ref)
            S += s
            D += d_
            I += i
            N += n
            OOV += sum(1 for g in ref if g.lower() not in set(idx2word.values()))
            rows.append({"file": p.name, "ref": ref, "hyp": hyp,
                         "wer": (s + d_ + i) / max(n, 1)})
    out = {"n": len(files), "wer": (S + D + I) / max(N, 1),
           "sub": S, "del": D, "ins": I, "ref_words": N, "oov_ref_words": OOV,
           "window": args.window, "stride": args.stride, "blank_conf": args.blank_conf,
           "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(f"WER: {out['wer']*100:.1f}% (sub={S} del={D} ins={I} ref={N} oov={OOV})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
