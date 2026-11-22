#!/usr/bin/env python
"""Few-shot prototype decoder: frozen multi-head encoder + cosine prototypes.

Prototypes: mean pooled embedding of train word clips per gloss.
Decode: sliding windows over test sentences -> nearest prototype -> collapse -> WER.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.continuous_baseline import wer


def embed_clip(model, feats: np.ndarray, device) -> np.ndarray:
    f = np.asarray(feats, dtype=np.float32)[:128]
    m = (np.abs(f).sum(axis=1) > 1e-6)
    if not m.any():
        return np.zeros(256, dtype=np.float32)
    with torch.no_grad():
        v = model.encode(torch.from_numpy(f[None]).to(device),
                         torch.from_numpy(m[None]).to(device)).cpu().numpy()[0]
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-9 else v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/multihead/best.pt")
    parser.add_argument("--word-dir", default="data/processed/isl-csltr/word_landmarks")
    parser.add_argument("--sent-dir", default="data/processed/isl-csltr/landmarks")
    parser.add_argument("--output", default="results/prototype_decode.json")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_cls = ckpt.get("config", {}).get("model", {}).get("num_classes", 263)
    htm = HTMAConfig(num_classes=n_cls)
    model = MultiHeadHTMAClassifier(htm, sources=tuple(ckpt["sources"])).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    bank: dict[str, list[np.ndarray]] = defaultdict(list)
    for p in sorted(Path(args.word_dir).glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        if str(d.get("split", "")) != "train":
            continue
        bank[str(d["label"]).upper()].append(embed_clip(model, d["features"], device))
    protos = {g: np.mean(v, axis=0) for g, v in bank.items() if len(v)}
    protos = {g: v / np.linalg.norm(v) for g, v in protos.items()}
    names = sorted(protos)
    P = np.stack([protos[g] for g in names])
    print(f"prototypes: {len(names)} glosses")

    files = []
    for p in sorted(Path(args.sent_dir).glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        if str(d.get("split", "")) == "test":
            files.append(p)
    print(f"test sentences: {len(files)}")

    S = D = I = N = 0
    rows = []
    for p in files:
        d = np.load(p, allow_pickle=True)
        feats = np.asarray(d["features"], dtype=np.float32)
        ref = [str(g).upper() for g in d["glosses"]]
        seq = []
        for s in range(0, max(len(feats) - args.window + 1, 1), args.stride):
            w = feats[s:s + args.window]
            if len(w) < args.window:
                w = np.concatenate([w, np.zeros((args.window - len(w), 168), np.float32)])
            v = embed_clip(model, w, device)
            sims = P @ v
            top = int(sims.argmax())
            seq.append(names[top] if sims[top] >= args.min_score else "<blank>")
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
        rows.append({"file": p.name, "ref": ref, "hyp": hyp, "wer": (s + d_ + i) / max(n, 1)})
    out = {"n": len(files), "prototypes": len(names), "wer": (S + D + I) / max(N, 1),
           "sub": S, "del": D, "ins": I, "ref_words": N, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(f"WER: {out['wer']*100:.1f}% (sub={S} del={D} ins={I} ref={N})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
