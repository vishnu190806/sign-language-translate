#!/usr/bin/env python
"""Evaluate the frozen extra head on ISL500-only words (signer-disjoint test).

Reads split/label from the extra .npz files; reports overall + per-word top-1.
Pass criterion for Phase A learnability: overall extra-test top-1 >= 30%.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/multihead_extra/best.pt")
    parser.add_argument("--base", default="models/multihead/best.pt")
    parser.add_argument("--npz-dir", default="data/processed/landmarks/isl500_extra")
    parser.add_argument("--label-map", default="models/multihead_extra/label_map_extra.json")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="results/extra_eval.json")
    args = parser.parse_args()

    from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extra_map: dict[str, int] = json.loads(Path(args.label_map).read_text())
    n_joint = len(json.loads(Path(args.base).parent.joinpath("label_map.json").read_text()))
    htm = HTMAConfig(input_dim=168, embedding_dim=256, num_blocks=2, num_heads=8,
                     dropout=0.1, num_classes=n_joint)
    model = MultiHeadHTMAClassifier(
        htm, sources=("include", "isl500", "extra"),
        head_classes={"include": n_joint, "isl500": n_joint, "extra": len(extra_map)},
    ).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    idx2word = {v: k for k, v in extra_map.items()}

    files = sorted(Path(args.npz_dir).glob("*.npz"))
    test = []
    for p in files:
        try:
            a = np.load(p, allow_pickle=True)
            if str(a.get("split", "")) != "test":
                continue
            lab = str(a.get("label", ""))
            if lab in extra_map:
                test.append((p, extra_map[lab]))
        except Exception:
            continue
    print(f"extra test clips: {len(test)}")

    def prep(f: np.ndarray):
        T = f.shape[0]
        if T < args.seq_len:
            pad = np.zeros((args.seq_len - T, f.shape[1]), np.float32)
            return np.concatenate([f, pad])
        return f[np.linspace(0, T - 1, args.seq_len).astype(int)]

    per_word: dict[str, list] = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for s in range(0, len(test), args.batch_size):
            chunk = test[s:s + args.batch_size]
            xb = torch.from_numpy(np.stack(
                [prep(np.load(p)["features"].astype(np.float32)) for p, _ in chunk])).to(device)
            mb = torch.ones(xb.shape[0], args.seq_len, dtype=torch.bool).to(device)
            pred = model(xb, mb, source="extra").argmax(1).cpu().numpy()
            for (p, y), q in zip(chunk, pred):
                w = idx2word[y]
                per_word[w][1] += 1
                per_word[w][0] += int(q == y)
    tot_ok = sum(v[0] for v in per_word.values())
    tot = sum(v[1] for v in per_word.values())
    out = {"n": tot, "top1": round(100 * tot_ok / max(tot, 1), 2),
           "per_word": {w: round(100 * v[0] / v[1], 1) for w, v in sorted(per_word.items())}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(f"extra-test top-1: {out['top1']}% on {tot} clips -> {args.output}")


if __name__ == "__main__":
    main()
