#!/usr/bin/env python
"""Episodic per-actor TTA: adapt SAME-MoE on k clips, eval rest. Reset per episode.

Actors: ISL500 users (signer field isl500_userNNN). Default: unseen users 014/015.
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


def load_clip(p: Path, seq_len: int = 128):
    a = np.load(p, allow_pickle=True)
    f = np.asarray(a["features"], dtype=np.float32)
    if f.shape[0] > seq_len:
        f = f[np.linspace(0, f.shape[0] - 1, seq_len).astype(int)]
    m = (np.abs(f).sum(axis=1) > 1e-6)
    return f, m, str(a.get("label", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/multihead/best.pt")
    parser.add_argument("--npz-dir", default="data/processed/landmarks/isl500")
    parser.add_argument("--label-map", default="data/processed/landmarks/joint_label_map.json")
    parser.add_argument("--actors", default="014,015")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/tta_episodic.json")
    args = parser.parse_args()

    from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier
    from src.models.same_adapter import SAMEAdapter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map: dict[str, int] = json.loads(Path(args.label_map).read_text())
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    htm = HTMAConfig(num_classes=len(label_map))

    actors = [a.strip().zfill(3) for a in args.actors.split(",")]
    pools: dict[str, list] = defaultdict(list)
    wanted = {f"isl500_{a}_" for a in actors}
    for p in sorted(Path(args.npz_dir).glob("*.npz")):
        if not any(p.name.startswith(w) for w in wanted):
            continue
        a = np.load(p, allow_pickle=True)
        lab = str(a.get("label", ""))
        if lab not in label_map:
            continue
        f, m, _ = load_clip(p)
        pools[str(a.get("signer", "unknown"))].append((f, m, label_map[lab]))
    print("pool sizes:", {k: len(v) for k, v in pools.items()})

    rng = np.random.default_rng(args.seed)
    results = []
    for actor in [f"isl500_user{a}" for a in actors]:
        clips = pools.get(actor, [])
        if len(clips) < args.k + 1:
            print(f"{actor}: only {len(clips)} clips, skipping")
            continue
        idx = rng.permutation(len(clips))
        adapt_set = [clips[i] for i in idx[:args.k]]
        eval_set = [clips[i] for i in idx[args.k:]]

        def batch(samples):
            T = max(f.shape[0] for f, _, _ in samples)
            X = np.zeros((len(samples), T, 168), np.float32)
            M = np.zeros((len(samples), T), bool)
            y = []
            for i, (f, m, lab) in enumerate(samples):
                X[i, :f.shape[0]] = f
                M[i, :f.shape[0]] = m
                y.append(lab)
            return (torch.from_numpy(X).to(device), torch.from_numpy(M).to(device),
                    torch.tensor(y).to(device))

        def eval_acc(fn, samples, chunk=256):
            ok = tot = 0
            with torch.no_grad():
                for s in range(0, len(samples), chunk):
                    Xc, Mc, yc = batch(samples[s:s + chunk])
                    ok += (fn(Xc, Mc).argmax(1) == yc).sum().item()
                    tot += len(yc)
                    del Xc, Mc, yc
            return ok / max(tot, 1)

        Xa, Ma, _ = batch(adapt_set)
        base = MultiHeadHTMAClassifier(htm, sources=("include", "isl500")).to(device).eval()
        base.load_state_dict(ckpt["model_state_dict"])
        base_acc = eval_acc(lambda X, M: base(X, M, source="isl500"), eval_set)
        adapter = SAMEAdapter(base).to(device)
        snap = adapter.snapshot()
        adapter.adapt(Xa, Ma, source="isl500", num_steps=args.steps, lr=args.lr)
        ad_acc = eval_acc(lambda X, M: adapter(X, M, source="isl500"), eval_set)
        adapter.reset(snap)
        del Xa, Ma
        results.append({"actor": actor, "n_adapt": len(adapt_set), "n_eval": len(eval_set),
                        "base_top1": base_acc, "adapted_top1": ad_acc, "delta": ad_acc - base_acc})
        print(f"{actor}: base={base_acc*100:.1f}% adapted={ad_acc*100:.1f}% delta={ad_acc-base_acc:+.3f}")

    deltas = [r["delta"] for r in results]
    out = {"results": results, "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
           "win_rate": sum(1 for d in deltas if d > 0) / max(len(deltas), 1),
           "k": args.k, "steps": args.steps, "lr": args.lr}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"mean delta={out['mean_delta']:+.4f} win_rate={out['win_rate']:.0%} -> {args.output}")


if __name__ == "__main__":
    main()
