#!/usr/bin/env python
"""Fit per-head temperature scaling (Guo et al., ICML 2017) on validation logits.

Writes results/temperature.json: {source: {temperature, nll_before, nll_after, ece_before, ece_after}}.
Calibrated temperatures feed the stability-gate operating point (risk-coverage tuning).
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


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            out += abs(acc[m].mean() - conf[m].mean()) * m.sum() / len(labels)
    return float(out)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float]:
    from torch import nn

    nll = nn.CrossEntropyLoss()
    with torch.no_grad():
        before = nll(logits, labels).item()
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = nll(logits / T.clamp_min(1e-3), labels)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(T.clamp_min(1e-3).item())
    with torch.no_grad():
        after = nll(logits / t, labels).item()
    return t, before, after


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/multihead_v2/best.pt")
    parser.add_argument("--label-map", default="models/multihead_v2/label_map.json")
    parser.add_argument("--extra-map", default="models/multihead_extra/label_map_extra.json")
    parser.add_argument("--output", default="results/temperature_v2.json")
    args = parser.parse_args()

    from src.data.dataset import create_dataloader
    from src.data.metadata import load_label_map
    from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier
    from scripts.train_multihead import canon

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(args.label_map)
    extra_map = load_label_map(args.extra_map)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    head_classes: dict[str, int] = dict(ckpt.get("head_classes") or {})
    for src in ckpt.get("sources", ["include"]):
        for suffix in ("heads." + src + ".3.weight", "heads." + src + ".weight"):
            if suffix in state:
                head_classes[src] = state[suffix].shape[0]
                break
    maps = {"include": label_map, "isl500": label_map, "extra": extra_map}
    htm = HTMAConfig(num_classes=max(head_classes.values(), default=len(label_map)))
    model = MultiHeadHTMAClassifier(htm, sources=tuple(ckpt.get("sources", ["include"])),
                                    head_classes=head_classes).to(device).eval()
    own = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items()
                           if k in own and own[k].shape == v.shape}, strict=False)

    dirs = {"include": "data/processed/landmarks/isl",
            "isl500": "data/processed/landmarks/isl500",
            "extra": "data/processed/landmarks/isl500_extra"}
    out: dict = {}
    with torch.no_grad():
        for source in ckpt.get("sources", ["include"]):
            lm = maps.get(source, label_map)
            loader = create_dataloader(dirs[source], "val", lm, 32, 512, 0,
                                       augment=False, normalize_fn=canon)
            Ls, ys = [], []
            for f, m, l in loader:
                Ls.append(model(f.to(device), m.to(device), source=source).cpu())
                ys.append(l)
            logits = torch.cat(Ls)
            labels = torch.cat(ys)
            t, nll_b, nll_a = fit_temperature(logits, labels)
            probs = torch.softmax(logits, 1).numpy()
            cal = torch.softmax(logits / t, 1).numpy()
            y = labels.numpy()
            out[source] = {"temperature": t, "nll_before": nll_b, "nll_after": nll_a,
                           "ece_before": ece(probs, y), "ece_after": ece(cal, y),
                           "samples": len(y)}
            print(f"{source}: T={t:.3f} NLL {nll_b:.4f}->{nll_a:.4f} ECE {out[source]['ece_before']:.4f}->{out[source]['ece_after']:.4f}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"temperatures": {s: v["temperature"] for s, v in out.items()},
         "default_source": "include", "details": out}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
