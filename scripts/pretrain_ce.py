#!/usr/bin/env python
"""CE pre-training of the CTC head on isolated word clips (single-gloss targets).

Mean-pools per-frame logits -> CE. Initializes emission behavior before CTC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="models/ctc_ce")
    args = parser.parse_args()

    from src.data.csltr_dataset import CSLTRDataset, ctc_collate
    from src.models.ctc_head import HTMACTCModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab: dict[str, int] = json.loads(Path("models/ctc/gloss_vocab.json").read_text())
    model = HTMACTCModel("models/multihead/best.pt", len(vocab) - 1).to(device)

    Liab = Path("data/processed/isl-csltr/word_landmarks")
    files = [p for p in sorted(Liab.glob("*.npz"))
             if str(np.load(p, allow_pickle=True).get("split", "")) == "train"]

    X, y = [], []
    for p in files:
        a = np.load(p, allow_pickle=True)
        lab = str(a.get("label", "")).strip().upper()
        if lab not in vocab:
            continue
        f = np.asarray(a["features"], dtype=np.float32)[:128]
        X.append(f)
        y.append(vocab[lab] - 1)
    n_cls = len(vocab) - 1
    logger.info(f"Word clips: {len(X)} covering {len(set(y))} glosses")

    opt = AdamW(model.head.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        tot, n = 0.0, 0
        for i in range(0, len(X), args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch = [torch.from_numpy(X[j]) for j in idx]
            T = max(f.shape[0] for f in batch)
            Xb = torch.zeros(len(batch), T, 168)
            Mb = torch.zeros(len(batch), T, dtype=torch.bool)
            for b, f in enumerate(batch):
                Xb[b, :f.shape[0]] = f
                Mb[b, :f.shape[0]] = True
            yb = torch.tensor([y[j] for j in idx], dtype=torch.long).to(device)
            opt.zero_grad()
            with torch.no_grad():
                h = model.encoder.input_proj(Xb.to(device))
                h = model.encoder.input_norm(h)
                h = model.encoder.appe(h)
                cos, sin = model.encoder._get_rope(h)
                for block in model.encoder.blocks:
                    h = block(h, cos, sin, Mb.to(device))
            logits = model.head(h)
            pooled = (logits * Mb.to(device).unsqueeze(-1)).sum(1) / Mb.to(device).sum(1, keepdim=True).clamp_min(1)
            loss = ce(pooled[:, 1:], yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | loss={tot/max(n,1):.4f}")
    torch.save({"head_state_dict": model.head.state_dict()}, out / "head_ce.pt")
    logger.info(f"Saved {out / 'head_ce.pt'}")


if __name__ == "__main__":
    main()
