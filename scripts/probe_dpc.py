#!/usr/bin/env python
"""Transfer probe: frozen encoder + linear head (10 epochs, INCLUDE val).
Compares DPC checkpoints against a scratch (random) encoder control.
Gate: proceed to full fine-tune only if best DPC beats control by >=1pp.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def run_probe(encoder_state: dict | None, label_map, npz_dir: str, epochs: int = 10) -> float:
    from src.data.dataset import create_dataloader
    from src.models.htma import HTMAClassifier, HTMAConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = HTMAConfig(input_dim=168, embedding_dim=256, num_blocks=2,
                     num_heads=8, dropout=0.1, num_classes=264)
    enc = HTMAClassifier(cfg).to(device)
    if encoder_state is not None:
        enc_state = {k: v for k, v in encoder_state.items() if not k.startswith("head.")}
        enc.load_state_dict(enc_state, strict=False)
    for p in enc.parameters():
        p.requires_grad = False
    head = nn.Linear(256, len(label_map)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    train_loader = create_dataloader(npz_dir, "train", label_map, 32, 256, 4, augment=False)
    val_loader = create_dataloader(npz_dir, "val", label_map, 32, 256, 4, augment=False)
    enc.eval()
    for _ in range(epochs):
        head.train()
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            with torch.no_grad():
                h = enc.input_proj(x)
                h = enc.input_norm(h)
                h = enc.appe(h)
                rc, rs = enc._get_rope(h)
                for b in enc.blocks:
                    h = b(h, rc, rs, m)
                z = enc.pool(h, m)
            opt.zero_grad()
            loss = crit(head(z), y)
            loss.backward()
            opt.step()
    head.eval()
    ok = tot = 0
    with torch.no_grad():
        for x, m, y in val_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            h = enc.input_proj(x)
            h = enc.input_norm(h)
            h = enc.appe(h)
            rc, rs = enc._get_rope(h)
            for b in enc.blocks:
                h = b(h, rc, rs, m)
            pred = head(enc.pool(h, m)).argmax(1).cpu()
            ok += (pred == y.cpu()).sum().item()
            tot += y.numel()
    return 100.0 * ok / max(tot, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-dir", default="models/dpc")
    parser.add_argument("--npz-dir", default="data/processed/landmarks/isl")
    parser.add_argument("--label-map", default="data/processed/landmarks/isl_label_map.json")
    parser.add_argument("--output", default="results/dpc_probe.json")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    from src.data.metadata import load_label_map

    label_map = load_label_map(args.label_map)
    rows = [{"step": "scratch", "val_top1": round(run_probe(None, label_map, args.npz_dir, args.epochs), 2)}]
    print(rows[-1], flush=True)
    for ckpt in sorted(Path(args.encoder_dir).glob("encoder_step*.pt")):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        acc = run_probe(state, label_map, args.npz_dir, args.epochs)
        rows.append({"step": ckpt.stem, "val_top1": round(acc, 2)})
        print(rows[-1], flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
