#!/usr/bin/env python
"""Multi-head co-training (Logos recipe): shared HTMA encoder, one head per source.

Alternates equal batches from INCLUDE and ISL500; loss = mean of per-head CE.
Best checkpoint selected on mean of per-source val accuracy.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import zip_longest
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYN = {"alright": "okay", "thankyou": "thank you", "big large": "big",
       "small little": "small", "store or shop": "store",
       "race (ethnicity)": "race", "you (plural)": "you plural"}


def canon(lab: str) -> str:
    lab = lab.split(".")[-1].strip().lower() if "." in lab else lab.strip().lower()
    return SYN.get(lab, lab)


def run_epoch(model, loaders, criterion, optimizer, device, train: bool):
    model.train(train)
    stats: dict[str, dict] = {}
    with torch.set_grad_enabled(train):
        for source, loader in loaders.items():
            total_loss, correct, total = 0.0, 0, 0
            for features, masks, labels in loader:
                features = features.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if train:
                    optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                    logits = model(features, masks, source=source)
                    loss = criterion(logits, labels)
                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                total_loss += loss.item() * features.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += features.size(0)
            stats[source] = {"loss": total_loss / max(total, 1), "top1": 100.0 * correct / max(total, 1)}
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-head INCLUDE+ISL500 co-training")
    parser.add_argument("--include-dir", default="data/processed/landmarks/isl")
    parser.add_argument("--isl500-dir", default="data/processed/landmarks/isl500")
    parser.add_argument("--label-map", default="data/processed/landmarks/joint_label_map.json")
    parser.add_argument("--output-dir", default="models/multihead")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    from src.config import load_config
    from src.data.dataset import create_dataloader
    from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map: dict[str, int] = json.loads(Path(args.label_map).read_text())
    config = load_config("configs/default.yaml", "configs/joint.yaml")

    htm = HTMAConfig(input_dim=168, embedding_dim=256, num_blocks=2, num_heads=8,
                     dropout=0.1, num_classes=len(label_map))
    model = MultiHeadHTMAClassifier(htm, sources=("include", "isl500")).to(device)
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    def make_loader(d, split, aug):
        return create_dataloader(d, split, label_map, 32, args.batch_size, 6,
                                 augment=aug, augmentation_mode="basic" if aug else "none",
                                 prefetch_factor=2, persistent_workers=True,
                                 normalize_fn=canon)

    train_loaders = {"include": make_loader(args.include_dir, "train", True),
                     "isl500": make_loader(args.isl500_dir, "train", True)}
    val_loaders = {"include": make_loader(args.include_dir, "val", False),
                   "isl500": make_loader(args.isl500_dir, "val", False)}
    logger.info("Train sizes: " + str({k: len(v.dataset) for k, v in train_loaders.items()}))
    logger.info("Val sizes: " + str({k: len(v.dataset) for k, v in val_loaders.items()}))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best, patience = 0.0, 0
    for epoch in range(args.epochs):
        for loader in train_loaders.values():
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(epoch)
        tr = run_epoch(model, train_loaders, criterion, optimizer, device, True)
        with torch.no_grad():
            va = run_epoch(model, val_loaders, criterion, optimizer, device, False)
        mean_val = sum(v["top1"] for v in va.values()) / len(va)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | train=" +
                    " ".join(f"{k}:{v['top1']:.1f}%" for k, v in tr.items()) +
                    " | val=" + " ".join(f"{k}:{v['top1']:.1f}%" for k, v in va.items()) +
                    f" | mean={mean_val:.2f}%")
        if mean_val > best:
            best, patience = mean_val, 0
            torch.save({"model_state_dict": model.state_dict(),
                        "sources": model.sources,
                        "metrics": va, "mean_val": mean_val},
                       out / "best.pt")
            with open(out / "label_map.json", "w") as f:
                json.dump(label_map, f, indent=1)
            logger.info(f"New best saved: {mean_val:.2f}%")
        else:
            patience += 1
        if patience >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    logger.info(f"Done. Best mean val: {best:.2f}%")


if __name__ == "__main__":
    main()
