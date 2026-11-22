#!/usr/bin/env python
"""Phase B: joint 3-source fine-tune (include + isl500 + extra), warm-started.

Ship rule (hard): custom test >= 81.8 AND official test >= 85.22.
Training guard: abort if include-val drops >3pp below its starting value
for 3 straight epochs (prevents drift before wasting GPU).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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
            stats[source] = {"loss": total_loss / max(total, 1),
                             "top1": 100.0 * correct / max(total, 1)}
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="models/multihead/best.pt")
    parser.add_argument("--extra-ckpt", default="models/multihead_extra/best.pt")
    parser.add_argument("--include-dir", default="data/processed/landmarks/isl")
    parser.add_argument("--isl500-dir", default="data/processed/landmarks/isl500")
    parser.add_argument("--extra-dir", default="data/processed/landmarks/isl500_extra")
    parser.add_argument("--joint-map", default="data/processed/landmarks/joint_label_map.json")
    parser.add_argument("--extra-map", default="models/multihead_extra/label_map_extra.json")
    parser.add_argument("--output-dir", default="models/multihead_v2")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--floor-drop", type=float, default=3.0)
    args = parser.parse_args()

    from src.data.dataset import create_dataloader
    from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joint_map: dict[str, int] = json.loads(Path(args.joint_map).read_text())
    extra_map: dict[str, int] = json.loads(Path(args.extra_map).read_text())
    n_joint, n_extra = len(joint_map), len(extra_map)

    htm = HTMAConfig(input_dim=168, embedding_dim=256, num_blocks=2, num_heads=8,
                     dropout=0.1, num_classes=n_joint)
    model = MultiHeadHTMAClassifier(
        htm, sources=("include", "isl500", "extra"),
        head_classes={"include": n_joint, "isl500": n_joint, "extra": n_extra}).to(device)
    base = torch.load(args.base, map_location=device, weights_only=False)
    model.load_state_dict(base["model_state_dict"], strict=False)
    ex = torch.load(args.extra_ckpt, map_location=device, weights_only=False)
    extra_head = {k: v for k, v in ex["model_state_dict"].items() if k.startswith("heads.extra")}
    model.load_state_dict(extra_head, strict=False)
    logger.info(f"Warm-started encoder+old heads and extra head ({len(extra_head)} tensors)")

    def make_loader(d, split, lm, aug, norm=None):
        return create_dataloader(d, split, lm, 32, args.batch_size, 2, augment=aug,
                                 augmentation_mode="basic" if aug else "none",
                                 prefetch_factor=2, persistent_workers=False,
                                 normalize_fn=norm)
    train_loaders = {"include": make_loader(args.include_dir, "train", joint_map, True, canon),
                     "isl500": make_loader(args.isl500_dir, "train", joint_map, True, canon),
                     "extra": make_loader(args.extra_dir, "train", extra_map, True)}
    val_loaders = {"include": make_loader(args.include_dir, "val", joint_map, False, canon),
                   "isl500": make_loader(args.isl500_dir, "val", joint_map, False, canon),
                   "extra": make_loader(args.extra_dir, "val", extra_map, False)}
    logger.info("Train: " + str({k: len(v.dataset) for k, v in train_loaders.items()}))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        start_include = run_epoch(model, {"include": val_loaders["include"]},
                                  criterion, optimizer, device, False)["include"]["top1"]
    logger.info(f"Starting include-val: {start_include:.2f}% (floor: -{args.floor_drop}pp)")
    floor_hits = 0
    best, patience = 0.0, 0
    for epoch in range(args.epochs):
        for loader in train_loaders.values():
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(epoch)
        tr = run_epoch(model, train_loaders, criterion, optimizer, device, True)
        with torch.no_grad():
            va = run_epoch(model, val_loaders, criterion, optimizer, device, False)
        mean_val = sum(v["top1"] for v in va.values()) / len(va)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | val=" +
                    " ".join(f"{k}:{v['top1']:.1f}%" for k, v in va.items()) +
                    f" | mean={mean_val:.2f}%")
        if va["include"]["top1"] < start_include - args.floor_drop:
            floor_hits += 1
            logger.warning(f"Include-val floor breach ({floor_hits}x)")
            if floor_hits >= 3:
                logger.warning("Aborting Phase B: include-val drifting")
                break
        else:
            floor_hits = 0
        if mean_val > best:
            best, patience = mean_val, 0
            torch.save({"model_state_dict": model.state_dict(),
                        "sources": model.sources,
                        "head_classes": {"extra": n_extra}, "mean_val": mean_val}, out / "best.pt")
        else:
            patience += 1
        if patience >= args.patience:
            break
    logger.info(f"Done. Best mean val: {best:.2f}%")


if __name__ == "__main__":
    main()
