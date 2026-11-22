#!/usr/bin/env python
"""Phase A: frozen-encoder extra head for ISL500-only words (zero-risk).

Loads models/multihead/best.pt, attaches a new `extra` head, freezes everything
else (encoder + old heads bit-identical => 263-class behavior CANNOT change),
trains only the new head. Output isolated to models/multihead_extra/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-A frozen extra head")
    parser.add_argument("--base", default="models/multihead/best.pt")
    parser.add_argument("--npz-dir", default="data/processed/landmarks/isl500_extra")
    parser.add_argument("--label-map", default="models/multihead_extra/label_map_extra.json")
    parser.add_argument("--output-dir", default="models/multihead_extra")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()

    from src.data.dataset import create_dataloader
    from src.data.metadata import load_label_map
    from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map: dict[str, int] = load_label_map(args.label_map)
    n_extra = len(label_map)
    logger.info(f"Extra classes: {n_extra}")

    base = torch.load(args.base, map_location=device, weights_only=False)
    n_joint = len(json.loads(Path(args.base).parent.joinpath("label_map.json").read_text()))
    htm = HTMAConfig(input_dim=168, embedding_dim=256, num_blocks=2, num_heads=8,
                     dropout=0.1, num_classes=n_joint)
    model = MultiHeadHTMAClassifier(
        htm, sources=("include", "isl500", "extra"),
        head_classes={"include": n_joint, "isl500": n_joint, "extra": n_extra},
    ).to(device)
    missing, unexpected = model.load_state_dict(base["model_state_dict"], strict=False)
    logger.info(f"Base weights loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("heads.extra")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params (extra head only): {n_trainable:,}")

    def make_loader(split, aug):
        return create_dataloader(args.npz_dir, split, label_map, 32, args.batch_size, 6,
                                 augment=aug, augmentation_mode="basic" if aug else "none")
    train_loader, val_loader = make_loader("train", True), make_loader("val", False)
    logger.info(f"Train {len(train_loader.dataset)} / val {len(val_loader.dataset)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=0.01)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    epochs = args.smoke or args.epochs
    best, patience = 0.0, 0
    for epoch in range(epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                loss = criterion(model(x, m, source="extra"), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        ok = tot = 0
        with torch.no_grad():
            for x, m, y in val_loader:
                pred = model(x.to(device), m.to(device), source="extra").argmax(1).cpu()
                ok += (pred == y).sum().item()
                tot += y.numel()
        acc = 100.0 * ok / max(tot, 1)
        logger.info(f"Epoch {epoch+1}/{epochs} | val top-1: {acc:.2f}%")
        if acc > best:
            best, patience = acc, 0
            torch.save({"model_state_dict": model.state_dict(),
                        "sources": model.sources,
                        "head_classes": {"extra": n_extra}, "best_val": best}, out / "best.pt")
        else:
            patience += 1
        if patience >= args.patience:
            break
    logger.info(f"Done. Best extra-val: {best:.2f}%")


if __name__ == "__main__":
    main()
