#!/usr/bin/env python
"""Train a DSTAdapter on top of a frozen HTMA base model."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train DSTAdapter")
    parser.add_argument("--base-checkpoint", required=True, help="Path to base model .pt")
    parser.add_argument("--dataset-dir", default="data/processed/landmarks/isl")
    parser.add_argument("--output-dir", default="models/adapter")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--adapter-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--label-map", default=None)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    from src.config import load_config
    from src.data.metadata import load_label_map
    from src.data.dataset import create_dataloader
    from src.models.htma import HTMAClassifier, HTMAConfig
    from src.models.dst_adapter import DSTAdapter

    config_data = {}
    if args.config:
        with open(args.config) as f:
            import yaml
            config_data = yaml.safe_load(f) or {}

    adapter_cfg = config_data.get("adapter", {})

    adapter_dim = args.adapter_dim or adapter_cfg.get("adapter_dim", 64)
    epochs = args.epochs or adapter_cfg.get("epochs", 30)
    lr = args.lr or adapter_cfg.get("learning_rate", 0.001)
    weight_decay = adapter_cfg.get("weight_decay", 0.01)
    warmup_epochs = adapter_cfg.get("warmup_epochs", 3)
    use_amp = adapter_cfg.get("use_amp", True)
    seed = adapter_cfg.get("seed", 42)

    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    if "config" in ckpt and "model" in ckpt["config"]:
        model_cfg_dict = ckpt["config"]["model"]
    else:
        model_cfg_dict = {"input_dim": 168, "embedding_dim": 256, "htma_blocks": 2, "num_heads": 8, "dropout": 0.1, "num_classes": 263}

    htm_config = HTMAConfig(
        input_dim=model_cfg_dict.get("input_dim", 168),
        embedding_dim=model_cfg_dict.get("embedding_dim", 256),
        num_blocks=model_cfg_dict.get("htma_blocks", 2),
        num_heads=model_cfg_dict.get("num_heads", 8),
        dropout=model_cfg_dict.get("dropout", 0.1),
        num_classes=model_cfg_dict.get("num_classes", 263),
    )
    base_model = HTMAClassifier(htm_config)
    base_model.load_state_dict(ckpt["model_state_dict"])

    model = DSTAdapter(base_model, adapter_dim=adapter_dim).to(device)

    total, trainable = model.count_parameters()
    logger.info(f"Parameters: total={total:,}, trainable={trainable:,} ({100*trainable/total:.1f}%)")

    if args.label_map:
        label_map_path = Path(args.label_map)
    else:
        label_map_path = Path(args.dataset_dir).parent / "isl_label_map.json"
    label_map = load_label_map(label_map_path)
    logger.info(f"Loaded {len(label_map)} classes")

    seq_len = config_data.get("processing", {}).get("sequence_length", 32)
    batch_size = adapter_cfg.get("batch_size", 64)

    train_loader = create_dataloader(
        args.dataset_dir, "train", label_map, seq_len, batch_size,
        augment=True, augmentation_mode="basic",
    )
    val_loader = create_dataloader(
        args.dataset_dir, "val", label_map, seq_len, batch_size,
        augment=False,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=adapter_cfg.get("label_smoothing", 0.1))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    cosine_epochs = epochs - warmup_epochs
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(cosine_epochs, 1), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler(enabled=False)
    best_val_acc = 0.0
    patience_counter = 0
    patience = adapter_cfg.get("early_stopping_patience", 15)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, masks, labels in train_loader:
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = model(features, masks)
                loss = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += features.size(0)

        scheduler.step()
        train_loss = total_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, masks, labels in val_loader:
                features = features.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                logits = model(features, masks)
                loss = criterion(logits, labels)
                val_loss += loss.item() * features.size(0)
                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += features.size(0)

        val_loss = val_loss / max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "adapter_state_dict": {k: v for k, v in model.state_dict().items() if not k.startswith("model.")},
                "full_state_dict": model.state_dict(),
                "adapter_dim": adapter_dim,
                "metrics": {"val_acc": val_acc, "val_loss": val_loss},
            }, output_dir / "best_adapter.pt")
            logger.info(f"New best adapter saved: {val_acc:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
