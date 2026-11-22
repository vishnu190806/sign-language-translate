#!/usr/bin/env python
"""Knowledge distillation: train a smaller student model from a teacher."""
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
    parser = argparse.ArgumentParser(description="Knowledge distillation")
    parser.add_argument("--teacher-checkpoint", required=True, help="Path to teacher .pt")
    parser.add_argument("--dataset-dir", default="data/processed/landmarks/isl")
    parser.add_argument("--output-dir", default="models/student")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--label-map", default=None)
    args = parser.parse_args()

    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    from src.config import load_config
    from src.data.metadata import load_label_map
    from src.data.dataset import create_dataloader
    from src.models.htma import HTMAClassifier, HTMAConfig
    from src.train.distiller import Distiller

    config_data = {}
    if args.config:
        with open(args.config) as f:
            import yaml
            config_data = yaml.safe_load(f) or {}

    distill_cfg = config_data.get("distillation", {})
    teacher_cfg = config_data.get("teacher", {})
    student_cfg = config_data.get("student", {})

    temperature = args.temperature or distill_cfg.get("temperature", 4.0)
    alpha = args.alpha or distill_cfg.get("alpha", 0.7)
    feature_weight = distill_cfg.get("feature_weight", 0.0)
    epochs = args.epochs or distill_cfg.get("epochs", 50)
    lr = args.lr or distill_cfg.get("learning_rate", 0.0003)
    weight_decay = distill_cfg.get("weight_decay", 0.01)
    use_amp = distill_cfg.get("use_amp", True)
    seed = distill_cfg.get("seed", 42)

    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ckpt = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    if "config" in ckpt and "model" in ckpt["config"]:
        t_cfg = ckpt["config"]["model"]
    else:
        t_cfg = {"input_dim": 168, "embedding_dim": 256, "htma_blocks": 2, "num_heads": 8, "dropout": 0.1, "num_classes": 263}

    teacher_config = HTMAConfig(
        input_dim=t_cfg.get("input_dim", 168),
        embedding_dim=t_cfg.get("embedding_dim", 256),
        num_blocks=t_cfg.get("htma_blocks", 2),
        num_heads=t_cfg.get("num_heads", 8),
        dropout=t_cfg.get("dropout", 0.1),
        num_classes=t_cfg.get("num_classes", 263),
    )
    teacher = HTMAClassifier(teacher_config)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher = teacher.to(device)

    student_config = HTMAConfig(
        input_dim=student_cfg.get("input_dim", t_cfg.get("input_dim", 168)),
        embedding_dim=student_cfg.get("embedding_dim", 128),
        num_blocks=student_cfg.get("htma_blocks", 1),
        num_heads=student_cfg.get("num_heads", 8),
        dropout=student_cfg.get("dropout", 0.1),
        num_classes=student_cfg.get("num_classes", 263),
    )
    student = HTMAClassifier(student_config).to(device)

    logger.info(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    logger.info(f"Student params: {sum(p.numel() for p in student.parameters()):,}")

    if args.label_map:
        label_map_path = Path(args.label_map)
    else:
        label_map_path = Path(args.dataset_dir).parent / "isl_label_map.json"
    label_map = load_label_map(label_map_path)
    logger.info(f"Loaded {len(label_map)} classes")

    seq_len = config_data.get("processing", {}).get("sequence_length", 32) if config_data.get("processing") else 32
    batch_size = distill_cfg.get("batch_size", 64)

    train_loader = create_dataloader(
        args.dataset_dir, "train", label_map, seq_len, batch_size,
        augment=True, augmentation_mode="basic",
    )
    val_loader = create_dataloader(
        args.dataset_dir, "val", label_map, seq_len, batch_size,
        augment=False,
    )

    distiller = Distiller(
        teacher=teacher,
        student=student,
        temperature=temperature,
        alpha=alpha,
        feature_weight=feature_weight,
    )

    trainable_params = list(student.parameters())
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    warmup_epochs = max(1, int(epochs * 0.1))
    cosine_epochs = epochs - warmup_epochs
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(cosine_epochs, 1), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0
    patience = distill_cfg.get("early_stopping_patience", 15)

    for epoch in range(epochs):
        train_metrics = distiller.train_epoch(train_loader, optimizer, device)
        val_metrics = distiller.validate(val_loader, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train KD: {train_metrics['kd_loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            distiller.export_student(str(output_dir / "best_student"))
            torch.save({
                "epoch": epoch,
                "student_state_dict": student.state_dict(),
                "metrics": val_metrics,
                "config": {
                    "teacher": {
                        "input_dim": teacher_config.input_dim,
                        "embedding_dim": teacher_config.embedding_dim,
                        "num_blocks": teacher_config.num_blocks,
                        "num_heads": teacher_config.num_heads,
                        "num_classes": teacher_config.num_classes,
                    },
                    "student": {
                        "input_dim": student_config.input_dim,
                        "embedding_dim": student_config.embedding_dim,
                        "num_blocks": student_config.num_blocks,
                        "num_heads": student_config.num_heads,
                        "num_classes": student_config.num_classes,
                    },
                    "temperature": temperature,
                    "alpha": alpha,
                },
            }, output_dir / "best_student.pt")
            logger.info(f"New best student saved: {val_metrics['accuracy']:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
