from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.config import Config
from src.models.builder import build_model
from src.data.dataset import create_dataloader
from src.train.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: Config, label_map: dict[str, int], npz_dir: str):
        self.config = config
        self.label_map = label_map
        self.npz_dir = npz_dir

        self._apply_seed(config.train.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model = build_model(config.model).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.train.label_smoothing
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

        warmup_epochs = max(1, int(config.train.epochs * 0.1))
        cosine_epochs = max(1, config.train.epochs - warmup_epochs)
        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(self.optimizer, T_max=cosine_epochs, eta_min=1e-6)
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_epochs])

        self.scaler = torch.amp.GradScaler(enabled=False)
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.global_step = 0
        self._amp_device = self.device.type if self.device.type in ("cuda", "cpu") else "cuda"
        try:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def _apply_seed(self, seed: int) -> None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed applied: {seed}")

    def _load_oversample(self) -> dict[int, int] | None:
        import json

        path = self.config.train.oversample_targets
        if not path:
            return None
        try:
            targets = json.loads(Path(path).read_text())
            return {t["idx"]: targets["repeat"] for t in targets["targets"]}
        except Exception as e:
            logger.warning(f"Ignoring oversample_targets {path}: {e}")
            return None

    def train(self) -> dict:
        aug_mode = self.config.augmentation.mode
        aug_probs = self.config.augmentation.probs or None
        oversample = self._load_oversample()
        train_loader = create_dataloader(
            self.npz_dir, "train", self.label_map,
            self.config.processing.sequence_length,
            self.config.train.batch_size,
            self.config.dataset.num_workers,
            augment=True,
            augmentation_mode=aug_mode,
            augmentation_probs=aug_probs,
            balanced=self.config.train.balanced_sampler,
            seed=self.config.train.seed,
            oversample_labels=oversample,
        )
        val_loader = create_dataloader(
            self.npz_dir, "val", self.label_map,
            self.config.processing.sequence_length,
            self.config.train.batch_size,
            self.config.dataset.num_workers,
            augment=False,
        )

        for epoch in range(self.config.train.epochs):
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                try:
                    train_loader.sampler.set_epoch(epoch)
                except Exception:
                    pass
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._validate(val_loader)
            self.scheduler.step()

            logger.info(
                f"Epoch {epoch+1}/{self.config.train.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['top1']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['top1']:.2f}%"
            )

            if val_metrics["top1"] > self.best_val_acc:
                self.best_val_acc = val_metrics["top1"]
                self.patience_counter = 0
                self._save_checkpoint("best.pt", epoch, val_metrics)
                logger.info(f"New best model saved: {val_metrics['top1']:.2f}%")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.train.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return {"best_val_acc": self.best_val_acc}

    def _train_epoch(self, loader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (features, masks, labels) in enumerate(loader):
            features = features.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=self._amp_device,
                dtype=torch.bfloat16,
                enabled=self.config.train.use_amp and self.device.type == "cuda",
            ):
                logits = self.model(features, masks)
                loss = self.criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * features.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += features.size(0)
            self.global_step += 1

        return {
            "loss": total_loss / max(total, 1),
            "top1": 100.0 * correct / max(total, 1),
        }

    @torch.no_grad()
    def _validate(self, loader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        total = 0

        for features, masks, labels in loader:
            features = features.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type=self._amp_device,
                dtype=torch.bfloat16,
                enabled=self.config.train.use_amp and self.device.type == "cuda",
            ):
                logits = self.model(features, masks)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * features.size(0)
            all_preds.append(logits.cpu())
            all_labels.append(labels.cpu())
            total += features.size(0)

        all_preds = torch.cat(all_preds) if all_preds else torch.zeros(0)
        all_labels = torch.cat(all_labels) if all_labels else torch.zeros(0, dtype=torch.long)
        if len(all_labels) == 0:
            return {"loss": 0.0, "top1": 0.0, "top5": 0.0, "macro": 0.0, "n": 0}
        metrics = compute_metrics(all_preds, all_labels)
        metrics["loss"] = total_loss / max(total, 1)
        return metrics

    def _save_checkpoint(self, filename: str, epoch: int, metrics: dict) -> None:
        import json
        from pathlib import Path

        ckpt_dir = Path(self.config.paths.checkpoints)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": {
                "model": self.config.model.__dict__,
                "processing": {
                    "sequence_length": self.config.processing.sequence_length,
                },
            },
        }, ckpt_dir / filename)

        with open(ckpt_dir / "label_map.json", "w") as f:
            json.dump(self.label_map, f, indent=2)
