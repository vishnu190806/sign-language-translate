from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from src.models.htma import HTMAClassifier


class FeatureDistillLoss(nn.Module):
    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        if teacher_dim != student_dim:
            self.proj = nn.Linear(student_dim, teacher_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        student_feat = self.proj(student_feat)
        return F.mse_loss(student_feat, teacher_feat.detach())


class Distiller:
    def __init__(
        self,
        teacher: HTMAClassifier,
        student: HTMAClassifier,
        temperature: float = 4.0,
        alpha: float = 0.7,
        feature_weight: float = 0.0,
    ):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = nn.KLDivLoss(reduction="batchmean")

        self.feature_weight = feature_weight
        self.feature_loss: Optional[FeatureDistillLoss] = None

        if feature_weight > 0:
            teacher_dim = teacher.config.embedding_dim
            student_dim = student.config.embedding_dim
            self.feature_loss = FeatureDistillLoss(teacher_dim, student_dim)

    def _compute_kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        return self.kd_loss(soft_student, soft_teacher) * (T * T)

    def _compute_combined_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        kd = self._compute_kd_loss(student_logits, teacher_logits)
        ce = self.ce_loss(student_logits, labels)
        loss = self.alpha * kd + (1.0 - self.alpha) * ce

        breakdown = {"kd_loss": kd.item(), "ce_loss": ce.item()}

        if self.feature_loss is not None and student_features is not None and teacher_features is not None:
            fl = self.feature_loss(student_features, teacher_features)
            loss = loss + self.feature_weight * fl
            breakdown["feature_loss"] = fl.item()

        breakdown["total_loss"] = loss.item()
        return loss, breakdown

    def _teacher_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher(x, mask)

    def train_epoch(self, dataloader, optimizer: torch.optim.Optimizer, device: torch.device) -> dict:
        self.student.train()
        self.teacher.eval()

        total_loss = 0.0
        total_kd = 0.0
        total_ce = 0.0
        correct = 0
        total = 0

        for features, masks, labels in dataloader:
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            teacher_logits = self._teacher_forward(features, masks)
            student_logits = self.student(features, masks)

            loss, breakdown = self._compute_combined_loss(
                student_logits, teacher_logits, labels
            )

            loss.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += breakdown["total_loss"] * features.size(0)
            total_kd += breakdown["kd_loss"] * features.size(0)
            total_ce += breakdown["ce_loss"] * features.size(0)
            pred = student_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += features.size(0)

        return {
            "loss": total_loss / max(total, 1),
            "kd_loss": total_kd / max(total, 1),
            "ce_loss": total_ce / max(total, 1),
            "accuracy": 100.0 * correct / max(total, 1),
        }

    @torch.no_grad()
    def validate(self, dataloader, device: torch.device) -> dict:
        self.student.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for features, masks, labels in dataloader:
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            logits = self.student(features, masks)
            loss = self.ce_loss(logits, labels)

            total_loss += loss.item() * features.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += features.size(0)

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": 100.0 * correct / max(total, 1),
        }

    def export_student(self, path: str) -> None:
        export_dir = Path(path)
        export_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.student.state_dict(), export_dir / "student_model.pt")

        config = {
            "input_dim": self.student.config.input_dim,
            "embedding_dim": self.student.config.embedding_dim,
            "num_blocks": self.student.config.num_blocks,
            "num_heads": self.student.config.num_heads,
            "num_classes": self.student.config.num_classes,
            "dropout": self.student.config.dropout,
            "max_seq_len": self.student.config.max_seq_len,
        }
        with open(export_dir / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)
