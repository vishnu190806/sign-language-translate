from __future__ import annotations

import numpy as np
import torch


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    top_k: tuple[int, ...] = (1, 5),
) -> dict:
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        metrics = {}

        for k in top_k:
            if k <= probs.size(1):
                _, topk_pred = probs.topk(k, dim=1)
                correct = (topk_pred == labels.unsqueeze(1)).any(dim=1)
                metrics[f"top{k}"] = 100.0 * correct.float().mean().item()
            else:
                metrics[f"top{k}"] = 0.0

        pred = probs.argmax(dim=1)
        correct_mask = pred == labels
        metrics["accuracy"] = 100.0 * correct_mask.float().mean().item()

        num_classes = probs.size(1)
        per_class_correct = torch.zeros(num_classes)
        per_class_total = torch.zeros(num_classes)
        for c in range(num_classes):
            class_mask = labels == c
            if class_mask.any():
                per_class_correct[c] = (pred[class_mask] == c).float().sum()
                per_class_total[c] = class_mask.float().sum()

        valid = per_class_total > 0
        if valid.any():
            metrics["macro_accuracy"] = 100.0 * (
                per_class_correct[valid] / per_class_total[valid]
            ).mean().item()
        else:
            metrics["macro_accuracy"] = 0.0

        return metrics


def confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(labels.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1
        return cm
