#!/usr/bin/env python
"""Offline evaluation script for the HTMA model.

Runs model on held-out split, computes all metrics, saves reports.
"""
from __future__ import annotations

import argparse
import logging
import platform
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate HTMA model on test split")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--npz-dir", default="data/processed/landmarks/isl")
    parser.add_argument("--label-map", default=None, help="Path to label map JSON")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import json

    import numpy as np
    import torch
    from sklearn.metrics import classification_report

    from src.config import load_config
    from src.data.metadata import load_label_map
    from src.data.dataset import create_dataloader
    from src.models.builder import build_model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})
    ckpt_model = ckpt_config.get("model", {})

    if args.label_map:
        label_map_path = Path(args.label_map)
    else:
        label_map_path = Path(args.npz_dir).parent / "isl_label_map.json"
    if not label_map_path.exists():
        logger.error(f"Label map not found at {label_map_path}")
        sys.exit(1)

    label_map = load_label_map(label_map_path)
    num_classes = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}
    logger.info(f"Loaded {num_classes} classes from label map")

    config = load_config(args.config, args.override)
    if ckpt_model.get("num_classes") and ckpt_model["num_classes"] != config.model.num_classes:
        logger.info(f"Overriding num_classes: {config.model.num_classes} -> {ckpt_model['num_classes']} (from checkpoint)")
        config.model.num_classes = ckpt_model["num_classes"]

    model = build_model(config.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    dataloader = create_dataloader(
        npz_dir=args.npz_dir,
        split=args.split,
        label_map=label_map,
        sequence_length=config.processing.sequence_length,
        batch_size=args.batch_size,
        num_workers=config.dataset.num_workers,
        augment=False,
    )

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for features, masks, labels in dataloader:
            features = features.to(device)
            masks = masks.to(device)
            logits = model(features, masks)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits) if all_logits else torch.zeros(0)
    if logits.shape[0] == 0:
        logger.error("No samples evaluated (empty split or label-map mismatch)")
        sys.exit(1)
    labels = torch.cat(all_labels)

    from src.train.metrics import compute_metrics, confusion_matrix as cm_func

    metrics = compute_metrics(logits, labels, top_k=(1, 5))
    cm = cm_func(logits, labels, num_classes)

    logger.info(f"Top-1 Accuracy: {metrics['top1']:.2f}%")
    logger.info(f"Top-5 Accuracy: {metrics['top5']:.2f}%")
    logger.info(f"Macro Accuracy: {metrics['macro_accuracy']:.2f}%")

    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1).numpy()
    labels_np = labels.numpy()

    all_class_indices = list(range(num_classes))
    target_names = [idx_to_label.get(i, f"class_{i}") for i in all_class_indices]
    per_class_str = classification_report(
        labels_np, preds, labels=all_class_indices, target_names=target_names, zero_division=0
    )

    with open(output_dir / "per_class_metrics.csv", "w") as f:
        report = classification_report(labels_np, preds, labels=all_class_indices, target_names=target_names, zero_division=0, output_dict=True)
        f.write("class,precision,recall,f1-score,support\n")
        for class_name in target_names:
            if class_name in report:
                r = report[class_name]
                f.write(f"{class_name},{r['precision']:.4f},{r['recall']:.4f},{r['f1-score']:.4f},{r['support']:.0f}\n")

    np.savetxt(output_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10))
        short_names = [n[:12] for n in target_names]
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(target_names)),
            yticks=np.arange(len(target_names)),
            xticklabels=short_names,
            yticklabels=short_names,
            xlabel="Predicted",
            ylabel="True",
            title=f"Confusion Matrix ({args.split} split)",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
        plt.setp(ax.get_yticklabels(), fontsize=6)
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved confusion_matrix.png")
    except ImportError:
        logger.warning("matplotlib not available, skipping confusion matrix plot")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        per_class_acc = np.diag(cm) / cm.sum(axis=1).clip(min=1)
        sorted_idx = np.argsort(per_class_acc)
        sorted_names = [target_names[i][:12] for i in sorted_idx]
        sorted_accs = per_class_acc[sorted_idx]

        fig, ax = plt.subplots(figsize=(14, 6))
        colors = ["#e74c3c" if a < 0.8 else "#f39c12" if a < 0.9 else "#2ecc71" for a in sorted_accs]
        ax.barh(range(len(sorted_names)), sorted_accs, color=colors)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=6)
        ax.set_xlabel("Accuracy")
        ax.set_title(f"Accuracy by Class ({args.split} split)")
        ax.set_xlim(0, 1.05)
        fig.tight_layout()
        fig.savefig(output_dir / "accuracy_by_class.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved accuracy_by_class.png")
    except ImportError:
        logger.warning("matplotlib not available, skipping accuracy plot")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics_output = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_classes": num_classes,
        "total_samples": int(labels.shape[0]),
        "top1_accuracy": round(metrics["top1"], 2),
        "top5_accuracy": round(metrics["top5"], 2),
        "macro_accuracy": round(metrics["macro_accuracy"], 2),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device": str(device),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    logger.info(f"Saved metrics.json")

    report_lines = [
        f"# Evaluation Report — {args.split.upper()} Split",
        "",
        f"- **Checkpoint**: `{args.checkpoint}`",
        f"- **Split**: {args.split}",
        f"- **Samples**: {labels.shape[0]}",
        f"- **Classes**: {num_classes}",
        f"- **Device**: {device}",
        f"- **Platform**: {platform.platform()}",
        f"- **PyTorch**: {torch.__version__}",
        "",
        "## Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Top-1 Accuracy | {metrics['top1']:.2f}% |",
        f"| Top-5 Accuracy | {metrics['top5']:.2f}% |",
        f"| Macro Accuracy | {metrics['macro_accuracy']:.2f}% |",
        "",
        "## Model",
        "",
        f"- Total parameters: {total_params:,}",
        f"- Trainable parameters: {trainable_params:,}",
        "",
        "## Per-Class Report",
        "",
        "```",
        per_class_str,
        "```",
    ]

    with open(output_dir / "evaluation_report.md", "w") as f:
        f.write("\n".join(report_lines))
    logger.info(f"Saved evaluation_report.md")

    logger.info(f"All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
