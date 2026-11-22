#!/usr/bin/env python
"""Ablation study: measure impact of each M8 improvement.

Evaluates six conditions on the same test set:
  baseline, +augmentation, +TTA, +adapters, +distillation, +all

Each condition reports top-1, top-3, top-5 accuracy.
Results are printed as a table and saved as JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_CONDITIONS = ["baseline", "+augmentation", "+TTA", "+adapters", "+distillation", "+all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation study for M8 improvements")
    parser.add_argument("--model-path", required=True, help="Path to base model checkpoint")
    parser.add_argument("--dataset-dir", required=True, help="Directory with .npz test files")
    parser.add_argument("--output-dir", default="results/ablation", help="Output directory")
    parser.add_argument("--config", default=None, help="YAML config path or JSON ablation config")
    parser.add_argument("--label-map", default=None, help="Path to label map JSON")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--conditions", nargs="+", default=None,
                        choices=ALL_CONDITIONS, help="Subset of conditions to run")
    parser.add_argument("--adapter-checkpoint", default=None,
                        help="Path to adapter checkpoint (for +adapters and +all)")
    parser.add_argument("--distilled-checkpoint", default=None,
                        help="Path to distilled student checkpoint (for +distillation)")
    parser.add_argument("--augmented-checkpoint", default=None,
                        help="Path to augmentation-trained checkpoint (for +augmentation)")
    parser.add_argument("--tta-num-augments", type=int, default=5,
                        help="Number of TTA augmentation passes")
    parser.add_argument("--dry-run", action="store_true", help="Print config without running")
    return parser.parse_args()


def load_label_map(path: str | Path) -> dict[str, int]:
    with open(path) as f:
        return json.load(f)


def load_base_model(model_path: str | Path, config, device):
    import torch
    from src.models.builder import build_model

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    ckpt_model = ckpt.get("config", {}).get("model", {})
    if ckpt_model.get("num_classes"):
        config.model.num_classes = ckpt_model["num_classes"]

    model = build_model(config.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_adapter_model(model_path: str | Path, adapter_path: str | Path, config, device):
    import torch
    from src.models.builder import build_model
    from src.models.dst_adapter import DSTAdapter

    base_model = load_base_model(model_path, config, device)

    adapter_ckpt = torch.load(adapter_path, map_location=device, weights_only=False)
    adapter_dim = adapter_ckpt.get("adapter_dim", 64)
    adapter_model = DSTAdapter(base_model, adapter_dim=adapter_dim).to(device)
    state_key = "model_state_dict" if "model_state_dict" in adapter_ckpt else "full_state_dict"
    adapter_model.load_state_dict(adapter_ckpt[state_key])
    adapter_model.eval()
    return adapter_model


def load_distilled_model(distilled_path: str | Path, config, device):
    import torch
    from src.models.builder import build_model

    ckpt = torch.load(distilled_path, map_location=device, weights_only=False)
    model_config = ckpt.get("config", {})
    student_cfg = model_config.get("student", model_config)

    orig_emb = config.model.embedding_dim
    orig_blocks = config.model.htma_blocks
    orig_nc = config.model.num_classes

    if student_cfg.get("num_classes"):
        config.model.num_classes = student_cfg["num_classes"]
    if student_cfg.get("embedding_dim"):
        config.model.embedding_dim = student_cfg["embedding_dim"]
    if student_cfg.get("num_blocks"):
        config.model.htma_blocks = student_cfg["num_blocks"]

    model = build_model(config.model).to(device)
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "student_state_dict"
    model.load_state_dict(ckpt[state_key])

    config.model.embedding_dim = orig_emb
    config.model.htma_blocks = orig_blocks
    config.model.num_classes = orig_nc

    model.eval()
    return model


def evaluate_condition(model, dataloader, device, tta_augment=False, tta_passes=5):
    import numpy as np
    import torch
    from src.features.augment_pipeline import AugmentationPipeline

    all_logits = []
    all_labels = []

    augmentor = None
    if tta_augment:
        augmentor = AugmentationPipeline(seed=42)

    with torch.no_grad():
        for features, masks, labels in dataloader:
            features = features.to(device)
            masks = masks.to(device)

            if tta_augment and augmentor is not None:
                logits_list = []
                rng = np.random.default_rng(42)
                for _ in range(tta_passes):
                    aug_features = features.cpu().numpy().copy()
                    for b in range(aug_features.shape[0]):
                        aug_features[b] = augmentor(aug_features[b], mode="train")
                    aug_tensor = torch.from_numpy(aug_features).to(device)
                    logits_list.append(model(aug_tensor, masks))
                logits = torch.stack(logits_list).mean(dim=0)
            else:
                logits = model(features, masks)

            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return compute_accuracy_metrics(logits, labels)


def compute_accuracy_metrics(logits, labels):
    import torch
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        result = {}

        for k in (1, 3, 5):
            if k <= probs.size(1):
                _, topk_pred = probs.topk(k, dim=1)
                correct = (topk_pred == labels.unsqueeze(1)).any(dim=1)
                result[f"top{k}"] = round(100.0 * correct.float().mean().item(), 2)
            else:
                result[f"top{k}"] = 0.0

        result["total_samples"] = int(labels.shape[0])
        return result


def create_label_map_from_npz_dir(npz_dir):
    import json
    import numpy as np

    labels = set()
    npz_path = Path(npz_dir)
    for npz_file in sorted(npz_path.glob("*.npz")):
        data = np.load(npz_file, allow_pickle=True)
        label = str(data.get("label", ""))
        if label:
            labels.add(label)

    label_map = {label: idx for idx, label in enumerate(sorted(labels))}
    return label_map


def print_ablation_table(results: dict[str, dict]) -> str:
    header = f"{'Condition':<20} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'Samples':>8}"
    separator = "-" * len(header)
    lines = [header, separator]

    baseline_top1 = results.get("baseline", {}).get("top1", 0.0)

    for cond in ALL_CONDITIONS:
        if cond not in results:
            continue
        r = results[cond]
        delta = r["top1"] - baseline_top1 if cond != "baseline" else 0.0
        delta_str = f" ({delta:+.1f})" if cond != "baseline" else ""
        lines.append(
            f"{cond:<20} {r['top1']:>7.2f}% {r['top3']:>7.2f}% {r['top5']:>7.2f}% {r['total_samples']:>8}{delta_str}"
        )

    table = "\n".join(lines)
    print(table)
    return table


def main():
    args = parse_args()

    import torch
    from src.config import load_config
    from src.data.dataset import create_dataloader

    config = load_config(args.config) if args.config else load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    npz_dir = Path(args.dataset_dir)

    conditions = args.conditions if args.conditions else ALL_CONDITIONS

    if args.dry_run:
        print(json.dumps({
            "model_path": args.model_path,
            "dataset_dir": str(npz_dir),
            "output_dir": args.output_dir,
            "split": args.split,
            "conditions": conditions,
            "device": str(device),
            "batch_size": args.batch_size,
            "tta_passes": args.tta_num_augments,
            "adapter_checkpoint": args.adapter_checkpoint,
            "distilled_checkpoint": args.distilled_checkpoint,
            "augmented_checkpoint": args.augmented_checkpoint,
        }, indent=2))
        return

    if not npz_dir.exists():
        logger.error(f"Dataset directory not found: {npz_dir}")
        sys.exit(1)

    if args.label_map:
        label_map = load_label_map(args.label_map)
    else:
        label_map_path = npz_dir.parent / "isl_label_map.json"
        if label_map_path.exists():
            label_map = load_label_map(label_map_path)
        else:
            logger.info("No label map found; building from .npz files")
            label_map = create_label_map_from_npz_dir(npz_dir)

    num_classes = len(label_map)
    config.model.num_classes = num_classes
    logger.info(f"Classes: {num_classes}")

    logger.info(f"Running conditions: {conditions}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = load_base_model(args.model_path, config, device)
    logger.info(f"Base model loaded: {sum(p.numel() for p in base_model.parameters()):,} params")

    results = {}
    t_start = time.time()

    for condition in conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running condition: {condition}")
        logger.info(f"{'='*60}")

        t_cond = time.time()

        if condition == "baseline":
            model = base_model

        elif condition == "+augmentation":
            aug_path = args.augmented_checkpoint
            if aug_path and Path(aug_path).exists():
                model = load_base_model(aug_path, config, device)
                logger.info(f"Loaded augmentation-trained model from {aug_path}")
            else:
                logger.warning("No augmentation checkpoint found; using base model with augmentation pipeline")
                model = base_model

        elif condition == "+TTA":
            model = base_model

        elif condition == "+adapters":
            adapter_path = args.adapter_checkpoint
            if adapter_path and Path(adapter_path).exists():
                model = load_adapter_model(args.model_path, adapter_path, config, device)
                logger.info(f"Loaded adapter model from {adapter_path}")
            else:
                logger.warning("No adapter checkpoint found; skipping condition")
                continue

        elif condition == "+distillation":
            dist_path = args.distilled_checkpoint
            if dist_path and Path(dist_path).exists():
                model = load_distilled_model(dist_path, config, device)
                logger.info(f"Loaded distilled model from {dist_path}")
            else:
                logger.warning("No distilled checkpoint found; skipping condition")
                continue

        elif condition == "+all":
            parts = []
            if args.adapter_checkpoint and Path(args.adapter_checkpoint).exists():
                model = load_adapter_model(args.model_path, args.adapter_checkpoint, config, device)
                parts.append("adapter")
            elif args.distilled_checkpoint and Path(args.distilled_checkpoint).exists():
                model = load_distilled_model(args.distilled_checkpoint, config, device)
                parts.append("distilled")
            else:
                model = base_model

            if args.augmented_checkpoint and Path(args.augmented_checkpoint).exists():
                model = load_base_model(args.augmented_checkpoint, config, device)
                parts.append("augmented")

            logger.info(f"+all combined: {', '.join(parts) if parts else 'base model'}")

        else:
            logger.warning(f"Unknown condition: {condition}; skipping")
            continue

        use_tta = condition in ("+TTA", "+all")
        tta_passes = args.tta_num_augments if use_tta else 0

        dataloader = create_dataloader(
            npz_dir=str(npz_dir),
            split=args.split,
            label_map=label_map,
            sequence_length=config.processing.sequence_length,
            batch_size=args.batch_size,
            num_workers=config.dataset.num_workers,
            augment=False,
        )

        metrics = evaluate_condition(
            model=model,
            dataloader=dataloader,
            device=device,
            tta_augment=use_tta,
            tta_passes=tta_passes,
        )

        elapsed = time.time() - t_cond
        results[condition] = {**metrics, "elapsed_seconds": round(elapsed, 1)}

        logger.info(
            f"  Top-1: {metrics['top1']:.2f}%  Top-3: {metrics['top3']:.2f}%  "
            f"Top-5: {metrics['top5']:.2f}%  ({elapsed:.1f}s)"
        )

    total_elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*60}")
    table_text = print_ablation_table(results)

    baseline_top1 = results.get("baseline", {}).get("top1", 0.0)
    improvements = {}
    for cond in results:
        if cond != "baseline":
            improvements[cond] = round(results[cond]["top1"] - baseline_top1, 2)

    report = {
        "metadata": {
            "model_path": args.model_path,
            "dataset_dir": str(npz_dir),
            "split": args.split,
            "num_classes": num_classes,
            "total_samples": results.get("baseline", {}).get("total_samples", 0),
            "device": str(device),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "total_elapsed_seconds": round(total_elapsed, 1),
            "tta_passes": args.tta_num_augments,
        },
        "conditions_run": list(results.keys()),
        "results": results,
        "improvements_over_baseline": improvements,
        "table": table_text,
    }

    output_path = output_dir / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    report_path = output_dir / "ablation_report.md"
    md_lines = [
        "# Ablation Study Report",
        "",
        f"- **Model**: `{args.model_path}`",
        f"- **Dataset**: `{npz_dir}`",
        f"- **Split**: {args.split}",
        f"- **Classes**: {num_classes}",
        f"- **Device**: {device}",
        f"- **TTA passes**: {args.tta_num_augments}",
        "",
        "## Results",
        "",
        "```",
        table_text,
        "```",
        "",
        "## Improvements Over Baseline",
        "",
        "| Condition | Top-1 Delta |",
        "|-----------|------------|",
    ]
    for cond, delta in improvements.items():
        md_lines.append(f"| {cond} | {delta:+.2f}% |")

    md_lines.extend([
        "",
        "## Condition Descriptions",
        "",
        "- **baseline**: Original HTMA model, no modifications",
        "- **+augmentation**: Model trained with AugmentationPipeline",
        "- **+TTA**: Test-time augmentation with AugmentationPipeline at inference",
        "- **+adapters**: DSTAdapter (spatial + temporal) wrapped around base model",
        "- **+distillation**: Knowledge-distilled student model",
        "- **+all**: All improvements combined",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
