#!/usr/bin/env python
"""Confusion-driven retrain: champion recipe + oversampled augmented views of
low-recall/high-confusion classes (see results/confusion_targets.json).

Output isolated to models/confusion/ via configs/confusion.yaml — never
overwrites models/baseline.pt. Ship rule: promote only if custom-test top-1 >
81.8 AND official-split top-1 >= 85.22; otherwise document abort, keep champion.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Confusion-driven retrain")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default="configs/confusion.yaml")
    parser.add_argument("--npz-dir", default="data/processed/landmarks/isl")
    parser.add_argument("--label-map", default="data/processed/landmarks/isl_label_map.json")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    from src.config import load_config
    from src.data.metadata import load_label_map
    from src.train.trainer import Trainer

    label_map = load_label_map(args.label_map)
    logger.info(f"Loaded {len(label_map)} classes from {args.label_map}")

    config = load_config(args.config, args.override)
    if args.epochs is not None:
        config.train.epochs = args.epochs

    trainer = Trainer(config, label_map, args.npz_dir)
    results = trainer.train()
    logger.info(f"Confusion retrain complete. Best val: {results['best_val_acc']:.2f}%")


if __name__ == "__main__":
    main()
