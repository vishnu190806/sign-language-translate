#!/usr/bin/env python
"""Train the HTMA model on ISL dataset."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_warm_start(model, checkpoint_path: str) -> tuple[int, int]:
    import torch

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    enc_state = {k: v for k, v in state.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(enc_state, strict=False)
    loaded = len(enc_state) - len(unexpected)
    skipped = len(missing)
    logger.info(f"Warm-start: loaded {loaded} encoder tensors, skipped {skipped} (head)")
    if loaded == 0:
        raise ValueError(f"No encoder weights loaded from {checkpoint_path}")
    return loaded, skipped


def main():
    parser = argparse.ArgumentParser(description="Train HTMA model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--npz-dir", default="data/processed/landmarks/isl")
    parser.add_argument("--label-map", default=None, help="Path to label map JSON")
    parser.add_argument("--warm-start", default=None,
                        help="Encoder checkpoint for DPC warm-start (head randomly initialized)")
    parser.add_argument("--output-subdir", default=None,
                        help="Write checkpoints to models/<subdir> instead of models/")
    args = parser.parse_args()

    from src.config import load_config
    from src.data.metadata import load_label_map
    from src.train.trainer import Trainer

    config = load_config(args.config, args.override)

    if args.label_map:
        label_map_path = Path(args.label_map)
    else:
        label_map_path = Path(args.npz_dir).parent / "isl_label_map.json"
    if not label_map_path.exists():
        logger.error(f"Label map not found at {label_map_path}. Run extract_landmarks.py first.")
        return

    label_map = load_label_map(label_map_path)
    logger.info(f"Loaded {len(label_map)} classes from {label_map_path}")

    trainer = Trainer(config, label_map, args.npz_dir)
    if args.output_subdir:
        from pathlib import Path as _P
        config.paths.checkpoints = str(_P(config.paths.checkpoints) / args.output_subdir)
        _P(config.paths.checkpoints).mkdir(parents=True, exist_ok=True)
    if args.warm_start:
        load_warm_start(trainer.model, args.warm_start)
    results = trainer.train()

    logger.info(f"Training complete. Best val accuracy: {results['best_val_acc']:.2f}%")


if __name__ == "__main__":
    main()
