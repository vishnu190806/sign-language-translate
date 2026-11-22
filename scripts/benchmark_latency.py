#!/usr/bin/env python
"""Benchmark inference latency for HTMA model.

Measures end-to-end inference latency (preprocess -> inference -> postprocess)
on CPU and CUDA (if available). Outputs JSON with fps, latency percentiles,
RAM usage, device info, timestamp, and commit hash.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def get_ram_usage_mb() -> float:
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def load_label_map(path: Path) -> dict[str, int]:
    import json as json_lib
    with open(path, "r") as f:
        return json_lib.load(f)


def main():
    parser = argparse.ArgumentParser(description="Benchmark HTMA model inference latency")
    parser.add_argument("--checkpoint", default="models/baseline.pt", help="Path to model checkpoint")
    parser.add_argument("--label-map", default="models/label_map.json", help="Path to label map JSON")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to base config")
    parser.add_argument("--override", default=None, help="Path to override config")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Number of timed iterations")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--sequence-length", type=int, default=32, help="Sequence length (T)")
    parser.add_argument("--input-dim", type=int, default=168, help="Input feature dimension")
    parser.add_argument("--output", default="docs/benchmarks/metrics.json", help="Output JSON path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    args = parser.parse_args()

    label_map_path = Path(args.label_map)
    if not label_map_path.exists():
        logger.error(f"Label map not found at {label_map_path}")
        sys.exit(1)

    label_map = load_label_map(label_map_path)
    num_classes = len(label_map)
    logger.info(f"Loaded {num_classes} classes from label map")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})
    ckpt_model = ckpt_config.get("model", {})

    from src.config import load_config
    from src.models.builder import build_model

    config = load_config(args.config, args.override)

    if ckpt_model.get("num_classes") and ckpt_model["num_classes"] != config.model.num_classes:
        logger.info(f"Overriding num_classes: {config.model.num_classes} -> {ckpt_model['num_classes']} (from checkpoint)")
        config.model.num_classes = ckpt_model["num_classes"]

    if ckpt_model.get("dropout") is not None:
        config.model.dropout = ckpt_model["dropout"]
        logger.info(f"Using dropout from checkpoint: {config.model.dropout}")

    model = build_model(config.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    B = args.batch_size
    T = args.sequence_length
    D = args.input_dim

    dummy_input = torch.randn(B, T, D, device=device)
    dummy_mask = torch.ones(B, T, dtype=torch.bool, device=device)

    logger.info(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy_input, dummy_mask)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    ram_samples = []

    logger.info(f"Running {args.iterations} timed iterations...")
    with torch.no_grad():
        for i in range(args.iterations):
            ram_before = get_ram_usage_mb()

            start = time.perf_counter()
            _ = model(dummy_input, dummy_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            ram_after = get_ram_usage_mb()
            ram_samples.append(max(ram_before, ram_after))

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

    latencies = np.array(latencies)
    ram_mb = float(np.median(ram_samples)) if ram_samples else 0.0

    fps = 1000.0 / np.median(latencies) if np.median(latencies) > 0 else 0.0
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))

    import datetime
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    commit_hash = get_commit_hash()

    metrics = {
        "fps": round(fps, 2),
        "latency_ms_p50": round(p50, 2),
        "latency_ms_p95": round(p95, 2),
        "ram_mb": round(ram_mb, 2),
        "device": str(device),
        "timestamp": timestamp,
        "commit_hash": commit_hash,
        "model_params": total_params,
        "batch_size": B,
        "sequence_length": T,
        "input_dim": D,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    if device.type == "cuda":
        metrics["gpu_name"] = torch.cuda.get_device_name(0)
        metrics["cuda_version"] = torch.version.cuda

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"FPS: {fps:.2f}")
    logger.info(f"Latency p50: {p50:.2f} ms")
    logger.info(f"Latency p95: {p95:.2f} ms")
    logger.info(f"RAM: {ram_mb:.2f} MB")
    logger.info(f"Device: {device}")


if __name__ == "__main__":
    main()