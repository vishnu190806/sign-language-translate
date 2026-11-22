#!/usr/bin/env python
"""Evaluate a checkpoint on the OFFICIAL INCLUDE test split (HF metadata).

Downloads the tiny public parquet metadata (no auth), matches official test
video_paths to local landmark .npz files by video stem, runs inference, and
writes results/official_split_metrics.json. Answers whether the gap to
Khartheesvar 87.4% / Varanasi 95.88% is split-protocol or model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

PARQUET_URLS = {
    "test": "https://huggingface.co/api/datasets/ai4bharat/INCLUDE/parquet/default/test/0.parquet",
    "train": "https://huggingface.co/api/datasets/ai4bharat/INCLUDE/parquet/default/train/0.parquet",
}


def load_official_split(split: str = "test") -> "pd.DataFrame":
    import pandas as pd

    return pd.read_parquet(PARQUET_URLS[split])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/baseline.pt")
    parser.add_argument("--label-map", default="models/label_map.json")
    parser.add_argument("--npz-dir", default="data/processed/landmarks/isl")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="results/official_split_metrics.json")
    args = parser.parse_args()

    df = load_official_split(args.split)
    print(f"Official {args.split} rows: {len(df)}")

    label_map: dict[str, int] = json.loads(Path(args.label_map).read_text())
    print(f"Label map classes: {len(label_map)}, sample keys: {list(label_map)[:3]}")

    stem_to_npz: dict[str, Path] = {}
    for p in Path(args.npz_dir).glob("*.npz"):
        stem = p.name.rsplit("_", 1)[0]
        stem_to_npz.setdefault(stem, p)

    matched: list[tuple[Path, int]] = []
    missing = ambiguous = oov = 0
    seen: dict[str, str] = {}
    for _, row in df.iterrows():
        vp = str(row["video_path"])
        stem = Path(vp).stem
        lab = str(row["label"])
        if stem in seen and seen[stem] != lab:
            ambiguous += 1
            continue
        seen[stem] = lab
        npz = stem_to_npz.get(stem)
        if npz is None:
            missing += 1
            continue
        if lab not in label_map:
            oov += 1
            continue
        matched.append((npz, label_map[lab]))
    print(f"Matched: {len(matched)}, missing npz: {missing}, ambiguous dupes: {ambiguous}, "
          f"label OOV: {oov}")

    from src.config import load_config
    from src.models.builder import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = load_config("configs/default.yaml", "configs/isl.yaml")
    n_cls = ckpt.get("config", {}).get("model", {}).get("num_classes", len(label_map))
    config.model.num_classes = n_cls
    model = build_model(config.model).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    def prep(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = feats.shape[0]
        if T < args.seq_len:
            pad = np.zeros((args.seq_len - T, feats.shape[1]), np.float32)
            return np.concatenate([feats, pad]), np.concatenate(
                [np.ones(T, bool), np.zeros(args.seq_len - T, bool)])
        idx = np.linspace(0, T - 1, args.seq_len).astype(int)
        return feats[idx], np.ones(args.seq_len, bool)

    top1 = top5 = 0
    with torch.no_grad():
        for s in range(0, len(matched), args.batch_size):
            chunk = matched[s:s + args.batch_size]
            xs, ms, ys = [], [], []
            for p, y in chunk:
                f = np.load(p)["features"].astype(np.float32)
                x, m = prep(f)
                xs.append(x)
                ms.append(m)
                ys.append(y)
            xb = torch.from_numpy(np.stack(xs)).to(device)
            mb = torch.from_numpy(np.stack(ms)).to(device)
            logits = model(xb, mb)
            pred = logits.argmax(1).cpu().numpy()
            topk = logits.topk(5, dim=1).indices.cpu().numpy()
            y = np.array(ys)
            top1 += (pred == y).sum()
            top5 += sum(y[i] in topk[i] for i in range(len(y)))

    n = len(matched)
    out = {
        "checkpoint": args.checkpoint,
        "official_split": args.split,
        "n_official_rows": len(df),
        "n_matched": n,
        "n_missing_npz": missing,
        "n_ambiguous_dupes": ambiguous,
        "n_label_oov": oov,
        "top1": round(100 * top1 / n, 2),
        "top5": round(100 * top5 / n, 2),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
