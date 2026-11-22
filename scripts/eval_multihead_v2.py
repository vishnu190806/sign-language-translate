#!/usr/bin/env python
"""Dual-gate eval for a 3-head multihead checkpoint (include head only).

Gates: custom test >= 81.8 AND official test >= 85.22.
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

SYN = {"alright": "okay", "thankyou": "thank you", "big large": "big",
       "small little": "small", "store or shop": "store",
       "race (ethnicity)": "race", "you (plural)": "you plural"}


def canon(lab: str) -> str:
    lab = lab.split(".")[-1].strip().lower() if "." in lab else lab.strip().lower()
    return SYN.get(lab, lab)


def prep(f: np.ndarray, seq_len: int = 32):
    T = f.shape[0]
    if T < seq_len:
        pad = np.zeros((seq_len - T, f.shape[1]), np.float32)
        return np.concatenate([f, pad])
    return f[np.linspace(0, T - 1, seq_len).astype(int)]


def run(model, pairs, device, batch_size=128, seq_len=32) -> tuple[float, float]:
    top1 = top5 = 0
    with torch.no_grad():
        for s in range(0, len(pairs), batch_size):
            chunk = pairs[s:s + batch_size]
            xb = torch.from_numpy(np.stack(
                [prep(np.load(p)["features"].astype(np.float32), seq_len) for p, _ in chunk])).to(device)
            mb = torch.ones(xb.shape[0], seq_len, dtype=torch.bool).to(device)
            logits = model(xb, mb, source="include")
            pred = logits.argmax(1).cpu().numpy()
            topk = logits.topk(5, dim=1).indices.cpu().numpy()
            y = np.array([y for _, y in chunk])
            top1 += (pred == y).sum()
            top5 += sum(y[i] in topk[i] for i in range(len(y)))
    n = len(pairs)
    return float(round(100 * top1 / n, 2)), float(round(100 * top5 / n, 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/multihead_v2/best.pt")
    parser.add_argument("--joint-map", default="data/processed/landmarks/joint_label_map.json")
    parser.add_argument("--extra-map", default="models/multihead_extra/label_map_extra.json")
    parser.add_argument("--npz-dir", default="data/processed/landmarks/isl")
    parser.add_argument("--output", default="results/multihead_v2_gates.json")
    args = parser.parse_args()

    from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joint_map: dict[str, int] = json.loads(Path(args.joint_map).read_text())
    extra_map: dict[str, int] = json.loads(Path(args.extra_map).read_text())
    n_joint, n_extra = len(joint_map), len(extra_map)
    htm = HTMAConfig(input_dim=168, embedding_dim=256, num_blocks=2, num_heads=8,
                     dropout=0.1, num_classes=n_joint)
    model = MultiHeadHTMAClassifier(
        htm, sources=("include", "isl500", "extra"),
        head_classes={"include": n_joint, "isl500": n_joint, "extra": n_extra}).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    custom = []
    for p in sorted(Path(args.npz_dir).glob("*.npz")):
        try:
            a = np.load(p, allow_pickle=True)
            if str(a.get("split", "")) != "test":
                continue
            lab = canon(str(a.get("label", "")))
            if lab in joint_map:
                custom.append((p, joint_map[lab]))
        except Exception:
            continue
    c1, c5 = run(model, custom, device)

    import pandas as pd
    df = pd.read_parquet(
        "https://huggingface.co/api/datasets/ai4bharat/INCLUDE/parquet/default/test/0.parquet")
    official = []
    for _, row in df.iterrows():
        stem = Path(str(row["video_path"])).stem
        lab = canon(str(row["label"]))
        if lab not in joint_map:
            continue
        matches = sorted(Path(args.npz_dir).glob(f"{stem}_*.npz"))
        if matches:
            official.append((matches[0], joint_map[lab]))
    o1, o5 = run(model, official, device)

    out = {"custom_top1": c1, "custom_top5": c5, "custom_n": len(custom),
           "official_top1": o1, "official_top5": o5, "official_n": len(official),
           "gate_custom_ge_81_8": c1 >= 81.8, "gate_official_ge_85_22": o1 >= 85.22,
           "ship": bool(c1 >= 81.8 and o1 >= 85.22)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
