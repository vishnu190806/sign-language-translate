import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_multihead_v2 import canon, prep, run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
buggy: dict[str, int] = json.loads(Path("models/label_map.json").read_text())
joint: dict[str, int] = json.loads(Path("data/processed/landmarks/joint_label_map.json").read_text())
extra: dict[str, int] = json.loads(Path("models/multihead_extra/label_map_extra.json").read_text())

from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier
htm = HTMAConfig(input_dim=168, embedding_dim=256, num_blocks=2, num_heads=8,
                 dropout=0.1, num_classes=len(joint))


def load_multi(ckpt_path):
    m = MultiHeadHTMAClassifier(
        htm, sources=("include", "isl500", "extra"),
        head_classes={"include": len(joint), "isl500": len(joint), "extra": len(extra)}).to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(ck["model_state_dict"], strict=False)
    return m


df = pd.read_parquet(
    "https://huggingface.co/api/datasets/ai4bharat/INCLUDE/parquet/default/test/0.parquet")
pairs_974, pairs_1008 = [], []
for _, row in df.iterrows():
    stem = Path(str(row["video_path"])).stem
    raw = str(row["label"])
    ms = sorted(Path("data/processed/landmarks/isl").glob(f"{stem}_*.npz"))
    if not ms:
        continue
    c = canon(raw)
    if raw in buggy or c in buggy:
        pairs_974.append((ms[0], joint[c]))
    if c in joint:
        pairs_1008.append((ms[0], joint[c]))
print(f"974-protocol pairs: {len(pairs_974)}, 1008-protocol pairs: {len(pairs_1008)}")

for name, ckpt in [("v1-deployed", "models/multihead/best.pt"),
                   ("v2-new", "models/multihead_v2/best.pt")]:
    m = load_multi(ckpt)
    a, _ = run(m, pairs_974, device)
    b, _ = run(m, pairs_1008, device)
    print(f"{name}: 974-protocol={a}% | 1008-protocol={b}%")
