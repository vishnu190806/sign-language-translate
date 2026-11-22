import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

CKPT = sys.argv[1] if len(sys.argv) > 1 else "models/ctc_v2/best_ctc.pt"

from src.data.csltr_dataset import CSLTRDataset
from src.models.ctc_head import HTMACTCModel

v = json.load(open("models/ctc/gloss_vocab.json"))
iv = {i: k for k, i in v.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = HTMACTCModel("models/multihead/best.pt", len(v) - 1, device=device).to(device).eval()
m.head.load_state_dict(torch.load(CKPT, map_location=device)["head_state_dict"])

for split in ("train", "val"):
    ds = CSLTRDataset("data/processed/isl-csltr/landmarks", split, v)
    print(f"--- {split} ({len(ds)} files) ---")
    for idx in [0, 1, 2]:
        f, ids = ds[idx]
        with torch.no_grad():
            lp = m(f[None].to(device), torch.ones(1, f.shape[0], dtype=torch.bool).to(device))
        pred = lp.argmax(-1)[0].tolist()
        hyp, prev = [], None
        for t in pred:
            if t != 0 and t != prev:
                hyp.append(iv[t])
            prev = t
        ref = [iv[i] for i in ids.tolist()]
        print("ref:", ref)
        print("hyp:", hyp)
