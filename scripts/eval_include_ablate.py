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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/joint/best.pt")
    parser.add_argument("--npz-dir", default="data/processed/landmarks/isl")
    parser.add_argument("--label-map", default="data/processed/landmarks/joint_label_map.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--zero-pose", action="store_true")
    parser.add_argument("--swap-hands", action="store_true")
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()

    from src.config import load_config
    from src.models.builder import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map: dict[str, int] = json.loads(Path(args.label_map).read_text())
    config = load_config("configs/default.yaml", "configs/joint.yaml")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config.model.num_classes = ckpt["config"]["model"]["num_classes"]
    model = build_model(config.model).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    files = []
    for p in sorted(Path(args.npz_dir).glob("*.npz")):
        try:
            d = np.load(p, allow_pickle=True)
            if str(d.get("split", "")) != args.split:
                continue
            if canon(str(d.get("label", ""))) not in label_map:
                continue
        except Exception:
            continue
        files.append(p)
    print(f"files: {len(files)} zero_pose={args.zero_pose}")

    top1 = top5 = total = 0
    with torch.no_grad():
        for i in range(0, len(files), args.batch):
            X, M, y = [], [], []
            for p in files[i:i + args.batch]:
                d = np.load(p, allow_pickle=True)
                f = np.asarray(d["features"], dtype=np.float32)
                T = f.shape[0]
                if T < 32:
                    f = np.concatenate([f, np.zeros((32 - T, 168), np.float32)])
                    m = np.concatenate([np.ones(T, bool), np.zeros(32 - T, bool)])
                elif T > 32:
                    f = f[np.linspace(0, T - 1, 32).astype(int)]
                    m = np.ones(32, bool)
                else:
                    m = np.ones(32, bool)
                if args.zero_pose:
                    f[:, 128:] = 0.0
                if args.swap_hands:
                    f = np.concatenate([f[:, 64:128], f[:, :64], f[:, 128:]], axis=1)
                X.append(f)
                M.append(m)
                y.append(label_map[canon(str(d["label"]))])
            logits = model(torch.from_numpy(np.stack(X)).to(device),
                           torch.from_numpy(np.stack(M)).to(device))
            pred = logits.argsort(dim=1, descending=True).cpu()
            yy = torch.tensor(y)
            top1 += (pred[:, 0] == yy).sum().item()
            top5 += sum(yy[j].item() in pred[j, :5].tolist() for j in range(len(y)))
            total += len(y)
    print(f"INCLUDE-{args.split}: top-1 {top1/total*100:.2f}% top-5 {top5/total*100:.2f}% ({total})")


if __name__ == "__main__":
    main()
