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


def load_seq(path: Path, seq_len: int):
    data = np.load(path, allow_pickle=True)
    feats = np.asarray(data["features"], dtype=np.float32)
    label = str(data["label"])
    T = feats.shape[0]
    if T < seq_len:
        pad = np.zeros((seq_len - T, feats.shape[1]), dtype=np.float32)
        feats = np.concatenate([feats, pad], axis=0)
        mask = np.concatenate([np.ones(T, dtype=bool), np.zeros(seq_len - T, dtype=bool)])
    elif T > seq_len:
        feats = feats[np.linspace(0, T - 1, seq_len).astype(int)]
        mask = np.ones(seq_len, dtype=bool)
    else:
        mask = np.ones(seq_len, dtype=bool)
    return feats, mask, label


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-source accuracy of joint model on joint test split")
    parser.add_argument("--checkpoint", default="models/joint/best.pt")
    parser.add_argument("--npz-dir", default="data/processed/landmarks/joint")
    parser.add_argument("--label-map", default="data/processed/landmarks/joint_label_map.json")
    parser.add_argument("--output", default="results/joint_by_source.json")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()

    from src.config import load_config
    from src.models.builder import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map: dict[str, int] = json.loads(Path(args.label_map).read_text())

    def canon(lab: str) -> str:
        lab = lab.split(".")[-1].strip().lower() if "." in lab else lab.strip().lower()
        return SYN.get(lab, lab)

    SYN = {"alright": "okay", "thankyou": "thank you", "big large": "big",
           "small little": "small", "store or shop": "store",
           "race (ethnicity)": "race", "you (plural)": "you plural"}
    config = load_config("configs/default.yaml", "configs/joint.yaml")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    use_source = isinstance(ckpt.get("sources"), list)
    if use_source:
        from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier

        htm = HTMAConfig(input_dim=168, embedding_dim=256, num_blocks=2, num_heads=8,
                         dropout=0.1, num_classes=len(label_map))
        model = MultiHeadHTMAClassifier(htm, sources=tuple(ckpt["sources"])).to(device).eval()
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"multi-head sources: {ckpt['sources']}")
    else:
        if ckpt.get("config", {}).get("model", {}).get("num_classes"):
            config.model.num_classes = ckpt["config"]["model"]["num_classes"]
        model = build_model(config.model).to(device).eval()
        model.load_state_dict(ckpt["model_state_dict"])

    groups: dict[str, list[Path]] = {"include": [], "isl500": []}
    for p in sorted(Path(args.npz_dir).glob("*.npz")):
        try:
            d = np.load(p, allow_pickle=True)
            if str(d.get("split", "")) != "test":
                continue
            if canon(str(d.get("label", ""))) not in label_map:
                continue
        except Exception:
            continue
        groups["isl500" if p.name.startswith("isl500_") else "include"].append(p)

    out: dict = {}
    with torch.no_grad():
        for name, files in groups.items():
            top1 = top5 = total = 0
            for i in range(0, len(files), args.batch):
                batch = files[i:i + args.batch]
                X, M, y = [], [], []
                for p in batch:
                    f, m, lab = load_seq(p, args.seq_len)
                    X.append(f)
                    M.append(m)
                    y.append(label_map[canon(lab)])
                logits = model(
                    torch.from_numpy(np.stack(X)).to(device),
                    torch.from_numpy(np.stack(M)).to(device),
                    **({"source": name} if use_source else {}),
                )
                pred = logits.argsort(dim=1, descending=True).cpu()
                yy = torch.tensor(y)
                top1 += (pred[:, 0] == yy).sum().item()
                top5 += sum(yy[j].item() in pred[j, :5].tolist() for j in range(len(y)))
                total += len(y)
            out[name] = {"total": total,
                           "top1": top1 / total if total else 0.0,
                           "top5": top5 / total if total else 0.0}
            if total:
                print(f"{name}: {total} clips, top-1 {top1/total*100:.2f}%, top-5 {top5/total*100:.2f}%")
            else:
                print(f"{name}: no test files")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
