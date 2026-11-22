#!/usr/bin/env python
"""Per-signer-grouped eval on deployed v2 (max-over-heads routing).

Groups: ISL500 test users 014/015 (all their clips, all heads).
Writes results/per_user_v2.json {user: {n, top1, per_head_top1}}.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

HAND_PREFIX = {"include": 0, "isl500": 263, "extra": 526}


def prep(f: np.ndarray, seq_len: int = 32):
    T = f.shape[0]
    if T < seq_len:
        pad = np.zeros((seq_len - T, f.shape[1]), np.float32)
        return np.concatenate([f, pad])
    return f[np.linspace(0, T - 1, seq_len).astype(int)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/multihead_v2/best.pt")
    parser.add_argument("--users", default="014,015")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="results/per_user_v2.json")
    parser.add_argument("--rows-out", default=None)
    args = parser.parse_args()

    from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joint: dict[str, int] = json.loads(Path("data/processed/landmarks/joint_label_map.json").read_text())
    ex_path = Path("models/multihead_extra/label_map_extra.json")
    extra: dict[str, int] = json.loads(ex_path.read_text())
    jnorm = {k.strip().lower(): v for k, v in joint.items()}
    enorm = {k.strip().lower(): v for k, v in extra.items()}

    htm = HTMAConfig(input_dim=168, embedding_dim=256, num_blocks=2, num_heads=8,
                     dropout=0.1, num_classes=len(joint))
    model = MultiHeadHTMAClassifier(
        htm, sources=("include", "isl500", "extra"),
        head_classes={"include": len(joint), "isl500": len(joint), "extra": len(extra)}).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    own = model.state_dict()
    model.load_state_dict({k: v for k, v in ckpt["model_state_dict"].items()
                           if k in own and own[k].shape == v.shape}, strict=False)

    def label_of(a) -> tuple[str, int, str] | None:
        raw = str(a.get("label", "")).strip()
        key = raw.split(".")[-1].strip().lower() if "." in raw else raw.strip().lower()
        if key in jnorm:
            return "joint", jnorm[key], key
        if key in enorm:
            return "extra", enorm[key], key
        return None

    per_user: dict[str, list] = defaultdict(lambda: [0, 0, defaultdict(lambda: [0, 0])])
    rows_out: list = []
    for user in args.users.split(","):
        for d in ("data/processed/landmarks/isl500", "data/processed/landmarks/isl500_extra"):
            files = sorted(Path(d).glob(f"isl500_{user}_*.npz"))
            batch, metas = [], []
            with torch.no_grad():
                for p in files:
                    try:
                        a = np.load(p, allow_pickle=True)
                    except Exception:
                        continue
                    lab = label_of(a)
                    if lab is None:
                        continue
                    batch.append(prep(np.asarray(a["features"], dtype=np.float32)))
                    metas.append((user, lab))
                    if len(batch) == args.batch_size:
                        _score(model, batch, metas, per_user, device, rows_out)
                        batch, metas = [], []
                if batch:
                    _score(model, batch, metas, per_user, device, rows_out)
    out = {}
    for user, (ok, tot, heads) in per_user.items():
        out[user] = {"n": tot, "top1": round(100 * ok / max(tot, 1), 2),
                     "per_head": {h: round(100 * v[0] / max(v[1], 1), 2) for h, v in heads.items()}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    if args.rows_out:
        Path(args.rows_out).write_text(json.dumps(rows_out))
    print(json.dumps(out, indent=1))


def _score(model, batch, metas, per_user, device, rows_out=None):
    xb = torch.from_numpy(np.stack(batch)).to(device)
    mb = torch.ones(xb.shape[0], 32, dtype=torch.bool).to(device)
    with torch.no_grad():
        logits = {s: model(xb, mb, source=s) for s in ("include", "isl500", "extra")}
    outs = {s: lg.argmax(1).cpu().numpy() for s, lg in logits.items()}
    conf = {s: lg.softmax(1).max(1).values.cpu().numpy() for s, lg in logits.items()}
    for i, (user, (space, idx, _)) in enumerate(metas):
        best, best_c, best_s = -1, -1.0, ""
        for s in ("include", "isl500", "extra"):
            gid = int(outs[s][i])
            c = float(conf[s][i])
            if c > best_c:
                best, best_c, best_s = gid, c, s
        ok, tot, heads = per_user[user]
        h = heads[best_s]
        h[1] += 1
        tot += 1
        is_ok = (best_s in ("include", "isl500") and space == "joint" and best == idx) or \
                (best_s == "extra" and space == "extra" and best == idx)
        if is_ok:
            ok += 1
            h[0] += 1
        per_user[user] = [ok, tot, heads]
        if rows_out is not None:
            rows_out.append({"user": user, "space": space, "idx": idx,
                             "pred": {s: int(outs[s][i]) for s in outs},
                             "conf": {s: round(float(conf[s][i]), 4) for s in conf}})


if __name__ == "__main__":
    main()
