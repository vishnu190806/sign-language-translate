#!/usr/bin/env python
"""Build the ISL500-extra label map from downloaded .h5 filenames (no loading).

Keeps normalized words with >= --min-train clips from train users 001-011.
Writes <out-map> ({word: idx}) and prints coverage stats.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

TRAIN_USERS = {f"{i:03d}" for i in range(1, 12)}


def norm(w: str) -> str:
    return {"thankyou": "thank you"}.get(w.lower().strip(), w.lower().strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw/isl500")
    parser.add_argument("--joint-map", default="data/processed/landmarks/joint_label_map.json")
    parser.add_argument("--out-map", default="models/multihead_extra/label_map_extra.json")
    parser.add_argument("--min-train", type=int, default=10)
    args = parser.parse_args()

    joint = {norm(k) for k in json.loads(Path(args.joint_map).read_text())}
    per_word: dict[str, Counter] = defaultdict(Counter)
    for f in Path(args.src).rglob("*.h5"):
        word_raw = f.stem.split("__")[0]
        user = f.parent.name[-3:]
        per_word[norm(word_raw)][user] += 1
    rows = []
    for w, users in sorted(per_word.items()):
        if w in joint or not w:
            continue
        ntrain = sum(n for u, n in users.items() if u in TRAIN_USERS)
        rows.append((w, ntrain, sum(users.values())))
    kept = [(w, nt, tot) for w, nt, tot in rows if nt >= args.min_train]
    label_map = {w: i for i, (w, _, _) in enumerate(kept)}
    assert len(label_map) == len(kept), "normalized name collision in extra map"
    out = Path(args.out_map)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(label_map, indent=1))
    ov_path = out.parent / "extra_overlap.json"
    ov_path.write_text(json.dumps({"overlap": sorted(label_map)}, indent=1))
    print(f"extra words total={len(rows)} kept(>={args.min_train} train clips)={len(kept)} -> {out}")
    if kept:
        import numpy as np
        ns = np.array([nt for _, nt, _ in kept])
        print(f"train clips/word: min={ns.min()} median={int(np.median(ns))} mean={ns.mean():.1f}")


if __name__ == "__main__":
    main()
