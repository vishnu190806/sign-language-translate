#!/usr/bin/env python
"""Pseudo-sentence benchmark (M9/B2): stitch isolated clips into
sentences with KNOWN boundaries; score aligners with boundary tolerance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SYN = {"alright": "okay", "thankyou": "thank you", "big large": "big",
       "small little": "small", "store or shop": "store",
       "race (ethnicity)": "race", "you (plural)": "you plural"}


def canon(lab: str) -> str:
    lab = lab.split(".")[-1].strip().lower() if "." in lab else lab.strip().lower()
    return SYN.get(lab, lab)


def stitch(clips: list[tuple[np.ndarray, str]]) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    feats = np.concatenate([c for c, _ in clips], axis=0)
    bounds: list[tuple[str, int, int]] = []
    t = 0
    for c, gloss in clips:
        bounds.append((gloss, t, t + c.shape[0]))
        t += c.shape[0]
    return feats.astype(np.float32), bounds


def uniform_split(n_frames: int, glosses: list[str]) -> list[tuple[str, int, int]]:
    bounds = []
    t = 0
    for i, g in enumerate(glosses):
        n = n_frames // len(glosses) + (1 if i < n_frames % len(glosses) else 0)
        bounds.append((g, t, t + n))
        t += n
    return bounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-dir", default="data/processed/landmarks/isl")
    parser.add_argument("--label-map", default="data/processed/landmarks/joint_label_map.json")
    parser.add_argument("--n-sentences", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/pseudo_sentences.json")
    args = parser.parse_args()

    label_map: dict[str, int] = json.loads(Path(args.label_map).read_text())
    by_label: dict[str, list[Path]] = {}
    for p in sorted(Path(args.npz_dir).glob("*.npz")):
        try:
            a = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if str(a.get("split", "")) != "test":
            continue
        lab = canon(str(a.get("label", "")))
        if lab in label_map:
            by_label.setdefault(lab, []).append(p)
    labs = sorted(by_label)
    rng = np.random.default_rng(args.seed)
    out = []
    for i in range(args.n_sentences):
        k = int(rng.integers(2, 5))
        picks = [str(labs[j]) for j in rng.choice(len(labs), size=k, replace=False)]
        clips = []
        for lab in picks:
            p = by_label[lab][int(rng.integers(len(by_label[lab])))]
            clips.append((np.asarray(np.load(p)["features"], dtype=np.float32), lab))
        feats, bounds = stitch(clips)
        out.append({"id": i, "glosses": picks,
                    "bounds": [[g, s, e] for g, s, e in bounds],
                    "n_frames": feats.shape[0]})
        np.savez_compressed(Path(args.output).parent / f"pseudo_{i:03d}.npz",
                            features=feats,
                            glosses=np.array(picks))
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(f"{len(out)} pseudo-sentences -> {args.output}")


if __name__ == "__main__":
    main()
