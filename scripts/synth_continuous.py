#!/usr/bin/env python
"""B3 synthesis (M9): novel ISL-order gloss sentences from isolated banks.

Template SOV reordering + novelty filter + multi-signer clip assignment.
No L2 boundary smoothing (abrupt boundaries regularize; Robotka 2026).
LLM novel-sentence generation plugs in via --llm-sentences JSON (C2 Qwen).
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

VERBS = {"EAT", "GO", "DRINK", "READ", "WRITE", "PLAY", "WATCH", "BUY", "SELL",
         "HELP", "LOVE", "LIKE", "NEED", "WANT", "GIVE", "TAKE", "BRING", "BRINGWATER"}


def to_isl_order(glosses: list[str]) -> list[str]:
    out = [g for g in glosses if g not in VERBS]
    out += [g for g in glosses if g in VERBS]
    return out


def is_novel(seq: list[str], seen: set[tuple[str, ...]]) -> bool:
    return tuple(seq) not in seen


def stitch_clips(seq: list[str], bank: dict[str, list[tuple[np.ndarray, str]]],
                 seed: int = 0, speed_jitter: float = 0.0
                 ) -> tuple[np.ndarray, list[tuple[str, int, int]], list[str]]:
    rng = np.random.default_rng(seed)
    feats: list[np.ndarray] = []
    bounds: list[tuple[str, int, int]] = []
    signers: list[str] = []
    t = 0
    for g in seq:
        cands = bank.get(g, [])
        if not cands:
            continue
        f, s = cands[int(rng.integers(len(cands)))]
        f = np.asarray(f, dtype=np.float32)
        if speed_jitter > 0:
            r = 1.0 + float(rng.uniform(-speed_jitter, speed_jitter))
            n = max(8, int(f.shape[0] / r))
            idx = np.linspace(0, f.shape[0] - 1, n).astype(int)
            f = f[idx]
        feats.append(f)
        bounds.append((g, t, t + f.shape[0]))
        signers.append(s)
        t += f.shape[0]
    if not feats:
        return np.zeros((0, 168), np.float32), [], []
    return np.concatenate(feats).astype(np.float32), bounds, signers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-dirs", nargs="+",
                        default=["data/processed/landmarks/isl",
                                 "data/processed/landmarks/isl500",
                                 "data/processed/landmarks/isl-csltr/word_landmarks"])
    parser.add_argument("--label-map", default="data/processed/landmarks/joint_label_map.json")
    parser.add_argument("--csltr-inventory", default="data/processed/isl-csltr/inventory.json")
    parser.add_argument("--n-sentences", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="data/processed/synthetic")
    args = parser.parse_args()

    joint: dict[str, int] = json.loads(Path(args.label_map).read_text())
    paths_by_gloss: dict[str, list[tuple[str, str]]] = {}
    for d in args.bank_dirs:
        for p in sorted(Path(d).glob("*.npz")):
            try:
                with np.load(p, allow_pickle=True) as a:
                    lab = str(a["label"]).strip() if "label" in a else ""
                    signer = str(a["signer"]).strip() if "signer" in a else d
            except Exception:
                continue
            key = lab.split(".")[-1].strip().lower() if "." in lab else lab.strip().lower()
            if key in joint:
                paths_by_gloss.setdefault(key.upper(), []).append((str(p), signer))
    print(f"bank: {len(paths_by_gloss)} glosses")

    def load_clip(entry: tuple[str, str]) -> tuple[np.ndarray, str]:
        p, signer = entry
        with np.load(p, allow_pickle=True) as a:
            return np.asarray(a["features"], dtype=np.float32), signer

    inv = json.load(open(args.csltr_inventory))
    items = inv["items"] if isinstance(inv, dict) else inv
    seen = {tuple(str(g).strip().upper() for g in i["glosses"])
            for i in items if i.get("kind") == "sentence" and i.get("glosses")}
    rng = np.random.default_rng(args.seed)
    glosses = sorted(paths_by_gloss)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    manifest = []
    attempts = 0
    while made < args.n_sentences and attempts < args.n_sentences * 50:
        attempts += 1
        k = int(rng.integers(2, 5))
        seq = to_isl_order([str(glosses[j]) for j in rng.choice(len(glosses), size=k, replace=False)])
        if not is_novel(seq, seen):
            continue
        bank = {}
        for g in seq:
            entries = paths_by_gloss[g]
            bank[g] = [load_clip(entries[int(rng.integers(len(entries)))]) for _ in range(2)]
        feats, bounds, signers = stitch_clips(seq, bank, seed=args.seed + made, speed_jitter=0.15)
        if not len(feats):
            continue
        np.savez_compressed(out_dir / f"synth_{made:04d}.npz", features=feats,
                            glosses=np.array(seq))
        manifest.append({"id": made, "glosses": seq, "signers": signers,
                         "bounds": [[g, s, e] for g, s, e in bounds]})
        seen.add(tuple(seq))
        made += 1
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{made} synthetic sentences -> {out_dir}")


if __name__ == "__main__":
    main()
