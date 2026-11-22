#!/usr/bin/env python
"""Build CTC gloss vocabulary from train-split CSLTR sentences.

blank=0, glosses=1..N (sorted). Asserts test glosses are covered.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> None:
    d = Path("data/processed/isl-csltr/landmarks")
    wd = Path("data/processed/isl-csltr/word_landmarks")
    out = Path("models/ctc")
    out.mkdir(parents=True, exist_ok=True)
    train, test = set(), set()
    for p in sorted(d.glob("*.npz")):
        a = np.load(p, allow_pickle=True)
        bucket = train if str(a.get("split", "")) == "train" else test
        bucket.update(str(g).strip().upper() for g in a.get("glosses", []))
    missing = sorted(test - train)
    if missing:
        extra = 0
        for p in sorted(wd.glob("*.npz")):
            a = np.load(p, allow_pickle=True)
            if str(a.get("split", "")) != "train":
                continue
            lab = str(a.get("label", "")).strip().upper()
            if lab in missing and lab not in train:
                train.add(lab)
                extra += 1
        missing = sorted(test - train)
        print(f"covered from word clips: {extra}")
    assert not missing, f"Test glosses outside train vocab: {missing}"
    vocab = {"<blank>": 0}
    vocab.update({g: i + 1 for i, g in enumerate(sorted(train))})
    (out / "gloss_vocab.json").write_text(json.dumps(vocab, indent=1))
    print(f"glosses: {len(train)} (+blank), test covered: {len(test)}")
    print(f"Wrote {out / 'gloss_vocab.json'}")


if __name__ == "__main__":
    main()
