#!/usr/bin/env python
"""Evaluate C1 rules gloss->sentence on ISL-CSLTR gold glosses.

Metric: exact match + unigram F1 vs reference English sentence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.text import detect_flags, gloss_to_sentence


def f1(pred: str, ref: str) -> float:
    p = pred.lower().replace("?", "").replace(".", "").replace(",", "").split()
    r = ref.lower().replace("?", "").replace(".", "").replace(",", "").split()
    if not p or not r:
        return 0.0
    overlap = 0
    rc = {}
    for w in r:
        rc[w] = rc.get(w, 0) + 1
    for w in p:
        if rc.get(w, 0) > 0:
            overlap += 1
            rc[w] -= 1
    prec = overlap / len(p)
    rec = overlap / len(r)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def main() -> None:
    inv = json.loads(Path("data/processed/isl-csltr/inventory.json").read_text())
    seen, em, f1s, rows = set(), 0, [], []
    for item in inv["items"]:
        if item["kind"] != "sentence" or not item["glosses"] or item["sentence"] in seen:
            continue
        seen.add(item["sentence"])
        hyp = gloss_to_sentence(item["glosses"], detect_flags(item["glosses"]))
        ref = item["sentence"]
        m = hyp.rstrip(".?").lower() == ref.lower()
        em += m
        f = f1(hyp, ref)
        f1s.append(f)
        rows.append({"sentence": ref, "glosses": item["glosses"], "hyp": hyp, "exact": m, "f1": round(f, 3)})
    n = len(rows)
    if n == 0:
        raise ValueError("No sentences evaluated")
    out = {"n": n, "exact_match": em / n, "mean_f1": sum(f1s) / n, "rows": rows}
    Path("results").mkdir(exist_ok=True)
    Path("results/gloss2text_rules.json").write_text(json.dumps(out, indent=1))
    print(f"sentences: {n}, exact: {em/n*100:.1f}%, mean F1: {sum(f1s)/n*100:.1f}%")
    print(f"Wrote results/gloss2text_rules.json")


if __name__ == "__main__":
    main()
