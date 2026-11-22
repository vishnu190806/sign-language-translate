#!/usr/bin/env python
"""Category accuracy analysis (M7.3): group per-class metrics by word category."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CATEGORIES: dict[str, set[str]] = {
    "Greetings": {"hello", "thank you", "good morning", "goodbye", "welcome", "sorry", "please"},
    "Animals": {"dog", "bird", "cat", "fish", "horse", "cow", "lion", "tiger", "bear", "monkey"},
    "Colors": {"red", "black", "white", "blue", "green", "yellow", "pink", "brown", "grey", "orange"},
    "Family": {"father", "mother", "brother", "sister", "family", "baby", "husband", "wife"},
    "People": {"boy", "girl", "man", "woman", "child", "friend", "teacher", "student", "doctor"},
    "Food": {"food", "water", "tea", "eat", "drink", "fruit", "rice", "bread", "milk"},
    "Time": {"today", "tomorrow", "yesterday", "morning", "evening", "night", "week", "month", "year", "time"},
}


def _parse_label(label: str) -> tuple[int | None, str]:
    label = label.strip()
    if "." in label:
        prefix, _, rest = label.partition(".")
        prefix = prefix.strip()
        if prefix.isdigit():
            return int(prefix), rest.strip().lower()
    return None, label.lower()


def _build_category_mapping(label_map: dict[str, int]) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {c: [] for c in list(CATEGORIES) + ["Other"]}
    for label, idx in label_map.items():
        _, clean = _parse_label(label)
        placed = False
        for cat, words in CATEGORIES.items():
            if clean in words:
                mapping[cat].append(idx)
                placed = True
                break
        if not placed:
            mapping["Other"].append(idx)
    return {k: sorted(v) for k, v in mapping.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-category accuracy analysis")
    parser.add_argument("--metrics", default="results/full_dataset/metrics.json")
    parser.add_argument("--label-map", default="models/label_map.json")
    parser.add_argument("--per-class", default="results/full_dataset/per_class_metrics.csv")
    parser.add_argument("--output", default="results/category_accuracy.json")
    args = parser.parse_args()

    label_map = json.loads(Path(args.label_map).read_text())
    mapping = _build_category_mapping(label_map)
    per_class: dict[str, float] = {}
    try:
        import csv

        with open(args.per_class) as f:
            for row in csv.DictReader(f):
                key = row.get("label", row.get("class", ""))
                try:
                    per_class[key] = float(row.get("accuracy", row.get("acc", 0)))
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"No per-class file at {args.per_class}; reporting coverage only")

    inv = {v: k for k, v in label_map.items()}
    report = {}
    for cat, idxs in mapping.items():
        accs = [per_class.get(inv[i], per_class.get(str(i))) for i in idxs]
        accs = [a for a in accs if a is not None]
        report[cat] = {"n_classes": len(idxs),
                       "mean_accuracy": sum(accs) / len(accs) if accs else None}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1))
    for cat, r in report.items():
        print(f"{cat:<10} classes={r['n_classes']:>3} mean_acc={r['mean_accuracy']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
