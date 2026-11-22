from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SYNONYMS = {
    "alright": "okay",
    "thankyou": "thank you",
    "big large": "big",
    "small little": "small",
    "store or shop": "store",
    "race (ethnicity)": "race",
    "you (plural)": "you plural",
}


def normalize(label: str) -> str:
    if "." in label:
        label = label.split(".")[-1].strip()
    label = label.lower().strip()
    label = re.sub(r"\s+", " ", label)
    return SYNONYMS.get(label, label)


def main() -> None:
    parser = argparse.ArgumentParser(description="INCLUDE x ISL500 vocabulary overlap")
    parser.add_argument("--label-map", default="models/label_map.json")
    parser.add_argument("--isl500-vocab", default="isl500_vocab.json")
    parser.add_argument("--output", default="results/vocab_overlap.json")
    args = parser.parse_args()

    include_map: dict[str, int] = json.loads(Path(args.label_map).read_text())
    include_norm = {normalize(k): k for k in include_map}
    isl500_words: list[str] = json.loads(Path(args.isl500_vocab).read_text())
    isl500_norm = {normalize(w): w for w in isl500_words}

    overlap = sorted(set(include_norm) & set(isl500_norm))
    include_only = sorted(set(include_norm) - set(isl500_norm))
    isl500_only = sorted(set(isl500_norm) - set(include_norm))

    out = {
        "include_classes": len(include_norm),
        "isl500_words": len(isl500_norm),
        "overlap_words": len(overlap),
        "overlap": overlap,
        "include_only_count": len(include_only),
        "isl500_only_count": len(isl500_only),
        "note": "Overlap = words trainable jointly under one normalized label space (union training) or shared-head eval. ISL500-only words need a second head (Logos multi-head) or prototype/KNN fallback.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"INCLUDE: {len(include_norm)}, ISL500: {len(isl500_norm)}, overlap: {len(overlap)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
