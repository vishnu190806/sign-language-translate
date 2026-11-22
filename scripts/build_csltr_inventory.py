#!/usr/bin/env python
"""Inventory ISL-CSLTR: map sentences -> ordered glosses -> video files.

Separates sentence-level videos from word-level clips by filename match.
Writes data/processed/isl-csltr/inventory.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    base = Path("data/raw/isl-csltr/corpus/ISL_CSLRT_Corpus")
    gloss_csv = base / "corpus_csv_files" / "ISL Corpus sign glosses.csv"
    video_root = base / "Videos_Sentence_Level"
    out_dir = Path("data/processed/isl-csltr")
    out_dir.mkdir(parents=True, exist_ok=True)

    glosses: dict[str, list[str]] = {}
    with open(gloss_csv) as f:
        for row in csv.DictReader(f):
            glosses[row["Sentence"].strip().lower()] = row["SIGN GLOSSES"].strip().split()

    items = []
    for sent_dir in sorted(video_root.iterdir()):
        if not sent_dir.is_dir():
            continue
        sent = sent_dir.name.lower()
        for vf in sorted(sent_dir.iterdir()):
            if not vf.is_file() or vf.suffix.lower() != ".mp4":
                continue
            stem = vf.stem.lower().strip()
            stem_nosuffix = stem[:-4].strip() if stem.endswith(" (2)") or stem.endswith(" (3)") or stem.endswith(" (4)") or stem.endswith(" (5)") else stem
            import re

            stem_nosuffix = re.sub(r"\s*\(\d+\)$", "", stem).strip()
            kind = "sentence" if stem_nosuffix == sent else "word"
            items.append({"sentence": sent_dir.name, "glosses": glosses.get(sent, []),
                          "file": str(vf), "kind": kind, "word": None if kind == "sentence" else stem_nosuffix})

    n_sent = sum(1 for i in items if i["kind"] == "sentence")
    n_word = sum(1 for i in items if i["kind"] == "word")
    no_gloss = [i["sentence"] for i in items if i["kind"] == "sentence" and not i["glosses"]]
    (out_dir / "inventory.json").write_text(json.dumps(
        {"items": items, "n_sentence_videos": n_sent, "n_word_clips": n_word,
         "sentences_no_gloss": sorted(set(no_gloss))}, indent=1))
    print(f"sentence videos: {n_sent}, word clips: {n_word}, sentences w/o gloss: {len(set(no_gloss))}")
    print(f"Wrote {out_dir / 'inventory.json'}")


if __name__ == "__main__":
    main()
