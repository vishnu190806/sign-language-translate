#!/usr/bin/env python
"""Download ISL500/ISL-DATA MediaPipe landmarks for INCLUDE-overlapping words only.

Requires: HF login + accepted dataset terms (gated repo):
    .\\venv\\Scripts\\huggingface-cli.exe login
Then accept terms at https://huggingface.co/datasets/ISL500/ISL-DATA

Downloads Landmarks/MediaPipe/ISL_DATA_USER*/<word>__*.h5 for the ~180 words
in results/vocab_overlap.json. Resumable (snapshot_download skips cached files).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Selective ISL500 landmark download")
    parser.add_argument("--overlap", default="results/vocab_overlap.json")
    parser.add_argument("--dest", default="data/raw/isl500")
    parser.add_argument("--users", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    overlap = json.loads(Path(args.overlap).read_text())["overlap"]
    users = [f"ISL_DATA_USER{u.strip().zfill(3)}" for u in args.users.split(",")]

    def case_pattern(w: str) -> str:
        if not w:
            return w
        return f"[{w[0].lower()}{w[0].upper()}]{w[1:]}"

    patterns = [f"Landmarks/MediaPipe/{u}/{case_pattern(w)}__*" for u in users for w in overlap]
    print(f"Words: {len(overlap)}, users: {len(users)}, patterns: {len(patterns)}")
    if args.dry_run:
        print("Dry run — first 5 patterns:")
        for p in patterns[:5]:
            print("  " + p)
        return

    snapshot_download(
        "ISL500/ISL-DATA",
        repo_type="dataset",
        local_dir=args.dest,
        allow_patterns=patterns + ["MappingFiles/*", "README.md"],
    )
    print(f"Done: {args.dest}. Next: inspect .h5 structure, then convert to .npz (168-d).")


if __name__ == "__main__":
    main()
