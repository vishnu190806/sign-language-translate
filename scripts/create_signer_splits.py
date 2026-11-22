#!/usr/bin/env python
"""Create signer-disjoint train/val/test splits (LOSO protocol)."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

SYNONYMS = {
    "alright": "okay",
    "thankyou": "thank you",
    "big large": "big",
    "small little": "small",
    "store or shop": "store",
    "race (ethnicity)": "race",
    "you (plural)": "you plural",
}


def normalize_label(label: str) -> str:
    if "." in label:
        label = label.split(".")[-1].strip()
    label = label.lower().strip()
    return SYNONYMS.get(label, label)


def load_metadata(metadata_path: str | Path) -> list[dict]:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        print(f"Metadata not found at {metadata_path}")
        sys.exit(1)
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        metadata = list(reader)
    print(f"Loaded {len(metadata)} records from {metadata_path}")
    return metadata


def analyze_signers(metadata: list[dict]) -> dict[str, dict]:
    signer_classes: dict[str, set[str]] = defaultdict(set)
    signer_sources: dict[str, set[str]] = defaultdict(set)
    signer_videos: dict[str, list[dict]] = defaultdict(list)

    for row in metadata:
        signer = row.get("signer", "unknown")
        if not signer or signer.strip() == "":
            signer = "unknown"
        word = normalize_label(row.get("normalized_word", row.get("word", "")))
        source = row.get("dataset", "unknown")
        signer_classes[signer].add(word)
        signer_sources[signer].add(source)
        signer_videos[signer].append(row)

    signers = {}
    for signer in sorted(signer_classes.keys()):
        signers[signer] = {
            "num_videos": len(signer_videos[signer]),
            "num_classes": len(signer_classes[signer]),
            "sources": sorted(signer_sources[signer]),
            "classes": sorted(signer_classes[signer]),
        }
    return signers


def create_random_splits(
    metadata: list[dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    shuffled = list(metadata)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def create_loso_splits(metadata: list[dict]) -> dict[str, dict[str, list[dict]]]:
    signer_to_rows: dict[str, list[dict]] = defaultdict(list)
    for row in metadata:
        signer = row.get("signer", "unknown")
        if not signer or signer.strip() == "":
            signer = "unknown"
        signer_to_rows[signer].append(row)

    signers = sorted(signer_to_rows.keys())
    print(f"\nCreating LOSO splits for {len(signers)} signers: {signers}")

    folds = {}
    for held_out in signers:
        test_rows = signer_to_rows[held_out]
        remaining = [r for s, rows in signer_to_rows.items() if s != held_out for r in rows]
        rng = random.Random(42)
        rng.shuffle(remaining)
        n = len(remaining)
        n_val = max(1, int(n * 0.15))
        val_rows = remaining[:n_val]
        train_rows = remaining[n_val:]
        folds[held_out] = {
            "train": train_rows,
            "val": val_rows,
            "test": test_rows,
        }
    return folds


def validate_no_leakage(splits: dict[str, list[dict]]) -> bool:
    train_signers: set[str] = set()
    val_signers: set[str] = set()
    test_signers: set[str] = set()

    for row in splits.get("train", []):
        s = row.get("signer", "unknown") or "unknown"
        train_signers.add(s)
    for row in splits.get("val", []):
        s = row.get("signer", "unknown") or "unknown"
        val_signers.add(s)
    for row in splits.get("test", []):
        s = row.get("signer", "unknown") or "unknown"
        test_signers.add(s)

    train_val_overlap = train_signers & val_signers
    train_test_overlap = train_signers & test_signers
    val_test_overlap = val_signers & test_signers

    all_clean = True
    if train_val_overlap:
        print(f"  LEAKAGE: train/val overlap: {sorted(train_val_overlap)}")
        all_clean = False
    if train_test_overlap:
        print(f"  LEAKAGE: train/test overlap: {sorted(train_test_overlap)}")
        all_clean = False
    if val_test_overlap:
        print(f"  LEAKAGE: val/test overlap: {sorted(val_test_overlap)}")
        all_clean = False

    if all_clean:
        print("  Zero signer overlap: PASS")
    return all_clean


def print_stats(splits: dict[str, list[dict]], name: str = "") -> None:
    header = f"--- {name} ---" if name else "--- Splits ---"
    print(f"\n{header}")
    for split_name, rows in splits.items():
        classes = {normalize_label(r.get("normalized_word", r.get("word", ""))) for r in rows}
        sources = {r.get("dataset", "unknown") for r in rows}
        signers = {r.get("signer", "unknown") or "unknown" for r in rows}
        print(f"  {split_name}: {len(rows)} clips, {len(classes)} classes, "
              f"{len(signers)} signers, {len(sources)} sources")


def print_loso_fold_stats(folds: dict[str, dict[str, list[dict]]]) -> None:
    print(f"\n{'='*60}")
    print("LOSO FOLD SUMMARY")
    print(f"{'='*60}")
    print(f"{'Held-out':<20} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 40)
    for held_out, splits in folds.items():
        print(f"{held_out:<20} {len(splits['train']):>6} {len(splits['val']):>6} {len(splits['test']):>6}")


def main():
    parser = argparse.ArgumentParser(description="Create signer-disjoint splits (LOSO protocol)")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV")
    parser.add_argument("--output-dir", default="data/splits", help="Output directory for splits")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing files")
    parser.add_argument("--loso", action="store_true", help="Create LOSO (Leave-One-Signer-Out) folds")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)

    print("\nSigner Analysis:")
    signer_info = analyze_signers(metadata)
    for signer, info in signer_info.items():
        print(f"  {signer}: {info['num_videos']} videos, {info['num_classes']} classes, "
              f"sources={info['sources']}")

    total_signers = len(signer_info)
    single_video_signers = sum(1 for s in signer_info.values() if s["num_videos"] == 1)
    print(f"\nTotal signers: {total_signers}")
    print(f"Signers with 1 video: {single_video_signers}")

    if args.loso:
        folds = create_loso_splits(metadata)
        print_loso_fold_stats(folds)

        all_clean = True
        for held_out, splits in folds.items():
            print(f"\nValidation for fold (held-out={held_out}):")
            clean = validate_no_leakage(splits)
            if not clean:
                all_clean = False

        if not args.dry_run:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for held_out, splits in folds.items():
                fold_dir = output_dir / f"loso_{held_out}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                for split_name, rows in splits.items():
                    out_path = fold_dir / f"{split_name}.csv"
                    if rows:
                        with open(out_path, "w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                            writer.writeheader()
                            writer.writerows(rows)
                    print(f"  Wrote {out_path} ({len(rows)} rows)")
            print(f"\nAll LOSO folds saved to {output_dir}")

    else:
        splits = create_random_splits(
            metadata,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        print_stats(splits, "Random Splits")
        print("\nValidation:")
        validate_no_leakage(splits)

        if not args.dry_run:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for split_name, rows in splits.items():
                out_path = output_dir / f"{split_name}.csv"
                if rows:
                    with open(out_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)
                print(f"  Wrote {out_path} ({len(rows)} rows)")
            print(f"\nSplits saved to {output_dir}")

            manifest = {
                "metadata": str(args.metadata),
                "total": len(metadata),
                "splits": {k: len(v) for k, v in splits.items()},
                "signers": {s: info["num_videos"] for s, info in signer_info.items()},
            }
            manifest_path = output_dir / "split_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"Manifest saved to {manifest_path}")

    if args.dry_run:
        print("\nDry run: no files written.")


if __name__ == "__main__":
    main()
