#!/usr/bin/env python
"""Extract landmarks from dataset videos and save as .npz files."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract landmarks from videos")
    parser.add_argument("--dataset", default="isl", choices=["isl"])
    parser.add_argument("--input-dir", default="data/raw/isl")
    parser.add_argument("--output-dir", default="data/processed/landmarks/isl")
    parser.add_argument("--subset", default=None, help="Use include50 for INCLUDE-50 subset")
    parser.add_argument("--metadata-csv", default=None, help="Path to metadata CSV (bypasses registry)")
    parser.add_argument("--label-map-json", default=None, help="Path to label map JSON (bypasses registry)")
    parser.add_argument("--target-fps", type=int, default=30)
    args = parser.parse_args()

    from src.features.extractor import OfflineExtractor
    from src.data.metadata import save_landmarks

    import pandas as pd

    if args.metadata_csv:
        logger.info(f"Loading metadata from {args.metadata_csv}")
        metadata = pd.read_csv(args.metadata_csv)
    else:
        from src.data.registry import DatasetRegistry
        import src.data.download
        dataset_cls = DatasetRegistry.get(args.dataset)
        dataset = dataset_cls()
        metadata = dataset.build_metadata(subset=args.subset)

    if args.label_map_json:
        logger.info(f"Loading label map from {args.label_map_json}")
        with open(args.label_map_json) as f:
            label_map = json.load(f)
    else:
        from src.data.registry import DatasetRegistry
        import src.data.download
        dataset_cls = DatasetRegistry.get(args.dataset)
        dataset = dataset_cls()
        label_map = dataset.get_label_map(args.subset)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.label_map_json:
        label_map_path = Path(args.label_map_json)
    else:
        label_map_path = output_dir.parent / "isl_label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"Saved label map: {len(label_map)} classes -> {label_map_path}")

    stats = {"extracted": 0, "failed": 0, "skipped": 0}
    metadata_rows = []

    existing_videos = set()
    for existing_npz in output_dir.glob("*.npz"):
        try:
            data = np.load(existing_npz, allow_pickle=True)
            vp = str(data.get("video_path", ""))
            if vp:
                existing_videos.add(vp)
        except Exception:
            pass
    logger.info(f"Found {len(existing_videos)} already-extracted videos in {output_dir}")

    for idx, row in metadata.iterrows():
        video_path = Path(row["video_path"])
        if not video_path.exists():
            stats["failed"] += 1
            continue

        label_idx = label_map[row["label"]]
        npz_filename = f"{video_path.stem}_{label_idx}.npz"
        npz_path = output_dir / npz_filename

        if str(video_path) in existing_videos or npz_path.exists():
            stats["skipped"] += 1
            if not npz_path.exists():
                npz_filename = f"{video_path.stem}_{label_idx}.npz"
                npz_path = output_dir / npz_filename
            metadata_rows.append({
                "npz_path": str(npz_path),
                "label": row["label"],
                "label_idx": label_idx,
                "split": row["split"],
                "video_path": str(video_path),
            })
            continue

        extractor = None
        try:
            extractor = OfflineExtractor()
            features = extractor.extract_from_video(video_path, target_fps=args.target_fps)
            extractor.close()
            del extractor
            gc.collect()

            if features.shape[0] == 0:
                logger.warning(f"Empty features for {video_path}")
                stats["failed"] += 1
                continue

            save_landmarks(
                features=features,
                label=row["label"],
                split=row["split"],
                video_path=str(video_path),
                output_dir=output_dir,
                filename=npz_filename,
            )
            metadata_rows.append({
                "npz_path": str(npz_path),
                "label": row["label"],
                "label_idx": label_idx,
                "split": row["split"],
                "video_path": str(video_path),
            })
            stats["extracted"] += 1
            if stats["extracted"] % 25 == 0:
                logger.info(f"Extracted {stats['extracted']} videos")
        except Exception as e:
            logger.error(f"Failed {video_path}: {e}")
            stats["failed"] += 1
            if extractor is not None:
                try:
                    extractor.close()
                except Exception:
                    pass
                del extractor
                gc.collect()

    import pandas as pd
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = output_dir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    logger.info(f"Saved metadata: {len(meta_df)} entries -> {meta_path}")

    print(f"\nExtraction complete:")
    print(f"  Extracted: {stats['extracted']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped (already exists): {stats['skipped']}")


if __name__ == "__main__":
    main()
