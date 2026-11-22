from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_metadata_table(
    video_dir: str | Path,
    label_map_path: str | Path | None = None,
) -> pd.DataFrame:
    video_dir = Path(video_dir)
    records = []
    for video_path in sorted(video_dir.rglob("*.mp4")) + sorted(video_dir.rglob("*.avi")):
        label = video_path.parent.name
        records.append({
            "video_path": str(video_path),
            "label": label,
            "split": _infer_split(video_path),
        })
    return pd.DataFrame(records)


def _infer_split(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "test" in parts:
        return "test"
    elif "val" in parts or "valid" in parts:
        return "val"
    return "train"


def build_label_map(metadata: pd.DataFrame) -> Dict[str, int]:
    labels = sorted(metadata["label"].unique())
    return {label: idx for idx, label in enumerate(labels)}


def save_label_map(label_map: Dict[str, int], path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(label_map, f, indent=2)


def load_label_map(path: str | Path) -> Dict[str, int]:
    with open(path) as f:
        return json.load(f)


def save_landmarks(
    features: np.ndarray,
    label: str,
    split: str,
    video_path: str,
    output_dir: str | Path,
    filename: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = Path(video_path).stem + ".npz"

    output_path = output_dir / filename
    np.savez_compressed(
        output_path,
        features=features.astype(np.float16),
        label=label,
        split=split,
        video_path=video_path,
        num_frames=features.shape[0],
    )
    return output_path


def get_split_stats(metadata: pd.DataFrame) -> Dict[str, dict]:
    stats = {}
    for split_name, group in metadata.groupby("split"):
        stats[split_name] = {
            "num_videos": len(group),
            "num_classes": group["label"].nunique(),
            "classes": sorted(group["label"].unique().tolist()),
        }
    return stats
