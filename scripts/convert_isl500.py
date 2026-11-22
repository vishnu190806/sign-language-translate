#!/usr/bin/env python
"""Inspect and convert ISL500/ISL-DATA MediaPipe .h5 landmarks to training .npz.

Usage:
    # 1. Inspect layout of downloaded .h5 files (no conversion):
    .\\venv\\Scripts\\python.exe scripts/convert_isl500.py --inspect --src data/raw/isl500

    # 2. Convert overlap words to .npz (168-d, INCLUDE label space):
    .\\venv\\Scripts\\python.exe scripts/convert_isl500.py --src data/raw/isl500 --dst data/processed/landmarks/isl500

Signer-disjoint splits: users 001-011 train, 012-013 val, 014-015 test.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRAIN_USERS = {f"{i:03d}" for i in range(1, 12)}
VAL_USERS = {"012", "013"}
TEST_USERS = {"014", "015"}


def tree(h, prefix="", depth=0, max_depth=4):
    import h5py

    if depth > max_depth:
        return
    for k in h.keys():
        v = h[k]
        if isinstance(v, h5py.Dataset):
            logger.info(f"{prefix}{k}: dataset shape={v.shape} dtype={v.dtype}")
        else:
            logger.info(f"{prefix}{k}/: group attrs={dict(v.attrs)}")
            tree(v, prefix + k + "/", depth + 1, max_depth)


def inspect_layout(src: Path, n: int = 3) -> None:
    import h5py

    files = sorted(src.rglob("*.h5"))[:n]
    if not files:
        logger.error(f"No .h5 files under {src}")
        return
    for f in files:
        logger.info(f"=== {f.name} ===")
        with h5py.File(f, "r") as h:
            logger.info(f"root attrs: {dict(h.attrs)}")
            tree(h)


def find_landmark_arrays(h) -> dict:
    """Heuristic search for hand (21,3) and pose (33,3) arrays. Returns {kind: array-like accessor}."""
    import h5py

    found: dict[str, list[str]] = {"hand21": [], "pose33": [], "other": []}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind == "f":
            s = obj.shape
            if len(s) >= 2 and s[-2:] == (21, 3):
                found["hand21"].append(name)
            elif len(s) >= 2 and s[-2:] == (33, 3):
                found["pose33"].append(name)
            elif len(s) >= 2 and s[-1] == 3 and s[-2] in (21, 33):
                found["other"].append(f"{name} shape={s}")

    h.visititems(visit)
    return found


def convert(src: Path, dst: Path, label_map: dict[str, int], overlap: set[str]) -> dict:
    import h5py

    from src.features.normalize import build_frame_features

    dst.mkdir(parents=True, exist_ok=True)
    stats = {"converted": 0, "skipped_oov": 0, "failed": 0, "layouts": {}}
    norm_syn = {"thankyou": "thank you"}

    def norm(w: str) -> str:
        w = w.lower().strip()
        return norm_syn.get(w, w)

    files = sorted(src.rglob("*.h5"))
    logger.info(f"Found {len(files)} .h5 files")
    for i, f in enumerate(files):
        word_raw = f.stem.split("__")[0]
        word = norm(word_raw)
        user = f.parent.name[-3:]
        if word not in overlap or word not in label_map:
            stats["skipped_oov"] += 1
            continue
        split = "train" if user in TRAIN_USERS else ("val" if user in VAL_USERS else "test")
        try:
            with h5py.File(f, "r") as h:
                if i == 0:
                    found = find_landmark_arrays(h)
                    stats["layouts"] = {k: v[:6] for k, v in found.items()}
                    logger.info(f"Detected layout: {json.dumps(stats['layouts'])[:500]}")
                frames = extract_frames(h, word_raw)
            if frames is None or len(frames) == 0:
                stats["failed"] += 1
                continue
            feats = np.stack([
                build_frame_features(fr.get("left"), fr.get("right"), fr.get("pose"))
                for fr in frames
            ]).astype(np.float32)
            out = dst / f"isl500_{user}_{word_raw}_{i}.npz"
            np.savez_compressed(
                out, features=feats, label=word,
                label_idx=label_map[word], split=split,
                dataset="isl500", signer=f"isl500_user{user}",
            )
            stats["converted"] += 1
        except Exception as e:
            logger.warning(f"Failed {f.name}: {e}")
            stats["failed"] += 1
        if (i + 1) % 500 == 0:
            logger.info(f"Progress {i + 1}/{len(files)} converted={stats['converted']}")
    return stats


def extract_frames(h, word_raw: str) -> list[dict] | None:
    """Map an open .h5 group to per-frame {left, right, pose} (21,3)/(33,3) arrays.

    Tries common layouts; extend once --inspect reveals the true schema.
    Override via env ISL500_HAND_KEY / ISL500_POSE_KEY (slash-separated paths).
    """
    import os

    import h5py

    def get(path: str):
        try:
            v = h[path]
            return np.asarray(v) if isinstance(v, h5py.Dataset) else None
        except Exception:
            return None

    hand_keys = [k for k in [os.environ.get("ISL500_HAND_KEY"), "hand_landmarks", "hands", "landmarks/hands"] if k]
    pose_keys = [k for k in [os.environ.get("ISL500_POSE_KEY"), "pose_landmarks", "pose", "landmarks/pose"] if k]

    for hk in hand_keys:
        hands = get(hk)
        if hands is None:
            continue
        pose = None
        for pk in pose_keys:
            pose = get(pk)
            if pose is not None:
                break
        return frames_from_arrays(hands, pose)
    found = find_landmark_arrays(h)
    if found["hand21"]:
        hands = np.asarray(h[found["hand21"][0]])
        pose = np.asarray(h[found["pose33"][0]]) if found["pose33"] else None
        return frames_from_arrays(hands, pose)
    return None


def frames_from_arrays(hands: np.ndarray, pose: np.ndarray | None) -> list[dict]:
    """Normalize (T,2,21,3)/(T,21,3) hands and (T,33,3) pose to frame dicts."""
    hands = np.asarray(hands, dtype=np.float32)
    if hands.ndim == 3 and hands.shape[1:] == (21, 3):
        hands = hands[:, None, :, :]
    frames: list[dict] = []
    T = hands.shape[0] if hands.ndim == 4 else 0
    for t in range(T):
        hh = hands[t]
        left = hh[0] if hh.shape[0] > 0 else None
        right = hh[1] if hh.shape[0] > 1 else None
        if left is not None and not np.any(left):
            left = None
        if right is not None and not np.any(right):
            right = None
        p = None
        if pose is not None:
            pose = np.asarray(pose, dtype=np.float32)
            if pose.ndim == 3 and t < pose.shape[0]:
                p = pose[t] if np.any(pose[t]) else None
            elif pose.ndim == 2:
                p = pose if np.any(pose) else None
        frames.append({"left": left, "right": right, "pose": p})
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect/convert ISL500 .h5 landmarks")
    parser.add_argument("--src", default="data/raw/isl500")
    parser.add_argument("--dst", default="data/processed/landmarks/isl500")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--inspect-n", type=int, default=3)
    parser.add_argument("--label-map", default="models/label_map.json")
    parser.add_argument("--overlap", default="results/vocab_overlap.json")
    args = parser.parse_args()

    src = Path(args.src)
    if args.inspect:
        inspect_layout(src, args.inspect_n)
        return
    label_map_raw = json.loads(Path(args.label_map).read_text())
    label_map = {}
    for k, v in label_map_raw.items():
        nk = k.split(".")[-1].strip().lower() if "." in k else k.strip().lower()
        label_map[nk] = v
    overlap = set(json.loads(Path(args.overlap).read_text())["overlap"])
    stats = convert(src, Path(args.dst), label_map, overlap)
    logger.info(f"Done: {stats['converted']} converted, {stats['skipped_oov']} OOV-skipped, {stats['failed']} failed")


if __name__ == "__main__":
    main()
