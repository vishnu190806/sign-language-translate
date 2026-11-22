#!/usr/bin/env python
"""Convert OpenHands 75-kp HDF5 frames to 168-d memmap + clip index.

Layout (AI4Bharat/OpenHands scripts/mediapipe_extract.py): stored 75 kp =
pose[0:33] + left-hand[33:54] + right-hand[54:75], with a parallel
`visibility` group (pose visibility varies, hands are ones when detected).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

POSE_END = 33
LHAND = (33, 54)
RHAND = (54, 75)


def openhands_frame_to_168(kp75: np.ndarray) -> np.ndarray | None:
    from src.features.normalize import build_frame_features

    pose = np.asarray(kp75[0:POSE_END], dtype=np.float64)
    lh = np.asarray(kp75[LHAND[0]:LHAND[1]], dtype=np.float64)
    rh = np.asarray(kp75[RHAND[0]:RHAND[1]], dtype=np.float64)
    left = lh if np.abs(lh).sum() > 1e-9 else None
    right = rh if np.abs(rh).sum() > 1e-9 else None
    if pose.shape[0] <= 12:
        return None
    return build_frame_features(left, right, pose).astype(np.float32)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", nargs="+", required=True)
    parser.add_argument("--out-dir", default="data/processed/openhands")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    clips: list[dict] = []
    offset = 0
    for pattern in args.h5:
        for path in sorted(Path().glob(pattern)):
            with h5py.File(path, "r") as h:
                keys = sorted(h["keypoints"].keys())
                for key in keys:
                    kp = h["keypoints"][key][:]
                    n = kp.shape[0]
                    buf = np.zeros((n, 168), dtype=np.float32)
                    for t in range(n):
                        f = openhands_frame_to_168(kp[t])
                        if f is not None:
                            buf[t] = f
                    frames.append(buf)
                    clips.append({"file": str(path), "key": key,
                                  "start": offset, "length": n})
                    offset += n
    mem = np.memmap(out_dir / "sign_library.f32.memmap", dtype=np.float32,
                    mode="w+", shape=(offset, 168))
    pos = 0
    for buf in frames:
        mem[pos:pos + buf.shape[0]] = buf
        pos += buf.shape[0]
    mem.flush()
    (out_dir / "clips.json").write_text(json.dumps(clips))
    print(f"frames={offset} clips={len(clips)} -> {out_dir}")


if __name__ == "__main__":
    main()
