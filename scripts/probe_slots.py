from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def slot_stats(files: list[Path], n: int, tag: str) -> None:
    s0 = s1 = p0 = p1 = 0.0
    c = 0
    both = left_only = right_only = neither = 0
    for p in random.sample(files, min(n, len(files))):
        a = np.load(p, allow_pickle=True)
        f = np.asarray(a["features"], dtype=np.float32)
        h0 = np.abs(f[:, :63]).sum()
        h1 = np.abs(f[:, 63:127]).sum()
        pp = np.abs(f[:, 128:167]).sum()
        s0 += h0
        s1 += h1
        pp_sum = pp
        p0 += pp_sum
        c += f.shape[0]
        a0 = h0 > 1e-6
        a1 = h1 > 1e-6
        both += a0 and a1
        left_only += a0 and not a1
        right_only += (not a0) and a1
        neither += (not a0) and (not a1)
    if c == 0:
        print(f"{tag}: no frames found")
        return
    print(f"{tag}: files={min(n, len(files))} slot0_energy={s0/c:.4f} slot1_energy={s1/c:.4f} "
          f"pose_energy={p0/c:.4f} both={both} left_only={left_only} right_only={right_only} neither={neither}")


def main() -> None:
    random.seed(1)
    j = Path("data/processed/landmarks/joint")
    isl500 = [p for p in j.glob("isl500_*.npz")]
    include = [p for p in j.glob("*.npz") if not p.name.startswith("isl500_")]
    slot_stats(isl500, 500, "isl500 ")
    slot_stats(include, 500, "include")


if __name__ == "__main__":
    main()
