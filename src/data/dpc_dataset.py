from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

WINDOW = 70


class DPCWindowDataset(Dataset):
    def __init__(self, memmap_path: str | Path, clips_path: str | Path, seed: int = 42):
        mem = np.memmap(memmap_path, dtype=np.float32, mode="r")
        total = mem.shape[0] // 168
        self.mem = mem.reshape(total, 168)
        clips = json.loads(Path(clips_path).read_text())
        self.clips = [c for c in clips if c["length"] >= WINDOW]
        self.rng = np.random.default_rng(seed)
        self.num_clips = len(self.clips)

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, i: int) -> torch.Tensor:
        c = self.clips[i]
        s = int(self.rng.integers(0, c["length"] - WINDOW + 1))
        return torch.from_numpy(self.mem[c["start"] + s:c["start"] + s + WINDOW].copy())
