from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CSLTRDataset(Dataset):
    def __init__(self, npz_dir: str | Path, split: str, vocab: dict[str, int],
                 max_frames: int = 256, augment: bool = False, seed: int = 42,
                 val_sentences: set[str] | None = None, only_val: bool = False,
                 word_dir: str | Path | None = None):
        self.npz_dir = Path(npz_dir)
        self.max_frames = max_frames
        self.augment = augment and split == "train" and not only_val
        self.rng = np.random.default_rng(seed)
        self.samples: list[tuple[Path, list[int]]] = []
        for p in sorted(self.npz_dir.glob("*.npz")):
            try:
                a = np.load(p, allow_pickle=True)
            except Exception:
                continue
            if str(a.get("split", "")) != split:
                continue
            if val_sentences is not None:
                in_val = str(a.get("sentence", "")).lower() in val_sentences
                if only_val != in_val:
                    continue
            try:
                ids = [vocab[str(g).strip().upper()] for g in a.get("glosses", [])]
            except KeyError:
                continue
            if not ids:
                continue
            self.samples.append((p, ids))
        if word_dir is not None and split == "train" and not only_val:
            for p in sorted(Path(word_dir).glob("*.npz")):
                try:
                    a = np.load(p, allow_pickle=True)
                except Exception:
                    continue
                if str(a.get("split", "")) != "train":
                    continue
                lab = str(a.get("label", "")).strip().upper()
                if lab not in vocab:
                    continue
                self.samples.append((p, [vocab[lab]]))

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.rng = np.random.default_rng(42 + epoch)

    def __getitem__(self, idx: int):
        p, ids = self.samples[idx]
        f = np.asarray(np.load(p, allow_pickle=True)["features"], dtype=np.float32)
        if f.shape[0] > self.max_frames:
            f = f[:self.max_frames]
        if self.augment and self.rng.random() < 0.3:
            f = f + self.rng.normal(0, 0.01, f.shape).astype(np.float32)
        return torch.from_numpy(f.copy()), torch.tensor(ids, dtype=torch.long)


def ctc_collate(batch: list) -> tuple:
    feats, ids = zip(*batch)
    lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    T = int(lens.max().item())
    X = torch.zeros(len(feats), T, 168)
    M = torch.zeros(len(feats), T, dtype=torch.bool)
    for i, f in enumerate(feats):
        X[i, :f.shape[0]] = f
        M[i, :f.shape[0]] = True
    flat = torch.cat(ids)
    tlen = torch.tensor([len(t) for t in ids], dtype=torch.long)
    return X, M, flat, tlen, lens
