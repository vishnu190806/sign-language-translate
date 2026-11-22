from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from src.features.augment import LandmarkAugmentor

logger = logging.getLogger(__name__)


class LandmarkDataset(Dataset):
    def __init__(
        self,
        npz_dir: str | Path,
        split: str = "train",
        label_map: dict[str, int] | None = None,
        sequence_length: int = 32,
        augment: bool = False,
        augmentation_mode: str = "none",
        augmentation_probs: dict[str, float] | None = None,
        spec=None,
        seed: int = 42,
        use_manifest: bool = True,
        normalize_fn=None,
        oversample_labels: dict[int, int] | None = None,
    ):
        self.npz_dir = Path(npz_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.augment = augment and split == "train"
        self.augmentation_mode = augmentation_mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._epoch = 0
        self.oversample_labels = dict(oversample_labels or {}) if split == "train" else {}

        if self.augment and augmentation_mode == "full":
            from src.features.augment_pipeline import AugmentationPipeline
            self.augmentor = AugmentationPipeline(spec=spec, seed=seed)
            if augmentation_probs:
                for name, prob in augmentation_probs.items():
                    self.augmentor.set_prob(name, float(prob))
        elif self.augment and augmentation_mode == "basic":
            self.augmentor = LandmarkAugmentor(spec)
        else:
            self.augmentor = None

        self.samples: list[tuple[Path, int]] = self._build_index(label_map, use_manifest, normalize_fn)

        logger.info(f"Loaded {len(self.samples)} samples for split={split}")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self.rng = np.random.default_rng(self.seed + epoch)

    def _manifest_path(self) -> Path:
        return self.npz_dir / "_manifest.json"

    def _build_index(self, label_map: dict[str, int] | None, use_manifest: bool, normalize_fn=None) -> list[tuple[Path, int]]:
        import json

        paths = sorted(self.npz_dir.glob("*.npz"))
        manifest: dict[str, dict] = {}
        if use_manifest:
            try:
                mp = self._manifest_path()
                if mp.exists():
                    manifest = json.loads(mp.read_text())
            except Exception:
                manifest = {}
        samples: list[tuple[Path, int]] = []
        updated = False
        for npz_path in paths:
            key = npz_path.name
            entry = manifest.get(key)
            if entry is None:
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    entry = {"split": str(data.get("split", "train")), "label": str(data.get("label", ""))}
                except Exception:
                    continue
                manifest[key] = entry
                updated = True
            if entry.get("split") != self.split:
                continue
            label = entry.get("label", "")
            if normalize_fn is not None:
                label = normalize_fn(label)
            if label_map is not None:
                if label not in label_map:
                    continue
                label_idx = label_map[label]
            else:
                label_idx = hash(label) % 100000
            samples.append((npz_path, label_idx))
            repeat = self.oversample_labels.get(label_idx, 1) if self.split == "train" else 1
            for _ in range(max(1, repeat) - 1):
                samples.append((npz_path, label_idx))
        if use_manifest and updated:
            try:
                self._manifest_path().write_text(json.dumps(manifest))
            except Exception:
                pass
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npz_path, label_idx = self.samples[idx]
        data = np.load(npz_path)
        features = data["features"].astype(np.float32)
        T = features.shape[0]

        if T < self.sequence_length:
            pad = np.zeros(
                (self.sequence_length - T, features.shape[1]), dtype=np.float32
            )
            features = np.concatenate([features, pad], axis=0)
            mask = np.concatenate(
                [np.ones(T, dtype=bool), np.zeros(self.sequence_length - T, dtype=bool)],
                axis=0,
            )
        elif T > self.sequence_length:
            indices = np.linspace(0, T - 1, self.sequence_length).astype(int)
            features = features[indices]
            mask = np.ones(self.sequence_length, dtype=bool)
        else:
            mask = np.ones(self.sequence_length, dtype=bool)

        if self.augment and self.augmentor is not None:
            if self.augmentation_mode == "full":
                features = self.augmentor(features, mode="train")
            else:
                features = self.augmentor(features, self.rng)

        return (
            torch.from_numpy(features.copy()),
            torch.from_numpy(mask),
            label_idx,
        )


class BalancedBatchSampler(Sampler):
    def __init__(self, labels: list[int], batch_size: int, num_samples: int | None = None, seed: int = 42,
                 groups: list[str] | None = None):
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = num_samples or len(labels)
        self.seed = seed
        self._epoch = 0

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            key = (label, groups[idx]) if groups is not None else label
            self.class_to_indices[key].append(idx)

        self.num_classes = len(self.class_to_indices)
        if self.num_classes == 0:
            raise ValueError("BalancedBatchSampler: no classes found (empty dataset or label map mismatch)")
        self.classes = sorted(self.class_to_indices.keys(), key=str)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        indices = []
        while len(indices) < self.num_samples:
            rng.shuffle(self.classes)
            for cls in self.classes:
                cls_indices = self.class_to_indices[cls]
                chosen = rng.choice(cls_indices, size=min(len(cls_indices), self.batch_size // self.num_classes + 1), replace=False)
                indices.extend(chosen.tolist())
                if len(indices) >= self.num_samples:
                    break
        return iter(indices[:self.num_samples])

    def __len__(self):
        return self.num_samples


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return features, masks, labels


def create_dataloader(
    npz_dir: str | Path,
    split: str,
    label_map: dict[str, int] | None = None,
    sequence_length: int = 32,
    batch_size: int = 64,
    num_workers: int = 4,
    augment: bool = False,
    balanced: bool = False,
    augmentation_mode: str = "none",
    augmentation_probs: dict[str, float] | None = None,
    prefetch_factor: int | None = 2,
    persistent_workers: bool | None = None,
    seed: int = 42,
    normalize_fn=None,
    oversample_labels: dict[int, int] | None = None,
) -> DataLoader:
    dataset = LandmarkDataset(
        npz_dir=npz_dir,
        split=split,
        label_map=label_map,
        sequence_length=sequence_length,
        augment=augment,
        augmentation_mode=augmentation_mode,
        augmentation_probs=augmentation_probs,
        seed=seed,
        normalize_fn=normalize_fn,
        oversample_labels=oversample_labels,
    )

    sampler = None
    shuffle = (split == "train")
    if balanced and split == "train":
        labels = [label for _, label in dataset.samples]
        groups = [("isl500" if p.name.startswith("isl500_") else "include") for p, _ in dataset.samples]
        sampler = BalancedBatchSampler(labels, batch_size, seed=seed, groups=groups)
        shuffle = False

    use_persistent = False if num_workers == 0 else (persistent_workers if persistent_workers is not None else True)
    pf = None if num_workers == 0 else prefetch_factor

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
        sampler=sampler,
        prefetch_factor=pf,
        persistent_workers=use_persistent,
    )
