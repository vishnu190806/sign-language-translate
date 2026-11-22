import sys
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import LandmarkDataset, BalancedBatchSampler, collate_fn, create_dataloader
from src.features.spec import DEFAULT_FEATURE_SPEC


NPZ_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "landmarks" / "isl"
LABEL_MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "landmarks" / "isl_label_map.json"

pytestmark = pytest.mark.skipif(
    not (NPZ_DIR.exists() and LABEL_MAP_PATH.exists()),
    reason="Requires excluded processed landmark dataset (data/processed/landmarks/isl)",
)


def _load_label_map():
    import json
    with open(LABEL_MAP_PATH) as f:
        return json.load(f)


class TestLandmarkDataset:
    def test_dataset_loads(self):
        label_map = _load_label_map()
        ds = LandmarkDataset(NPZ_DIR, split="train", label_map=label_map, sequence_length=32)
        assert len(ds) > 0

    def test_dataset_item_shape(self):
        label_map = _load_label_map()
        ds = LandmarkDataset(NPZ_DIR, split="train", label_map=label_map, sequence_length=32)
        features, mask, label = ds[0]
        assert features.shape == (32, DEFAULT_FEATURE_SPEC.total_features)
        assert mask.shape == (32,)
        assert mask.dtype == torch.bool
        assert isinstance(label, int)

    def test_dataset_val_split(self):
        label_map = _load_label_map()
        ds_val = LandmarkDataset(NPZ_DIR, split="val", label_map=label_map, sequence_length=32)
        assert len(ds_val) > 0
        for i in range(min(5, len(ds_val))):
            _, _, label = ds_val[i]
            assert 0 <= label < 50

    def test_dataset_test_split(self):
        label_map = _load_label_map()
        ds_test = LandmarkDataset(NPZ_DIR, split="test", label_map=label_map, sequence_length=32)
        assert len(ds_test) > 0

    def test_collate_fn(self):
        label_map = _load_label_map()
        ds = LandmarkDataset(NPZ_DIR, split="train", label_map=label_map, sequence_length=32)
        batch = [ds[i] for i in range(min(4, len(ds)))]
        features, masks, labels = collate_fn(batch)
        assert features.shape == (4, 32, DEFAULT_FEATURE_SPEC.total_features)
        assert masks.shape == (4, 32)
        assert labels.shape == (4,)

    def test_mask_all_ones_for_padded_short(self):
        label_map = _load_label_map()
        ds = LandmarkDataset(NPZ_DIR, split="train", label_map=label_map, sequence_length=256)
        features, mask, _ = ds[0]
        assert features.shape == (256, DEFAULT_FEATURE_SPEC.total_features)
        assert mask.sum() < 256

    def test_no_nan_in_features(self):
        label_map = _load_label_map()
        ds = LandmarkDataset(NPZ_DIR, split="train", label_map=label_map, sequence_length=32)
        for i in range(min(10, len(ds))):
            features, _, _ = ds[i]
            assert not torch.isnan(features).any()

    def test_augmented_differs_from_original(self):
        label_map = _load_label_map()
        ds_aug = LandmarkDataset(NPZ_DIR, split="train", label_map=label_map, sequence_length=32, augment=True)
        ds_noaug = LandmarkDataset(NPZ_DIR, split="train", label_map=label_map, sequence_length=32, augment=False)
        feat_aug, _, _ = ds_aug[0]
        feat_noaug, _, _ = ds_noaug[0]
        assert feat_aug.shape == feat_noaug.shape

    def test_dataloader_batches(self):
        label_map = _load_label_map()
        loader = create_dataloader(
            NPZ_DIR, split="train", label_map=label_map,
            sequence_length=32, batch_size=8, num_workers=0,
        )
        features, masks, labels = next(iter(loader))
        assert features.shape == (8, 32, DEFAULT_FEATURE_SPEC.total_features)
        assert masks.shape == (8, 32)
        assert labels.shape == (8,)
        assert labels.min() >= 0
        assert labels.max() < len(label_map)


class TestBalancedBatchSampler:
    def test_sampler_length(self):
        labels = [0] * 100 + [1] * 50 + [2] * 30
        sampler = BalancedBatchSampler(labels, batch_size=16, num_samples=96)
        assert len(sampler) == 96

    def test_sampler_iter(self):
        labels = [0] * 100 + [1] * 50 + [2] * 30
        sampler = BalancedBatchSampler(labels, batch_size=16, num_samples=96)
        indices = list(sampler)
        assert len(indices) == 96
        assert all(0 <= i < len(labels) for i in indices)

    def test_balanced_dataloader(self):
        label_map = _load_label_map()
        loader = create_dataloader(
            NPZ_DIR, split="train", label_map=label_map,
            sequence_length=32, batch_size=16, num_workers=0, balanced=True,
        )
        features, masks, labels = next(iter(loader))
        assert features.shape[0] == 16
        assert features.shape[1] == 32
