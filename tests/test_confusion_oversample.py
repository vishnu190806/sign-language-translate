from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import LandmarkDataset, create_dataloader


def _make_npz(tmp_path, name: str, label: str, split: str, label_idx: int = 0):
    import numpy as np

    p = tmp_path / name
    feats = np.random.randn(10, 168).astype(np.float32)
    np.savez_compressed(p, features=feats, label=label, split=split)
    return p


@pytest.fixture
def tiny_dir(tmp_path):
    for i, lab in enumerate(["a", "b", "c"]):
        _make_npz(tmp_path, f"f{i}_{i}.npz", lab, "train")
    return tmp_path


def _label_map():
    return {"a": 0, "b": 1, "c": 2}


def test_oversample_repeats_targets(tiny_dir):
    ds = LandmarkDataset(str(tiny_dir), split="train", label_map=_label_map(),
                         oversample_labels={1: 3}, use_manifest=False)
    labels = [l for _, l in ds.samples]
    assert len(ds) == 5
    assert labels.count(1) == 3
    assert labels.count(0) == 1 and labels.count(2) == 1


def test_oversample_gated_to_train(tiny_dir):
    ds = LandmarkDataset(str(tiny_dir), split="val", label_map=_label_map(),
                         oversample_labels={1: 3}, use_manifest=False)
    assert len(ds) == 0
    _make_npz(tiny_dir, "v0_1.npz", "b", "val")
    ds = LandmarkDataset(str(tiny_dir), split="val", label_map=_label_map(),
                         oversample_labels={1: 3}, use_manifest=False)
    assert len(ds) == 1


def test_oversample_unknown_idx_ignored(tiny_dir):
    ds = LandmarkDataset(str(tiny_dir), split="train", label_map=_label_map(),
                         oversample_labels={99: 5}, use_manifest=False)
    assert len(ds) == 3


def test_oversample_deterministic_order(tiny_dir):
    kw = dict(split="train", label_map=_label_map(), oversample_labels={0: 2, 2: 2},
              use_manifest=False)
    a = LandmarkDataset(str(tiny_dir), **kw)
    b = LandmarkDataset(str(tiny_dir), **kw)
    assert [str(p) for p, _ in a.samples] == [str(p) for p, _ in b.samples]
    assert len(a) == 5


def test_dataloader_passthrough(tiny_dir):
    dl = create_dataloader(str(tiny_dir), split="train", label_map=_label_map(),
                           batch_size=2, num_workers=0, augment=False,
                           oversample_labels={2: 2})
    assert len(dl.dataset) == 4
