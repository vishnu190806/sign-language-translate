import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import BalancedBatchSampler
from src.features.augment import LandmarkAugmentor
from src.features.spec import DEFAULT_FEATURE_SPEC
from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier


class TestMultiHead:
    def _model(self):
        return MultiHeadHTMAClassifier(HTMAConfig(input_dim=168, embedding_dim=64, num_blocks=1,
                                                  num_heads=4, num_classes=10))

    def test_default_forward(self):
        m = self._model()
        assert m(torch.randn(2, 32, 168)).shape == (2, 10)

    def test_per_source_forward(self):
        m = self._model()
        x = torch.randn(2, 32, 168)
        for s in m.sources:
            assert m(x, source=s).shape == (2, 10)

    def test_heads_differ_after_init_copy(self):
        m = self._model()
        with torch.no_grad():
            for p in m.heads["isl500"].parameters():
                p.add_(1.0)
        x = torch.randn(2, 32, 168)
        assert not torch.allclose(m(x, source="include"), m(x, source="isl500"))

    def test_no_nan_masked(self):
        m = self._model()
        x = torch.randn(2, 32, 168)
        mask = torch.ones(2, 32, dtype=torch.bool)
        mask[:, 20:] = False
        out = m(x, mask, source="include")
        assert not torch.isnan(out).any()

    def test_state_dict_roundtrip(self):
        m = self._model().eval()
        x = torch.randn(1, 32, 168)
        with torch.no_grad():
            before = m(x, source="isl500")
        m2 = self._model().eval()
        m2.load_state_dict(m.state_dict())
        with torch.no_grad():
            assert torch.allclose(before, m2(x, source="isl500"))


class TestGroupedSampler:
    def test_groups_cover_both_sources(self):
        labels = [0] * 20 + [1] * 20
        groups = ["a"] * 10 + ["b"] * 10 + ["a"] * 10 + ["b"] * 10
        s = BalancedBatchSampler(labels, batch_size=8, num_samples=40, groups=groups)
        assert s.num_classes == 4
        idx = list(s)
        assert len(idx) == 40
        seen = {(labels[i], groups[i]) for i in idx}
        assert seen == {(0, "a"), (0, "b"), (1, "a"), (1, "b")}

    def test_no_groups_backward_compat(self):
        s = BalancedBatchSampler([0] * 10 + [1] * 10, batch_size=8, num_samples=20)
        assert s.num_classes == 2
        assert len(list(s)) == 20


class TestPoseDropout:
    def test_full_dropout_zeroes_pose(self):
        aug = LandmarkAugmentor(DEFAULT_FEATURE_SPEC)
        x = np.random.default_rng(0).normal(size=(32, 168)).astype(np.float32)
        out = aug.random_pose_dropout(x, dropout_prob=1.0)
        assert out.shape == x.shape
        assert (out[:, 128:] == 0).all()

    def test_zero_dropout_identity(self):
        aug = LandmarkAugmentor(DEFAULT_FEATURE_SPEC)
        x = np.random.default_rng(1).normal(size=(32, 168)).astype(np.float32)
        assert np.array_equal(aug.random_pose_dropout(x, dropout_prob=0.0), x)

    def test_pipeline_runs(self):
        aug = LandmarkAugmentor(DEFAULT_FEATURE_SPEC)
        x = np.random.default_rng(2).normal(size=(32, 168)).astype(np.float32)
        out = aug(x, np.random.default_rng(3))
        assert out.shape == x.shape
        assert not np.isnan(out).any()
