from __future__ import annotations

import numpy as np
import pytest

from src.features.spec import DEFAULT_FEATURE_SPEC
from src.features.normalize import (
    build_frame_features,
    normalize_hand_landmarks,
    SignerStatistics,
    SignerEstimator,
    PerSignerNormalizer,
)
from src.features.augment import LandmarkAugmentor


class TestFeatureSpec:
    def test_total_features(self):
        spec = DEFAULT_FEATURE_SPEC
        assert spec.total_features == 168

    def test_hand_features(self):
        spec = DEFAULT_FEATURE_SPEC
        assert spec.hand_features == (21 * 3 + 1) * 2  # 128

    def test_pose_features(self):
        spec = DEFAULT_FEATURE_SPEC
        assert spec.pose_features == 13 * 3 + 1  # 40


class TestNormalize:
    def test_hand_normalization_wrist_at_origin(self, spec, dummy_hand):
        normalized = normalize_hand_landmarks(dummy_hand, spec)
        wrist = normalized[:3]
        assert np.allclose(wrist, 0, atol=1e-6)

    def test_hand_normalization_scale(self, spec):
        hand = np.random.rand(spec.hand_landmarks, spec.hand_dims).astype(np.float32)
        hand *= 10.0
        normalized = normalize_hand_landmarks(hand, spec)
        norm_val = np.linalg.norm(normalized[9*3 : 9*3+3])
        assert np.isclose(norm_val, 1.0, atol=1e-4)

    def test_build_frame_features_shape(self, spec, dummy_hand, dummy_pose):
        features = build_frame_features(dummy_hand, dummy_hand, dummy_pose, spec)
        assert features.shape == (168,)

    def test_build_frame_features_missing_hand(self, spec, dummy_pose):
        features = build_frame_features(None, None, dummy_pose, spec)
        assert features.shape == (168,)
        assert np.all(features[:64] == 0)

    def test_build_frame_features_hand_presence_flag(self, spec, dummy_hand, dummy_pose):
        features = build_frame_features(dummy_hand, None, dummy_pose, spec)
        assert features[spec.hand_features_per_hand - 1] == 1.0
        assert features[spec.hand_features_per_hand * 2 - 1] == 0.0


class TestNewAugmentations:
    @pytest.fixture
    def augmentor(self):
        return LandmarkAugmentor()

    @pytest.fixture
    def sample_seq(self):
        rng = np.random.default_rng(42)
        return rng.random((20, 168), dtype=np.float32)

    def test_frame_permutation_preserves_shape(self, augmentor, sample_seq):
        rng = np.random.default_rng(42)
        out = augmentor.frame_permutation(sample_seq, rng=rng)
        assert out.shape == sample_seq.shape

    def test_frame_permutation_reproducible(self, augmentor, sample_seq):
        out1 = augmentor.frame_permutation(sample_seq, rng=np.random.default_rng(99))
        out2 = augmentor.frame_permutation(sample_seq, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(out1, out2)

    def test_frame_permutation_modifies_input(self, augmentor):
        rng = np.random.default_rng(7)
        seq = np.zeros((20, 168), dtype=np.float32)
        for t in range(20):
            seq[t, :] = float(t)
        base = rng.random(168, dtype=np.float32) * 0.0001
        for t in range(5, 15):
            seq[t, :] = base + float(t) * 0.0001
        out = augmentor.frame_permutation(seq, variation_threshold=0.5, min_segment=3, rng=rng)
        assert not np.array_equal(out, seq)

    def test_speed_variation_preserves_shape(self, augmentor, sample_seq):
        rng = np.random.default_rng(42)
        out = augmentor.speed_variation(sample_seq, rng=rng)
        assert out.shape == sample_seq.shape

    def test_speed_variation_reproducible(self, augmentor, sample_seq):
        out1 = augmentor.speed_variation(sample_seq, rng=np.random.default_rng(99))
        out2 = augmentor.speed_variation(sample_seq, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(out1, out2)

    def test_speed_variation_modifies_input(self, augmentor, sample_seq):
        out = augmentor.speed_variation(sample_seq, rng=np.random.default_rng(0))
        assert not np.array_equal(out, sample_seq)

    def test_random_frame_drop_preserves_shape(self, augmentor, sample_seq):
        rng = np.random.default_rng(42)
        out = augmentor.random_frame_drop(sample_seq, rng=rng)
        assert out.shape == sample_seq.shape

    def test_random_frame_drop_reproducible(self, augmentor, sample_seq):
        out1 = augmentor.random_frame_drop(sample_seq, rng=np.random.default_rng(99))
        out2 = augmentor.random_frame_drop(sample_seq, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(out1, out2)

    def test_random_frame_drop_modifies_input(self, augmentor, sample_seq):
        out = augmentor.random_frame_drop(sample_seq, rng=np.random.default_rng(0))
        assert not np.array_equal(out, sample_seq)

    def test_spatial_rotation_preserves_shape(self, augmentor, sample_seq):
        rng = np.random.default_rng(42)
        out = augmentor.spatial_rotation(sample_seq, rng=rng)
        assert out.shape == sample_seq.shape

    def test_spatial_rotation_reproducible(self, augmentor, sample_seq):
        out1 = augmentor.spatial_rotation(sample_seq, rng=np.random.default_rng(99))
        out2 = augmentor.spatial_rotation(sample_seq, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(out1, out2)

    def test_random_translation_updated_preserves_shape(self, augmentor, sample_seq):
        rng = np.random.default_rng(42)
        out = augmentor.random_translation(sample_seq, rng=rng)
        assert out.shape == sample_seq.shape

    def test_random_translation_shifts_all_landmarks(self, augmentor, sample_seq):
        rng = np.random.default_rng(42)
        out = augmentor.random_translation(sample_seq, rng=rng)
        hand_coords = 21 * 3
        assert not np.allclose(out[:, :hand_coords], sample_seq[:, :hand_coords])
        assert not np.allclose(out[:, 128:128 + 13 * 3], sample_seq[:, 128:128 + 13 * 3])

    def test_frame_fusion_preserves_shape(self, augmentor, sample_seq):
        rng = np.random.default_rng(42)
        out = augmentor.frame_fusion(sample_seq, rng=rng)
        assert out.shape == sample_seq.shape

    def test_frame_fusion_reproducible(self, augmentor, sample_seq):
        out1 = augmentor.frame_fusion(sample_seq, rng=np.random.default_rng(99))
        out2 = augmentor.frame_fusion(sample_seq, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(out1, out2)

    def test_frame_fusion_modifies_input(self, augmentor, sample_seq):
        out = augmentor.frame_fusion(sample_seq, rng=np.random.default_rng(0))
        assert not np.array_equal(out, sample_seq)

    def test_random_keypoint_removal_preserves_shape(self, augmentor, sample_seq):
        rng = np.random.default_rng(42)
        out = augmentor.random_keypoint_removal(sample_seq, rng=rng)
        assert out.shape == sample_seq.shape

    def test_random_keypoint_removal_reproducible(self, augmentor, sample_seq):
        out1 = augmentor.random_keypoint_removal(sample_seq, rng=np.random.default_rng(99))
        out2 = augmentor.random_keypoint_removal(sample_seq, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(out1, out2)

    def test_random_keypoint_removal_sets_presence_flags(self, augmentor):
        seq = np.ones((5, 168), dtype=np.float32)
        seq[:, 63] = 1.0
        seq[:, 127] = 1.0
        seq[:, 167] = 1.0
        rng = np.random.default_rng(42)
        out = augmentor.random_keypoint_removal(seq, removal_ratio=(0.3, 0.3), rng=rng)
        if np.any(out[:, :63] == 0):
            assert out[0, 63] == 0
        if np.any(out[:, 64:127] == 0):
            assert out[0, 127] == 0
        if np.any(out[:, 128:167] == 0):
            assert out[0, 167] == 0

    def test_probability_zero_returns_identity(self, augmentor, sample_seq):
        rng = np.random.default_rng(42)
        out = augmentor.frame_permutation(sample_seq, variation_threshold=0, rng=rng)
        np.testing.assert_array_equal(out, sample_seq)

    def test_all_augmentations_preserve_shape(self, augmentor, sample_seq):
        rng = np.random.default_rng(42)
        methods = [
            augmentor.frame_permutation,
            augmentor.speed_variation,
            augmentor.random_frame_drop,
            augmentor.spatial_rotation,
            augmentor.random_translation,
            augmentor.frame_fusion,
            augmentor.random_keypoint_removal,
        ]
        for method in methods:
            out = method(sample_seq, rng=np.random.default_rng(42))
            assert out.shape == sample_seq.shape, f"{method.__name__} changed shape"


class TestSignerStatistics:
    def test_compute_statistics_correct_mean(self):
        rng = np.random.default_rng(42)
        arr1 = rng.random((20, 168), dtype=np.float32)
        arr2 = rng.random((20, 168), dtype=np.float32)
        stats = SignerStatistics().compute_statistics([arr1, arr2])
        all_frames = np.concatenate([arr1, arr2], axis=0)
        expected_mean = all_frames.mean(axis=0)
        np.testing.assert_allclose(stats["mean"], expected_mean)

    def test_compute_statistics_correct_std(self):
        rng = np.random.default_rng(42)
        arr1 = rng.random((20, 168), dtype=np.float32)
        stats = SignerStatistics().compute_statistics([arr1])
        expected_std = arr1.std(axis=0)
        expected_std = np.where(expected_std < 1e-6, 1.0, expected_std)
        np.testing.assert_allclose(stats["std"], expected_std)

    def test_compute_statistics_clamps_small_std(self):
        arr = np.zeros((5, 168), dtype=np.float32)
        stats = SignerStatistics().compute_statistics([arr])
        assert np.all(stats["std"] == 1.0)

    def test_normalize_preserves_shape(self):
        rng = np.random.default_rng(42)
        features = rng.random((20, 168), dtype=np.float32)
        stats = {"mean": np.zeros(168, dtype=np.float32), "std": np.ones(168, dtype=np.float32)}
        result = SignerStatistics().normalize(features, stats)
        assert result.shape == features.shape

    def test_normalize_zero_mean_unity_std(self):
        rng = np.random.default_rng(42)
        features = rng.random((20, 168), dtype=np.float32)
        stats = {"mean": np.zeros(168, dtype=np.float32), "std": np.ones(168, dtype=np.float32)}
        result = SignerStatistics().normalize(features, stats)
        np.testing.assert_allclose(result, features)

    def test_normalize_shifts_mean(self):
        features = np.ones((10, 168), dtype=np.float32) * 5.0
        stats = {"mean": np.ones(168, dtype=np.float32) * 3.0, "std": np.ones(168, dtype=np.float32)}
        result = SignerStatistics().normalize(features, stats)
        np.testing.assert_allclose(result, 2.0)


class TestSignerEstimator:
    def test_estimates_correct_signer(self):
        rng = np.random.default_rng(42)
        signer_a = rng.random((50, 168), dtype=np.float32)
        signer_b = rng.random((50, 168), dtype=np.float32) + 10.0
        stats = SignerStatistics()
        signer_stats = {
            "signer_a": stats.compute_statistics([signer_a]),
            "signer_b": stats.compute_statistics([signer_b]),
        }
        estimator = SignerEstimator()
        query = signer_a[:3].copy()
        result = estimator.estimate_signer(query, signer_stats)
        assert result == "signer_a"

    def test_estimates_wrong_signer_with_large_offset(self):
        rng = np.random.default_rng(42)
        signer_a = rng.random((50, 168), dtype=np.float32)
        signer_b = rng.random((50, 168), dtype=np.float32) + 100.0
        stats = SignerStatistics()
        signer_stats = {
            "signer_a": stats.compute_statistics([signer_a]),
            "signer_b": stats.compute_statistics([signer_b]),
        }
        estimator = SignerEstimator()
        query = signer_b[:3].copy()
        result = estimator.estimate_signer(query, signer_stats)
        assert result == "signer_b"


class TestPerSignerNormalizer:
    def test_disabled_returns_features_unchanged(self):
        rng = np.random.default_rng(42)
        features = rng.random((20, 168), dtype=np.float32)
        normalizer = PerSignerNormalizer(use_per_signer=False)
        result = normalizer.normalize(features, signer_stats={"a": {"mean": np.zeros(168), "std": np.ones(168)}})
        np.testing.assert_array_equal(result, features)

    def test_no_signer_stats_returns_features_unchanged(self):
        rng = np.random.default_rng(42)
        features = rng.random((20, 168), dtype=np.float32)
        normalizer = PerSignerNormalizer(use_per_signer=True)
        result = normalizer.normalize(features, signer_stats=None)
        np.testing.assert_array_equal(result, features)

    def test_enabled_with_stats_applies_normalization(self):
        rng = np.random.default_rng(42)
        features = rng.random((20, 168), dtype=np.float32) * 10.0
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        stats = {"signer_0": {"mean": mean, "std": std}}
        normalizer = PerSignerNormalizer(use_per_signer=True)
        result = normalizer.normalize(features, signer_stats=stats, signer_id="signer_0")
        np.testing.assert_allclose(result, (features - mean) / std)

    def test_auto_estimates_signer(self):
        rng = np.random.default_rng(42)
        signer_a = rng.random((50, 168), dtype=np.float32)
        signer_b = rng.random((50, 168), dtype=np.float32) + 20.0
        stats_cls = SignerStatistics()
        signer_stats = {
            "a": stats_cls.compute_statistics([signer_a]),
            "b": stats_cls.compute_statistics([signer_b]),
        }
        normalizer = PerSignerNormalizer(use_per_signer=True)
        query = signer_a[:3].copy()
        result = normalizer.normalize(query, signer_stats=signer_stats)
        expected = (query - signer_stats["a"]["mean"]) / signer_stats["a"]["std"]
        np.testing.assert_allclose(result, expected)

    def test_preserves_shape(self):
        rng = np.random.default_rng(42)
        features = rng.random((30, 168), dtype=np.float32)
        mean = np.zeros(168, dtype=np.float32)
        std = np.ones(168, dtype=np.float32)
        stats = {"s": {"mean": mean, "std": std}}
        normalizer = PerSignerNormalizer(use_per_signer=True)
        result = normalizer.normalize(features, signer_stats=stats, signer_id="s")
        assert result.shape == features.shape


class TestBuildFrameFeaturesBackwardCompatibility:
    def test_no_normalizer_works(self, spec, dummy_hand, dummy_pose):
        features = build_frame_features(dummy_hand, dummy_hand, dummy_pose, spec)
        assert features.shape == (168,)

    def test_with_per_signer_normalizer(self, spec, dummy_hand, dummy_pose):
        normalizer = PerSignerNormalizer(use_per_signer=False)
        features = build_frame_features(
            dummy_hand, dummy_hand, dummy_pose, spec,
            per_signer_normalizer=normalizer,
        )
        assert features.shape == (168,)
