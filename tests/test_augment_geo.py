from __future__ import annotations

import numpy as np
import pytest

from src.features.augment_geo import GeometryAugmentor
from src.features.spec import DEFAULT_FEATURE_SPEC


@pytest.fixture
def sample_features():
    rng = np.random.default_rng(42)
    return rng.random((30, 168), dtype=np.float32)


@pytest.fixture
def augmentor():
    return GeometryAugmentor(seed=42)


class TestCameraDepthVariation:
    def test_preserves_shape(self, augmentor, sample_features):
        result = augmentor.camera_depth_variation(sample_features)
        assert result.shape == sample_features.shape

    def test_reproducible(self, sample_features):
        r1 = GeometryAugmentor(seed=42).camera_depth_variation(sample_features)
        r2 = GeometryAugmentor(seed=42).camera_depth_variation(sample_features)
        np.testing.assert_array_equal(r1, r2)

    def test_modifies_input(self, augmentor, sample_features):
        result = augmentor.camera_depth_variation(sample_features)
        assert not np.array_equal(result, sample_features)

    def test_only_z_modified(self, sample_features):
        aug = GeometryAugmentor(seed=42)
        result = aug.camera_depth_variation(
            sample_features, scale_range=(0.5, 0.5), rng=np.random.default_rng(0)
        )
        x_idx = np.concatenate([
            np.arange(0, 63, 3), np.arange(64, 127, 3), np.arange(128, 167, 3)
        ])
        y_idx = x_idx + 1
        np.testing.assert_array_equal(result[:, x_idx], sample_features[:, x_idx])
        np.testing.assert_array_equal(result[:, y_idx], sample_features[:, y_idx])


class TestHVShift:
    def test_preserves_shape(self, augmentor, sample_features):
        result = augmentor.hv_shift(sample_features)
        assert result.shape == sample_features.shape

    def test_reproducible(self, sample_features):
        r1 = GeometryAugmentor(seed=42).hv_shift(sample_features)
        r2 = GeometryAugmentor(seed=42).hv_shift(sample_features)
        np.testing.assert_array_equal(r1, r2)

    def test_modifies_input(self, augmentor, sample_features):
        result = augmentor.hv_shift(sample_features)
        assert not np.array_equal(result, sample_features)


class TestHandSizeVariation:
    def test_preserves_shape(self, augmentor, sample_features):
        result = augmentor.hand_size_variation(sample_features)
        assert result.shape == sample_features.shape

    def test_reproducible(self, sample_features):
        r1 = GeometryAugmentor(seed=42).hand_size_variation(sample_features)
        r2 = GeometryAugmentor(seed=42).hand_size_variation(sample_features)
        np.testing.assert_array_equal(r1, r2)

    def test_only_affects_hands(self, augmentor, sample_features):
        result = augmentor.hand_size_variation(sample_features)
        pose_start = DEFAULT_FEATURE_SPEC.hand_features
        np.testing.assert_array_equal(result[:, pose_start:], sample_features[:, pose_start:])

    def test_wrist_unchanged(self, augmentor, sample_features):
        result = augmentor.hand_size_variation(
            sample_features, scale_range=(0.5, 0.5), rng=np.random.default_rng(0)
        )
        for h in range(2):
            start = h * DEFAULT_FEATURE_SPEC.hand_features_per_hand
            np.testing.assert_array_equal(
                result[:, start:start + 3], sample_features[:, start:start + 3]
            )


class TestViewpointRotation:
    def test_preserves_shape(self, augmentor, sample_features):
        result = augmentor.viewpoint_rotation(sample_features)
        assert result.shape == sample_features.shape

    def test_reproducible(self, sample_features):
        r1 = GeometryAugmentor(seed=42).viewpoint_rotation(sample_features)
        r2 = GeometryAugmentor(seed=42).viewpoint_rotation(sample_features)
        np.testing.assert_array_equal(r1, r2)

    def test_zero_angle_identity(self, augmentor, sample_features):
        result = augmentor.viewpoint_rotation(
            sample_features, angle_range_deg=(0.0, 0.0), rng=np.random.default_rng(0)
        )
        np.testing.assert_array_almost_equal(result, sample_features)

    def test_rotation_matrix_90deg(self, sample_features):
        result = GeometryAugmentor(seed=42).viewpoint_rotation(
            sample_features, angle_range_deg=(90.0, 90.0), rng=np.random.default_rng(0)
        )
        for x_i, y_i in [(0, 1), (3, 4), (64, 65), (128, 129)]:
            np.testing.assert_array_almost_equal(
                result[:, x_i], -sample_features[:, y_i], decimal=5
            )
            np.testing.assert_array_almost_equal(
                result[:, y_i], sample_features[:, x_i], decimal=5
            )


class TestFingerArticulation:
    def test_preserves_shape(self, augmentor, sample_features):
        result = augmentor.finger_articulation(sample_features)
        assert result.shape == sample_features.shape

    def test_reproducible(self, sample_features):
        r1 = GeometryAugmentor(seed=42).finger_articulation(sample_features)
        r2 = GeometryAugmentor(seed=42).finger_articulation(sample_features)
        np.testing.assert_array_equal(r1, r2)

    def test_zero_angle_identity(self, augmentor, sample_features):
        result = augmentor.finger_articulation(
            sample_features, angle_range_deg=(0.0, 0.0), rng=np.random.default_rng(0)
        )
        np.testing.assert_array_almost_equal(result, sample_features)

    def test_modifies_input(self, augmentor, sample_features):
        result = augmentor.finger_articulation(sample_features)
        assert not np.array_equal(result, sample_features)


class TestElbowDisplacement:
    def test_preserves_shape(self, augmentor, sample_features):
        result = augmentor.elbow_displacement(sample_features)
        assert result.shape == sample_features.shape

    def test_reproducible(self, sample_features):
        r1 = GeometryAugmentor(seed=42).elbow_displacement(sample_features)
        r2 = GeometryAugmentor(seed=42).elbow_displacement(sample_features)
        np.testing.assert_array_equal(r1, r2)

    def test_only_affects_hands(self, augmentor, sample_features):
        result = augmentor.elbow_displacement(sample_features)
        pose_start = DEFAULT_FEATURE_SPEC.hand_features
        np.testing.assert_array_equal(result[:, pose_start:], sample_features[:, pose_start:])

    def test_modifies_input(self, augmentor, sample_features):
        result = augmentor.elbow_displacement(sample_features)
        hand_end = DEFAULT_FEATURE_SPEC.hand_features
        assert not np.array_equal(result[:, :hand_end], sample_features[:, :hand_end])


class TestGeometryAugmentorCall:
    def test_preserves_shape(self, augmentor, sample_features):
        result = augmentor(sample_features)
        assert result.shape == sample_features.shape

    def test_reproducible(self, sample_features):
        r1 = GeometryAugmentor(seed=42)(sample_features)
        r2 = GeometryAugmentor(seed=42)(sample_features)
        np.testing.assert_array_equal(r1, r2)

    def test_probability_zero_no_augmentation(self, sample_features):
        aug = GeometryAugmentor(seed=42)
        fake_rng = _FakeRNG([1.0] * 10)
        result = aug(sample_features, rng=fake_rng)
        np.testing.assert_array_equal(result, sample_features)

    def test_probability_one_always_augment(self, sample_features):
        aug = GeometryAugmentor(seed=42)
        fake_rng = _FakeRNG([0.0] * 10, uniform_val=0.25)
        result = aug(sample_features, rng=fake_rng)
        assert not np.array_equal(result, sample_features)


class _FakeRNG:
    def __init__(self, random_values: list[float], uniform_val: float = 0.0):
        self._values = iter(random_values)
        self._uniform_val = uniform_val

    def random(self) -> float:
        return next(self._values)

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self._uniform_val
