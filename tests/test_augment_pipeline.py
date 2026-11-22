from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.features.augment_pipeline import (
    AugmentationPipeline,
    create_augmentation_pipeline,
    visualize_augmentation,
    visualize_statistics,
)


@pytest.fixture
def sample_features():
    rng = np.random.default_rng(42)
    return rng.random((30, 168), dtype=np.float32)


@pytest.fixture
def pipeline():
    return AugmentationPipeline(seed=42)


class TestPipelineShapePreservation:
    def test_output_shape_matches_input(self, pipeline, sample_features):
        result = pipeline(sample_features, mode="train")
        assert result.shape == sample_features.shape

    def test_output_shape_various_lengths(self):
        pipeline = AugmentationPipeline(seed=42)
        for T in [1, 5, 60]:
            data = np.random.default_rng(0).random((T, 168), dtype=np.float32)
            result = pipeline(data, mode="train")
            assert result.shape == data.shape


class TestPipelineReproducibility:
    def test_same_seed_same_output(self, sample_features):
        r1 = AugmentationPipeline(seed=42)(sample_features)
        r2 = AugmentationPipeline(seed=42)(sample_features)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_different_output(self, sample_features):
        r1 = AugmentationPipeline(seed=1)(sample_features)
        r2 = AugmentationPipeline(seed=99)(sample_features)
        assert not np.array_equal(r1, r2)


class TestPipelineValMode:
    def test_val_returns_copy(self, pipeline, sample_features):
        result = pipeline(sample_features, mode="val")
        np.testing.assert_array_equal(result, sample_features)

    def test_test_returns_copy(self, pipeline, sample_features):
        result = pipeline(sample_features, mode="test")
        np.testing.assert_array_equal(result, sample_features)

    def test_val_does_not_modify_original(self, pipeline, sample_features):
        original_copy = sample_features.copy()
        pipeline(sample_features, mode="val")
        np.testing.assert_array_equal(sample_features, original_copy)

    def test_train_may_modify(self, pipeline, sample_features):
        pipeline.set_prob("gaussian_noise", 1.0)
        result = pipeline(sample_features, mode="train")
        assert not np.array_equal(result, sample_features)


class TestStatisticsTracking:
    def test_statistics_empty_initially(self, pipeline):
        stats = pipeline.get_statistics()
        assert stats == {}

    def test_statistics_recorded_after_augmentation(self, sample_features):
        pipeline = AugmentationPipeline(seed=42)
        pipeline.set_prob("gaussian_noise", 1.0)
        pipeline.set_prob("temporal_jitter", 1.0)
        pipeline(sample_features, mode="train")
        stats = pipeline.get_statistics()
        assert "gaussian_noise" in stats
        assert stats["gaussian_noise"]["count"] >= 1

    def test_reset_statistics(self, sample_features):
        pipeline = AugmentationPipeline(seed=42)
        pipeline.set_prob("gaussian_noise", 1.0)
        pipeline(sample_features, mode="train")
        pipeline.reset_statistics()
        stats = pipeline.get_statistics()
        assert stats == {}

    def test_statistics_severity_values(self, sample_features):
        pipeline = AugmentationPipeline(seed=42)
        pipeline.set_prob("gaussian_noise", 1.0)
        pipeline(sample_features, mode="train")
        stats = pipeline.get_statistics()
        assert "mean_severity" in stats["gaussian_noise"]
        assert "std_severity" in stats["gaussian_noise"]


class TestConfiguration:
    def test_set_prob(self, pipeline):
        pipeline.set_prob("gaussian_noise", 0.9)
        assert pipeline.probs["gaussian_noise"] == 0.9

    def test_set_prob_unknown_raises(self, pipeline):
        with pytest.raises(ValueError, match="Unknown augmentation"):
            pipeline.set_prob("nonexistent_aug", 0.5)

    def test_disable_augmentation(self, pipeline, sample_features):
        pipeline.disable("gaussian_noise")
        pipeline.disable("temporal_jitter")
        pipeline.disable("temporal_crop")
        pipeline.disable("temporal_dropout")
        pipeline.disable("random_hand_dropout")
        pipeline.disable("spatial_scale")
        pipeline.disable("random_translation")
        pipeline.disable("frame_permutation")
        pipeline.disable("speed_variation")
        pipeline.disable("random_frame_drop")
        pipeline.disable("spatial_rotation")
        pipeline.disable("frame_fusion")
        pipeline.disable("random_keypoint_removal")
        pipeline.disable("camera_depth_variation")
        pipeline.disable("hv_shift")
        pipeline.disable("hand_size_variation")
        pipeline.disable("viewpoint_rotation")
        pipeline.disable("finger_articulation")
        pipeline.disable("elbow_displacement")
        result = pipeline(sample_features, mode="train")
        np.testing.assert_array_equal(result, sample_features)

    def test_enable_restores_default_prob(self, pipeline):
        pipeline.disable("gaussian_noise")
        assert pipeline.probs["gaussian_noise"] == 0.0
        pipeline.enable("gaussian_noise")
        assert pipeline.probs["gaussian_noise"] == 0.5


class TestFactoryFunction:
    def test_create_with_defaults(self):
        pipeline = create_augmentation_pipeline({"seed": 99})
        assert isinstance(pipeline, AugmentationPipeline)
        assert pipeline.seed == 99

    def test_create_with_custom_probs(self):
        config = {"seed": 42, "probs": {"gaussian_noise": 0.9, "temporal_jitter": 0.1}}
        pipeline = create_augmentation_pipeline(config)
        assert pipeline.probs["gaussian_noise"] == 0.9
        assert pipeline.probs["temporal_jitter"] == 0.1

    def test_create_empty_config(self):
        pipeline = create_augmentation_pipeline({})
        assert isinstance(pipeline, AugmentationPipeline)


class TestVisualization:
    @patch("matplotlib.pyplot", create=True)
    def test_visualize_augmentation(self, mock_plt):
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        orig = np.random.default_rng(0).random((30, 168))
        aug = np.random.default_rng(1).random((30, 168))
        visualize_augmentation(orig, aug, "test title")
        mock_plt.subplots.assert_called_once_with(1, 2, figsize=(14, 5))
        mock_fig.suptitle.assert_called_once_with("test title")
        mock_plt.tight_layout.assert_called_once()
        mock_plt.show.assert_called_once()

    @patch("matplotlib.pyplot", create=True)
    def test_visualize_statistics(self, mock_plt):
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        stats = {
            "gaussian_noise": {"count": 5, "mean_severity": 0.4, "std_severity": 0.1},
            "temporal_jitter": {"count": 3, "mean_severity": 0.6, "std_severity": 0.2},
        }
        visualize_statistics(stats)
        mock_plt.subplots.assert_called_once_with(1, 2, figsize=(14, 5))
        mock_plt.tight_layout.assert_called_once()
        mock_plt.show.assert_called_once()

    @patch("matplotlib.pyplot", create=True)
    def test_visualize_statistics_empty(self, mock_plt):
        visualize_statistics({})
        mock_plt.subplots.assert_not_called()
