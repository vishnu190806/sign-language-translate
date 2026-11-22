from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import ModelConfig
from src.models.htma import HTMAClassifier, HTMAConfig
from src.models.builder import build_model


class TestHTMAClassifier:
    def test_forward_shape(self):
        config = HTMAConfig(
            input_dim=168,
            embedding_dim=128,
            num_blocks=1,
            num_heads=4,
            num_classes=10,
        )
        model = HTMAClassifier(config)
        x = torch.randn(2, 32, 168)
        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_with_mask(self):
        config = HTMAConfig(input_dim=168, embedding_dim=128, num_blocks=1, num_heads=4, num_classes=5)
        model = HTMAClassifier(config)
        x = torch.randn(2, 32, 168)
        mask = torch.ones(2, 32, dtype=torch.bool)
        mask[:, 20:] = False
        out = model(x, mask)
        assert out.shape == (2, 5)

    def test_build_model(self):
        config = ModelConfig(
            name="htma",
            input_dim=168,
            embedding_dim=128,
            htma_blocks=1,
            num_heads=4,
            num_classes=10,
        )
        model = build_model(config)
        assert isinstance(model, HTMAClassifier)
        x = torch.randn(1, 32, 168)
        out = model(x)
        assert out.shape == (1, 10)

    def test_default_config_forward(self):
        config = HTMAConfig()
        model = HTMAClassifier(config)
        x = torch.randn(1, 32, 168)
        out = model(x)
        assert out.shape == (1, config.num_classes)

    def test_no_nan(self):
        config = HTMAConfig(input_dim=168, embedding_dim=64, num_blocks=1, num_heads=4, num_classes=5)
        model = HTMAClassifier(config)
        x = torch.randn(2, 10, 168)
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestEarlyFusion:
    def test_early_fusion_default(self):
        config = HTMAConfig(
            input_dim=168, embedding_dim=64, num_blocks=1, num_heads=4, num_classes=5,
            fusion_method="early",
        )
        model = HTMAClassifier(config)
        x = torch.randn(2, 16, 168)
        out = model(x)
        assert out.shape == (2, 5)

    def test_early_fusion_with_mask(self):
        config = HTMAConfig(
            input_dim=168, embedding_dim=64, num_blocks=1, num_heads=4, num_classes=5,
            fusion_method="early",
        )
        model = HTMAClassifier(config)
        x = torch.randn(2, 16, 168)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[:, 10:] = False
        out = model(x, mask)
        assert out.shape == (2, 5)

    def test_early_fusion_no_nan(self):
        config = HTMAConfig(
            input_dim=168, embedding_dim=64, num_blocks=1, num_heads=4, num_classes=5,
            fusion_method="early",
        )
        model = HTMAClassifier(config)
        x = torch.randn(2, 16, 168)
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestLateFusion:
    def test_late_fusion_shape(self):
        config = HTMAConfig(
            input_dim=168, embedding_dim=64, num_blocks=1, num_heads=4, num_classes=5,
            fusion_method="late", use_separate_encoders=True,
            hand_input_dim=128, pose_input_dim=40,
        )
        model = HTMAClassifier(config)
        x = torch.randn(2, 16, 168)
        out = model(x)
        assert out.shape == (2, 5)

    def test_late_fusion_with_mask(self):
        config = HTMAConfig(
            input_dim=168, embedding_dim=64, num_blocks=1, num_heads=4, num_classes=5,
            fusion_method="late", use_separate_encoders=True,
        )
        model = HTMAClassifier(config)
        x = torch.randn(2, 16, 168)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[:, 10:] = False
        out = model(x, mask)
        assert out.shape == (2, 5)

    def test_late_fusion_no_nan(self):
        config = HTMAConfig(
            input_dim=168, embedding_dim=64, num_blocks=2, num_heads=4, num_classes=5,
            fusion_method="late", use_separate_encoders=True,
        )
        model = HTMAClassifier(config)
        x = torch.randn(2, 16, 168)
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestMultiscaleFusion:
    def test_multiscale_shape(self):
        config = HTMAConfig(
            input_dim=168, embedding_dim=64, num_blocks=2, num_heads=4, num_classes=5,
            fusion_method="multiscale",
        )
        model = HTMAClassifier(config)
        x = torch.randn(2, 16, 168)
        out = model(x)
        assert out.shape == (2, 5)

    def test_multiscale_with_separate_encoders(self):
        config = HTMAConfig(
            input_dim=168, embedding_dim=64, num_blocks=2, num_heads=4, num_classes=5,
            fusion_method="multiscale", use_separate_encoders=True,
        )
        model = HTMAClassifier(config)
        x = torch.randn(2, 16, 168)
        out = model(x)
        assert out.shape == (2, 5)

    def test_multiscale_with_mask(self):
        config = HTMAConfig(
            input_dim=168, embedding_dim=64, num_blocks=2, num_heads=4, num_classes=5,
            fusion_method="multiscale",
        )
        model = HTMAClassifier(config)
        x = torch.randn(2, 16, 168)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[:, 10:] = False
        out = model(x, mask)
        assert out.shape == (2, 5)

    def test_multiscale_no_nan(self):
        config = HTMAConfig(
            input_dim=168, embedding_dim=64, num_blocks=2, num_heads=4, num_classes=5,
            fusion_method="multiscale",
        )
        model = HTMAClassifier(config)
        x = torch.randn(2, 16, 168)
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestBackwardCompatibility:
    def test_default_config_unchanged(self):
        config = HTMAConfig()
        assert config.fusion_method == "early"
        assert config.use_separate_encoders is False

    def test_default_produces_same_output_shape(self):
        config = HTMAConfig(input_dim=168, embedding_dim=128, num_blocks=1, num_heads=4, num_classes=10)
        model = HTMAClassifier(config)
        x = torch.randn(2, 32, 168)
        out = model(x)
        assert out.shape == (2, 10)

    def test_separate_encoders_disabled_by_default(self):
        config = HTMAConfig(input_dim=168, embedding_dim=64, num_blocks=1, num_heads=4, num_classes=5)
        model = HTMAClassifier(config)
        assert not hasattr(model, "hand_encoder") or not model.use_separate_encoders
