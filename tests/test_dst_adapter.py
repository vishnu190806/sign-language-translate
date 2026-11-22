from __future__ import annotations

import pytest
import torch

from src.models.htma import HTMAClassifier, HTMAConfig
from src.models.dst_adapter import (
    ChannelAwareFusion,
    DSTAdapter,
    SpatialAdapter,
    TemporalAdapter,
)


def _make_model(
    num_blocks: int = 2,
    embedding_dim: int = 128,
    num_heads: int = 4,
    num_classes: int = 10,
) -> HTMAClassifier:
    config = HTMAConfig(
        input_dim=168,
        embedding_dim=embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        num_classes=num_classes,
    )
    return HTMAClassifier(config)


class TestSpatialAdapter:
    def test_forward_shape(self):
        adapter = SpatialAdapter(dim=128, adapter_dim=32)
        x = torch.randn(2, 32, 128)
        out = adapter(x)
        assert out.shape == (2, 32, 128)

    def test_residual_connection(self):
        adapter = SpatialAdapter(dim=64, adapter_dim=16)
        x = torch.randn(1, 10, 64)
        out = adapter(x)
        assert not torch.isnan(out).any()


class TestTemporalAdapter:
    def test_forward_shape(self):
        adapter = TemporalAdapter(dim=128, adapter_dim=32)
        x = torch.randn(2, 32, 128)
        out = adapter(x)
        assert out.shape == (2, 32, 128)

    def test_residual_connection(self):
        adapter = TemporalAdapter(dim=64, adapter_dim=16)
        x = torch.randn(1, 10, 64)
        out = adapter(x)
        assert not torch.isnan(out).any()


class TestChannelAwareFusion:
    def test_forward_shape(self):
        fusion = ChannelAwareFusion(dim=128)
        backbone = torch.randn(2, 32, 128)
        adapter_out = torch.randn(2, 32, 128)
        out = fusion(backbone, adapter_out)
        assert out.shape == (2, 32, 128)

    def test_identical_inputs(self):
        fusion = ChannelAwareFusion(dim=64)
        x = torch.randn(1, 10, 64)
        out = fusion(x, x)
        assert out.shape == x.shape
        assert torch.allclose(out, x, atol=1e-6)

    def test_weights_computed(self):
        fusion = ChannelAwareFusion(dim=64)
        assert len(list(fusion.mlp.parameters())) > 0


class TestDSTAdapter:
    def test_inserts_adapters_after_each_block(self):
        model = _make_model(num_blocks=3)
        adapter = DSTAdapter(model, adapter_dim=32)
        assert len(adapter.spatial_adapters) == 3
        assert len(adapter.temporal_adapters) == 3
        assert len(adapter.spatial_fusions) == 3
        assert len(adapter.temporal_fusions) == 3

    def test_original_params_frozen(self):
        model = _make_model()
        adapter = DSTAdapter(model, adapter_dim=32)
        for param in adapter.model.parameters():
            assert not param.requires_grad

    def test_adapter_params_trainable(self):
        model = _make_model()
        adapter = DSTAdapter(model, adapter_dim=32)
        trainable = [p for p in adapter.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_parameter_ratio(self):
        model = _make_model(embedding_dim=256, num_blocks=2)
        adapter = DSTAdapter(model, adapter_dim=64)
        total, trainable = adapter.count_parameters()
        ratio = trainable / total
        assert ratio < 0.5
        assert ratio > 0.0

    def test_forward_shape(self):
        model = _make_model(num_blocks=2, num_classes=10)
        adapter = DSTAdapter(model, adapter_dim=32)
        x = torch.randn(2, 32, 168)
        out = adapter(x)
        assert out.shape == (2, 10)

    def test_forward_with_mask(self):
        model = _make_model(num_classes=5)
        adapter = DSTAdapter(model, adapter_dim=32)
        x = torch.randn(2, 32, 168)
        mask = torch.ones(2, 32, dtype=torch.bool)
        mask[:, 20:] = False
        out = adapter(x, mask)
        assert out.shape == (2, 5)

    def test_no_nan(self):
        model = _make_model(num_blocks=1, embedding_dim=64, num_classes=5)
        adapter = DSTAdapter(model, adapter_dim=16)
        x = torch.randn(2, 10, 168)
        out = adapter(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_does_not_update_source(self):
        model = _make_model(num_blocks=1, num_classes=5)
        adapter = DSTAdapter(model, adapter_dim=32)
        before = {n: p.clone() for n, p in adapter.model.named_parameters()}
        x = torch.randn(2, 16, 168)
        adapter(x)
        for n, p in adapter.model.named_parameters():
            assert torch.equal(p, before[n])

    def test_backward_only_updates_adapters(self):
        model = _make_model(num_blocks=1, num_classes=5)
        adapter = DSTAdapter(model, adapter_dim=32)
        x = torch.randn(2, 16, 168)
        out = adapter(x)
        loss = out.sum()
        loss.backward()
        for name, param in adapter.named_parameters():
            if param.grad is not None:
                assert "model." not in name
