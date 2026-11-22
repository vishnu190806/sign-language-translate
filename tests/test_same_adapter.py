from __future__ import annotations

import pytest
import torch

from src.models.htma import HTMAClassifier, HTMAConfig
from src.models.same_adapter import MoELayer, SAMEAdapter


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


class TestMoELayer:
    def test_forward_shape(self):
        layer = MoELayer(dim=128, num_experts=4, bottleneck_dim=64)
        x = torch.randn(2, 32, 128)
        signer = torch.randn(2, 6)
        out = layer(x, signer)
        assert out.shape == (2, 32, 128)

    def test_expert_count(self):
        layer = MoELayer(dim=64, num_experts=3, bottleneck_dim=32)
        assert len(layer.experts) == 3


class TestSAMEAdapter:
    def test_inserts_moe_after_each_block(self):
        model = _make_model(num_blocks=3)
        adapter = SAMEAdapter(model, num_experts=2)
        assert len(adapter.moe_layers) == 3

    def test_original_params_frozen(self):
        model = _make_model()
        adapter = SAMEAdapter(model)
        for param in adapter.model.parameters():
            assert not param.requires_grad

    def test_adapter_params_trainable(self):
        model = _make_model()
        adapter = SAMEAdapter(model)
        trainable = [p for p in adapter.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_forward_shape(self):
        model = _make_model(num_blocks=2, num_classes=10)
        adapter = SAMEAdapter(model, num_experts=2)
        x = torch.randn(2, 32, 168)
        out = adapter(x)
        assert out.shape == (2, 10)

    def test_forward_with_mask(self):
        model = _make_model(num_classes=5)
        adapter = SAMEAdapter(model, num_experts=2)
        x = torch.randn(2, 32, 168)
        mask = torch.ones(2, 32, dtype=torch.bool)
        mask[:, 20:] = False
        out = adapter(x, mask)
        assert out.shape == (2, 5)

    def test_adapt_runs(self):
        model = _make_model(num_blocks=1, num_classes=5)
        adapter = SAMEAdapter(model, num_experts=2)
        x = torch.randn(2, 16, 168)
        out = adapter.adapt(x, num_steps=2)
        assert out.shape == (2, 5)
        assert not torch.isnan(out).any()


class TestSAMESourceRouting:
    def _make_multihead(self):
        from src.models.htma import MultiHeadHTMAClassifier

        config = HTMAConfig(input_dim=168, embedding_dim=64, num_blocks=1,
                            num_heads=4, num_classes=10)
        return MultiHeadHTMAClassifier(config, sources=("include", "isl500"))

    def test_forward_routes_to_source_head(self):
        model = self._make_multihead()
        adapter = SAMEAdapter(model, num_experts=2)
        x = torch.randn(2, 16, 168)
        with torch.no_grad():
            for p in adapter.model.heads["isl500"].parameters():
                p.add_(1.0)
        out_inc = adapter(x, source="include")
        out_isl = adapter(x, source="isl500")
        assert out_inc.shape == (2, 10)
        assert not torch.allclose(out_inc, out_isl)

    def test_adapt_accepts_mask_lr_weights(self):
        model = self._make_multihead()
        adapter = SAMEAdapter(model, num_experts=2)
        x = torch.randn(2, 16, 168)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[:, 10:] = False
        out = adapter.adapt(x, mask=mask, source="isl500", num_steps=2,
                            lr=3e-5, w_ent=1.0, w_mcc=0.5, w_kl=2.0)
        assert out.shape == (2, 10)
        assert not torch.isnan(out).any()

    def test_reset_restores_snapshot(self):
        model = self._make_multihead()
        adapter = SAMEAdapter(model, num_experts=2)
        x = torch.randn(2, 16, 168)
        snap = adapter.snapshot()
        adapter.adapt(x, source="isl500", num_steps=2)
        changed = any(not torch.equal(a, b) for (_, a), (_, b) in
                      zip(adapter.moe_layers.state_dict().items(), snap.items()))
        assert changed
        adapter.reset(snap)
        for (_, a), (_, b) in zip(adapter.moe_layers.state_dict().items(), snap.items()):
            assert torch.equal(a, b)

    def test_device_mismatch_raises(self):
        model = self._make_multihead()
        adapter = SAMEAdapter(model, num_experts=2)
        x = torch.randn(2, 16, 168)
        if torch.cuda.is_available():
            adapter = adapter.cuda()
            with pytest.raises(AssertionError):
                adapter.adapt(x)
        else:
            out = adapter.adapt(x, num_steps=1)
            assert out.shape == (2, 10)

    def test_adapt_does_not_update_source_params(self):
        model = _make_model(num_blocks=1, num_classes=5)
        adapter = SAMEAdapter(model, num_experts=2)
        x = torch.randn(2, 16, 168)
        before = {n: p.clone() for n, p in model.named_parameters()}
        adapter.adapt(x, num_steps=3)
        for n, p in model.named_parameters():
            assert torch.equal(p, before[n])

    def test_signer_features_shape(self):
        model = _make_model()
        adapter = SAMEAdapter(model)
        x = torch.randn(4, 20, 168)
        feats = adapter._extract_signer_features(x)
        assert feats.shape == (4, 6)

    def test_signer_features_values(self):
        model = _make_model()
        adapter = SAMEAdapter(model)
        x = torch.zeros(1, 10, 168)
        x[:, :, 0:64] = 2.0
        feats = adapter._extract_signer_features(x)
        assert feats.shape == (1, 6)
        assert feats[0, 0].item() == pytest.approx(2.0, abs=1e-5)

    def test_unsupervised_losses(self):
        model = _make_model(num_blocks=1, num_classes=5)
        adapter = SAMEAdapter(model, num_experts=2)
        x = torch.randn(2, 16, 168)

        with torch.no_grad():
            source_logits = adapter.model(x)
            source_probs = torch.softmax(source_logits, dim=-1)

        logits = adapter(x)
        probs = torch.softmax(logits, dim=-1)

        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.mean(torch.sum(probs * log_probs, dim=-1))
        assert entropy.item() > 0

        q = probs**2
        q = q / (q.sum(dim=0, keepdim=True) + 1e-8)
        mcc = -torch.sum(torch.sqrt(probs * q + 1e-8))
        assert mcc.item() < 0

        kl = torch.nn.functional.kl_div(log_probs, source_probs, reduction="batchmean")
        assert kl.item() >= 0
