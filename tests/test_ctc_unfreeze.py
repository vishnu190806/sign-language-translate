from __future__ import annotations

from pathlib import Path

import pytest

_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "multihead" / "best.pt"
pytestmark = pytest.mark.skipif(
    not _CHECKPOINT.exists(),
    reason="Requires excluded models/multihead/best.pt (not published in the release)",
)


def test_ctc_frozen_by_default():
    from src.models.ctc_head import HTMACTCModel
    m = HTMACTCModel("models/multihead/best.pt", n_glosses=10)
    enc_params = list(m.encoder.parameters())
    assert enc_params and all(not p.requires_grad for p in enc_params)
    assert any(p.requires_grad for p in m.head.parameters())


def test_ctc_train_encoder_unfreezes_blocks_not_norms():
    from src.models.ctc_head import HTMACTCModel
    m = HTMACTCModel("models/multihead/best.pt", n_glosses=10, train_encoder=True)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any("encoder.blocks" in n for n in trainable)
    assert any(n.startswith("head.") for n in trainable)
    assert not any("encoder.input_proj" in n for n in trainable)


def test_ctc_unfrozen_step_changes_blocks():
    import torch
    from src.models.ctc_head import HTMACTCModel
    m = HTMACTCModel("models/multihead/best.pt", n_glosses=5, train_encoder=True)
    before = m.encoder.blocks[0].conv_mixer.conv.weight.detach().clone()
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(2, 40, 168)
    loss = m(x).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert not torch.equal(before, m.encoder.blocks[0].conv_mixer.conv.weight.detach())
