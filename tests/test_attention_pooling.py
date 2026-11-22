from __future__ import annotations

import torch

from src.models.common import AttentionPooling


def test_all_masked_row_produces_finite_output() -> None:
    pool = AttentionPooling(d_model=8)
    x = torch.zeros(2, 5, 8)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    out = pool(x, mask)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_partial_mask_matches_unmasked_valid_positions() -> None:
    pool = AttentionPooling(d_model=8)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    out_masked = pool(x, mask)
    out_sliced = pool(x[:, :2])
    assert torch.allclose(out_masked, out_sliced, atol=1e-6)
