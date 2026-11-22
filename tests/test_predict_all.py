from __future__ import annotations


def _tiny_multi():
    from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier
    cfg = HTMAConfig(input_dim=168, embedding_dim=32, num_blocks=1,
                     num_heads=4, num_classes=6)
    return MultiHeadHTMAClassifier(cfg, sources=("a", "b"),
                                   head_classes={"a": 6, "b": 4})


def test_predict_all_matches_per_source_loop():
    import torch
    m = _tiny_multi().eval()
    x = torch.randn(2, 16, 168)
    with torch.no_grad():
        per_src = {s: m(x, source=s) for s in ("a", "b")}
        batched = m.predict_all(x)
    assert set(batched) == {"a", "b"}
    for s in ("a", "b"):
        assert torch.allclose(per_src[s], batched[s], atol=1e-5)


def test_predict_all_respects_head_sizes():
    import torch
    m = _tiny_multi().eval()
    x = torch.randn(2, 16, 168)
    with torch.no_grad():
        out = m.predict_all(x)
    assert tuple(out["a"].shape) == (2, 6)
    assert tuple(out["b"].shape) == (2, 4)
