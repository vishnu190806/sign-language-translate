from __future__ import annotations


def _tiny_cfg():
    from src.models.htma import HTMAConfig
    return HTMAConfig(input_dim=168, embedding_dim=64, num_blocks=1,
                      num_heads=4, num_classes=10)


def test_dpc_shapes_and_loss_finite():
    import torch
    from src.models.htma import HTMAClassifier
    from src.models.dpc import DPCPretrainer
    dpc = DPCPretrainer(HTMAClassifier(_tiny_cfg()))
    x = torch.randn(4, 7, 10, 168)
    loss, acc = dpc(x)
    assert loss.shape == ()
    assert bool(torch.isfinite(loss))
    assert 0.0 <= float(acc) <= 1.0
    z = dpc.encode_window(torch.randn(4, 10, 168))
    assert tuple(z.shape) == (4, 64)


def test_dpc_loss_decreases_on_toy():
    import torch
    from src.models.htma import HTMAConfig, HTMAClassifier
    from src.models.dpc import DPCPretrainer
    cfg = HTMAConfig(input_dim=168, embedding_dim=32, num_blocks=1,
                     num_heads=4, num_classes=10)
    dpc = DPCPretrainer(HTMAClassifier(cfg))
    opt = torch.optim.Adam(dpc.parameters(), lr=1e-3)
    x = torch.sin(torch.linspace(0, 20, 140)).reshape(2, 7, 10, 1).repeat(1, 1, 1, 168)
    l0, _ = dpc(x)
    for _ in range(5):
        opt.zero_grad()
        l, _ = dpc(x)
        l.backward()
        opt.step()
    l1, _ = dpc(x)
    assert float(l1) < float(l0)


def test_batched_windows_match_stacked():
    import torch
    from src.models.dpc import DPCPretrainer
    from src.models.htma import HTMAClassifier
    dpc = DPCPretrainer(HTMAClassifier(_tiny_cfg()))
    dpc.eval()
    torch.manual_seed(0)
    x = torch.randn(4, 7, 10, 168)
    with torch.no_grad():
        stacked = torch.stack([dpc.encode_window(x[:, i]) for i in range(7)], dim=1)
        batched = dpc.encode_windows(x)
    assert tuple(batched.shape) == (4, 7, 64)
    assert torch.allclose(stacked, batched, atol=1e-5)
