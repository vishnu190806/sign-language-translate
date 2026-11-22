from __future__ import annotations


def _make_base():
    from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier
    cfg = HTMAConfig(input_dim=168, embedding_dim=64, num_blocks=1,
                     num_heads=4, num_classes=10)
    return MultiHeadHTMAClassifier(cfg, sources=("include", "isl500"),
                                   head_classes={"include": 10, "isl500": 10})


def test_extra_head_output_size():
    import torch
    from src.models.htma import MultiHeadHTMAClassifier
    base = _make_base()
    base.heads.update({"extra": __import__("torch").nn.Sequential(
        __import__("torch").nn.Linear(64, 64),
        __import__("torch").nn.GELU(),
        __import__("torch").nn.Linear(64, 7))})
    out = base(torch.randn(2, 16, 168), source="extra")
    assert tuple(out.shape) == (2, 7)


def test_freeze_all_but_extra():
    import torch
    from src.models.htma import MultiHeadHTMAClassifier
    base = _make_base()
    base.heads.update({"extra": torch.nn.Linear(64, 7)})
    for name, p in base.named_parameters():
        p.requires_grad = name.startswith("heads.extra")
    trainable = [n for n, p in base.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("heads.extra") for n in trainable)
    frozen = [n for n, p in base.named_parameters() if not p.requires_grad]
    assert any("input_proj" in n for n in frozen)
    assert any(n.startswith("heads.include") for n in frozen)


def test_old_heads_unchanged_after_extra_step():
    import torch
    base = _make_base()
    base.heads.update({"extra": torch.nn.Linear(64, 7)})
    for name, p in base.named_parameters():
        p.requires_grad = name.startswith("heads.extra")
    before = {n: p.detach().clone() for n, p in base.named_parameters()
              if not p.requires_grad}
    opt = torch.optim.Adam([p for p in base.parameters() if p.requires_grad], lr=1e-3)
    x = torch.randn(4, 16, 168)
    loss = base(x, source="extra").sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    for n, old in before.items():
        assert torch.equal(dict(base.named_parameters())[n].detach(), old)
