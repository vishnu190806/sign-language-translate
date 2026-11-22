from __future__ import annotations


def test_warm_start_loads_encoder_not_head(tmp_path):
    import torch
    from src.models.htma import HTMAClassifier, HTMAConfig

    cfg = HTMAConfig(input_dim=168, embedding_dim=32, num_blocks=1,
                     num_heads=4, num_classes=264)
    src = HTMAClassifier(cfg)
    for p in src.parameters():
        p.data.fill_(7.0)
    ckpt = tmp_path / "enc.pt"
    torch.save({"model_state_dict": src.state_dict()}, ckpt)

    from scripts.train import load_warm_start
    dst = HTMAClassifier(cfg)
    loaded, skipped = load_warm_start(dst, str(ckpt))
    assert loaded > 0 and skipped > 0
    assert bool((dst.input_proj.weight == 7.0).all())
    assert not bool((dst.head[0].weight == 7.0).all())


def test_warm_start_missing_file_raises(tmp_path):
    import pytest
    from src.models.htma import HTMAClassifier, HTMAConfig
    from scripts.train import load_warm_start
    cfg = HTMAConfig(input_dim=168, embedding_dim=32, num_blocks=1,
                     num_heads=4, num_classes=264)
    with pytest.raises(FileNotFoundError):
        load_warm_start(HTMAClassifier(cfg), str(tmp_path / "nope.pt"))
