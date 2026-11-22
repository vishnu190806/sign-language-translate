from __future__ import annotations


def test_dpc_dataset_shapes_and_short_clip_filter(tmp_path):
    import json

    import numpy as np

    mem = np.memmap(tmp_path / "m.mm", dtype=np.float32, mode="w+", shape=(200, 168))
    mem[:] = np.random.randn(200, 168).astype(np.float32)
    mem.flush()
    (tmp_path / "clips.json").write_text(json.dumps([
        {"file": "a", "key": "long", "start": 0, "length": 130},
        {"file": "a", "key": "short", "start": 130, "length": 70},
        {"file": "a", "key": "tiny", "start": 130, "length": 10},
    ]))
    from src.data.dpc_dataset import DPCWindowDataset
    ds = DPCWindowDataset(str(tmp_path / "m.mm"), str(tmp_path / "clips.json"))
    assert ds.num_clips == 2
    x = ds[0]
    assert tuple(x.shape) == (70, 168)
    assert str(x.dtype) == "torch.float32"
