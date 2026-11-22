from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _TwoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(4, 3)
        self.fc_b = nn.Linear(4, 2)
        with torch.no_grad():
            self.fc_a.weight.fill_(0.0)
            self.fc_a.bias.copy_(torch.tensor([3.0, 1.0, 0.0]))
            self.fc_b.weight.fill_(0.0)
            self.fc_b.bias.copy_(torch.tensor([2.0, 0.5]))

    def forward(self, x, mask=None, source=None):
        if source == "a":
            return self.fc_a(x.mean(1))
        if source == "b":
            return self.fc_b(x.mean(1))
        raise AssertionError("source required")


def test_unified_predict_max_across_heads():
    from src.infer.routing import unified_predict
    m = _TwoHead().eval()
    x = torch.randn(1, 8, 4)
    mk = torch.ones(1, 8, dtype=torch.bool)
    probs, united = unified_predict(m, x, mk, ["a", "b"],
                                    {0: "w0", 1: "w1", 2: "w2"})
    assert probs.shape == (5,)
    assert abs(probs.sum() - 2.0) < 1e-5
    assert united[0] == "w0" and united[3] == "b:0" or united[3] != "w0"
    assert united[4] != "w2"


def test_unified_predict_label_maps_per_head():
    from src.infer.routing import unified_predict, top_k
    m = _TwoHead().eval()
    x = torch.randn(1, 8, 4)
    mk = torch.ones(1, 8, dtype=torch.bool)
    probs, united = unified_predict(
        m, x, mk, ["a", "b"],
        {"a": {0: "apple", 1: "pear", 2: "fig"}, "b": {0: "dog", 1: "cat"}})
    assert united == {0: "apple", 1: "pear", 2: "fig", 3: "dog", 4: "cat"}
    top = top_k(probs, united, k=2)
    assert top[0][0] == "apple"
    assert top[0][1] > top[1][1]


def test_top_k_sorted_desc():
    from src.infer.routing import top_k
    probs = np.array([0.1, 0.7, 0.2])
    assert [w for w, _ in top_k(probs, {0: "a", 1: "b", 2: "c"}, k=2)] == ["b", "c"]


def test_per_head_temperatures_applied():
    import torch
    from src.infer.routing import unified_predict
    m = _TwoHead().eval()
    x = torch.randn(1, 8, 4)
    mk = torch.ones(1, 8, dtype=torch.bool)
    maps = {"a": {0: "apple", 1: "pear", 2: "fig"}, "b": {0: "dog", 1: "cat"}}
    p1, _ = unified_predict(m, x, mk, ["a", "b"], maps, temperatures={"a": 1.0, "b": 1.0})
    p2, _ = unified_predict(m, x, mk, ["a", "b"], maps, temperatures={"a": 1.0, "b": 0.5})
    assert not bool((p1 == p2).all())
    assert abs(p1[:3] - p2[:3]).max() < 1e-6
