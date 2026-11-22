from __future__ import annotations

import numpy as np


def _gate():
    from src.infer.gate import StabilityGate
    return StabilityGate(threshold=0.4, consecutive_frames=5, cooldown_frames=10)


def test_gate_status_reports_streak_and_cooldown():
    g = _gate()
    s = g.status()
    assert s["streak"] == 0 and s["cooldown"] == 0 and s["dominant"] is None
    for _ in range(3):
        g.update(np.array([0.8, 0.1, 0.1]))
    s = g.status()
    assert s["streak"] == 3 and s["dominant"] == 0
    assert 0.0 <= s["stability"] <= 1.0


def test_gate_status_after_emit_shows_cooldown():
    g = _gate()
    outs = [g.update(np.array([0.9, 0.05, 0.05])) for _ in range(6)]
    assert outs[4] == 0
    s = g.status()
    assert s["cooldown"] == 9 and s["last_emitted"] == 0
