from __future__ import annotations

import numpy as np
import pytest

from src.infer.gate import StabilityGate


class TestStabilityGate:
    def test_no_emission_without_stability(self):
        gate = StabilityGate(threshold=0.85, consecutive_frames=5, cooldown_frames=10)
        for _ in range(4):
            result = gate.update(np.array([0.4, 0.3, 0.3]))
            assert result is None

    def test_emission_on_stable_prediction(self):
        gate = StabilityGate(threshold=0.85, consecutive_frames=3, cooldown_frames=5)
        probs = np.array([0.9, 0.05, 0.05])
        for _ in range(3):
            result = gate.update(probs)
        assert result == 0

    def test_cooldown_prevents_re_emission(self):
        gate = StabilityGate(threshold=0.85, consecutive_frames=3, cooldown_frames=3)
        probs = np.array([0.9, 0.05, 0.05])
        for _ in range(3):
            gate.update(probs)
        for _ in range(3):
            result = gate.update(probs)
            assert result is None

    def test_low_confidence_blocks_emission(self):
        gate = StabilityGate(threshold=0.85, consecutive_frames=3, cooldown_frames=5)
        probs = np.array([0.6, 0.2, 0.2])
        for _ in range(10):
            result = gate.update(probs)
            assert result is None

    def test_reset(self):
        gate = StabilityGate(threshold=0.85, consecutive_frames=3, cooldown_frames=5)
        probs = np.array([0.9, 0.05, 0.05])
        for _ in range(2):
            gate.update(probs)
        gate.reset()
        result = gate.update(probs)
        assert result is None
