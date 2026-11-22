from __future__ import annotations

import pytest
import numpy as np

from src.infer.buffer import RingBuffer


class TestRingBuffer:
    def test_initial_state(self):
        buf = RingBuffer(10, 168)
        assert buf.count == 0
        assert not buf.is_full
        assert buf.valid_ratio == 0.0

    def test_add_and_retrieve(self):
        buf = RingBuffer(4, 5)
        for i in range(4):
            buf.add(np.ones(5) * (i + 1))
        assert buf.is_full
        seq, mask = buf.get_sequence()
        assert seq.shape == (4, 5)
        assert mask.all()

    def test_wraparound(self):
        buf = RingBuffer(3, 5)
        buf.add(np.ones(5) * 1)
        buf.add(np.ones(5) * 2)
        buf.add(np.ones(5) * 3)
        buf.add(np.ones(5) * 4)
        seq, mask = buf.get_sequence()
        assert buf.pos == 1

    def test_reset(self):
        buf = RingBuffer(4, 5)
        buf.add(np.ones(5))
        buf.reset()
        assert buf.count == 0
        assert buf.pos == 0

    def test_valid_ratio(self):
        buf = RingBuffer(4, 5)
        buf.add(np.ones(5))
        buf.add(np.zeros(5))
        assert buf.valid_ratio == pytest.approx(0.5)
