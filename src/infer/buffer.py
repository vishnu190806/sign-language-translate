from __future__ import annotations

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int, feature_dim: int = 168):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.buffer = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.mask = np.zeros(capacity, dtype=bool)
        self.pos = 0
        self.count = 0

    def add(self, features: np.ndarray) -> None:
        valid = not np.all(features == 0)
        self.buffer[self.pos] = features
        self.mask[self.pos] = valid
        self.pos = (self.pos + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def get_sequence(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < self.capacity:
            seq = np.zeros((self.capacity, self.feature_dim), dtype=np.float32)
            mask = np.zeros(self.capacity, dtype=bool)
            seq[:self.count] = self.buffer[:self.count]
            mask[:self.count] = self.mask[:self.count]
            return seq, mask
        else:
            idx = np.arange(self.capacity) + self.pos
            idx = idx % self.capacity
            return self.buffer[idx].copy(), self.mask[idx].copy()

    @property
    def is_full(self) -> bool:
        return self.count >= self.capacity

    def reset(self) -> None:
        self.buffer[:] = 0
        self.mask[:] = False
        self.pos = 0
        self.count = 0

    @property
    def valid_ratio(self) -> float:
        if self.count == 0:
            return 0.0
        return float(np.sum(self.mask[:self.count])) / self.count
