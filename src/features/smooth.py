from __future__ import annotations

import numpy as np


class OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 0.05,
        beta: float = 80.0,
        derivate_cutoff: float = 1.0,
        freq: float = 30.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivate_cutoff = derivate_cutoff
        self.freq = freq
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def __call__(self, x: np.ndarray, t: float | None = None) -> np.ndarray:
        if t is None:
            t = self._t_prev + 1.0 / self.freq if self._t_prev is not None else 0.0
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x

        dt = t - self._t_prev
        if dt <= 0:
            dt = 1.0 / self.freq

        dx = (x - self._x_prev) / dt
        alpha_d = self._smoothing_factor(dt, self.derivate_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self._smoothing_factor(dt, cutoff)
        x_hat = alpha * x + (1 - alpha) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat

    def _smoothing_factor(self, dt: float, cutoff: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None


class LandmarkSmoother:
    def __init__(self, feature_dim: int = 168, **kwargs):
        self.filters = [OneEuroFilter(**kwargs) for _ in range(feature_dim)]

    def smooth(self, features: np.ndarray) -> np.ndarray:
        out = np.empty_like(features)
        for i, f in enumerate(self.filters):
            out[i] = f(features[i])
        return out

    def reset(self) -> None:
        for f in self.filters:
            f.reset()
