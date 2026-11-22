from __future__ import annotations

import numpy as np

from src.features.spec import FeatureSpec, DEFAULT_FEATURE_SPEC


class GeometryAugmentor:
    def __init__(self, spec: FeatureSpec | None = None, seed: int = 42):
        self.spec = spec if spec is not None else DEFAULT_FEATURE_SPEC
        self.rng = np.random.default_rng(seed)
        s = self.spec
        self._left_start = 0
        self._right_start = s.hand_features_per_hand
        self._pose_start = s.hand_features
        self._left_coord_end = self._left_start + s.hand_landmarks * s.hand_dims
        self._right_coord_end = self._right_start + s.hand_landmarks * s.hand_dims
        self._pose_coord_end = self._pose_start + s.pose_landmarks * s.pose_dims

    def camera_depth_variation(
        self,
        features: np.ndarray,
        scale_range: tuple[float, float] = (0.8, 1.2),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = self.rng
        s = rng.uniform(*scale_range)
        out = features.copy()
        z_indices = np.concatenate([
            np.arange(self._left_start + 2, self._left_coord_end, 3),
            np.arange(self._right_start + 2, self._right_coord_end, 3),
            np.arange(self._pose_start + 2, self._pose_coord_end, 3),
        ])
        out[:, z_indices] *= s
        return out

    def hv_shift(
        self,
        features: np.ndarray,
        max_shift: float = 0.15,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = self.rng
        dx = rng.uniform(-max_shift, max_shift)
        dy = rng.uniform(-max_shift, max_shift)
        out = features.copy()
        x_indices = np.concatenate([
            np.arange(self._left_start, self._left_coord_end, 3),
            np.arange(self._right_start, self._right_coord_end, 3),
            np.arange(self._pose_start, self._pose_coord_end, 3),
        ])
        y_indices = x_indices + 1
        out[:, x_indices] += dx
        out[:, y_indices] += dy
        return out

    def hand_size_variation(
        self,
        features: np.ndarray,
        scale_range: tuple[float, float] = (0.85, 1.15),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = self.rng
        s = rng.uniform(*scale_range)
        out = features.copy()
        lm = self.spec.hand_landmarks
        d = self.spec.hand_dims
        for h in range(self.spec.num_hands):
            start = h * self.spec.hand_features_per_hand
            coords = out[:, start:start + lm * d].reshape(-1, lm, d)
            wrist = coords[:, 0:1, :].copy()
            coords[:, 1:] = wrist + (coords[:, 1:] - wrist) * s
        return out

    def viewpoint_rotation(
        self,
        features: np.ndarray,
        angle_range_deg: tuple[float, float] = (-15.0, 15.0),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = self.rng
        theta = np.deg2rad(rng.uniform(*angle_range_deg))
        c, sn = np.cos(theta), np.sin(theta)
        out = features.copy()
        x_indices = np.concatenate([
            np.arange(self._left_start, self._left_coord_end, 3),
            np.arange(self._right_start, self._right_coord_end, 3),
            np.arange(self._pose_start, self._pose_coord_end, 3),
        ])
        y_indices = x_indices + 1
        x_vals = out[:, x_indices].copy()
        y_vals = out[:, y_indices].copy()
        out[:, x_indices] = x_vals * c - y_vals * sn
        out[:, y_indices] = x_vals * sn + y_vals * c
        return out

    def finger_articulation(
        self,
        features: np.ndarray,
        angle_range_deg: tuple[float, float] = (-10.0, 10.0),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = self.rng
        out = features.copy()
        lm = self.spec.hand_landmarks
        d = self.spec.hand_dims
        finger_chains = [
            (1, [2, 3, 4]),
            (5, [6, 7, 8]),
            (9, [10, 11, 12]),
            (13, [14, 15, 16]),
            (17, [18, 19, 20]),
        ]
        for h in range(self.spec.num_hands):
            start = h * self.spec.hand_features_per_hand
            coords = out[:, start:start + lm * d].reshape(-1, lm, d)
            for pivot_idx, distal_indices in finger_chains:
                theta = np.deg2rad(rng.uniform(*angle_range_deg))
                co, si = np.cos(theta), np.sin(theta)
                px = coords[:, pivot_idx, 0].copy()
                py = coords[:, pivot_idx, 1].copy()
                for j in distal_indices:
                    rx = coords[:, j, 0] - px
                    ry = coords[:, j, 1] - py
                    coords[:, j, 0] = px + rx * co - ry * si
                    coords[:, j, 1] = py + rx * si + ry * co
        return out

    def elbow_displacement(
        self,
        features: np.ndarray,
        max_shift: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = self.rng
        dx = rng.uniform(-max_shift, max_shift)
        dy = rng.uniform(-max_shift, max_shift)
        out = features.copy()
        lm = self.spec.hand_landmarks
        d = self.spec.hand_dims
        for h in range(self.spec.num_hands):
            start = h * self.spec.hand_features_per_hand
            coords = out[:, start:start + lm * d].reshape(-1, lm, d)
            coords[:, :, 0] += dx
            coords[:, :, 1] += dy
        return out

    def __call__(
        self,
        features: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = self.rng
        out = features.copy()
        if rng.random() < 0.3:
            out = self.camera_depth_variation(out, rng=rng)
        if rng.random() < 0.3:
            out = self.hv_shift(out, rng=rng)
        if rng.random() < 0.3:
            out = self.hand_size_variation(out, rng=rng)
        if rng.random() < 0.3:
            out = self.viewpoint_rotation(out, rng=rng)
        if rng.random() < 0.3:
            out = self.finger_articulation(out, rng=rng)
        if rng.random() < 0.3:
            out = self.elbow_displacement(out, rng=rng)
        return out
