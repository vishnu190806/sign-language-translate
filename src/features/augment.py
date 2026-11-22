from __future__ import annotations

import numpy as np

from src.features.spec import FeatureSpec, DEFAULT_FEATURE_SPEC


class LandmarkAugmentor:
    def __init__(self, spec: FeatureSpec | None = None):
        self.spec = spec if spec is not None else DEFAULT_FEATURE_SPEC

    def temporal_jitter(
        self,
        sequence: np.ndarray,
        max_shift: int = 2,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        shift = rng.integers(-max_shift, max_shift + 1)
        return np.roll(sequence, shift, axis=0)

    def temporal_crop(
        self,
        sequence: np.ndarray,
        min_ratio: float = 0.7,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        T = sequence.shape[0]
        crop_len = int(T * rng.uniform(min_ratio, 1.0))
        start = rng.integers(0, T - crop_len + 1)
        cropped = sequence[start : start + crop_len]
        if cropped.shape[0] < T:
            pad = np.zeros((T - cropped.shape[0], sequence.shape[1]), dtype=sequence.dtype)
            cropped = np.concatenate([cropped, pad], axis=0)
        return cropped[:T]

    def temporal_dropout(
        self,
        sequence: np.ndarray,
        keep_ratio: float = 0.85,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        T = sequence.shape[0]
        num_keep = int(T * keep_ratio)
        indices = np.sort(rng.choice(T, size=num_keep, replace=False))
        out = np.zeros_like(sequence)
        out[indices] = sequence[indices]
        return out

    def gaussian_noise(
        self,
        sequence: np.ndarray,
        std: float = 0.01,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(0, std, sequence.shape).astype(sequence.dtype)
        return sequence + noise

    def random_hand_dropout(
        self,
        sequence: np.ndarray,
        dropout_prob: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        out = sequence.copy()
        hand_size = self.spec.hand_features_per_hand
        for t in range(out.shape[0]):
            for h in range(self.spec.num_hands):
                if rng.random() < dropout_prob:
                    offset = h * hand_size
                    out[t, offset : offset + hand_size] = 0
        return out

    def random_pose_dropout(
        self,
        sequence: np.ndarray,
        dropout_prob: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        out = sequence.copy()
        if rng.random() < dropout_prob:
            out[:, self.spec.hand_features:] = 0
        return out

    def spatial_scale(
        self,
        sequence: np.ndarray,
        scale_range: tuple[float, float] = (0.9, 1.1),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        scale = rng.uniform(*scale_range)
        return sequence * scale

    def random_translation(
        self,
        sequence: np.ndarray,
        max_shift: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        shift_x = rng.uniform(-max_shift, max_shift)
        shift_y = rng.uniform(-max_shift, max_shift)
        out = sequence.copy()
        hand_size = self.spec.hand_features_per_hand
        for h in range(self.spec.num_hands):
            offset = h * hand_size
            coords = out[:, offset:offset + self.spec.hand_landmarks * self.spec.hand_dims].reshape(
                out.shape[0], self.spec.hand_landmarks, self.spec.hand_dims
            )
            coords[:, :, 0] += shift_x
            coords[:, :, 1] += shift_y
        pose_offset = self.spec.hand_features
        pose_coords = out[:, pose_offset:pose_offset + self.spec.pose_landmarks * self.spec.pose_dims].reshape(
            out.shape[0], self.spec.pose_landmarks, self.spec.pose_dims
        )
        pose_coords[:, :, 0] += shift_x
        pose_coords[:, :, 1] += shift_y
        return out

    def frame_permutation(
        self,
        sequence: np.ndarray,
        variation_threshold: float = 0.01,
        min_segment: int = 3,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        T = sequence.shape[0]
        if T < min_segment:
            return sequence.copy()
        diff = np.abs(np.diff(sequence, axis=0))
        variation = diff.mean(axis=1)
        low_var = variation < variation_threshold
        segments = []
        start = 0
        for i in range(len(low_var)):
            if low_var[i]:
                if start <= i:
                    pass
            else:
                if start <= i:
                    segments.append((start, i + 1))
                start = i + 1
        if start < T:
            segments.append((start, T))
        out = sequence.copy()
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len >= min_segment:
                perm = rng.permutation(seg_len)
                out[seg_start:seg_end] = sequence[seg_start:seg_end][perm]
        return out

    def speed_variation(
        self,
        sequence: np.ndarray,
        speed_range: tuple[float, float] = (0.8, 1.2),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        T = sequence.shape[0]
        speed = rng.uniform(*speed_range)
        src_idx = np.linspace(0, T - 1, int(T * speed))
        src_idx = np.clip(src_idx, 0, T - 1)
        out = np.zeros_like(sequence)
        for i in range(min(len(src_idx), T)):
            idx = int(src_idx[i])
            out[i] = sequence[idx]
        return out

    def random_frame_drop(
        self,
        sequence: np.ndarray,
        drop_ratio: tuple[float, float] = (0.1, 0.2),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        T = sequence.shape[0]
        ratio = rng.uniform(*drop_ratio)
        num_drop = max(1, int(T * ratio))
        drop_indices = rng.choice(T, size=num_drop, replace=False)
        out = sequence.copy()
        out[drop_indices] = 0
        left_presence_idx = self.spec.hand_landmarks * self.spec.hand_dims
        right_presence_idx = self.spec.hand_features_per_hand + self.spec.hand_landmarks * self.spec.hand_dims
        pose_presence_idx = self.spec.hand_features + self.spec.pose_landmarks * self.spec.pose_dims
        out[drop_indices, left_presence_idx] = 0
        out[drop_indices, right_presence_idx] = 0
        out[drop_indices, pose_presence_idx] = 0
        return out

    def spatial_rotation(
        self,
        sequence: np.ndarray,
        angle_range: float = 15.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        angle_x = rng.uniform(-angle_range, angle_range)
        angle_y = rng.uniform(-angle_range, angle_range)
        angle_z = rng.uniform(-angle_range, angle_range)
        rx = np.radians(angle_x)
        ry = np.radians(angle_y)
        rz = np.radians(angle_z)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)],
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)],
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1],
        ])
        R = Rz @ Ry @ Rx
        out = sequence.copy()
        hand_size = self.spec.hand_landmarks * self.spec.hand_dims
        for h in range(self.spec.num_hands):
            offset = h * self.spec.hand_features_per_hand
            coords = out[:, offset:offset + hand_size].reshape(
                out.shape[0], self.spec.hand_landmarks, self.spec.hand_dims
            )
            coords[:] = coords @ R.T
        pose_offset = self.spec.hand_features
        pose_coords = out[:, pose_offset:pose_offset + self.spec.pose_landmarks * self.spec.pose_dims].reshape(
            out.shape[0], self.spec.pose_landmarks, self.spec.pose_dims
        )
        pose_coords[:] = pose_coords @ R.T
        return out

    def frame_fusion(
        self,
        sequence: np.ndarray,
        blend_range: tuple[float, float] = (0.3, 0.7),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        T = sequence.shape[0]
        if T < 2:
            return sequence.copy()
        out = sequence.copy()
        for i in range(T - 1):
            alpha = rng.uniform(*blend_range)
            out[i] = alpha * sequence[i] + (1 - alpha) * sequence[i + 1]
        out[T - 1] = sequence[T - 1]
        return out

    def random_keypoint_removal(
        self,
        sequence: np.ndarray,
        removal_ratio: tuple[float, float] = (0.1, 0.3),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        D = sequence.shape[1]
        ratio = rng.uniform(*removal_ratio)
        num_remove = max(1, int(D * ratio))
        remove_indices = rng.choice(D, size=num_remove, replace=False)
        out = sequence.copy()
        out[:, remove_indices] = 0
        hand_size = self.spec.hand_features_per_hand
        hand_coords = self.spec.hand_landmarks * self.spec.hand_dims
        remove_set = set(remove_indices.tolist())
        for h in range(self.spec.num_hands):
            offset = h * hand_size
            hand_indices = set(range(offset, offset + hand_coords))
            if remove_set & hand_indices:
                out[:, offset + hand_coords] = 0
        pose_offset = self.spec.hand_features
        pose_indices = set(range(pose_offset, pose_offset + self.spec.pose_landmarks * self.spec.pose_dims))
        if remove_set & pose_indices:
            out[:, pose_offset + self.spec.pose_landmarks * self.spec.pose_dims] = 0
        return out

    def __call__(
        self,
        sequence: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        out = sequence.copy()
        if rng.random() < 0.5:
            out = self.temporal_jitter(out, rng=rng)
        if rng.random() < 0.3:
            out = self.temporal_crop(out, rng=rng)
        if rng.random() < 0.3:
            out = self.temporal_dropout(out, rng=rng)
        if rng.random() < 0.5:
            out = self.gaussian_noise(out, rng=rng)
        if rng.random() < 0.3:
            out = self.random_hand_dropout(out, rng=rng)
        if rng.random() < 0.3:
            out = self.spatial_scale(out, rng=rng)
        if rng.random() < 0.3:
            out = self.random_translation(out, rng=rng)
        if rng.random() < 0.2:
            out = self.frame_permutation(out, rng=rng)
        if rng.random() < 0.2:
            out = self.speed_variation(out, rng=rng)
        if rng.random() < 0.3:
            out = self.random_frame_drop(out, rng=rng)
        if rng.random() < 0.2:
            out = self.spatial_rotation(out, rng=rng)
        if rng.random() < 0.2:
            out = self.frame_fusion(out, rng=rng)
        if rng.random() < 0.2:
            out = self.random_keypoint_removal(out, rng=rng)
        if rng.random() < 0.5:
            out = self.random_pose_dropout(out, rng=rng)
        return out
