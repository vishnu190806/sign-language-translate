from __future__ import annotations

import numpy as np
from src.features.spec import FeatureSpec, DEFAULT_FEATURE_SPEC


def normalize_hand_landmarks(landmarks: np.ndarray, spec: FeatureSpec = DEFAULT_FEATURE_SPEC) -> np.ndarray:
    out = landmarks.copy().reshape(-1, spec.hand_dims)
    wrist = out[0]
    out -= wrist
    middle_mcp = out[9]
    scale = np.linalg.norm(middle_mcp)
    if scale > 1e-6:
        out /= scale
    return out.flatten()


def normalize_pose_landmarks(landmarks: np.ndarray, spec: FeatureSpec = DEFAULT_FEATURE_SPEC) -> np.ndarray:
    out = landmarks.copy().reshape(-1, spec.pose_dims)
    left_shoulder = out[spec.POSE_INDICES.index(11) - min(spec.POSE_INDICES)]
    right_shoulder = out[spec.POSE_INDICES.index(12) - min(spec.POSE_INDICES)]
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    out -= mid_shoulder
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width > 1e-6:
        out /= shoulder_width
    return out.flatten()


class SignerStatistics:
    def compute_statistics(self, features_list: list[np.ndarray]) -> dict[str, np.ndarray]:
        all_frames = np.concatenate([f.reshape(-1, f.shape[-1]) for f in features_list], axis=0)
        mean = all_frames.mean(axis=0)
        std = all_frames.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return {"mean": mean, "std": std}

    def normalize(self, features: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
        return (features - stats["mean"]) / stats["std"]


class SignerEstimator:
    def estimate_signer(
        self, features: np.ndarray, signer_stats: dict[str, dict]
    ) -> str:
        num_frames = min(features.shape[0], 5)
        query = features[:num_frames].mean(axis=0)

        best_id: str | None = None
        best_score = np.inf

        for signer_id, stats in signer_stats.items():
            centered = (query - stats["mean"]) / stats["std"]
            mse = float(np.mean(centered ** 2))
            if mse < best_score:
                best_score = mse
                best_id = signer_id

        if best_id is None:
            raise ValueError("No signer statistics provided")
        return best_id


class PerSignerNormalizer:
    def __init__(
        self,
        use_per_signer: bool = False,
        estimator: SignerEstimator | None = None,
        signer_statistics: SignerStatistics | None = None,
    ) -> None:
        self.use_per_signer = use_per_signer
        self._estimator = estimator or SignerEstimator()
        self._signer_stats = signer_statistics or SignerStatistics()

    def normalize(
        self,
        features: np.ndarray,
        signer_stats: dict[str, dict] | None = None,
        signer_id: str | None = None,
    ) -> np.ndarray:
        if not self.use_per_signer or signer_stats is None:
            return features

        if signer_id is None:
            signer_id = self._estimator.estimate_signer(features, signer_stats)

        stats = signer_stats[signer_id]
        return self._signer_stats.normalize(features, stats)


def build_frame_features(
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    pose: np.ndarray | None,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
    per_signer_normalizer: PerSignerNormalizer | None = None,
    signer_stats: dict[str, dict] | None = None,
    signer_id: str | None = None,
) -> np.ndarray:
    features = np.zeros(spec.total_features, dtype=np.float32)

    hands = [left_hand, right_hand]
    for i, hand in enumerate(hands):
        offset = i * spec.hand_features_per_hand
        if hand is not None and hand.shape == (spec.hand_landmarks, spec.hand_dims):
            normalized = normalize_hand_landmarks(hand, spec)
            features[offset:offset + spec.hand_landmarks * spec.hand_dims] = normalized
            features[offset + spec.hand_landmarks * spec.hand_dims] = 1.0

    hand_offset = spec.hand_features
    if pose is not None and pose.shape[0] > max(spec.POSE_INDICES):
        pose_subset = pose[list(spec.POSE_INDICES)]
        normalized_pose = normalize_pose_landmarks(pose_subset, spec)
        pose_size = spec.pose_landmarks * spec.pose_dims
        features[hand_offset:hand_offset + pose_size] = normalized_pose
        features[hand_offset + pose_size] = 1.0

    if per_signer_normalizer is not None and per_signer_normalizer.use_per_signer:
        features = per_signer_normalizer.normalize(features, signer_stats, signer_id)

    return features
