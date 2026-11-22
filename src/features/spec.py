from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    hand_landmarks: int = 21
    hand_dims: int = 3
    hand_presence: int = 1
    num_hands: int = 2
    pose_landmarks: int = 13
    pose_dims: int = 3
    pose_presence: int = 1

    @property
    def hand_features_per_hand(self) -> int:
        return self.hand_landmarks * self.hand_dims + self.hand_presence

    @property
    def hand_features(self) -> int:
        return self.hand_features_per_hand * self.num_hands

    @property
    def pose_features(self) -> int:
        return self.pose_landmarks * self.pose_dims + self.pose_presence

    @property
    def total_features(self) -> int:
        return self.hand_features + self.pose_features

    POSE_INDICES: tuple[int, ...] = (
        0,   # nose
        2,   # left_eye
        5,   # right_eye
        7,   # left_ear
        8,   # right_ear
        9,   # mouth_left
        10,  # mouth_right
        11,  # left_shoulder
        12,  # right_shoulder
        13,  # left_elbow
        14,  # right_elbow
        15,  # left_wrist
        16,  # right_wrist
    )


DEFAULT_FEATURE_SPEC = FeatureSpec()
assert DEFAULT_FEATURE_SPEC.total_features == 168
