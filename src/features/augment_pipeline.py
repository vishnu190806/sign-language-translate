from __future__ import annotations

from typing import Any

import numpy as np

from src.features.augment import LandmarkAugmentor
from src.features.augment_geo import GeometryAugmentor
from src.features.spec import FeatureSpec, DEFAULT_FEATURE_SPEC

DEFAULT_PROBS: dict[str, float] = {
    "temporal_jitter": 0.5,
    "temporal_crop": 0.3,
    "temporal_dropout": 0.3,
    "gaussian_noise": 0.5,
    "random_hand_dropout": 0.3,
    "spatial_scale": 0.3,
    "random_translation": 0.3,
    "frame_permutation": 0.2,
    "speed_variation": 0.2,
    "random_frame_drop": 0.3,
    "spatial_rotation": 0.2,
    "frame_fusion": 0.2,
    "random_keypoint_removal": 0.2,
    "camera_depth_variation": 0.3,
    "hv_shift": 0.3,
    "hand_size_variation": 0.3,
    "viewpoint_rotation": 0.3,
    "finger_articulation": 0.3,
    "elbow_displacement": 0.3,
}


class AugmentationPipeline:
    def __init__(self, spec: FeatureSpec | None = None, seed: int = 42) -> None:
        self.spec = spec if spec is not None else DEFAULT_FEATURE_SPEC
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._landmark_aug = LandmarkAugmentor(self.spec)
        self._geo_aug = GeometryAugmentor(self.spec, seed=seed)
        self._probs = dict(DEFAULT_PROBS)
        self._stats: dict[str, dict[str, Any]] = {}

    @property
    def probs(self) -> dict[str, float]:
        return dict(self._probs)

    def set_prob(self, name: str, prob: float) -> None:
        if name not in self._probs:
            raise ValueError(f"Unknown augmentation: {name}")
        self._probs[name] = prob

    def enable(self, name: str) -> None:
        self.set_prob(name, DEFAULT_PROBS[name])

    def disable(self, name: str) -> None:
        self.set_prob(name, 0.0)

    def _record(self, name: str, severity: float) -> None:
        if name not in self._stats:
            self._stats[name] = {"count": 0, "severities": []}
        self._stats[name]["count"] += 1
        self._stats[name]["severities"].append(severity)

    def reset_statistics(self) -> None:
        self._stats.clear()

    def get_statistics(self) -> dict[str, dict[str, Any]]:
        result = {}
        for name, data in self._stats.items():
            sevs = np.array(data["severities"]) if data["severities"] else np.array([0.0])
            result[name] = {
                "count": data["count"],
                "mean_severity": float(sevs.mean()),
                "std_severity": float(sevs.std()),
            }
        return result

    _LANDMARK_METHODS = {
        "temporal_jitter": lambda aug, seq, rng: aug.temporal_jitter(seq, rng=rng),
        "temporal_crop": lambda aug, seq, rng: aug.temporal_crop(seq, rng=rng),
        "temporal_dropout": lambda aug, seq, rng: aug.temporal_dropout(seq, rng=rng),
        "gaussian_noise": lambda aug, seq, rng: aug.gaussian_noise(seq, rng=rng),
        "random_hand_dropout": lambda aug, seq, rng: aug.random_hand_dropout(seq, rng=rng),
        "spatial_scale": lambda aug, seq, rng: aug.spatial_scale(seq, rng=rng),
        "random_translation": lambda aug, seq, rng: aug.random_translation(seq, rng=rng),
        "frame_permutation": lambda aug, seq, rng: aug.frame_permutation(seq, rng=rng),
        "speed_variation": lambda aug, seq, rng: aug.speed_variation(seq, rng=rng),
        "random_frame_drop": lambda aug, seq, rng: aug.random_frame_drop(seq, rng=rng),
        "spatial_rotation": lambda aug, seq, rng: aug.spatial_rotation(seq, rng=rng),
        "frame_fusion": lambda aug, seq, rng: aug.frame_fusion(seq, rng=rng),
        "random_keypoint_removal": lambda aug, seq, rng: aug.random_keypoint_removal(seq, rng=rng),
    }

    _GEO_METHODS = {
        "camera_depth_variation": lambda aug, seq, rng: aug.camera_depth_variation(seq, rng=rng),
        "hv_shift": lambda aug, seq, rng: aug.hv_shift(seq, rng=rng),
        "hand_size_variation": lambda aug, seq, rng: aug.hand_size_variation(seq, rng=rng),
        "viewpoint_rotation": lambda aug, seq, rng: aug.viewpoint_rotation(seq, rng=rng),
        "finger_articulation": lambda aug, seq, rng: aug.finger_articulation(seq, rng=rng),
        "elbow_displacement": lambda aug, seq, rng: aug.elbow_displacement(seq, rng=rng),
    }

    def __call__(self, features: np.ndarray, mode: str = "train") -> np.ndarray:
        if mode != "train":
            return features.copy()
        out = features.copy()
        for name, fn in self._LANDMARK_METHODS.items():
            if self._probs.get(name, 0.0) > 0 and self._rng.random() < self._probs[name]:
                severity = float(self._rng.random())
                out = fn(self._landmark_aug, out, self._rng)
                self._record(name, severity)
        for name, fn in self._GEO_METHODS.items():
            if self._probs.get(name, 0.0) > 0 and self._rng.random() < self._probs[name]:
                severity = float(self._rng.random())
                out = fn(self._geo_aug, out, self._rng)
                self._record(name, severity)
        return out


def create_augmentation_pipeline(config: dict[str, Any]) -> AugmentationPipeline:
    spec = config.get("spec", None)
    seed = config.get("seed", 42)
    pipeline = AugmentationPipeline(spec=spec, seed=seed)
    probs = config.get("probs", {})
    for name, prob in probs.items():
        pipeline.set_prob(name, float(prob))
    return pipeline


def visualize_augmentation(
    original: np.ndarray,
    augmented: np.ndarray,
    title: str = "Augmentation Comparison",
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(original.T, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0].set_title("Original")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Feature dim")
    axes[1].imshow(augmented.T, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[1].set_title("Augmented")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Feature dim")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_statistics(stats: dict[str, dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    names = list(stats.keys())
    counts = [stats[n]["count"] for n in names]
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(names, counts)
    axes[0].set_xlabel("Count")
    axes[0].set_title("Augmentation Frequency")
    mean_sev = [stats[n]["mean_severity"] for n in names]
    std_sev = [stats[n]["std_severity"] for n in names]
    axes[1].barh(names, mean_sev, xerr=std_sev, capsize=3)
    axes[1].set_xlabel("Severity")
    axes[1].set_title("Augmentation Severity (mean ± std)")
    plt.tight_layout()
    plt.show()
