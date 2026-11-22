from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class LandmarkReplayer:
    def __init__(self, npz_dir: str | Path, fps: int = 30):
        self.npz_dir = Path(npz_dir)
        self.fps = fps
        self._samples: list[tuple[Path, str]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        for npz_path in sorted(self.npz_dir.glob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            label = str(data.get("label", "unknown"))
            self._samples.append((npz_path, label))
        logger.info(f"Loaded {len(self._samples)} landmark samples from {self.npz_dir}")

    def get_samples_by_words(self, words: list[str]) -> list[tuple[np.ndarray, str]]:
        word_set = {w.lower().strip() for w in words}
        results = []
        for npz_path, label in self._samples:
            if label.lower().strip() in word_set:
                data = np.load(npz_path)
                features = data["features"].astype(np.float32)
                results.append((features, label))
        return results

    def get_all_samples(self) -> list[tuple[np.ndarray, str]]:
        results = []
        for npz_path, label in self._samples:
            data = np.load(npz_path)
            features = data["features"].astype(np.float32)
            results.append((features, label))
        return results

    def yield_frames(self, features: np.ndarray) -> list[np.ndarray]:
        return [features[i] for i in range(features.shape[0])]

    def get_sample_info(self, features: np.ndarray, label: str) -> dict:
        return {
            "label": label,
            "num_frames": features.shape[0],
            "feature_dim": features.shape[1] if features.ndim > 1 else 0,
            "duration_s": round(features.shape[0] / self.fps, 2),
        }

    def __len__(self) -> int:
        return len(self._samples)
