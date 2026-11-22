from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class StabilityGate:
    def __init__(
        self,
        threshold: float = 0.40,
        consecutive_frames: int = 12,
        cooldown_frames: int = 30,
        agreement_ratio: float = 0.60,
        min_confidence_delta: float = 0.08,
    ):
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.cooldown_frames = cooldown_frames
        self.agreement_ratio = agreement_ratio
        self.min_confidence_delta = min_confidence_delta

        self._history: deque[int] = deque(maxlen=consecutive_frames)
        self._confidence_history: deque[float] = deque(maxlen=consecutive_frames)
        self._cooldown_counter = 0
        self._last_emitted: int | None = None

    def update(self, probs: np.ndarray) -> int | None:
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            if self._cooldown_counter == 0:
                self._last_emitted = None
            return None

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        self._history.append(predicted_class)
        self._confidence_history.append(confidence)

        if len(self._history) < self.consecutive_frames:
            return None

        class_counts: dict[int, int] = {}
        for cls in self._history:
            class_counts[cls] = class_counts.get(cls, 0) + 1

        dominant_class, dominant_count = max(class_counts.items(), key=lambda x: x[1])
        stability = dominant_count / self.consecutive_frames

        dominant_confs = [c for cls, c in zip(self._history, self._confidence_history) if cls == dominant_class]
        other_confs = [c for cls, c in zip(self._history, self._confidence_history) if cls != dominant_class]
        mean_confidence = np.mean(dominant_confs)
        second_best = max(other_confs) if other_confs else 0.0
        confidence_delta = mean_confidence - second_best

        if (
            stability >= self.agreement_ratio
            and mean_confidence >= self.threshold
            and confidence_delta >= self.min_confidence_delta
            and dominant_class != self._last_emitted
        ):
            self._last_emitted = dominant_class
            self._cooldown_counter = self.cooldown_frames
            self._history.clear()
            self._confidence_history.clear()
            return dominant_class

        return None

    def status(self) -> dict:
        counts: dict[int, int] = {}
        for cls in self._history:
            counts[cls] = counts.get(cls, 0) + 1
        dominant = max(counts, key=counts.get) if counts else None
        return {
            "streak": len(self._history),
            "dominant": dominant,
            "stability": (counts[dominant] / self.consecutive_frames) if dominant is not None else 0.0,
            "cooldown": self._cooldown_counter,
            "last_emitted": self._last_emitted,
        }

    def reset(self) -> None:
        self._history.clear()
        self._confidence_history.clear()
        self._cooldown_counter = 0
        self._last_emitted = None
