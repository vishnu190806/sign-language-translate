from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional


class LatencyProfiler:
    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha
        self._stage_ema: dict[str, float] = {}
        self._stage_sum: dict[str, float] = defaultdict(float)
        self._stage_count: dict[str, int] = defaultdict(int)
        self._current_stage: Optional[str] = None
        self._current_start: float = 0.0
        self._total_start: float = 0.0
        self._total_ema: float = 0.0
        self._frame_count: int = 0

    def begin(self) -> None:
        self._total_start = time.perf_counter()

    def start(self, stage: str) -> None:
        self._current_stage = stage
        self._current_start = time.perf_counter()

    def stop(self, stage: str) -> None:
        if self._current_stage != stage or self._current_start == 0.0:
            return
        elapsed = time.perf_counter() - self._current_start
        self._current_stage = None
        self._current_start = 0.0

        if stage not in self._stage_ema:
            self._stage_ema[stage] = elapsed
        else:
            self._stage_ema[stage] = (
                self.ema_alpha * elapsed
                + (1.0 - self.ema_alpha) * self._stage_ema[stage]
            )
        self._stage_sum[stage] += elapsed
        self._stage_count[stage] += 1

    def end_frame(self) -> None:
        total = time.perf_counter() - self._total_start
        self._frame_count += 1
        if self._total_ema == 0.0:
            self._total_ema = total
        else:
            self._total_ema = self.ema_alpha * total + (1.0 - self.ema_alpha) * self._total_ema

    @property
    def total_ms(self) -> float:
        return self._total_ema * 1000

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0

    def stage_ms(self, stage: str) -> float:
        return self._stage_ema.get(stage, 0.0) * 1000

    def stage_avg_ms(self, stage: str) -> float:
        count = self._stage_count.get(stage, 0)
        if count == 0:
            return 0.0
        return (self._stage_sum[stage] / count) * 1000

    def summary(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for stage in sorted(self._stage_ema.keys()):
            result[f"{stage}_ms"] = self.stage_ms(stage)
        result["total_ms"] = self.total_ms
        result["fps"] = self.fps
        return result

    def reset(self) -> None:
        self._stage_ema.clear()
        self._stage_sum.clear()
        self._stage_count.clear()
        self._current_stage = None
        self._current_start = 0.0
        self._total_start = 0.0
        self._total_ema = 0.0
        self._frame_count = 0
