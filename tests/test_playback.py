from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.tts.base import TTSEngine
from src.tts.playback import AudioPlaybackManager


class MockTTSEngine(TTSEngine):
    def __init__(self):
        self._call_count = 0

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self._call_count += 1
        return np.zeros(1000, dtype=np.float32), 22050

    @property
    def is_available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "mock"


class TestAudioPlaybackManager:
    def test_init(self):
        engine = MockTTSEngine()
        manager = AudioPlaybackManager(engine)
        assert manager.tts is engine
        assert manager.is_playing is False
        assert manager.current_text == ""

    def test_init_with_params(self):
        engine = MockTTSEngine()
        manager = AudioPlaybackManager(engine, max_queue_size=2, device=0)
        assert manager.device == 0

    def test_speak_non_blocking(self):
        engine = MockTTSEngine()
        manager = AudioPlaybackManager(engine)
        manager.start()

        t_start = time.perf_counter()
        result = manager.speak("hello")
        t_end = time.perf_counter()

        assert result is True
        assert (t_end - t_start) < 0.01

        manager.stop(wait=True, timeout=2.0)

    def test_speak_empty_returns_false(self):
        engine = MockTTSEngine()
        manager = AudioPlaybackManager(engine)
        result = manager.speak("")
        assert result is False
        result = manager.speak("  ")
        assert result is False

    def test_queue_maxsize_1(self):
        engine = MockTTSEngine()
        manager = AudioPlaybackManager(engine, max_queue_size=1)
        manager.start()

        result1 = manager.speak("first")
        result2 = manager.speak("second")
        assert result1 is True
        assert result2 is True

        manager.stop(wait=True, timeout=2.0)

    def test_interrupt(self):
        engine = MockTTSEngine()
        manager = AudioPlaybackManager(engine)
        manager.start()

        manager.speak("hello")
        time.sleep(0.05)
        manager.interrupt()

        assert manager.is_playing is False
        assert manager.current_text == ""

        manager.stop(wait=True, timeout=2.0)

    def test_stop_graceful(self):
        engine = MockTTSEngine()
        manager = AudioPlaybackManager(engine)
        manager.start()

        manager.speak("hello")
        time.sleep(0.05)
        manager.stop(wait=True, timeout=2.0)

        assert manager._running is False

    def test_stop_without_wait(self):
        engine = MockTTSEngine()
        manager = AudioPlaybackManager(engine)
        manager.start()

        manager.stop(wait=False)
        time.sleep(0.1)

        assert manager._running is False

    def test_last_speak_ms_default(self):
        engine = MockTTSEngine()
        manager = AudioPlaybackManager(engine)
        assert manager.last_speak_ms == 0.0

    def test_start_idempotent(self):
        engine = MockTTSEngine()
        manager = AudioPlaybackManager(engine)
        manager.start()
        manager.start()

        assert manager._running is True
        manager.stop(wait=True, timeout=2.0)

    def test_stop_before_start(self):
        engine = MockTTSEngine()
        manager = AudioPlaybackManager(engine)
        manager.stop(wait=True, timeout=1.0)

        assert manager._running is False
