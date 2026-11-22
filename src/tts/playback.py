from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

from src.tts.base import TTSEngine

logger = logging.getLogger(__name__)


class AudioPlaybackManager:
    def __init__(self, tts_engine: TTSEngine, max_queue_size: int = 1, device: int | None = None):
        self.tts = tts_engine
        self.device = device
        self._queue: queue.Queue[str] = queue.Queue(maxsize=max_queue_size)
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._is_playing = False
        self._current_text = ""
        self._lock = threading.Lock()
        self._last_speak_ms: float = 0.0
        self._level: float = 0.0

    def start(self) -> None:
        if self._running:
            return
        if self._worker is not None and self._worker.is_alive():
            logger.warning("Playback worker still alive; not starting another")
            self._running = True
            return
        self._running = True
        self._worker = threading.Thread(target=self._play_loop, daemon=True)
        self._worker.start()
        logger.info("Audio playback manager started")

    def stop(self, wait: bool = True, timeout: float = 2.0) -> None:
        self._running = False
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker and wait:
            self._worker.join(timeout=timeout)
            if self._worker.is_alive():
                logger.warning("Playback worker did not exit within timeout")
        self._worker = None

    def speak(self, text: str) -> bool:
        if not text.strip():
            return False
        try:
            self._queue.put_nowait(text)
            return True
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(text)
            return True

    def interrupt(self) -> None:
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._is_playing = False
            self._current_text = ""

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._is_playing

    @property
    def current_text(self) -> str:
        with self._lock:
            return self._current_text

    @property
    def last_speak_ms(self) -> float:
        return self._last_speak_ms

    @property
    def level(self) -> float:
        with self._lock:
            return self._level

    def _play_loop(self) -> None:
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:
                break

            try:
                t_start = time.perf_counter()
                with self._lock:
                    self._is_playing = True
                    self._current_text = text

                audio, sample_rate = self.tts.synthesize(text)
                self._play_audio(audio, sample_rate)

                self._last_speak_ms = (time.perf_counter() - t_start) * 1000
            except Exception as e:
                logger.error(f"TTS playback error: {e}")
            finally:
                with self._lock:
                    self._is_playing = False
                    self._current_text = ""

    def _play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        try:
            with self._lock:
                self._level = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
            import sounddevice as sd
            if self.device is not None:
                sd.play(audio, sample_rate, device=self.device, blocking=True)
            else:
                sd.play(audio, sample_rate, blocking=True)
        except Exception as e:
            logger.warning(f"sounddevice playback failed: {e}")
        finally:
            with self._lock:
                self._level = 0.0
