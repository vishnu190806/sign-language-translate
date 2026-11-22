from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from src.tts.base import TTSEngine

logger = logging.getLogger(__name__)


class PiperEngine(TTSEngine):
    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        sample_rate: int = 22050,
        length_scale: float = 1.0,
        sentence_silence: float = 0.0,
        voice_dir: str | None = None,
    ):
        self.voice = voice
        self.sample_rate = sample_rate
        self.length_scale = length_scale
        self.sentence_silence = sentence_silence
        self.voice_dir = Path(voice_dir) if voice_dir else None
        self._piper = None
        self._available = False
        self._last_synthesis_ms: float = 0.0
        self._offline_mode = os.environ.get("PIPER_OFFLINE", "0") == "1"
        self._syn_config = None
        try:
            from piper.config import SynthesisConfig

            if self.length_scale != 1.0:
                self._syn_config = SynthesisConfig(length_scale=self.length_scale)
        except Exception:
            self._syn_config = None
        self._try_init()

    def _try_init(self) -> None:
        try:
            from piper import PiperVoice
            if self.voice_dir and self.voice_dir.exists():
                onnx_path = self.voice_dir / f"{self.voice}.onnx"
                if onnx_path.exists():
                    self._piper = PiperVoice.load(str(onnx_path))
                    self._available = True
                    logger.info(f"Piper TTS loaded with voice from local path: {onnx_path}")
                elif not self._offline_mode:
                    raise FileNotFoundError(
                        f"Piper voice file not found at {onnx_path}. "
                        f"Download it (e.g. rhasspy/piper-voices) into voice_dir."
                    )
                else:
                    logger.warning(f"Piper TTS offline mode: voice file not found at {onnx_path}")
                    self._available = False
            elif not self._offline_mode:
                raise FileNotFoundError(
                    f"Piper voice '{self.voice}' has no local files and voice_dir is not set. "
                    f"Download the voice (e.g. rhasspy/piper-voices) and set tts.voice_dir, "
                    f"or place <voice>.onnx[.json] where PiperVoice can find it."
                )
            else:
                logger.warning("Piper TTS offline mode: no voice_dir provided")
                self._available = False
        except Exception as e:
            logger.warning(f"Piper TTS not available: {e}")
            self._available = False

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not self._available or self._piper is None:
            raise RuntimeError("Piper TTS is not available")

        t_start = time.perf_counter()

        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._piper.synthesize_wav(text, wf, syn_config=self._syn_config)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            sample_rate = wf.getframerate()

        self._last_synthesis_ms = (time.perf_counter() - t_start) * 1000

        if self.sentence_silence > 0:
            silence_samples = int(sample_rate * self.sentence_silence)
            silence = np.zeros(silence_samples, dtype=np.float32)
            audio_f32 = audio.astype(np.float32) / 32768.0
            return np.concatenate([audio_f32, silence]), sample_rate

        return audio.astype(np.float32) / 32768.0, sample_rate

    def stream_chunks(self, text: str):
        if not self._available or self._piper is None:
            raise RuntimeError("Piper TTS is not available")
        for chunk in self._piper.synthesize(text, syn_config=self._syn_config):
            raw = chunk.audio_int16_bytes
            if raw is None:
                raw = (chunk.audio_float_array * 32767).astype(np.int16).tobytes()
            yield np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def synthesize_stream(self, text: str) -> tuple[np.ndarray, int]:
        if not self._available or self._piper is None:
            raise RuntimeError("Piper TTS is not available")

        t_start = time.perf_counter()

        audio_chunks = []
        rates = set()
        for chunk in self._piper.synthesize(text, syn_config=self._syn_config):
            raw = chunk.audio_int16_bytes
            if raw is None:
                raw = (chunk.audio_float_array * 32767).astype(np.int16).tobytes()
            audio_chunks.append(raw)
            rates.add(chunk.sample_rate)

        self._last_synthesis_ms = (time.perf_counter() - t_start) * 1000

        if not audio_chunks:
            return np.zeros(0, dtype=np.float32), self.sample_rate

        raw_audio = b"".join(audio_chunks)
        audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        stream_rate = rates.pop() if len(rates) == 1 else self.sample_rate

        if self.sentence_silence > 0:
            silence_samples = int(stream_rate * self.sentence_silence)
            silence = np.zeros(silence_samples, dtype=np.float32)
            audio = np.concatenate([audio, silence])

        return audio, stream_rate

    @property
    def last_synthesis_ms(self) -> float:
        return self._last_synthesis_ms

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def name(self) -> str:
        return "piper"
