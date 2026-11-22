from __future__ import annotations

import numpy as np

from src.tts.base import TTSEngine


class KokoroEngine(TTSEngine):
    SAMPLE_RATE = 24000

    def __init__(
        self,
        voice: str = "af_heart",
        lang_code: str = "a",
        pipeline=None,
        **_ignored,
    ):
        self.voice = voice
        self.lang_code = lang_code
        self._pipeline = pipeline

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code=self.lang_code)
        return self._pipeline

    def stream_chunks(self, text: str):
        import numpy as np

        if not text or not text.strip():
            raise ValueError("KokoroEngine.stream_chunks requires non-empty text")
        pipe = self._ensure_pipeline()
        for _, _, audio in pipe(text, voice=self.voice):
            yield np.asarray(audio, dtype=np.float32)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not text or not text.strip():
            raise ValueError("KokoroEngine.synthesize requires non-empty text")
        pipe = self._ensure_pipeline()
        chunks = [audio for _, _, audio in pipe(text, voice=self.voice)]
        if not chunks:
            raise RuntimeError("Kokoro returned no audio")
        return np.concatenate(chunks).astype(np.float32), self.SAMPLE_RATE

    @property
    def is_available(self) -> bool:
        try:
            import kokoro  # noqa: F401
        except ImportError:
            return False
        return True

    @property
    def name(self) -> str:
        return "kokoro"
