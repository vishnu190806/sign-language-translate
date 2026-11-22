from __future__ import annotations

import logging

from src.tts.base import TTSEngine

logger = logging.getLogger(__name__)


def create_tts_engine(engine_name: str = "piper", **kwargs) -> TTSEngine:
    if engine_name == "piper":
        from src.tts.piper_engine import PiperEngine
        return PiperEngine(**kwargs)
    if engine_name == "kokoro":
        from src.tts.kokoro_engine import KokoroEngine
        return KokoroEngine(**kwargs)
    raise ValueError(f"Unknown TTS engine: {engine_name}")
