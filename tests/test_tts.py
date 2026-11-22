from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.tts.piper_engine import PiperEngine
from src.tts.factory import create_tts_engine


class TestPiperEngine:
    def test_init_with_voice(self):
        engine = PiperEngine(voice="en_US-lessac-medium")
        assert engine.voice == "en_US-lessac-medium"
        assert engine.sample_rate == 22050
        assert engine.name == "piper"

    def test_init_with_params(self):
        engine = PiperEngine(
            voice="en_US-lessac-medium",
            sample_rate=16000,
            length_scale=1.2,
            sentence_silence=0.5,
        )
        assert engine.sample_rate == 16000
        assert engine.length_scale == 1.2
        assert engine.sentence_silence == 0.5

    def test_synthesize_requires_piper(self):
        engine = PiperEngine(voice="nonexistent_voice")
        if not engine.is_available:
            with pytest.raises(RuntimeError, match="Piper TTS is not available"):
                engine.synthesize("hello")

    def test_last_synthesis_ms_default(self):
        engine = PiperEngine(voice="en_US-lessac-medium")
        assert engine.last_synthesis_ms == 0.0

    def test_factory_creates_piper(self):
        engine = create_tts_engine("piper", voice="en_US-lessac-medium")
        assert isinstance(engine, PiperEngine)

    def test_factory_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            create_tts_engine("unknown_engine")

    def test_factory_edge_raises(self):
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            create_tts_engine("edge")


class TestTTSEngineInterface:
    def test_has_is_available(self):
        engine = PiperEngine(voice="en_US-lessac-medium")
        assert hasattr(engine, "is_available")
        assert isinstance(engine.is_available, bool)

    def test_has_name(self):
        engine = PiperEngine(voice="en_US-lessac-medium")
        assert hasattr(engine, "name")
        assert isinstance(engine.name, str)

    def test_has_synthesize(self):
        engine = PiperEngine(voice="en_US-lessac-medium")
        assert hasattr(engine, "synthesize")
        assert callable(engine.synthesize)
