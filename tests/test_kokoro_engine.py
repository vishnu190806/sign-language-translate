from __future__ import annotations

import numpy as np


class _FakePipe:
    def __init__(self, sr: int = 24000):
        self.sr = sr
        self.texts: list[str] = []

    def __call__(self, text, voice="af_heart"):
        self.texts.append(text)
        yield None, None, np.zeros(2400, dtype=np.float32)


def test_kokoro_engine_synthesize_shape_and_rate():
    from src.tts.kokoro_engine import KokoroEngine
    eng = KokoroEngine(voice="af_heart", pipeline=_FakePipe())
    audio, sr = eng.synthesize("Hello world.")
    assert sr == 24000
    assert isinstance(audio, np.ndarray) and audio.shape == (2400,)
    assert eng.is_available and eng.name == "kokoro"


def test_kokoro_engine_empty_text_raises():
    import pytest
    from src.tts.kokoro_engine import KokoroEngine
    eng = KokoroEngine(pipeline=_FakePipe())
    with pytest.raises(ValueError):
        eng.synthesize("   ")


def test_factory_creates_kokoro():
    from src.tts.base import TTSEngine
    from src.tts.factory import create_tts_engine
    from src.tts.kokoro_engine import KokoroEngine
    eng = create_tts_engine("kokoro", pipeline=_FakePipe())
    assert isinstance(eng, KokoroEngine)
    assert isinstance(eng, TTSEngine)


def test_kokoro_stream_chunks_first_audio_fast():
    from src.tts.kokoro_engine import KokoroEngine
    eng = KokoroEngine(pipeline=_FakePipe())
    chunks = list(eng.stream_chunks("Hello world. This is a second sentence."))
    assert len(chunks) >= 1
    assert all(isinstance(c, np.ndarray) for c in chunks)
    assert sum(len(c) for c in chunks) == 2400


def test_piper_stream_chunks_yields_int16_as_float():
    from unittest.mock import MagicMock
    from src.tts.piper_engine import PiperEngine
    chunk = MagicMock()
    chunk.audio_int16_bytes = (np.arange(100, dtype=np.int16)).tobytes()
    chunk.sample_rate = 22050
    eng = PiperEngine.__new__(PiperEngine)
    eng._piper = MagicMock()
    eng._piper.synthesize.return_value = iter([chunk])
    eng._available = True
    eng.sample_rate = 22050
    eng.sentence_silence = 0.0
    eng._syn_config = None
    out = list(eng.stream_chunks("hi"))
    assert len(out) == 1 and out[0].shape == (100,)


def test_kokoro_accepts_piper_style_kwargs():
    from src.tts.factory import create_tts_engine
    from src.tts.kokoro_engine import KokoroEngine
    eng = create_tts_engine(
        "kokoro", voice="af_heart", sample_rate=22050,
        length_scale=1.0, sentence_silence=0.0, pipeline=_FakePipe(),
    )
    assert isinstance(eng, KokoroEngine)
    audio, sr = eng.synthesize("Hello.")
    assert sr == 24000 and len(audio) == 2400
