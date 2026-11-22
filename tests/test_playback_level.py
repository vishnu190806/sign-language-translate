from __future__ import annotations

import numpy as np


def _manager():
    from src.tts.playback import AudioPlaybackManager
    from unittest.mock import MagicMock
    return AudioPlaybackManager(MagicMock())


def test_level_defaults_zero():
    assert _manager().level == 0.0


def test_level_reflects_played_audio_rms():
    import sys
    from unittest.mock import MagicMock
    fake_sd = MagicMock()
    sys.modules["sounddevice"] = fake_sd
    try:
        m = _manager()
        seen = []
        fake_sd.play.side_effect = lambda *a, **k: seen.append(m.level)
        m._play_audio(np.ones(1000, dtype=np.float32), 22050)
        assert seen and seen[0] == 1.0
        assert m.level == 0.0
    finally:
        del sys.modules["sounddevice"]
