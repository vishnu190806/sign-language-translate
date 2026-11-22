from __future__ import annotations

import numpy as np


def _frame(h=480, w=640):
    rng = np.random.default_rng(0)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def test_panels_darken_frame():
    from src.app.overlay import draw_predictions
    f = _frame()
    out = draw_predictions(f, [("hello", 0.9)], "hi.")
    assert out.shape == f.shape
    assert abs(int(out[:100, :].mean()) - int(f[:100, :].mean())) > 1


def test_confidence_color_thresholds():
    from src.app.overlay import conf_color
    assert conf_color(0.9) == (0, 210, 0)
    assert conf_color(0.5) == (0, 165, 255)
    assert conf_color(0.2) == (60, 60, 220)


def test_sentence_wraps_to_two_lines():
    from src.app.overlay import draw_predictions
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    long_sent = " ".join(["word"] * 40)
    out = draw_predictions(f, [], long_sent)
    rows = [r for r in range(400, 480) if (out[r, 10:630] > 0).any()]
    assert len(rows) > 14


def test_orb_grows_with_level():
    from src.app.overlay import draw_predictions
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    a = draw_predictions(f, [], "", orb_level=0.0, tts_playing=True)
    b = draw_predictions(f, [], "", orb_level=1.0, tts_playing=True)
    assert (b > 0).sum() > (a > 0).sum()


def test_debug_panel_adds_content():
    from src.app.overlay import draw_predictions
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    plain = draw_predictions(f, [("hi", 0.5)], "s.")
    debug = draw_predictions(f, [("hi", 0.5)], "s.", debug={"fps": 30.0})
    assert (debug > 0).sum() > (plain > 0).sum()


def test_overlay_fast_enough():
    import time
    from src.app.overlay import draw_predictions
    f = _frame()
    t0 = time.perf_counter()
    for _ in range(10):
        draw_predictions(f, [("a", 0.9), ("b", 0.4), ("c", 0.2)], "hello world.",
                         fps=30.0, debug={"fps": 30.0}, orb_level=0.5, tts_playing=True)
    ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"\noverlay ms/frame: {ms:.2f}")
    assert ms < 100
