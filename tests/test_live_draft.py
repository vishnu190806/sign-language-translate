from __future__ import annotations


def test_rules_draft_differs_from_glosses():
    from src.text.rules import gloss_to_sentence
    draft = gloss_to_sentence(["YOU", "HAPPY"])
    assert draft != "YOU HAPPY"
    assert isinstance(draft, str) and len(draft) > 0


def test_overlay_renders_draft_line():
    import numpy as np
    from src.app.overlay import draw_predictions
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_predictions(f, [], "YOU HAPPY", draft="You are happy.")
    assert (out[380:470, 10:630] > 0).any()


def test_countdown_bar_scales_with_progress():
    import numpy as np
    from src.app.overlay import draw_predictions
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    a = draw_predictions(f, [], "HI", auto_speak_progress=0.2)
    b = draw_predictions(f, [], "HI", auto_speak_progress=0.9)
    assert (b > 0).sum() > (a > 0).sum()
    c = draw_predictions(f, [], "HI", auto_speak_progress=None)
    assert (a > 0).sum() > (c > 0).sum()
