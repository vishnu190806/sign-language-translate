from __future__ import annotations


def test_isl_order_sov():
    from scripts.synth_continuous import to_isl_order
    assert to_isl_order(["I", "EAT", "APPLE"]) == ["I", "APPLE", "EAT"]
    assert to_isl_order(["YOU", "GO", "HOME"]) == ["YOU", "HOME", "GO"]


def test_novelty_filter():
    from scripts.synth_continuous import is_novel
    seen = {("I", "APPLE", "EAT")}
    assert is_novel(["YOU", "HOME", "GO"], seen)
    assert not is_novel(["I", "APPLE", "EAT"], seen)


def test_stitch_assigns_signers():
    import numpy as np
    from scripts.synth_continuous import stitch_clips
    bank = {"A": [(np.zeros((10, 168), np.float32), "s1"),
                  (np.zeros((12, 168), np.float32), "s2")],
            "B": [(np.zeros((14, 168), np.float32), "s1")]}
    feats, bounds, signers = stitch_clips(["A", "B"], bank, seed=0)
    assert [b[0] for b in bounds] == ["A", "B"]
    assert feats.shape[0] == bounds[-1][2] == 12 + 14
    assert len(signers) == 2 and signers[1] == "s1"
