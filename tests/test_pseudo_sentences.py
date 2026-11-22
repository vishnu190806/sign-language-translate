from __future__ import annotations


def test_pseudo_sentence_boundaries_exact(tmp_path):
    import numpy as np
    from scripts.build_pseudo_sentences import stitch
    a = np.zeros((30, 168), np.float32) + 1.0
    b = np.zeros((40, 168), np.float32) + 2.0
    c = np.zeros((50, 168), np.float32) + 3.0
    feats, bounds = stitch([(a, "A"), (b, "B"), (c, "C")])
    assert feats.shape == (120, 168)
    assert bounds == [("A", 0, 30), ("B", 30, 70), ("C", 70, 120)]
    assert (feats[:30] == 1.0).all() and (feats[70:] == 3.0).all()


def test_uniform_baseline_scores_perfect_on_exact_bounds():
    from scripts.align_viterbi import uniform_split, score_bounds
    pred = uniform_split(120, ["A", "B", "C"])
    assert pred == [("A", 0, 40), ("B", 40, 80), ("C", 80, 120)]
    assert score_bounds(pred, pred, tol=5) == 1.0
    truth = [("A", 0, 30), ("B", 30, 70), ("C", 70, 120)]
    assert score_bounds(pred, truth, tol=5) < 1.0
