#!/usr/bin/env python
"""Boundary utilities (M9/B2): uniform-split baseline + tolerance scorer + CTC Viterbi."""
from __future__ import annotations


def uniform_split(n_frames: int, glosses: list[str]) -> list[tuple[str, int, int]]:
    bounds = []
    t = 0
    for i, g in enumerate(glosses):
        n = n_frames // len(glosses) + (1 if i < n_frames % len(glosses) else 0)
        bounds.append((g, t, t + n))
        t += n
    return bounds


def score_bounds(pred: list[tuple[str, int, int]],
                 truth: list[tuple[str, int, int]], tol: int = 5) -> float:
    if len(pred) != len(truth):
        return 0.0
    ok = 0
    for (pg, ps, pe), (tg, ts, te) in zip(pred, truth):
        if pg == tg and abs(ps - ts) <= tol and abs(pe - te) <= tol:
            ok += 1
    return ok / max(len(truth), 1)


def viterbi_align(log_probs, gloss_ids: list[int], blank: int = 0) -> list[tuple[int, int, int]]:
    import numpy as np

    lp = np.asarray(log_probs, dtype=np.float64)
    T, V = lp.shape
    G = len(gloss_ids)
    states = [blank] + [g for gid in gloss_ids for g in (gid, blank)]
    S = len(states)
    NEG = -1e9
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int64)
    dp[0, 0] = lp[0, states[0]]
    if S > 1:
        dp[0, 1] = lp[0, states[1]]
    for t in range(1, T):
        for s in range(S):
            cand = [(dp[t - 1, s], s)]
            if s > 0:
                cand.append((dp[t - 1, s - 1], s - 1))
            best_prev = max(cand, key=lambda c: c[0])
            if best_prev[0] > NEG / 2:
                dp[t, s] = best_prev[0] + lp[t, states[s]]
                bp[t, s] = best_prev[1]
    path = np.zeros(T, dtype=np.int64)
    path[T - 1] = int(np.argmax(dp[T - 1, S - 2:]) + S - 2)
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    bounds: list[tuple[int, int, int]] = []
    cur, start = states[path[0]], 0
    for t in range(1, T):
        if states[path[t]] != cur:
            if cur != blank:
                bounds.append((cur, start, t))
            cur, start = states[path[t]], t
    if cur != blank:
        bounds.append((cur, start, T))
    return bounds
