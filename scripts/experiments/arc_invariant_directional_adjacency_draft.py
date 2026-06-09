"""ARC invariant family: DIRECTIONAL ADJACENCY (the GAP-1 verifier — orientation discrimination).

ops/verifier_gaps.md GAP-1: object_count + palette_histogram_shape are PROVABLY transpose-invariant
(a grid and its transpose have identical object stats and color histograms), so the always-on backbone
cannot tell the correct output from its transpose — a measured, designed-in SHARED null space
(confirmed on TRM real mis-votes: 2/5 uncaptured, results/arc3_trm_verifier_rerank.json).

This family is the missing always-on ORIENTED signal: the distribution of HORIZONTAL adjacent color
pairs (left->right transitions) and VERTICAL adjacent color pairs (top->bottom), scored against the
same distributions over the train OUTPUTS. A transpose SWAPS the H and V distributions, so when the
train outputs are directionally asymmetric (H != V) a transposed candidate scores a high violation.
It is NOT transpose-invariant (unlike the existing always-on families), and it never abstains (defined
for any grid of size >= 1x2 / 2x1), so it slots into the union_max ensemble as an always-on family.

Honest scope: for directionally SYMMETRIC tasks (H ~= V) the transpose is genuinely ambiguous and this
family cannot discriminate it — that residual is irreducible (a square transpose preserving H==V is
indistinguishable from the output by any local statistic). family_score returns LOWER = more consistent,
matching the object_count / palette_histogram convention, in [0,1]. No oracle: uses only train pairs.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def _dir_pair_dists(grid) -> tuple[dict, dict]:
    """(H_dist, V_dist): normalized distributions of (color, right-neighbor) and (color, down-neighbor)
    ordered color pairs. Ordered => a transpose maps H<->V (a vertical pair becomes a horizontal pair)."""
    g = np.asarray(grid)
    if g.ndim != 2:
        return {}, {}
    h, w = g.shape
    H, V = Counter(), Counter()
    if w >= 2:
        for r in range(h):
            row = g[r]
            for c in range(w - 1):
                H[(int(row[c]), int(row[c + 1]))] += 1
    if h >= 2:
        for r in range(h - 1):
            a, b = g[r], g[r + 1]
            for c in range(w):
                V[(int(a[c]), int(b[c]))] += 1

    def _norm(cnt):
        tot = sum(cnt.values())
        return {k: v / tot for k, v in cnt.items()} if tot else {}

    return _norm(H), _norm(V)


def _avg_dists(dists: list[dict]) -> dict:
    if not dists:
        return {}
    keys = set().union(*(d.keys() for d in dists))
    n = len(dists)
    return {k: sum(d.get(k, 0.0) for d in dists) / n for k in keys}


def _l1_half(d1: dict, d2: dict) -> float:
    """L1 distance between two prob dicts, /2 -> [0,1]."""
    keys = set(d1) | set(d2)
    return sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in keys) / 2.0


def _build_signature(train_pairs) -> tuple[dict, dict, float]:
    """Average H and V directional-pair distributions over the train OUTPUTS, plus the train's own
    H-vs-V asymmetry (so a symmetric task -> small asymmetry -> this family naturally discriminates
    less, which is the honest behaviour)."""
    Hs, Vs = [], []
    for p in train_pairs:
        h, v = _dir_pair_dists(p["output"])
        Hs.append(h)
        Vs.append(v)
    sigH, sigV = _avg_dists(Hs), _avg_dists(Vs)
    asymmetry = _l1_half(sigH, sigV)  # how directional the train outputs are (0 = symmetric)
    return sigH, sigV, asymmetry


def family_score(candidate_grid, inv, train_pairs, test_input) -> float:
    """Violation in [0,1]: how far the candidate's directional-adjacency distributions are from the
    train-OUTPUT signature. LOWER = more consistent. Transpose-sensitive (the GAP-1 signal)."""
    sigH, sigV, _asym = _build_signature(train_pairs)
    if not sigH and not sigV:
        return 0.0  # degenerate (1x1 train outputs) -> no signal, stay neutral
    candH, candV = _dir_pair_dists(candidate_grid)
    vH = _l1_half(candH, sigH)
    vV = _l1_half(candV, sigV)
    return (vH + vV) / 2.0


def transpose_violation(candidate_grid, train_pairs) -> float:
    """Targeted GAP-1 probe: how much MORE consistent is the candidate than its transpose? Positive =>
    the family prefers this orientation over its transpose (the orientation-discrimination signal).
    Returns family_score(transpose) - family_score(candidate); > 0 means the candidate is preferred."""
    g = np.asarray(candidate_grid)
    cand = family_score(g, None, train_pairs, None)
    trans = family_score(g.T, None, train_pairs, None)
    return float(trans - cand)
