"""Tests for the inert-click-signature move-pruner (Reki's dead-signature mechanism,
gated by HazardMovePruner's trust+specificity discipline instead of Reki's greedy K=2).

Contract: InertClickSigPruner learns, per structural signature (color, size, is_rect,
twin_count), which clicked components never change the frame. It prunes ONLY signatures
that (a) have been observed >= min_observations times, (b) have OBSERVED inert fraction
>= min_specificity, and (c) NEVER produced a level-up. Twin blobs sharing a signature
transfer evidence to each other. It is conservative (no prune when unproven / below the
specificity bar) and never prunes a level-completing signature. verifier_is_oracle stays
False.

Spec refs: REQ-ARC-FCP-5595, SCENARIO-ARC-FCP-5595-SIGNATURE-CLASSIFIED-ON-REAL-DATA,
SCENARIO-ARC-FCP-5595-RANK-CANDIDATES-SANITY-CHECK. ops/known-issues.md 2026-07-11 task 9.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_color_blob_salience import connected_color_blobs
from carnot.agentic.arc_inert_click_pruner import InertClickSigPruner, click_signature


def _gof(frame):
    return np.asarray(frame)


def _blank(shape=(8, 8)):
    return np.zeros(shape, dtype=int)


def _grid_with_two_inert_blobs():
    g = _blank()
    g[1:3, 1:3] = 3  # blob A: 2x2 solid rect, color 3
    g[1:3, 5:7] = 3  # blob B: 2x2 solid rect, color 3 -- a TWIN of A
    g[5, 5] = 5  # blob C: single pixel, color 5, unique signature
    return g


def test_click_signature_twin_count_matches_shared_shape_only():
    g = _grid_with_two_inert_blobs()
    blobs = connected_color_blobs(g, min_pixels=1, max_component_fraction=1.0)
    by_color = {b.color: [] for b in blobs}
    for b in blobs:
        by_color[b.color].append(b)
    sig_a = click_signature(by_color[3][0], blobs)
    sig_b = click_signature(by_color[3][1], blobs)
    sig_c = click_signature(by_color[5][0], blobs)
    assert sig_a == sig_b  # twins: identical (color, size, is_rect) -> same signature
    assert sig_a[3] == 1  # twin_count=1 (exactly one OTHER blob shares the signature)
    assert sig_c[3] == 0  # blob C is unique -> no twins
    assert sig_a != sig_c


def test_prunes_signature_confidently_inert_after_evidence_floor():
    p = InertClickSigPruner(_gof, min_observations=4, min_specificity=0.9)
    g0 = _grid_with_two_inert_blobs()
    click_a = '{"action": 6, "data": {"x": 1, "y": 1}}'
    click_b = '{"action": 6, "data": {"x": 5, "y": 1}}'
    for _ in range(4):
        p.observe(g0, click_a, g0.copy(), False)  # no change -> inert
    assert p.should_prune(g0, click_a) is True
    # blob B is a TWIN of blob A (same signature) -> evidence transfers, prunable too
    assert p.should_prune(g0, click_b) is True
    assert p.stats()["pruned"] >= 2


def test_does_not_prune_below_evidence_floor():
    p = InertClickSigPruner(_gof, min_observations=4, min_specificity=0.9)
    g0 = _grid_with_two_inert_blobs()
    click_a = '{"action": 6, "data": {"x": 1, "y": 1}}'
    for _ in range(3):  # 3 < min_observations=4
        p.observe(g0, click_a, g0.copy(), False)
    assert p.should_prune(g0, click_a) is False


def test_does_not_prune_when_specificity_too_low():
    p = InertClickSigPruner(_gof, min_observations=4, min_specificity=0.9)
    g0 = _grid_with_two_inert_blobs()
    click_a = '{"action": 6, "data": {"x": 1, "y": 1}}'
    changed = g0.copy()
    changed[7, 7] = 9  # a real (non-levelup) content change elsewhere on the frame
    p.observe(g0, click_a, g0.copy(), False)  # inert
    p.observe(g0, click_a, g0.copy(), False)  # inert
    p.observe(g0, click_a, changed, False)  # effective, not inert
    p.observe(g0, click_a, changed, False)  # effective, not inert
    # specificity = 2/4 = 0.5 < 0.9 -> tolerated as noisy, NOT pruned (unlike a binary
    # ever-effective-is-sacred rule, which would also not prune, but for a different
    # reason -- here it is the continuous bar that keeps it unpruned)
    assert p.should_prune(g0, click_a) is False


def test_never_prunes_a_signature_that_ever_leveled_up():
    p = InertClickSigPruner(_gof, min_observations=4, min_specificity=0.9)
    g0 = _grid_with_two_inert_blobs()
    click_a = '{"action": 6, "data": {"x": 1, "y": 1}}'
    p.observe(g0, click_a, g0.copy(), True)  # one real level-up -> permanently sacred
    for _ in range(5):
        p.observe(g0, click_a, g0.copy(), False)  # even mostly-inert afterward
    assert p.should_prune(g0, click_a) is False


def test_keyboard_action_never_pruned_and_undecodable_label_safe():
    p = InertClickSigPruner(_gof, min_observations=1, min_specificity=0.5)
    g0 = _grid_with_two_inert_blobs()
    p.observe(g0, '{"action": 2}', g0.copy(), False)  # keyboard nav -> observe no-ops
    assert p.should_prune(g0, '{"action": 2}') is False
    assert p.should_prune(g0, "not-json") is False
    assert p.observed == 0


def test_rank_candidates_drops_only_confidently_inert_click_rows():
    p = InertClickSigPruner(_gof, min_observations=4, min_specificity=0.9)
    g0 = _grid_with_two_inert_blobs()
    click_a = '{"action": 6, "data": {"x": 1, "y": 1}}'
    for _ in range(4):
        p.observe(g0, click_a, g0.copy(), False)
    rows = [
        {"action": 2, "data": None},  # keyboard -> always kept
        {"action": 6, "data": {"x": 1, "y": 1}},  # confidently inert (blob A) -> dropped
        {"action": 6, "data": {"x": 5, "y": 5}},  # unproven blob C -> kept
    ]
    kept = p.rank_candidates(g0, rows)
    assert {"action": 2, "data": None} in kept
    assert {"action": 6, "data": {"x": 5, "y": 5}} in kept
    assert {"action": 6, "data": {"x": 1, "y": 1}} not in kept
    assert len(kept) == 2
