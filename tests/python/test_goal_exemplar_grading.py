"""LEVER #2 (REQ-ARC-WMTE-5593-4): goal-exemplar grading closes GAP-ARCH-GOAL-NOT-VERIFIED.

The goal-consistency veto in `execute_bounded_llm_reinduction` only fires when the grading window
contains >=1 real level-up (a correct FALSE_NEGATIVE_RISK guard). But the toward-NEXT-level window has
zero real level-ups (the episode-transition-start reset at each boundary), so the veto is structurally
inert exactly at a deepening boundary -- a correct-dynamics / WRONG-win-predicate model is still trusted.

The lever injects the already-captured PRIOR-LEVEL win-state grid as one synthetic ground-truth POSITIVE
so the veto becomes informative. These tests exercise that injection's effect on
`score_goal_predicate_consistency` directly (the load-bearing logic), plus that it is default-off.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    Transition,
    score_goal_predicate_consistency,
)

BG = 5
WIN = np.array([[BG, BG], [BG, 9]], dtype=np.int16)  # a "win-state" exemplar grid
MID = np.array([[BG, BG], [9, BG]], dtype=np.int16)  # a non-win in-progress grid


def _noop_window():
    """A toward-next-level window: real transitions, NONE of which is a level-up (level_after==level_before)."""
    return [
        Transition(grid=MID, action=1, data=None, next_grid=MID, level_before=1, level_after=1),
        Transition(grid=MID, action=2, data=None, next_grid=MID, level_before=1, level_after=1),
    ]


def _inject_exemplar(window, exemplar):
    """Mirror the injection in arc_llm_reinduction.execute_bounded_llm_reinduction (lever #2)."""
    ex = np.asarray(exemplar)
    return [
        Transition(grid=ex, action=0, data=None, next_grid=ex, level_before=0, level_after=1),
        *window,
    ]


def test_veto_inert_without_exemplar():
    # Without the exemplar the window has no real level-up -> n_real_levelups==0 -> the veto cannot fire,
    # so even a garbage constant-True predicate scores a trivial accuracy on uninformative data.
    always_true = lambda _g: True  # noqa: E731
    c = score_goal_predicate_consistency(always_true, _noop_window())
    assert c.n_real_levelups == 0  # the inert condition the lever fixes


def test_wrong_predicate_caught_with_exemplar():
    # A WRONG win-predicate (never returns True) must be caught once the real win-state exemplar is present:
    # the exemplar is a real level-up, the predicate returns False on it -> mismatch -> accuracy < 1.0 AND
    # n_real_levelups >= 1 -> the veto in execute_bounded_llm_reinduction WOULD fire (skip -> refactor).
    wrong = lambda _g: False  # noqa: E731
    window = _inject_exemplar(_noop_window(), WIN)
    c = score_goal_predicate_consistency(wrong, window)
    assert c.n_real_levelups >= 1
    assert c.accuracy < 1.0  # veto fires (informative now)


def test_correct_predicate_passes_with_exemplar():
    # A CORRECT win-predicate (True exactly on the win state, False on the in-progress no-op states) must
    # NOT be vetoed -- accuracy 1.0.
    def correct(g):
        return bool(np.array_equal(np.asarray(g), WIN))

    window = _inject_exemplar(_noop_window(), WIN)
    c = score_goal_predicate_consistency(correct, window)
    assert c.n_real_levelups >= 1
    assert c.accuracy == 1.0  # no false veto on a correct predicate


def test_injection_is_prepended_positive():
    # The injected synthetic transition is a real level-up (level_after > level_before) with the exemplar
    # as its next_grid -- so score_goal_predicate_consistency reads it as one ground-truth positive.
    window = _inject_exemplar(_noop_window(), WIN)
    inj = window[0]
    assert inj.level_after > inj.level_before
    assert np.array_equal(inj.next_grid, WIN)
