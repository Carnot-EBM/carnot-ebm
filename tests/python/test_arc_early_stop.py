"""Unit tests for the StepwiseExplorer smart grace-period early-stop.

REQ: arc-action-efficiency-early-stop / SCENARIO: grace-period-no-level-cap

WHY (2026-06-20): the live agent reaches a level then burns thousands of actions hunting unreachable
deeper levels (lp85: L1 at action 20, then ran to 7792). The early-stop cuts that fruitless tail to free
WALL-CLOCK (more games within the 12h compute cap) WITHOUT capping levels -- a new level-up resets the
grace window, so reachable consecutive levels are still solved. (Under the real PER-LEVEL scoring metric
the tail does not hurt a solved level's score; the win is compute budget, not score -- see
arc_local_submission_gate.py.) Default is OFF (grace=None) so the live config is unchanged.
"""
from __future__ import annotations

from carnot.agentic.arc_competition_agent import StepwiseExplorer


def _explorer(grace):
    # target_levels high so the target-reached check does not short-circuit the grace logic
    e = StepwiseExplorer(target_levels=5, early_stop_grace=grace)
    e.start_level = 0
    return e


def test_disabled_never_early_stops():
    """grace=None (the live default): never stops early; only explored_out terminates."""
    e = _explorer(None)
    e.best_level = 1
    assert e.is_done(list(range(10_000)), None) is False
    assert e.early_stopped is False


def test_does_not_fire_before_first_level():
    """The grace only arms AFTER >=1 level -- it must NOT cut the productive first-level search
    (sp80 needs ~7200 actions to find L1, all with best_level==start)."""
    e = _explorer(100)
    e.best_level = 0  # no level reached yet (best == start)
    assert e.is_done(list(range(10_000)), None) is False


def test_fires_after_grace_without_new_level():
    """After a level-up, stop once `grace` moves pass with no new level."""
    e = _explorer(100)
    e.best_level = 1
    assert e.is_done(list(range(50)), None) is False    # arms the window at frame 50
    assert e.is_done(list(range(120)), None) is False   # 120-50=70 <= 100
    assert e.is_done(list(range(160)), None) is True     # 160-50=110 > 100 -> stop
    assert e.early_stopped is True


def test_rides_consecutive_levels_no_cap():
    """A new level-up RESETS the window -> multi-level games are NOT capped; only the tail after the
    LAST findable level is cut."""
    e = _explorer(100)
    e.best_level = 1
    assert e.is_done(list(range(50)), None) is False     # window armed at 50 for level 1
    e.best_level = 2                                       # reached level 2 within grace
    assert e.is_done(list(range(120)), None) is False    # new level -> window re-armed at 120
    assert e.is_done(list(range(200)), None) is False    # 200-120=80 <= 100
    assert e.is_done(list(range(230)), None) is True      # 230-120=110 > 100 -> stop AFTER level 2


def test_target_levels_still_short_circuits():
    """The existing target-reached termination is unaffected by the grace."""
    e = StepwiseExplorer(target_levels=1, early_stop_grace=100)
    e.start_level = 0
    e.best_level = 1  # reached start+target -> done regardless of grace
    assert e.is_done([], None) is True
