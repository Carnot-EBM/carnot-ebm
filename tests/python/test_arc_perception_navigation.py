"""Tests for the autonomous perception-navigation policy (REQ-ARC-WMTE-5839): the plan-then-replay state
machine that carries a self-discovered navigation solve behind the scored run_game interface."""

from __future__ import annotations

from carnot.agentic.arc_perception_navigation import PerceptionNavigationPolicy


def _policy_with_plan(plan):
    pol = object.__new__(PerceptionNavigationPolicy)  # bypass __init__'s (slow) solve
    pol.plan = list(plan)
    pol.i = 0
    pol.reset_sent = False
    pol.target = 3
    return pol


def test_reset_then_replays_plan_then_done():
    pol = _policy_with_plan([4, 2, 3])
    assert pol.next_move() == ("RESET", None)
    assert not pol.is_done()
    assert pol.next_move() == (4, None)
    assert pol.next_move() == (2, None)
    assert not pol.is_done()
    assert pol.next_move() == (3, None)
    assert pol.is_done()
    assert pol.next_move() == (None, None)


def test_empty_plan_is_done_after_reset():
    pol = _policy_with_plan([])
    assert not pol.is_done()  # reset not sent yet
    assert pol.next_move() == ("RESET", None)
    assert pol.is_done()
