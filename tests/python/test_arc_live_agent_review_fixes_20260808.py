"""Regression tests for two Phase 0c fixes from the 2026-08-08 adversarial live-agent
review (`docs/research-notes/live-agent-adversarial-review-2026-08-08.md`, Correctness/Minor).

Spec refs: REQ-ARC-WMTE-6231 (graded-goal-bias exemplar), REQ-ARC-WMTE-6232 (plan/pi reset).

1. `CARNOT_ARC_GRADED_GOAL_BIAS=1`'s "win-state exemplar" was captured from `grid_of(latest)`
   -- the frame RETURNED BY the winning action, which for these games already shows the next
   level's opening board, not the just-completed level's win state. That made the graded goal
   bias an inverted gradient pulling search back toward the start instead of toward the goal.
   Fixed in `_observe_level_boundary` to prefer the last-admitted transition's PRE-action grid.

2. Under `CARNOT_ARC_ACTIVE_PROBE=1`, a one-step probe plan leaves `self.pi` stale at 1 after
   it is consumed. A subsequent reinduction (`_induce_and_plan` re-entered directly, bypassing
   the normal level-boundary reset) installed a fresh `self.plan` without resetting `self.pi`,
   so the policy either silently skipped the new plan's first step or raised IndexError on a
   second one-step plan. Fixed by resetting `self.plan`/`self.pi` at `_induce_and_plan` entry,
   plus a bounds guard on the induce branch's direct `_next_plan_move()` call as defense in
   depth for any future regression of the reset.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_executable_world_model import Transition


class _Frame:
    """Minimal frame stub -- the policy reads `levels_completed` and the grid, nothing else."""

    def __init__(self, grid: np.ndarray, level: int = 0) -> None:
        self.frame = [grid.tolist()]
        self.levels_completed = level
        self.state = "NOT_FINISHED"
        self.score = 0
        self.available_actions = [1, 2, 3, 4, 5, 6]


def _grid(seedval: int, n: int = 4) -> np.ndarray:
    rng = np.random.RandomState(seedval)
    return rng.randint(0, 4, size=(n, n)).astype(int)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Keep every test in this file off the LLM -- these tests are about plan/pi and exemplar
    bookkeeping, not induction quality, so induction is short-circuited (or monkeypatched)
    rather than exercised for real."""
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")


def test_win_state_exemplar_prefers_pre_action_transition_grid_over_post_transition_frame():
    """The fixed `_observe_level_boundary` must capture the LAST TRANSITION's pre-action grid
    (the true last-observed board of the level just won), not `grid_of(latest)` (the frame
    returned by the winning action itself, which already shows the next level's board)."""
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)

    # Establish the baseline level (0) -- the first call only initializes bookkeeping.
    pol._observe_level_boundary(_Frame(_grid(1), level=0), frames_seen=1)
    assert pol._previous_level_complete_grid is None

    pre_win_grid = _grid(2)
    post_transition_grid = _grid(3)
    # Simulate the transition-collection block appending the winning action's transition
    # just before the boundary check fires, exactly as `_next_move_routed` does.
    pol.transitions.append(Transition(pre_win_grid, 1, None, post_transition_grid, 0, 1))

    latest = _Frame(post_transition_grid, level=1)
    events = pol._observe_level_boundary(latest, frames_seen=2)

    assert events, "expected a level-up event to fire"
    assert pol._previous_level_complete_grid is not None
    assert np.array_equal(pol._previous_level_complete_grid, pre_win_grid)
    assert not np.array_equal(pol._previous_level_complete_grid, post_transition_grid)


def test_win_state_exemplar_falls_back_to_latest_frame_when_no_transition_recorded():
    """When no transition has been recorded yet (edge case -- e.g. a level-up detected on the
    very first observed frame), the old `grid_of(latest)` capture is still used rather than
    leaving the exemplar permanently None. Not the common case, but must not crash."""
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    pol._observe_level_boundary(_Frame(_grid(1), level=0), frames_seen=1)
    assert not pol.transitions

    latest = _Frame(_grid(4), level=1)
    events = pol._observe_level_boundary(latest, frames_seen=2)

    assert events
    assert pol._previous_level_complete_grid is not None


def test_induce_and_plan_resets_stale_plan_and_pi_even_on_early_return():
    """The real (non-monkeypatched) `_induce_and_plan`, even on its `disabled_by_env` early
    return, must clear a stale `self.plan`/`self.pi` left behind by a just-consumed one-step
    active-probe plan -- pre-fix, the early return touched neither field, so a subsequent
    `phase == "execute"` step would resume indexing from the stale `pi`."""
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    # Simulate exactly what CARNOT_ARC_ACTIVE_PROBE=1 leaves behind after its one-step plan
    # (`self.plan = [chosen_probe.action.as_plan_step()]; self.pi = 0`) is consumed once.
    pol.plan = [{"action": 3, "data": None}]
    pol.pi = 1

    pol._induce_and_plan()

    assert pol.plan == []
    assert pol.pi == 0


def test_next_move_induce_branch_survives_stale_pi_if_induce_and_plan_ever_regresses(
    monkeypatch,
):
    """Defense-in-depth: even if a future edit to `_induce_and_plan` reintroduces the stale-pi
    defect (installs a fresh plan without resetting `self.pi`), the induce branch's own bounds
    guard must not let `_next_plan_move` index out of range. This monkeypatches
    `_induce_and_plan` to reproduce the pre-fix shape directly, independent of whether the
    entry-reset above still holds, so it stays a real regression test for the guard itself."""
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    pol.phase = "induce"
    pol.induced = False
    pol._execute_plan_from_current = True

    def _fake_induce_and_plan(self):
        # Reproduces the exact pre-fix defect: installs a one-step plan without resetting
        # `self.pi`, which a caller left stale at 1 from a just-consumed probe plan.
        self.plan = [{"action": 2, "data": None}]
        self.pi = 1

    monkeypatch.setattr(E3AgentPolicy, "_induce_and_plan", _fake_induce_and_plan)

    latest = _Frame(_grid(5), level=0)
    kind, data = pol.next_move([], latest)  # must not raise IndexError

    assert pol._prov_top == "induce.no_plan.explorer"
