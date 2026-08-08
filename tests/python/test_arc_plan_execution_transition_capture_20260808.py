"""Spec: REQ-ARC-WMTE-6223.

Regression tests for the plan-execution transition-capture bug.

docs/research-notes/live-agent-adversarial-review-2026-08-08.md, "Correctness" section,
major finding 1:

  "Plan execution is fully open-loop despite the class contract 'on divergence, back to
  EXPLORE.' No divergence check exists anywhere; self._prev is nulled at induce time and
  never re-set during execute, so executed plan steps produce no Transition rows and the
  agent learns nothing from them. Corollary: a plan-won level records a stale pre-plan
  explore row as _win_transition under a comment asserting the opposite."

THE BUG. `E3AgentPolicy._next_move_routed` has an EXPLORE branch that, after picking a
move, records `self._prev = (grid_before_action, action, data)` so the transition-
collection block at the top of the NEXT `next_move()` call can pair it with the frame the
action produced and append a real `Transition` to `self.transitions`. The two EXECUTE
branches (continuing a plan, and taking the first step right after induction) called
`self._next_plan_move()` to get the move but never set `self._prev` the same way -- so
`self._prev` stayed `None` for the whole `execute` phase and no plan step ever produced a
`Transition`. `_begin_level_goal_episode` captures `self.transitions[-1]` as the win
example the instant a level completes; with no plan-step transitions ever appended, a
level won by replaying a plan captured a STALE pre-plan explore row instead of the actual
winning action.

THE FIX. `E3AgentPolicy._track_prev_for_transition` mirrors the explore branch's own
`self._prev` / `self._prev_level` / `self.cell` bookkeeping and is now called from both
plan-step call sites in `_next_move_routed`. These tests drive the real policy through a
scripted plan-execution sequence (no LLM, no GPU) and check both halves of the finding:
transitions actually accumulate during `execute`, and a plan-won level's `_win_transition`
reflects the real winning plan step rather than a stale row.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic.arc_competition_agent import E3AgentPolicy


# ----------------------------------------------------------------------------------------
# A minimal frame stub, matching the one in test_arc_action_provenance.py: the policy only
# reads `levels_completed` (via `_level_of`) and the grid (via `grid_of`).
# ----------------------------------------------------------------------------------------


class _Frame:
    def __init__(self, grid: np.ndarray, level: int = 0) -> None:
        self.frame = [grid.tolist()]
        self.levels_completed = level
        self.state = "NOT_FINISHED"
        self.score = 0
        self.available_actions = [1, 2, 3, 4, 5, 6]


def _grid(seedval: int, n: int = 8) -> np.ndarray:
    rng = np.random.RandomState(seedval)
    return rng.randint(0, 4, size=(n, n)).astype(int)


def _drive(policy: E3AgentPolicy, n_actions: int) -> None:
    """Step the real explorer a few times first, so the policy carries genuine explorer
    state (root grid, HUD-mask resolution, etc.) before the test overrides `.plan` /
    `.phase` to reach the execute branches directly. Same warm-up pattern as
    test_arc_action_provenance.py's `_drive`, trimmed to what this file needs (the action
    sequence itself is not asserted on here)."""
    frames: list[_Frame] = []
    latest = None
    for i in range(n_actions):
        kind, _ = policy.next_move(frames, latest)
        if kind is None:
            break
        latest = _Frame(_grid(i + 1), level=0)
        frames.append(latest)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test off the LLM and off any GPU -- these tests must run standalone."""
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _policy_ready_for_plan_execution() -> E3AgentPolicy:
    """Build a policy with warmed-up explorer state, then install a 2-step plan and drop
    it directly into the execute phase -- the only way to reach the execute branches
    without a live LLM (mirrors test_arc_action_provenance.py's plan-branch tests)."""
    pol = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    _drive(pol, 3)
    pol.plan = [{"action": 1, "data": None}, {"action": 2, "data": None}]
    pol.pi = 0
    pol.phase = "execute"
    pol.induced = True
    # Clean slate: isolate this test from whatever `_prev` the warm-up explore steps left
    # behind, matching what `_induce_and_plan`'s own caller does right before entering
    # execute (`self._prev = None` at the induce-branch call site in `_next_move_routed`).
    pol._prev = None
    return pol


class TestPlanStepsNowProduceTransitions:
    def test_first_plan_step_records_prev_but_appends_no_transition_yet(self) -> None:
        """Nothing precedes the first plan step (self._prev was cleared above), so there is
        nothing to pair it WITH yet -- but the step must record its OWN pre-action state
        for the *next* call to complete. That recording is exactly what was missing."""
        pol = _policy_ready_for_plan_execution()
        transitions_before = len(pol.transitions)

        frame0 = _Frame(_grid(101), level=0)
        mv1, _ = pol.next_move([], frame0)

        assert mv1 == 1
        assert len(pol.transitions) == transitions_before, (
            "the first plan step cannot yet have a completed transition to append"
        )
        assert pol._prev is not None, (
            "a plan step must leave (grid, action, data) in self._prev so the "
            "transition-collection block can pair it with the NEXT frame -- before the fix "
            "this stayed None for the whole execute phase"
        )
        assert pol._prev[1] == 1, "the recorded action must be the plan step just taken"

    def test_second_call_completes_the_first_plan_steps_transition(self) -> None:
        """The bug's headline consequence: `self.transitions` must actually grow while a
        plan is executing, not stay frozen at its pre-plan length."""
        pol = _policy_ready_for_plan_execution()
        transitions_before = len(pol.transitions)

        frame0 = _Frame(_grid(101), level=0)
        frame1 = _Frame(_grid(102), level=0)
        pol.next_move([], frame0)  # picks action 1; records self._prev
        mv2, _ = pol.next_move([frame0], frame1)  # completes action 1's transition

        assert mv2 == 2, "the plan's second step must still be reached normally"
        assert len(pol.transitions) == transitions_before + 1, (
            "executing a plan step must append a real Transition on the following call, "
            "exactly like an explore step already does"
        )
        t = pol.transitions[-1]
        assert t.action == 1
        assert t.level_before == 0
        assert t.level_after == 0


class TestWinTransitionReflectsTheActualPlanStep:
    def test_plan_won_level_captures_the_winning_plan_step_not_a_stale_explore_row(
        self,
    ) -> None:
        """The corollary in the finding: with transitions now accumulating during
        `execute`, `_begin_level_goal_episode`'s `self.transitions[-1]` capture must land
        on the plan step that ACTUALLY completed the level -- with NO separate code change
        at that capture site (step 2 of the fix: verify this comes out correct
        automatically once self.transitions grows correctly during execute)."""
        pol = _policy_ready_for_plan_execution()
        # The warm-up in `_policy_ready_for_plan_execution` already ran a few explore steps
        # (to give the policy genuine explorer state), which may itself have appended
        # transitions -- so this must assert a DELTA, not an absolute count.
        transitions_before = len(pol.transitions)

        frame0 = _Frame(_grid(201), level=0)
        frame1 = _Frame(_grid(202), level=0)
        # action 2 (the SECOND plan step) is the one that levels the game up
        frame2 = _Frame(_grid(203), level=1)

        pol.next_move([], frame0)  # picks action 1 (no transition yet)
        pol.next_move([frame0], frame1)  # completes action 1 -> transition #1 (0 -> 0)
        pol.next_move([frame0, frame1], frame2)  # completes action 2 -> transition #2 (0 -> 1)

        assert len(pol.transitions) == transitions_before + 2, (
            "both plan steps must have produced a transition"
        )
        win = pol._win_transition
        assert win is not None, "a level-up must capture a win transition"
        assert win is pol.transitions[-1], (
            "_begin_level_goal_episode reads self.transitions[-1] -- this must be the "
            "SAME object, not a copy, so the two stay in lockstep by construction"
        )
        assert win.action == 2, (
            "the win transition must be the plan step that ACTUALLY leveled up the game "
            "(action 2). Before the fix, self.transitions never grew during execute, so "
            "this would have captured whatever stale explore-phase row happened to be "
            "last in self.transitions, contradicting the comment asserting otherwise."
        )
        assert win.level_before == 0
        assert win.level_after == 1


class TestExistingPlanBranchLabelsAreUnaffected:
    def test_plan_step_provenance_labels_and_action_sequence_unchanged(self) -> None:
        """The fix must not perturb WHICH action a plan step returns or how it is labelled
        -- only that a transition gets recorded alongside it. A regression here would mean
        the fix leaked into routing, not just bookkeeping."""
        pol = _policy_ready_for_plan_execution()
        frame0 = _Frame(_grid(301), level=0)
        frame1 = _Frame(_grid(302), level=0)

        mv1, data1 = pol.next_move([], frame0)
        mv2, data2 = pol.next_move([frame0], frame1)

        assert (mv1, data1) == (1, None)
        assert (mv2, data2) == (2, None)
        assert pol.pi == 2
        assert pol.phase == "execute"
