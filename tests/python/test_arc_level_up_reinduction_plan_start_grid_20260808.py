"""Spec: REQ-ARC-WMTE-6224.

Regression tests for the level-up reinduction plan-start-grid bug.

docs/research-notes/live-agent-adversarial-review-2026-08-08.md, "Correctness" section,
major finding 2:

  "Level-up reinduction plans are computed from the level's opening grid but executed from
  the current displaced state with no RESET. The gate requires at least one post-boundary
  transition, so by construction the agent has already moved when
  _execute_plan_from_current=True replays a plan valid only from the opening grid -- wrong
  from step 0. Fix: plan from the actual current grid (self.transitions[-1].next_grid) when
  executing from current, reserving root_grid for the RESET-replay path."

THE BUG. Several branches inside `_induce_and_plan` hand a "where does the plan start"
grid to a planner (`execute_bounded_llm_reinduction`'s `root_grid=` kwarg, and the
structured-nav / TTT-prior-engine `_call_plan_in_model` calls) by reading `self.root_grid`
directly. `root_grid` is captured the MOMENT a level opens. When
`self._execute_plan_from_current` is True (set only for a level-up reinduction --
see `_begin_level_goal_episode`), `next_move()` replays the produced plan starting from
wherever the agent already is, with no RESET first. A level-up reinduction only fires
after at least one post-boundary transition (exploration that happens after the level
opens, before the induction gate trips), so by the time `_induce_and_plan` runs,
`root_grid` is already stale.

THE FIX. `_induce_and_plan` now resolves a `_plan_start_grid` local once, near the top of
the method, before any branch reads it: `self.transitions[-1].next_grid` when
`self._execute_plan_from_current` is True and at least one transition exists, else
`self.root_grid` unchanged (the RESET-replay path, where the game really will be back at
`root_grid` by the time the plan's first step runs).

These tests drive the real policy through a scenario shaped exactly like the one the
finding describes -- `root_grid` still holding the level's opening grid, and one extra
transition recorded after it (standing in for the required post-boundary exploration) --
then intercept `execute_bounded_llm_reinduction` (the call site every level-up
reinduction reaches, whether or not the cheaper nav / TTT-prior / trajectory-transfer
tiers fire first) and assert it receives the grid from that latest transition, not the
stale opening grid. A second test confirms the RESET-replay path (`_execute_plan_from_
current=False`) is unaffected: there, `root_grid` IS the correct plan-start grid, and the
fix must not change that.

GAP CLOSED 2026-08-08 (found by the hostile verifier reviewing this same fix). The first
version of this fix scoped `_plan_start_grid` to the `level_up_reinduction` branch only,
on the claim that `_execute_plan_from_current` is "guaranteed False" everywhere else. That
claim is false: `_observe_active_probe_transition` (fired from `next_move()` right after
recording any transition, independent of `self.phase`) sets `phase="induce"` directly,
bypassing the two `next_move()` reset sites -- both of which only run inside `if self.phase
== "explore":`. So a policy can re-enter `_induce_and_plan` via the STALL / non-level-up
path with `_execute_plan_from_current` still True from an earlier level-up or active-probe
round. `TestStallPathReentryPlansFromCurrentGrid` below drives exactly that scenario against
the STALL path's own `execute_bounded_llm_reinduction` call site (gated by
`CARNOT_ARC_STALL_REFACTOR_LOOP`, default on) and confirms it too now reads `_plan_start_
grid`, not the stale `self.root_grid`.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_executable_world_model import Transition


def _grid(marker: int, n: int = 4) -> np.ndarray:
    """A tiny, cheaply-distinguishable logical grid. `marker` is stamped into cell (0, 0)
    so two grids built with different markers are never accidentally equal."""
    g = np.zeros((n, n), dtype=np.int16)
    g[0, 0] = marker
    return g


def _capturing_llm_reinduction_stub(calls: list[dict]):
    """A drop-in replacement for `execute_bounded_llm_reinduction` that records the kwargs
    it was called with and returns a harmless not-planned result -- the rest of
    `_induce_and_plan`'s bookkeeping (`attempt.update(...)`) reads attributes off the
    return value, so the stub must carry the same shape as a real `LlmReinductionResult`,
    same pattern `test_arc_trajectory_transfer_cascade.py` already uses for this."""

    def _fake(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model_specs="test",
            planned=False,
            skipped="test_stub",
            plan=[],
            selected_candidate_name="",
            goal_candidate_names=[],
            dynamics_candidate_names=[],
            refinement_rounds_used=0,
            rounds=[],
            engine_retention={},
            counterexamples=[],
            verifier_is_oracle=False,
            goal_predicate=None,
            goal_predicate_satisfiable=False,
            goal_satisfiability={},
            goal_expression="",
            structural_goal_diagnostics={},
            subgoal_search_used=False,
            subgoal_decomposition=[],
            per_subgoal_reachable=[],
            factored_planner_used=False,
            expert_trust_weights=[],
        )

    return _fake


@pytest.fixture(autouse=True)
def _isolate_from_other_induction_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the structured-nav tier off (its env flag is unset by default, but pin it so a
    dirty environment cannot change that) and the trajectory-transfer tier off (same
    reason) -- this test is about the `execute_bounded_llm_reinduction` call site
    specifically, and both of those tiers `return` before reaching it when they fire."""
    monkeypatch.delenv("CARNOT_ARC_STRUCTURED_NAV", raising=False)
    monkeypatch.delenv("CARNOT_ARC_TRAJECTORY_TRANSFER", raising=False)


def _policy_after_post_boundary_move(*, execute_plan_from_current: bool):
    """A policy positioned exactly like the real agent right before a level-up
    reinduction call fires: `root_grid` still holds the OLD (level-opening) grid, and
    `self.transitions` carries one extra transition recorded AFTER the boundary -- the
    post-boundary exploration the reinduction gate requires before it fires -- whose
    `next_grid` is a different grid, standing in for "the agent has already moved".

    Returns `(policy, stale_opening_grid, current_grid)`.
    """
    policy = E3AgentPolicy("xx11", proposer=SimpleNamespace(), target_levels=3, value_head=None)
    stale_opening_grid = _grid(1)
    current_grid = _grid(2)
    policy.root_grid = stale_opening_grid
    policy.transitions = [
        Transition(
            grid=stale_opening_grid,
            action=1,
            data=None,
            next_grid=current_grid,
            level_before=1,
            level_after=1,
        )
    ]
    policy._episode_transition_start = 0
    policy._pending_induction_reason = "level_up_reinduction"
    policy._execute_plan_from_current = execute_plan_from_current
    return policy, stale_opening_grid, current_grid


class TestLevelUpReinductionPlansFromTheCurrentGrid:
    def test_execute_from_current_uses_the_latest_transitions_grid_not_stale_root_grid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy, stale_opening_grid, current_grid = _policy_after_post_boundary_move(
            execute_plan_from_current=True
        )
        calls: list[dict] = []
        monkeypatch.setattr(
            agent, "execute_bounded_llm_reinduction", _capturing_llm_reinduction_stub(calls)
        )

        policy._induce_and_plan()

        assert len(calls) == 1, "execute_bounded_llm_reinduction must fire exactly once"
        received = calls[0]["root_grid"]
        assert np.array_equal(received, current_grid), (
            "a level-up reinduction plan is replayed starting from wherever the agent "
            "already is, with no RESET -- it must be planned from the CURRENT grid (the "
            "latest transition's outcome), not the grid the level happened to open with"
        )
        assert not np.array_equal(received, stale_opening_grid), (
            "sanity check: the stale opening grid and the current grid must actually "
            "differ, or this test cannot tell the fix apart from the bug"
        )

    def test_reset_replay_path_still_plans_from_root_grid_unaffected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When `_execute_plan_from_current` is False, `next_move()` issues a RESET before
        replaying the plan (see the `induce.plan_needs_reset` branch), so the game really
        will be back at `root_grid` by the time the first plan step runs -- the fix must
        leave this path byte-identical to before."""
        policy, stale_opening_grid, _current_grid = _policy_after_post_boundary_move(
            execute_plan_from_current=False
        )
        calls: list[dict] = []
        monkeypatch.setattr(
            agent, "execute_bounded_llm_reinduction", _capturing_llm_reinduction_stub(calls)
        )

        policy._induce_and_plan()

        assert len(calls) == 1
        received = calls[0]["root_grid"]
        assert np.array_equal(received, stale_opening_grid), (
            "the RESET-replay path must be unaffected by this fix: with no post-boundary "
            "movement to account for (the game gets reset before the plan runs), "
            "root_grid IS the correct plan-start grid and must still be used"
        )

    def test_empty_transitions_bails_out_before_any_grid_is_resolved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_plan_start_grid`'s ternary also has an `else self.root_grid` arm for "no
        transition to read a current grid from", matching the finding's own fallback
        wording. That arm is NOT separately exercised here: `_active_transitions()` is a
        suffix of `self.transitions`, so an empty `self.transitions` always means an empty
        `active_transitions`, and `_induce_and_plan` already bails out at its pre-existing
        "no_active_transitions" precondition before `_plan_start_grid` is ever computed --
        this test documents and locks in that ordering (the fallback arm exists for
        robustness against a future change to that ordering, not because it is reachable
        today)."""
        policy = E3AgentPolicy("xx11", proposer=SimpleNamespace(), target_levels=3, value_head=None)
        policy.root_grid = _grid(3)
        policy.transitions = []
        policy._episode_transition_start = 0
        policy._pending_induction_reason = "level_up_reinduction"
        policy._execute_plan_from_current = True
        calls: list[dict] = []
        monkeypatch.setattr(
            agent, "execute_bounded_llm_reinduction", _capturing_llm_reinduction_stub(calls)
        )

        policy._induce_and_plan()

        assert calls == [], "no active transitions -- the method must bail out before planning"
        attempt = policy.induction_attempts[-1]
        assert attempt["skipped"] == "no_active_transitions"


class TestActiveProbeReentryDoesNotResetExecutePlanFromCurrent:
    """The mechanism that makes the STALL-path gap reachable: `_observe_active_probe_transition`
    re-enters induction without going through either of `next_move()`'s two reset sites."""

    def test_observe_active_probe_transition_leaves_execute_plan_from_current_true(self) -> None:
        policy = E3AgentPolicy("xx11", proposer=SimpleNamespace(), target_levels=3, value_head=None)
        policy._execute_plan_from_current = True
        policy._active_probe_pending = SimpleNamespace()
        policy._active_probe_controller = SimpleNamespace(
            observe_transition=lambda *a, **k: SimpleNamespace(
                posterior_entropy_before=0.0,
                posterior_entropy_after=0.0,
                posterior_entropy_reduction=0.0,
                matched_hypotheses=[],
            ),
            diagnostics=lambda: {},
        )
        transition = Transition(
            grid=_grid(1), action=1, data=None, next_grid=_grid(2), level_before=1, level_after=1
        )

        policy._observe_active_probe_transition(transition)

        assert policy.phase == "induce"
        assert policy._pending_induction_reason == "active_probe_observed"
        assert policy._execute_plan_from_current is True, (
            "this re-entry path sets phase='induce' directly and must NOT reset "
            "_execute_plan_from_current -- if it did, the STALL-path gap this class "
            "documents could never actually occur"
        )


class TestStallPathReentryPlansFromCurrentGrid:
    """GAP CLOSED 2026-08-08: the STALL path's own `execute_bounded_llm_reinduction` call site
    (reached whenever `_pending_induction_reason != "level_up_reinduction"`) must also read
    `_plan_start_grid`, because `_execute_plan_from_current` can still be True there via the
    active-probe reentry `TestActiveProbeReentryDoesNotResetExecutePlanFromCurrent` proves."""

    def test_reentry_with_stale_flag_uses_current_grid_not_stale_root_grid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CARNOT_ARC_ACTIVE_PROBE", raising=False)
        monkeypatch.delenv("CARNOT_ARC_STALL_REFACTOR_LOOP", raising=False)
        stale_opening_grid = _grid(1)
        current_grid = _grid(2)
        policy = E3AgentPolicy(
            "xx11",
            proposer=SimpleNamespace(induce=lambda *a, **k: (False, None)),
            target_levels=3,
            value_head=None,
        )
        policy.root_grid = stale_opening_grid
        policy.transitions = [
            Transition(
                grid=stale_opening_grid,
                action=1,
                data=None,
                next_grid=current_grid,
                level_before=1,
                level_after=1,
            )
        ]
        policy._episode_transition_start = 0
        # Not "level_up_reinduction" -- this is the STALL / reentry path, reached e.g. after
        # `_observe_active_probe_transition` re-enters induction (see the sibling test class).
        policy._pending_induction_reason = "active_probe_observed"
        # The bug scenario: True from an earlier level-up or active-probe round, and never
        # reset because this reentry bypasses both of next_move()'s reset sites.
        policy._execute_plan_from_current = True
        calls: list[dict] = []
        monkeypatch.setattr(
            agent, "execute_bounded_llm_reinduction", _capturing_llm_reinduction_stub(calls)
        )

        policy._induce_and_plan()

        assert len(calls) == 1, "the STALL path's own execute_bounded_llm_reinduction must fire"
        received = calls[0]["root_grid"]
        assert np.array_equal(received, current_grid), (
            "the STALL path must also plan from the current grid when "
            "_execute_plan_from_current is still True on reentry, not the stale root_grid "
            "captured when the level opened"
        )
        assert not np.array_equal(received, stale_opening_grid)
