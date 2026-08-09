"""REQ-ARC-WMTE-TRAJ-TRANSFER-1 (lever #2, 2026-08-07): trace capture + live cascade wiring.

Two things are tested here, mirroring the split in the implementation:

1. `_begin_level_goal_episode` stashes the completed level's replayable action trace and its
   opening logical grid at the one instant both are still intact (see that method's docstring).
2. `_induce_and_plan`'s level_up_reinduction branch tries the cheap trajectory-transfer stage
   BEFORE paying for `execute_bounded_llm_reinduction`. Promoted to the SHIPPED DEFAULT
   2026-08-08 (REQ-ARC-WMTE-6234, exp6215's live-path A/B); with the flag explicitly OFF
   (`CARNOT_ARC_TRAJECTORY_TRANSFER=0`), the stage never touches it at all.

test_arc_object_relative_trajectory_transfer.py covers the underlying primitive in isolation;
these tests cover only the wiring around it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_executable_world_model import Transition


def _grid(h: int, w: int, cells: dict[tuple[int, int], int]) -> np.ndarray:
    g = np.zeros((h, w), dtype=np.int16)
    for (y, x), color in cells.items():
        g[y, x] = color
    return g


def _win_transitions(n: int = 3) -> list[Transition]:
    grid = np.zeros((4, 4), dtype=np.int16)
    return [
        Transition(grid=grid, action=1, data=None, next_grid=grid, level_before=0, level_after=0)
        for _ in range(n)
    ]


# --------------------------------------------------------------------------- #
# (1) trace capture in _begin_level_goal_episode                              #
# --------------------------------------------------------------------------- #
def test_completed_level_trace_captures_explore_transitions(monkeypatch):
    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
    policy = E3AgentPolicy("lp85", proposer=SimpleNamespace(), target_levels=3, value_head=None)
    policy.transitions = _win_transitions(3)
    policy._episode_transition_start = 0
    policy.plan = []
    policy.pi = 0
    old_root = np.array([[7]], dtype=np.int16)
    policy.root_grid = old_root
    policy.cell = 4

    # First call establishes the baseline (levels_completed=0 -> no boundary fires yet); the
    # SECOND call is the actual level-up, matching the two-call pattern
    # test_scenario_arc_wmte_4533_level_boundary_resets_induction already establishes.
    policy._observe_level_boundary(SimpleNamespace(levels_completed=0), frames_seen=0)
    policy._observe_level_boundary(SimpleNamespace(levels_completed=1), frames_seen=1)

    assert policy._completed_level_trace == [{"action": 1, "data": None} for _ in range(3)]
    assert policy._completed_level_first_grid is old_root  # captured BEFORE the boundary overwrite
    assert policy._completed_level_cell == 4
    # the episode window moved past the captured transitions, per the pre-existing contract
    assert policy._episode_transition_start == 3


def test_completed_level_trace_includes_executed_plan_steps(monkeypatch):
    """A level won by a PLAN replay (not exploration) must still be captured -- self.plan[:self.pi]
    is cleared two statements after the trace-capture point, so it must be read before that."""
    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
    policy = E3AgentPolicy("lp85", proposer=SimpleNamespace(), target_levels=3, value_head=None)
    policy.transitions = []
    policy._episode_transition_start = 0
    policy.plan = [{"action": 6, "data": {"x": 1, "y": 2}}, {"action": 1, "data": None}]
    policy.pi = 2  # both steps executed
    policy.root_grid = np.array([[1]], dtype=np.int16)

    policy._observe_level_boundary(SimpleNamespace(levels_completed=0), frames_seen=0)
    policy._observe_level_boundary(SimpleNamespace(levels_completed=1), frames_seen=1)

    assert policy._completed_level_trace == [
        {"action": 6, "data": {"x": 1, "y": 2}},
        {"action": 1, "data": None},
    ]
    # the clearing two statements later still happens -- the capture did not prevent it
    assert policy.plan == []
    assert policy.pi == 0


def test_completed_level_trace_defaults_empty_before_any_level_up():
    policy = E3AgentPolicy("lp85", proposer=SimpleNamespace(), target_levels=3, value_head=None)
    assert policy._completed_level_trace == []
    assert policy._completed_level_first_grid is None


# --------------------------------------------------------------------------- #
# (2) cascade stage in _induce_and_plan                                       #
# --------------------------------------------------------------------------- #
def _confident_setup(policy) -> None:
    """Old/new grids with one matched, unshifted object -> a trivially confident transfer."""
    policy._completed_level_first_grid = _grid(10, 10, {(1, 1): 5})
    policy.root_grid = _grid(10, 10, {(1, 1): 5})
    policy._completed_level_cell = 1
    policy._completed_level_trace = [{"action": 6, "data": {"x": 10, "y": 10}}]


def test_flag_explicitly_off_never_touches_trajectory_transfer_or_llm_reinduction(monkeypatch):
    """REQ-ARC-WMTE-6234 (2026-08-08): the lever's live-path A/B (exp6215, promotion_ready_score
    1.0, 0 harmful regressions across 4/4 games, mutation-proven) promoted this to the SHIPPED
    DEFAULT. This test now covers the EXPLICIT-OFF override (`CARNOT_ARC_TRAJECTORY_TRANSFER=0`)
    rather than the bare default -- the cascade stage must not run, and the expensive LLM tier
    must still be reachable exactly as before, the load-bearing byte-identity property."""
    called = {"llm": False}

    def _fake_llm_reinduction(**kwargs):
        called["llm"] = True
        return SimpleNamespace(
            model_specs="test",
            planned=False,
            skipped="test_stub",
            plan=[],
            reinduce_attempts=0,
            defects=[],
            goal_defects=[],
        )

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _fake_llm_reinduction)
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_TRANSFER", "0")

    policy = E3AgentPolicy("lp85", proposer=SimpleNamespace(), target_levels=3, value_head=None)
    assert policy.trajectory_transfer_enabled is False
    _confident_setup(policy)
    policy.transitions = _win_transitions(1)
    policy._episode_transition_start = 0
    policy._pending_induction_reason = "level_up_reinduction"

    policy._induce_and_plan()

    assert called["llm"] is True  # fell through to the LLM tier, exactly as before this lever
    attempt = policy.induction_attempts[-1]
    assert "trajectory_transfer" not in attempt
    assert attempt.get("engine_source") != "object_relative_trajectory_transfer"


def test_flag_on_confident_transfer_short_circuits_before_llm_reinduction(monkeypatch):
    called = {"llm": False}

    def _fake_llm_reinduction(**kwargs):
        called["llm"] = True
        raise AssertionError("must not reach the expensive LLM tier when the transfer is confident")

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _fake_llm_reinduction)
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_TRANSFER", "1")

    policy = E3AgentPolicy("lp85", proposer=SimpleNamespace(), target_levels=3, value_head=None)
    assert policy.trajectory_transfer_enabled is True
    _confident_setup(policy)
    policy.transitions = _win_transitions(1)
    policy._episode_transition_start = 0
    policy._pending_induction_reason = "level_up_reinduction"

    policy._induce_and_plan()

    assert called["llm"] is False
    attempt = policy.induction_attempts[-1]
    assert attempt["trajectory_transfer"]["transfer_confident"] is True
    assert attempt["planned"] is True
    assert attempt["engine_source"] == "object_relative_trajectory_transfer"
    assert policy.plan == [{"action": 6, "data": {"x": 10, "y": 10}}]


def test_bare_default_is_now_enabled_no_env_override(monkeypatch):
    """REQ-ARC-WMTE-6234 (2026-08-08): pins the SHIPPED DEFAULT itself, with no env override at
    all -- distinct from the two tests above, which pin the explicit-on and explicit-off
    overrides. A fresh policy with `CARNOT_ARC_TRAJECTORY_TRANSFER` unset must resolve to
    trajectory_transfer_enabled=True, matching SUBMITTED_OBJECT_RELATIVE_TRAJECTORY_TRANSFER_ENABLED."""
    monkeypatch.delenv("CARNOT_ARC_TRAJECTORY_TRANSFER", raising=False)
    assert agent.SUBMITTED_OBJECT_RELATIVE_TRAJECTORY_TRANSFER_ENABLED is True

    policy = E3AgentPolicy("lp85", proposer=SimpleNamespace(), target_levels=3, value_head=None)
    assert policy.trajectory_transfer_enabled is True


def test_flag_on_unconfident_transfer_falls_through_to_llm_reinduction(monkeypatch):
    """Flag on, but the two grids share no matchable objects -> transfer_confident is False and
    the cascade must fall through to the next tier, not fabricate a plan."""
    called = {"llm": False}

    def _fake_llm_reinduction(**kwargs):
        called["llm"] = True
        return SimpleNamespace(
            model_specs="test",
            planned=False,
            skipped="test_stub",
            plan=[],
            reinduce_attempts=0,
            defects=[],
            goal_defects=[],
        )

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _fake_llm_reinduction)
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_TRANSFER", "1")

    policy = E3AgentPolicy("lp85", proposer=SimpleNamespace(), target_levels=3, value_head=None)
    policy._completed_level_first_grid = _grid(10, 10, {(1, 1): 5})
    policy.root_grid = _grid(10, 10, {(8, 8): 9})  # different color entirely -- no match possible
    policy._completed_level_cell = 1
    policy._completed_level_trace = [{"action": 6, "data": {"x": 10, "y": 10}}]
    policy.transitions = _win_transitions(1)
    policy._episode_transition_start = 0
    policy._pending_induction_reason = "level_up_reinduction"

    policy._induce_and_plan()

    assert called["llm"] is True
    attempt = policy.induction_attempts[-1]
    assert attempt["trajectory_transfer"]["transfer_confident"] is False
    assert attempt.get("engine_source") != "object_relative_trajectory_transfer"


def test_flag_on_no_completed_trace_is_reported_not_silently_skipped(monkeypatch):
    """Flag on but nothing to transfer yet (e.g. the very first level) -- must be distinguishable
    from a ran-and-declined cell, per the fire-counter discipline (exp5836 lesson)."""
    monkeypatch.setattr(
        agent,
        "execute_bounded_llm_reinduction",
        lambda **kwargs: SimpleNamespace(
            model_specs="test",
            planned=False,
            skipped="test_stub",
            plan=[],
            reinduce_attempts=0,
            defects=[],
            goal_defects=[],
        ),
    )
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_TRANSFER", "1")

    policy = E3AgentPolicy("lp85", proposer=SimpleNamespace(), target_levels=3, value_head=None)
    assert policy._completed_level_trace == []
    policy.root_grid = _grid(4, 4, {(1, 1): 2})
    policy.transitions = _win_transitions(1)
    policy._episode_transition_start = 0
    policy._pending_induction_reason = "level_up_reinduction"

    policy._induce_and_plan()

    attempt = policy.induction_attempts[-1]
    assert attempt["trajectory_transfer"] == {"skipped": "no_completed_level_trace"}
