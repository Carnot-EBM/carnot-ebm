"""REQ-REPORT-7666: SDK-grounded confirmation of executed induced goals."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
from carnot.agentic.arc_goal_confirmation import GoalConfirmation


def frame(seed: int, level: int = 0, state: str = "NOT_FINISHED", layers: int = 1):
    grid = np.full((8, 8), seed % 4, dtype=int).tolist()
    return SimpleNamespace(
        frame=[grid for _ in range(layers)],
        levels_completed=level,
        state=state,
        available_actions=[1, 2, 3, 4, 5, 6],
        score=0,
    )


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    ("case", "level", "state", "layers", "expected"),
    [
        ("wrong_goal", 0, "NOT_FINISHED", 1, "contradiction"),
        ("true_goal", 1, "NOT_FINISHED", 1, "confirmed"),
        ("final_action_level_change", 2, "WIN", 1, "confirmed"),
        ("delayed_animation", 0, "NOT_FINISHED", 3, "unknown"),
        ("unknown_terminal", 0, "MYSTERY", 1, "unknown"),
        ("loss", 0, "GAME_OVER", 1, "contradiction"),
    ],
)
def test_independent_goal_fixture(
    seed: int, case: str, level: int, state: str, layers: int, expected: str
) -> None:
    """SCENARIO-REPORT-7666-OBSERVATION: 48 separate SDK action outcomes."""
    before = frame(seed)
    guard = GoalConfirmation()
    guard.arm(before, frames_seen=1, level=0, predicted_goal=True, plan_length=1)
    outcome = guard.observe(frame(seed + 1, level, state, layers), frames_seen=2)
    assert outcome["status"] == expected, case
    assert outcome["predicted_goal"] is True
    assert outcome["sdk_level"] == level
    assert outcome["contradiction"] is (expected == "contradiction")


def test_stale_frame_and_timeout_remain_unknown() -> None:
    """SCENARIO-REPORT-7666-OBSERVATION: no fresh frame is no negative label."""
    before = frame(7)
    guard = GoalConfirmation()
    guard.arm(before, frames_seen=1, level=0, predicted_goal=True, plan_length=1)
    assert guard.observe(before, frames_seen=1)["status"] == "unknown"
    assert guard.timeout()["status"] == "unknown"


def test_hidden_alias_and_goal_before_dedup() -> None:
    """SCENARIO-REPORT-7666-OBSERVATION: model state is not SDK truth."""
    before = frame(3)
    guard = GoalConfirmation()
    guard.arm(before, frames_seen=1, level=0, predicted_goal=False, plan_length=1)
    assert guard.observe(frame(3), frames_seen=2)["status"] == "unknown"
    assert guard.receipts[-1]["contradiction"] is False


def test_missing_endpoint_and_bad_level_remain_unknown() -> None:
    """SCENARIO-REPORT-7666-OBSERVATION: malformed SDK level is not a loss."""
    guard = GoalConfirmation()
    assert guard.observe(frame(1), frames_seen=2)["reason"] == "no_executed_endpoint"
    before = frame(1)
    guard.arm(before, frames_seen=1, level=0, predicted_goal=True, plan_length=1)
    bad = frame(2)
    bad.levels_completed = "not-an-integer"
    assert guard.observe(bad, frames_seen=2)["reason"] == "unknown_level"
    missing = frame(3)
    missing.levels_completed = None
    assert guard.observe(missing, frames_seen=3)["reason"] == "unknown_level"


def test_scored_factory_reaches_guard_and_recovers(monkeypatch) -> None:
    """SCENARIO-REPORT-7666-CONTRADICTION/PARITY: the real policy consumes SDK frames."""
    monkeypatch.setenv("CARNOT_ARC_GOAL_CONFIRMATION", "1")
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")

    class Base:
        game_id = "xx11"

        def __init__(self):
            pass

    agent = make_carnot_agent(Base)()
    policy = agent._policy
    assert isinstance(policy, E3AgentPolicy)
    initial = frame(1)
    policy.next_move([], initial)
    policy.phase = "execute"
    policy.plan = [{"action": 1, "data": None}]
    policy.pi = 0
    policy.induced = True
    policy._goal_confirmation_predicate = lambda grid: True
    policy._prev = None
    assert policy.next_move([initial], initial) == (1, None)
    observed = frame(2)
    move = policy.next_move([initial, observed], observed)
    assert move[0] is not None
    assert policy.goal_confirmation_receipts()[-1]["status"] == "contradiction"
    assert policy.phase == "explore"
    assert policy.plan == []
    assert policy._goal_confirmation_predicate is None


def test_disabled_flag_preserves_plan_action_and_random_state(monkeypatch) -> None:
    """SCENARIO-REPORT-7666-PARITY: default-off plans remain unchanged."""
    monkeypatch.delenv("CARNOT_ARC_GOAL_CONFIRMATION", raising=False)
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    policy = E3AgentPolicy("xx11", proposer=None, explore_budget=6)
    first = frame(1)
    policy.next_move([], first)
    policy.phase = "execute"
    policy.plan = [{"action": 1, "data": None}]
    policy.pi = 0
    policy._goal_confirmation_predicate = lambda grid: True
    state = random.getstate()
    assert policy.next_move([first], first) == (1, None)
    assert policy.goal_confirmation_receipts() == []
    assert random.getstate() == state
