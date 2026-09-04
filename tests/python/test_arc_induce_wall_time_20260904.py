"""Spec: REQ-ARC-WMTE-7011, SCENARIO-ARC-WMTE-7011-A, SCENARIO-ARC-WMTE-7011-B,
SCENARIO-ARC-WMTE-7011-C.

Every induction attempt carries its wall clock, and the policy tells an optional hook when an
induce starts and ends.

INCIDENT 2026-09-04. The r11l post-fix run took 24,998 s for 2,077 actions. The same day the
classical path measured 272 actions in one second with the LLM off, so the generator was the
whole cost -- and no attempt row said how long any call took. The wall time had to be inferred
from character counts and a decode rate.
"""

from __future__ import annotations

from typing import Any

import pytest

from carnot.agentic.arc_competition_agent import E3AgentPolicy

# Constructing E3AgentPolicy loads the explorer stack (+549MB measured on a fresh worker),
# which the conftest memory watchdog reads as a leak. Same marker the sibling policy tests use.
pytestmark = pytest.mark.memory_watchdog_skip


def _policy(monkeypatch: pytest.MonkeyPatch) -> E3AgentPolicy:
    monkeypatch.delenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", raising=False)
    return E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)


def _body_that_records(policy: E3AgentPolicy, *, raise_after: bool = False):
    def _body() -> str:
        policy.induction_attempts.append(
            {"reason": "stall", "planned": True, "skipped": "", "transition_count": 7}
        )
        if raise_after:
            raise RuntimeError("generator died")
        return "ok"

    return _body


# --- SCENARIO-A: the attempt row is stamped ------------------------------------------------


def test_the_attempt_row_carries_started_at_and_wall_s(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-7011-A: `wall_s` and `started_at` are on the row the body appended."""
    policy = _policy(monkeypatch)
    monkeypatch.setattr(policy, "_induce_and_plan", _body_that_records(policy))

    assert policy._induce_and_plan_timed() == "ok"

    (attempt,) = policy.induction_attempts
    assert isinstance(attempt["wall_s"], float) and attempt["wall_s"] >= 0.0
    assert attempt["started_at"].endswith("+00:00")
    assert attempt["reason"] == "stall" and attempt["planned"] is True


def test_a_raising_body_is_still_stamped_and_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7011-A: timing is a finally, not a happy-path branch. The exception
    keeps propagating exactly as before the wrapper existed."""
    policy = _policy(monkeypatch)
    monkeypatch.setattr(policy, "_induce_and_plan", _body_that_records(policy, raise_after=True))
    with pytest.raises(RuntimeError, match="generator died"):
        policy._induce_and_plan_timed()
    (attempt,) = policy.induction_attempts
    assert "wall_s" in attempt and "started_at" in attempt


def test_a_body_that_records_no_attempt_stamps_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-7011-A: the wrapper stamps only a row the body itself appended. It
    must never reach back and overwrite an OLDER attempt's timing."""
    policy = _policy(monkeypatch)
    policy.induction_attempts.append({"reason": "old", "wall_s": 99.0})
    monkeypatch.setattr(policy, "_induce_and_plan", lambda: None)
    policy._induce_and_plan_timed()
    assert policy.induction_attempts == [{"reason": "old", "wall_s": 99.0}]


# --- SCENARIO-B: the hook sees start and end -----------------------------------------------


def test_the_hook_is_told_start_then_finish_with_the_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7011-B: two calls, in order, and the finish carries the row's fields."""
    policy = _policy(monkeypatch)
    policy._pending_induction_reason = "level_up_reinduction"
    monkeypatch.setattr(policy, "_induce_and_plan", _body_that_records(policy))
    calls: list[tuple[str, dict[str, Any]]] = []
    policy.induction_progress_hook = lambda kind, payload: calls.append((kind, payload))

    policy._induce_and_plan_timed()

    assert [k for k, _ in calls] == ["induction_started", "induction_finished"]
    started, finished = calls[0][1], calls[1][1]
    assert started["attempt_index"] == 0
    assert started["reason"] == "level_up_reinduction"
    assert "transition_count" in started and started["started_at"].endswith("+00:00")
    assert finished["attempt_index"] == 0
    assert finished["wall_s"] == policy.induction_attempts[0]["wall_s"]
    assert finished["planned"] is True and finished["skipped"] == ""
    assert finished["transition_count"] == 7


def test_no_hook_means_no_calls_and_no_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-7011-B: the default is None; the scored path is unchanged."""
    policy = _policy(monkeypatch)
    assert policy.induction_progress_hook is None
    monkeypatch.setattr(policy, "_induce_and_plan", _body_that_records(policy))
    policy._induce_and_plan_timed()
    assert policy._induction_progress_hook_errors == 0


# --- SCENARIO-C: a broken hook cannot break the run ----------------------------------------


def test_a_raising_hook_is_counted_never_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-7011-C: instrumentation is diagnostics; the run is the deliverable."""
    policy = _policy(monkeypatch)
    monkeypatch.setattr(policy, "_induce_and_plan", _body_that_records(policy))

    def _hook(kind: str, payload: dict[str, Any]) -> None:
        raise ValueError("hook exploded")

    policy.induction_progress_hook = _hook
    assert policy._induce_and_plan_timed() == "ok"
    assert policy._induction_progress_hook_errors == 2  # start and finish both counted
    assert policy.induction_attempts[0]["wall_s"] >= 0.0
