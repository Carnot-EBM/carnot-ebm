"""Tests for wiring InertClickSigPruner.rank_candidates/observe into StepwiseExplorer's live
composition chain -- the loose end task 9's own DONE note flagged as not yet done.

Spec refs: REQ-ARC-FCP-5595, SCENARIO-ARC-FCP-5595-LIVE-WIRING-CANDIDATES,
SCENARIO-ARC-FCP-5595-LIVE-WIRING-OBSERVE, SCENARIO-ARC-FCP-5595-DEFAULT-OFF-PARITY.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.agentic import arc_competition_agent as comp
from carnot.agentic.arc_competition_agent import (
    SUBMITTED_AGENT_CONFIG,
    SUBMITTED_INERT_CLICK_PRUNER_ENABLED,
    E3AgentPolicy,
    StepwiseExplorer,
)
from carnot.agentic.arc_inert_click_pruner import InertClickSigPruner, coerce_inert_click_pruner


def test_coerce_inert_click_pruner_none_false_true_and_instance() -> None:
    assert coerce_inert_click_pruner(None) is None
    assert coerce_inert_click_pruner(False) is None
    default = coerce_inert_click_pruner(True)
    assert isinstance(default, InertClickSigPruner)
    instance = InertClickSigPruner(lambda frame: frame)
    assert coerce_inert_click_pruner(instance) is instance
    # anything else (a Mapping, a duck-typed spy, a stray string) falls through to None --
    # matching coerce_program_synthesis_filter's own strict-isinstance discipline.
    assert coerce_inert_click_pruner({"mode": "on"}) is None
    assert coerce_inert_click_pruner("on") is None


class _SpyRankPruner:
    """Records rank_candidates calls and drops the last row -- isolates "is the wiring
    calling the right method with the right args at the right point" from InertClickSigPruner's
    own signature-gating logic (separately covered by test_arc_inert_click_pruner.py)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, list[dict]]] = []

    def rank_candidates(self, frame: Any, rows: list[dict]) -> list[dict]:
        self.calls.append((frame, list(rows)))
        return rows[:-1]


def test_scenario_arc_fcp_5595_stepwise_explorer_uses_live_rank_candidates() -> None:
    """SCENARIO-ARC-FCP-5595-LIVE-WIRING-CANDIDATES: _candidates calls
    inert_click_pruner.rank_candidates and uses its (filtered) return value."""

    spy = _SpyRankPruner()
    explorer = StepwiseExplorer(
        online_discriminative=False,
        navigation_cost_tiebreak=False,
        frame_change_scorer=None,
        action_effect_expansion_prior=False,
    )
    # coerce_inert_click_pruner only accepts True/False/None/a real InertClickSigPruner
    # (matching coerce_program_synthesis_filter's own strict isinstance discipline) -- set the
    # attribute directly to isolate "does _candidates consume self.inert_click_pruner correctly"
    # from "does the coercion function accept this value" (separately covered above).
    explorer.inert_click_pruner = spy
    frame = SimpleNamespace(frame=np.array([[1]], dtype=np.int16), available_actions=[1, 2])

    candidates = explorer._candidates(frame, path=[])

    assert len(spy.calls) == 1
    seen_frame, seen_rows = spy.calls[0]
    assert seen_frame is frame
    assert {row["action"] for row in seen_rows} == {1, 2}
    # the spy drops the last row -- confirms _candidates propagates the filtered result, not
    # just calls the method and discards it.
    assert len(candidates) == len(seen_rows) - 1


def test_scenario_arc_fcp_5595_stepwise_explorer_skips_when_pruner_none() -> None:
    """No pruner configured -> no call, no crash, candidates pass through unchanged."""

    explorer = StepwiseExplorer(
        online_discriminative=False,
        navigation_cost_tiebreak=False,
        frame_change_scorer=None,
        action_effect_expansion_prior=False,
        inert_click_pruner=None,
    )
    frame = SimpleNamespace(frame=np.array([[1]], dtype=np.int16), available_actions=[1, 2])

    candidates = explorer._candidates(frame, path=[])

    assert {row["action"] for row in candidates} == {1, 2}


def test_scenario_arc_fcp_5595_stepwise_explorer_rank_candidates_failure_is_non_fatal() -> None:
    """A raising rank_candidates must not break candidate generation -- matches every sibling
    optional-filter hook's try/except discipline in _candidates."""

    class _Raises:
        def rank_candidates(self, frame: Any, rows: list[dict]) -> list[dict]:
            raise RuntimeError("boom")

    explorer = StepwiseExplorer(
        online_discriminative=False,
        navigation_cost_tiebreak=False,
        frame_change_scorer=None,
        action_effect_expansion_prior=False,
    )
    explorer.inert_click_pruner = _Raises()
    frame = SimpleNamespace(frame=np.array([[1]], dtype=np.int16), available_actions=[1, 2])

    candidates = explorer._candidates(frame, path=[])

    assert {row["action"] for row in candidates} == {1, 2}


class _FakeFrame:
    """Minimal stand-in for an arcengine frame: only .frame is read by grid_of."""

    def __init__(self, grid: np.ndarray) -> None:
        self.frame = grid
        self.state = "NOT_FINISHED"
        self.levels_completed = 0


class _SpyObservePruner:
    """Records observe() calls -- isolates "is _ingest feeding the pruner" from the pruner's
    own tally/gating logic."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any, Any, bool]] = []

    def observe(
        self, frame_before: Any, label: Any, frame_after: Any, leveled_up: bool = False
    ) -> None:
        self.calls.append((frame_before, label, frame_after, leveled_up))


def test_scenario_arc_fcp_5595_ingest_feeds_inert_click_pruner_observe() -> None:
    """SCENARIO-ARC-FCP-5595-LIVE-WIRING-OBSERVE: the same per-transition OBSERVE hook that
    feeds dense_curiosity/controllable_novelty_policy/action_prior also feeds
    inert_click_pruner.observe with the realized (before, label, after, leveled_up) transition."""

    spy = _SpyObservePruner()
    explorer = StepwiseExplorer()
    explorer.inert_click_pruner = spy
    grid0 = np.zeros((3, 3), dtype=int)
    explorer._ingest(_FakeFrame(grid0.copy()))
    origin = explorer.cur

    # Simulate having just served a click action from `origin` (mirrors the shape _serve()
    # populates -- see arc_competition_agent.py's `_serve` `self.awaiting = {...}` assignment).
    explorer.awaiting = {
        "origin": origin,
        "action": 6,
        "data": {"x": 1, "y": 1},
        "grid": _FakeFrame(grid0.copy()),
        "level_before": int(explorer.best_level),
        "previous_frame": _FakeFrame(grid0.copy()),
    }

    grid1 = grid0.copy()
    grid1[1, 1] = 5
    explorer._ingest(_FakeFrame(grid1))

    assert len(spy.calls) == 1
    frame_before, label, frame_after, leveled_up = spy.calls[0]
    assert np.array_equal(frame_before.frame, grid0)
    assert label == {"action": 6, "data": {"x": 1, "y": 1}}
    assert np.array_equal(frame_after.frame, grid1)
    assert leveled_up is False


def test_scenario_arc_fcp_5595_ingest_skips_observe_when_pruner_none() -> None:
    """No pruner configured -> _ingest still runs cleanly (no crash, no-op)."""

    explorer = StepwiseExplorer(inert_click_pruner=None)
    grid0 = np.zeros((3, 3), dtype=int)
    explorer._ingest(_FakeFrame(grid0.copy()))
    origin = explorer.cur
    explorer.awaiting = {
        "origin": origin,
        "action": 6,
        "data": {"x": 1, "y": 1},
        "grid": _FakeFrame(grid0.copy()),
        "level_before": int(explorer.best_level),
        "previous_frame": _FakeFrame(grid0.copy()),
    }
    grid1 = grid0.copy()
    grid1[1, 1] = 5
    explorer._ingest(_FakeFrame(grid1))  # must not raise


def test_scenario_arc_fcp_5595_default_off_parity() -> None:
    """SCENARIO-ARC-FCP-5595-DEFAULT-OFF-PARITY: tracks SUBMITTED_INERT_CLICK_PRUNER_ENABLED
    (currently False, per the solve_rate_dropped guardrail -- unvalidated by a matched-budget
    A/B yet) rather than a hardcoded literal, and SUBMITTED_AGENT_CONFIG agrees."""

    assert SUBMITTED_INERT_CLICK_PRUNER_ENABLED is False
    assert (
        SUBMITTED_AGENT_CONFIG["inert_click_pruner_enabled"] is SUBMITTED_INERT_CLICK_PRUNER_ENABLED
    )

    explorer = StepwiseExplorer()
    assert explorer.inert_click_pruner is None  # off by default -> coerced to None

    pol = E3AgentPolicy("paritytest", proposer=None, value_head=lambda _f: 0.0)
    assert pol.explorer.inert_click_pruner is None


def test_scenario_arc_fcp_5595_e3_agent_policy_inert_click_pruner_can_be_opted_in() -> None:
    """Opting in via E3AgentPolicy threads a real InertClickSigPruner to the explorer."""

    pol = E3AgentPolicy(
        "paritytest", proposer=None, value_head=lambda _f: 0.0, inert_click_pruner=True
    )
    assert isinstance(pol.explorer.inert_click_pruner, InertClickSigPruner)
