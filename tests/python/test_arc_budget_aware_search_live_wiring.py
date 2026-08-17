"""REQ-ARC-WMTE-6180 (lever #5, 2026-08-07): budget-exhaustion-meter wiring into the live path.

The estimator (`arc_hud_bar_detector.budget_exhaustion_estimate`) and its consumer
(`arc_solver_kit.budget_aware_path_cost_weight`) already existed, built and tested, but nothing
on the SCORED path called either -- REQ-ARC-WMTE-6180's own implementation-status row explicitly
declared wiring out of scope, a separate follow-up. This file covers that follow-up:

  1. The flag ladder (explicit kwarg > env override > the kit's own default).
  2. PRODUCER: `StepwiseExplorer._ingest` accumulates a raw-frame buffer and updates
     `self.actions_remaining_estimate`, only when the flag is on and a HUD mask is admitted.
  3. CONSUMER: `StepwiseExplorer._frontier` calls `budget_aware_path_cost_weight` in place of the
     bare `depth` cost term, ONLY when the flag is on -- verified by intercepting the call, not by
     an emergent node-selection outcome (depth_cost is monotonic in depth, so most branches never
     actually reorder nodes at different depths; the mechanism, not a specific reordering, is what
     this file pins).

REQ-ARC-WMTE-6235 (2026-08-08, ARC live-agent improvement plan Phase 1c): promoted to the
SHIPPED DEFAULT after exp6216's live-path A/B reproduced cleanly on a fresh, unflagged re-run
(deadline misses 6 -> 0 across 6 games, 0 harmful regressions, promotion_ready_score=1.0,
mutation-proven). Tests below that exercised the "flag off" path via the BARE default now pass
`budget_aware_search=False` explicitly; a new test in each section pins the new bare default.

2026-08-17 seam update (gate-timeout fix). `_ingest` no longer re-runs the batch
`budget_exhaustion_estimate` over the whole buffer each action (that was O(n^2) per run and
77% of a profiled gate game's wall clock). It now feeds `IncrementalBudgetExhaustionEstimator`
one frame at a time. The three producer tests that intercepted the old batch-function seam now
intercept the estimator class; each test's CONTRACT is unchanged (flag gates the work, no mask
means no work, the returned value is adopted, an estimator error yields None and never a crash).
Output equivalence between the two implementations is asserted separately, step for step, in
tests/python/test_arc_budget_estimator_incremental_equivalence.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_competition_agent import StepwiseExplorer


# --------------------------------------------------------------------------- #
# (1) flag ladder                                                             #
# --------------------------------------------------------------------------- #
def test_flag_defaults_on():
    """REQ-ARC-WMTE-6235: the bare default (no kwarg, no env override) is now True."""
    assert StepwiseExplorer().budget_aware_search_enabled is True


def test_flag_explicit_kwarg_off():
    assert StepwiseExplorer(budget_aware_search=False).budget_aware_search_enabled is False


def test_flag_explicit_kwarg_on():
    assert StepwiseExplorer(budget_aware_search=True).budget_aware_search_enabled is True


def test_flag_explicit_kwarg_off_overrides_env(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_BUDGET_AWARE_SEARCH", "1")
    assert StepwiseExplorer(budget_aware_search=False).budget_aware_search_enabled is False


def test_flag_env_override_on(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_BUDGET_AWARE_SEARCH", "1")
    assert StepwiseExplorer().budget_aware_search_enabled is True


def test_flag_env_override_explicit_off(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_BUDGET_AWARE_SEARCH", "0")
    assert StepwiseExplorer().budget_aware_search_enabled is False


def test_buffer_and_estimate_default_empty_before_any_frame():
    explorer = StepwiseExplorer()
    assert explorer._budget_frames == []
    assert explorer.actions_remaining_estimate is None


# --------------------------------------------------------------------------- #
# (2) producer: _ingest accumulates frames + estimate                         #
# --------------------------------------------------------------------------- #
def test_ingest_appends_every_frame_regardless_of_flag():
    """The buffer fill is unconditional (cheap -- a single list append) so it is ready the
    moment the flag is later flipped; only the ESTIMATE computation is gated."""
    explorer = StepwiseExplorer(budget_aware_search=False)
    assert explorer.budget_aware_search_enabled is False
    frame = np.zeros((4, 4), dtype=np.int16)
    explorer._ingest(frame)
    assert len(explorer._budget_frames) == 1
    assert explorer.actions_remaining_estimate is None  # flag off -> never computed


def test_ingest_computes_estimate_only_when_flag_on_and_mask_present(monkeypatch):
    # 2026-08-17 seam update: intercept the incremental estimator class instead of the
    # retired per-action batch call. The contract under test is the same: no admitted
    # mask means no estimator work; an admitted mask means the estimate is adopted.
    called = {"n": 0}

    class _FakeEstimator:
        def __init__(self, mask: object, **kwargs: object) -> None:
            pass

        def observe(self, frame: object) -> None:
            pass

        def estimate(self) -> dict:
            called["n"] += 1
            return {"actions_remaining_estimate": 42.0, "verdict": "estimate"}

    monkeypatch.setattr(agent, "IncrementalBudgetExhaustionEstimator", _FakeEstimator)

    explorer = StepwiseExplorer(budget_aware_search=True)
    frame = np.zeros((4, 4), dtype=np.int16)

    # No mask admitted yet -> estimator must not be called.
    explorer.hud_mask = None
    explorer._ingest(frame)
    assert called["n"] == 0
    assert explorer.actions_remaining_estimate is None

    # Mask admitted -> estimator is called and the estimate is adopted.
    explorer.hud_mask = np.ones((4, 4), dtype=bool)
    explorer._ingest(frame)
    assert called["n"] == 1
    assert explorer.actions_remaining_estimate == 42.0


def test_ingest_never_calls_estimator_when_flag_off(monkeypatch):
    # 2026-08-17 seam update: same contract as before (flag off -> zero estimator
    # work); the interception point is now the estimator class constructor.
    called = {"n": 0}

    class _FakeEstimator:
        def __init__(self, mask: object, **kwargs: object) -> None:
            called["n"] += 1

        def observe(self, frame: object) -> None:
            pass

        def estimate(self) -> dict:
            return {}

    monkeypatch.setattr(agent, "IncrementalBudgetExhaustionEstimator", _FakeEstimator)
    explorer = StepwiseExplorer(budget_aware_search=False)  # flag explicitly off
    explorer.hud_mask = np.ones((4, 4), dtype=bool)
    explorer._ingest(np.zeros((4, 4), dtype=np.int16))
    assert called["n"] == 0


def test_ingest_estimator_exception_yields_none_not_a_crash(monkeypatch):
    # 2026-08-17 seam update: same contract (an estimator error yields None, never a
    # crash), plus the new recovery detail: the broken estimator is discarded so the
    # next frame rebuilds from the raw buffer, matching the old stateless retry.
    class _RaisingEstimator:
        def __init__(self, mask: object, **kwargs: object) -> None:
            pass

        def observe(self, frame: object) -> None:
            raise RuntimeError("boom")

        def estimate(self) -> dict:
            raise RuntimeError("boom")

    monkeypatch.setattr(agent, "IncrementalBudgetExhaustionEstimator", _RaisingEstimator)
    explorer = StepwiseExplorer(budget_aware_search=True)
    explorer.hud_mask = np.ones((4, 4), dtype=bool)
    explorer._ingest(np.zeros((4, 4), dtype=np.int16))  # must not raise
    assert explorer.actions_remaining_estimate is None
    assert explorer._budget_estimator is None  # discarded -> next frame rebuilds


# --------------------------------------------------------------------------- #
# (3) consumer: _frontier's cost-term substitution                            #
# --------------------------------------------------------------------------- #
def _one_node_graph() -> dict:
    return {
        "n0": {
            "path": [{"action": 1, "data": None}, {"action": 2, "data": None}],
            "untested": [{"action": 3, "data": None}],
            "value": 0.0,
            "frame": "plain",
        }
    }


def test_frontier_never_calls_budget_aware_weight_when_flag_off(monkeypatch):
    called = {"n": 0}

    def _fake_weight(**kwargs):
        called["n"] += 1
        return float(kwargs["depth"])

    monkeypatch.setattr(agent, "budget_aware_path_cost_weight", _fake_weight)
    explorer = StepwiseExplorer(budget_aware_search=False)  # flag explicitly off
    explorer.cur = "root"
    explorer.graph = _one_node_graph()
    assert explorer._frontier() == "n0"
    assert called["n"] == 0


def test_frontier_calls_budget_aware_weight_with_estimate_when_flag_on(monkeypatch):
    calls: list[dict] = []

    def _fake_weight(**kwargs):
        calls.append(dict(kwargs))
        return float(kwargs["depth"])  # keep ordering behavior simple for this test

    monkeypatch.setattr(agent, "budget_aware_path_cost_weight", _fake_weight)
    explorer = StepwiseExplorer(budget_aware_search=True)
    explorer.actions_remaining_estimate = 7.0
    explorer.cur = "root"
    explorer.graph = _one_node_graph()

    assert explorer._frontier() == "n0"
    assert len(calls) == 1
    assert calls[0]["depth"] == 2  # len(path)
    assert calls[0]["plan_length"] == 2
    assert calls[0]["actions_remaining_estimate"] == 7.0


def test_frontier_byte_identical_ordering_when_flag_off_even_with_estimate_set():
    """Byte-identity discipline: with the flag off, the sort key must be built from the exact
    same expression as before this lever, even if actions_remaining_estimate happens to hold a
    stale value (e.g. left over from a flag flip mid-run in a test)."""
    explorer_off = StepwiseExplorer(budget_aware_search=False)
    explorer_off.actions_remaining_estimate = 1.0  # would matter if consulted; must not be
    explorer_off.cur = "root"
    explorer_off.graph = {
        "shallow": {
            "path": [{"action": 1, "data": None}],
            "untested": [{"action": 9, "data": None}],
            "value": 0.0,
            "frame": "a",
        },
        "deep": {
            "path": [{"action": 1, "data": None}, {"action": 2, "data": None}],
            "untested": [{"action": 9, "data": None}],
            "value": 0.0,
            "frame": "b",
        },
    }
    # Plain BFS: shallowest wins, regardless of the (unconsulted) estimate.
    assert explorer_off._frontier() == "shallow"
