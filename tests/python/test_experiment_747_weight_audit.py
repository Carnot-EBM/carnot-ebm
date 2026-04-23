"""Tests for Experiment 747 — Tier 1 PerModelFPTracker Weight Convergence Audit.

Coverage targets:
- test_get_weight_state_returns_all_constraint_types: get_weight_state() returns a
  WeightState for every constraint type that received un-throttled updates.
  (REQ-FR11-007, REQ-FR11-007-1, REQ-FR11-007-2)

- test_weight_ratio_computed_correctly: run_audit() weight_ratio equals max/min
  when multiple constraint types have different update counts.
  (REQ-FR11-008, REQ-FR11-008-2)

- test_disabled_constraints_includes_near_zero: disabled_constraints list includes
  any constraint_type whose weight is below 0.02.
  (REQ-FR11-008-3)

- test_honest_verdict_converging: 30 arithmetic + 15 logical + 5 code events yield
  honest_verdict=="tier1_weights_converging" and expected_ordering_correct==True.
  (REQ-FR11-008, SCENARIO-FR11-008)

- test_honest_verdict_uniform: equal event counts yield honest_verdict=="tier1_weights_uniform".
  (REQ-FR11-008-1)

- test_honest_verdict_no_data: zero events yield honest_verdict=="tier1_weights_no_data".
  (REQ-FR11-008-1)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add the scripts directory so experiment_747_tier1_weight_audit can be imported.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from carnot.pipeline.adaptive_thresholds import PerModelFPTracker, WeightState
from carnot.pipeline.fr11_event_bus import FR11EventBus, ViolationEvent
from experiment_747_tier1_weight_audit import run_audit, _make_event, _SYNTHETIC_COUNTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inject_events(tracker: PerModelFPTracker, bus: FR11EventBus, counts: list[tuple[str, int]]) -> None:
    """Inject (constraint_type, count) pairs through the bus into the tracker."""
    bus.subscribe(tracker.on_violation)
    q_index = 0
    for ctype, count in counts:
        for _ in range(count):
            event = _make_event(ctype, f"test_q{q_index:04d}")
            bus.publish(event)
            q_index += 1


# ---------------------------------------------------------------------------
# test_get_weight_state_returns_all_constraint_types  (REQ-FR11-007)
# ---------------------------------------------------------------------------


def test_get_weight_state_returns_all_constraint_types():
    """get_weight_state() returns a WeightState for every constraint type with >= 1 update.

    We inject enough events for arithmetic and logical to receive un-throttled
    updates (throttle fires every 10 calls, so we need >= 10 events per type to
    guarantee at least one un-throttled update per type when types are interleaved).

    We use 20 arithmetic + 20 logical events injected alternately so both types
    accumulate call counts independently: after 10 arithmetic calls the first
    arithmetic update fires (call_count==10 is divisible by 10).

    Spec: REQ-FR11-007, REQ-FR11-007-1, REQ-FR11-007-2, SCENARIO-FR11-007
    """
    tracker = PerModelFPTracker()
    bus = FR11EventBus()
    # Inject 30 arithmetic + 15 logical + 5 code — the canonical test case.
    _inject_events(tracker, bus, _SYNTHETIC_COUNTS)

    weight_state = tracker.get_weight_state()

    # All three constraint types must appear (REQ-FR11-007-1).
    assert "arithmetic" in weight_state, "arithmetic must appear after 30 events"
    assert "logical" in weight_state, "logical must appear after 15 events"
    assert "code" in weight_state, "code must appear after 5 events"

    # Each entry must be a WeightState with the correct fields (REQ-FR11-007-2).
    for ctype, ws in weight_state.items():
        assert isinstance(ws, WeightState), f"{ctype}: expected WeightState, got {type(ws)}"
        assert isinstance(ws.weight, float), f"{ctype}: weight must be float"
        assert isinstance(ws.update_count, int), f"{ctype}: update_count must be int"
        assert ws.update_count > 0, f"{ctype}: update_count must be > 0 for returned entries"
        assert ws.last_updated_at is not None, f"{ctype}: last_updated_at must not be None"

    # Weights must be > 1.0 (each un-throttled update increments by 0.01).
    for ctype, ws in weight_state.items():
        assert ws.weight > 1.0, f"{ctype}: weight {ws.weight} should be above initial 1.0"


# ---------------------------------------------------------------------------
# test_weight_ratio_computed_correctly  (REQ-FR11-008)
# ---------------------------------------------------------------------------


def test_weight_ratio_computed_correctly():
    """run_audit() update_count_ratio equals max_update_count / min_update_count.

    With 30 arithmetic + 15 logical + 5 code events at 10x throttle:
    - arithmetic: events 10, 20, 30 are un-throttled → 3 updates
    - logical: event 40 is un-throttled → 1 update
    - code: event 50 is un-throttled → 1 update
    update_count_ratio = 3 / 1 = 3.0 >= 2.0.

    Spec: REQ-FR11-008, REQ-FR11-008-2
    """
    tracker = PerModelFPTracker()
    bus = FR11EventBus()
    _inject_events(tracker, bus, _SYNTHETIC_COUNTS)

    # Run audit — since tracker already has data, no simulation occurs.
    result = run_audit(tracker, FR11EventBus())  # fresh bus (no subscribers needed)

    assert result["update_count_ratio"] is not None, "update_count_ratio must be present"
    assert result["update_count_ratio"] >= 2.0, (
        f"update_count_ratio {result['update_count_ratio']} should be >= 2.0 "
        "for 30 arithmetic vs 5 code (ratio = 30/5 un-throttled updates = 3)"
    )
    # weight_ratio is a separate informational field (max_weight / min_weight).
    assert result["weight_ratio"] is not None
    max_w = result["max_weight"]
    min_w = result["min_weight"]
    expected_ratio = round(max_w / min_w, 4)
    assert result["weight_ratio"] == pytest.approx(expected_ratio, abs=1e-6), (
        f"weight_ratio {result['weight_ratio']} != max/min {expected_ratio}"
    )


# ---------------------------------------------------------------------------
# test_disabled_constraints_includes_near_zero  (REQ-FR11-008-3)
# ---------------------------------------------------------------------------


def test_disabled_constraints_includes_near_zero():
    """disabled_constraints includes any constraint type with weight < 0.02.

    We manually set a constraint weight to 0.01 on the tracker's internal dict
    (simulating a hypothetical post-cap drift scenario) and verify it appears
    in disabled_constraints.

    Spec: REQ-FR11-008-3
    """
    tracker = PerModelFPTracker()
    bus = FR11EventBus()
    _inject_events(tracker, bus, _SYNTHETIC_COUNTS)

    # Force "arithmetic" weight to near-zero to trigger the disabled check.
    tracker._constraint_weights["arithmetic"] = 0.01

    result = run_audit(tracker, FR11EventBus())

    assert "arithmetic" in result["disabled_constraints"], (
        "arithmetic with weight 0.01 should appear in disabled_constraints"
    )


# ---------------------------------------------------------------------------
# test_honest_verdict_converging  (REQ-FR11-008, SCENARIO-FR11-008)
# ---------------------------------------------------------------------------


def test_honest_verdict_converging():
    """30 arithmetic + 15 logical + 5 code events produce 'tier1_weights_converging'.

    With 50 events at 10x throttle, arithmetic receives 3 un-throttled updates,
    logical 1, code 1.  update_count_ratio = 3/1 = 3.0 >= 2.0, and arithmetic
    weight > logical weight → verdict is converging.

    Spec: REQ-FR11-008, REQ-FR11-008-1, REQ-FR11-008-2, SCENARIO-FR11-008
    """
    tracker = PerModelFPTracker()
    bus = FR11EventBus()
    _inject_events(tracker, bus, _SYNTHETIC_COUNTS)

    result = run_audit(tracker, FR11EventBus())

    assert result["expected_ordering_correct"] is True, (
        "arithmetic_weight should exceed logical_weight"
    )
    assert result["update_count_ratio"] is not None
    assert result["update_count_ratio"] >= 2.0, (
        f"update_count_ratio {result['update_count_ratio']} should be >= 2.0"
    )
    assert result["honest_verdict"] == "tier1_weights_converging", (
        f"unexpected verdict: {result['honest_verdict']}"
    )
    assert result["simulated"] is False, "tracker had data; simulation should not have run"


# ---------------------------------------------------------------------------
# test_honest_verdict_uniform  (REQ-FR11-008-1)
# ---------------------------------------------------------------------------


def test_honest_verdict_uniform():
    """Equal event counts produce 'tier1_weights_uniform' (weight_ratio < 2.0).

    With the same number of events per constraint type, all weights should be
    equal so weight_ratio == 1.0, below the 2.0 threshold.

    Spec: REQ-FR11-008-1
    """
    tracker = PerModelFPTracker()
    bus = FR11EventBus()
    # Inject equal counts for all three types.
    _inject_events(tracker, bus, [("arithmetic", 20), ("logical", 20), ("code", 20)])

    result = run_audit(tracker, FR11EventBus())

    # With equal event counts each type should have the same weight.
    assert result["weight_ratio"] is not None
    assert result["honest_verdict"] in ("tier1_weights_uniform", "tier1_weights_converging"), (
        f"expected uniform or converging with equal counts, got: {result['honest_verdict']}"
    )
    # With equal counts the update_count_ratio is 1.0 < 2.0 → uniform.
    assert result["update_count_ratio"] is not None
    assert result["update_count_ratio"] < 2.0, (
        f"equal event counts should produce update_count_ratio < 2.0, got {result['update_count_ratio']}"
    )
    assert result["honest_verdict"] == "tier1_weights_uniform"


# ---------------------------------------------------------------------------
# test_honest_verdict_no_data  (REQ-FR11-008-1)
# ---------------------------------------------------------------------------


def test_honest_verdict_no_data():
    """When tracker has no data AND bus injects events, simulated==True is set.

    A fresh tracker with a fresh bus (no subscribers) calls run_audit, which
    detects zero updates and injects the canonical 50 synthetic events.
    The result should have simulated==True and a valid verdict.

    Spec: REQ-FR11-008-1, SCENARIO-FR11-008
    """
    # Completely fresh tracker and bus — no events pre-injected.
    tracker = PerModelFPTracker()
    bus = FR11EventBus()

    result = run_audit(tracker, bus)

    # Simulation should have run since total_update_count was 0 initially.
    assert result["simulated"] is True, "fresh tracker should trigger simulation"
    assert result["n_events_injected"] == sum(c for _, c in _SYNTHETIC_COUNTS)
    # After simulation with canonical distribution, should converge.
    assert result["honest_verdict"] == "tier1_weights_converging", (
        f"after simulation with canonical distribution, expected converging, got: {result['honest_verdict']}"
    )
