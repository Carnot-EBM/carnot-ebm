"""Tests for Experiment 734 — FR-11 EventBus relay.

Coverage targets (REQ-FR11-001/002/003/004):
- test_gated_blocked_path: gate fail → correct artifact written, status="gated_blocked"
- test_fr11_event_bus_delivers_to_all_subscribers: bus publishes to 2 subscribers,
  events_acked == 2 * events_published  (REQ-FR11-001)
- test_fp_tracker_increments_on_violation: on_violation increments constraint_weight
  (REQ-FR11-002)
- test_fp_tracker_caps_weight_at_2: weight never exceeds 2.0 (REQ-FR11-002)
- test_session_memory_calls_observe_pattern_after_5: after 5 violations of same type,
  ConstraintTemplateLibrary.observe_pattern is called (REQ-FR11-003)
- test_throttle_prevents_update_except_every_10: only every 10th call updates weight
  (REQ-FR11-004)
- test_bus_latency_under_200ms: latency for 2 no-op subscribers < 200ms (REQ-FR11-001)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_violation_event(
    constraint_type: str = "carry_check",
    question_domain: str = "arithmetic",
    query_id: str = "q_test",
) -> "object":
    """Build a ViolationEvent for testing without requiring a real model."""
    from carnot.pipeline.fr11_event_bus import ViolationEvent

    return ViolationEvent(
        query_id=query_id,
        step_index=0,
        energy_score=0.45,
        probe_confidence=0.81,
        constraint_type=constraint_type,
        question_domain=question_domain,
        timestamp="2026-04-22T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# REQ-FR11-001: FR11EventBus delivers to all subscribers
# ---------------------------------------------------------------------------


def test_fr11_event_bus_delivers_to_all_subscribers():
    """FR11EventBus.publish() calls all subscribers and increments events_acked per subscriber.

    Spec: REQ-FR11-001, SCENARIO-FR11-001
    """
    from carnot.pipeline.fr11_event_bus import FR11EventBus

    bus = FR11EventBus()
    received: list = []

    def sub_a(ev):
        received.append(("a", ev.constraint_type))

    def sub_b(ev):
        received.append(("b", ev.constraint_type))

    bus.subscribe(sub_a)
    bus.subscribe(sub_b)

    ev = _make_violation_event("carry_check")
    bus.publish(ev)

    # events_published == 1, events_acked == 2 (one per subscriber)
    assert bus.events_published == 1
    assert bus.events_acked == 2
    assert len(received) == 2
    assert ("a", "carry_check") in received
    assert ("b", "carry_check") in received


# ---------------------------------------------------------------------------
# REQ-FR11-001: latency under 200ms
# ---------------------------------------------------------------------------


def test_bus_latency_under_200ms():
    """measure_publish_latency_ms returns < 200ms for in-memory no-op subscribers.

    Spec: REQ-FR11-001, SCENARIO-FR11-002
    """
    from carnot.pipeline.fr11_event_bus import FR11EventBus

    bus = FR11EventBus()
    bus.subscribe(lambda ev: None)
    bus.subscribe(lambda ev: None)

    ev = _make_violation_event()
    latency_ms = bus.measure_publish_latency_ms(ev)

    assert latency_ms < 200.0, f"Latency {latency_ms:.2f}ms exceeded 200ms threshold"


# ---------------------------------------------------------------------------
# REQ-FR11-002: PerModelFPTracker.on_violation increments constraint_weight
# ---------------------------------------------------------------------------


def test_fp_tracker_increments_on_violation():
    """on_violation increments constraint_weight[constraint_type] by 0.01 on every 10th call.

    Spec: REQ-FR11-002
    """
    from carnot.pipeline.adaptive_thresholds import PerModelFPTracker

    tracker = PerModelFPTracker()
    ev = _make_violation_event("sign_check")

    # Call 10 times — only the 10th should update the weight (throttle rule).
    for _ in range(10):
        tracker.on_violation(ev)

    weights = getattr(tracker, "_constraint_weights", {})
    # Expected: initial 1.0 + 0.01 = 1.01 after the 10th call.
    assert "sign_check" in weights
    assert abs(weights["sign_check"] - 1.01) < 1e-9


def test_fp_tracker_caps_weight_at_2():
    """constraint_weight never exceeds 2.0 regardless of violation count.

    Spec: REQ-FR11-002
    """
    from carnot.pipeline.adaptive_thresholds import PerModelFPTracker

    tracker = PerModelFPTracker()
    ev = _make_violation_event("carry_check")

    # Force cap: call 10*200 = 2000 times → 200 weight updates of +0.01 each.
    # Without cap, weight would reach 1.0 + 2.0 = 3.0; with cap it stays at 2.0.
    for _ in range(2000):
        tracker.on_violation(ev)

    weights = getattr(tracker, "_constraint_weights", {})
    assert weights.get("carry_check", 1.0) <= 2.0


# ---------------------------------------------------------------------------
# REQ-FR11-003: SessionMemory calls observe_pattern after 5 violations
# ---------------------------------------------------------------------------


def test_session_memory_calls_observe_pattern_after_5():
    """SessionMemory.on_violation triggers observe_pattern at 5th violation of same type.

    Spec: REQ-FR11-003, SCENARIO-FR11-003
    """
    from carnot.pipeline.session_memory import SessionMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        mem = SessionMemory(storage_dir=tmpdir, model_id="test-model")

        template_lib = MagicMock()
        ev = _make_violation_event("carry_check")

        # 4 violations — observe_pattern NOT called yet.
        for _ in range(4):
            mem.on_violation(ev, template_lib)

        template_lib.observe_pattern.assert_not_called()

        # 5th violation — observe_pattern IS called.
        mem.on_violation(ev, template_lib)
        template_lib.observe_pattern.assert_called_once_with("carry_check", "test-model", 5)


def test_session_memory_caches_violations_by_type_and_domain():
    """SessionMemory accumulates violations_by_type and violations_by_domain.

    Spec: REQ-FR11-003
    """
    from carnot.pipeline.session_memory import SessionMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        mem = SessionMemory(storage_dir=tmpdir, model_id="test-model")
        template_lib = MagicMock()

        mem.on_violation(_make_violation_event("sign_check", "algebra"), template_lib)
        mem.on_violation(_make_violation_event("sign_check", "arithmetic"), template_lib)

        assert mem._violations_by_type.get("sign_check", 0) == 2
        assert mem._violations_by_domain.get("algebra", 0) == 1
        assert mem._violations_by_domain.get("arithmetic", 0) == 1


# ---------------------------------------------------------------------------
# REQ-FR11-004: weight update throttle
# ---------------------------------------------------------------------------


def test_throttle_prevents_update_except_every_10():
    """Only every 10th on_violation call updates constraint_weight; others are skipped.

    Spec: REQ-FR11-004, SCENARIO-FR11-004
    """
    from carnot.pipeline.adaptive_thresholds import PerModelFPTracker

    tracker = PerModelFPTracker()
    ev = _make_violation_event("unit_consistency")

    # 9 calls — no updates (call 1-9, all throttled; update fires at call 10).
    for _ in range(9):
        tracker.on_violation(ev)

    weights = getattr(tracker, "_constraint_weights", {})
    # After 9 calls, no update yet (update fires at call 10).
    assert "unit_consistency" not in weights or weights["unit_consistency"] == 1.0

    # 10th call — update fires.
    tracker.on_violation(ev)
    weights = getattr(tracker, "_constraint_weights", {})
    assert "unit_consistency" in weights
    assert abs(weights["unit_consistency"] - 1.01) < 1e-9

    # Calls 11-19 — no further update.
    for _ in range(9):
        tracker.on_violation(ev)

    weights = getattr(tracker, "_constraint_weights", {})
    # Still at 1.01 after calls 11-19 (next update at call 20).
    assert abs(weights["unit_consistency"] - 1.01) < 1e-9


# ---------------------------------------------------------------------------
# Gate blocked path
# ---------------------------------------------------------------------------


def test_gated_blocked_path(tmp_path):
    """When tier21_cascade_gate.json shows gate==fail, experiment writes gated_blocked artifact.

    Spec: REQ-FR11-001 (gate precondition)
    """
    import sys

    # Write a failing gate file.
    gate_path = tmp_path / "tier21_cascade_gate.json"
    gate_path.write_text(json.dumps({"gate": "fail", "reason": "test_forced_fail"}))

    deliverable = tmp_path / "results" / "experiment_734_fr11_tier21_relay.json"
    deliverable.parent.mkdir(parents=True, exist_ok=True)

    # Patch the module-level constants so the script uses our tmp paths.
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import importlib
    import experiment_734_fr11_tier21_relay as exp734

    with (
        patch.object(exp734, "_GATE_PATH", gate_path),
        patch.object(exp734, "_DELIVERABLE", str(deliverable.relative_to(tmp_path))),
        patch.object(exp734, "_REPO_ROOT", tmp_path),
    ):
        result = exp734._check_gate()

    assert result is False
    assert deliverable.exists(), "gated_blocked artifact was not written"

    artifact = json.loads(deliverable.read_text())
    assert artifact["status"] == "gated_blocked"
    assert artifact["honest_verdict"] == "gated_blocked_tier21_cascade_failed"
    assert artifact.get("gate_source") == "exp733"
