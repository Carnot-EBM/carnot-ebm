"""REQ-REPORT-7707 and REQ-CL-7707 cold audit checks."""

import json

import pytest

from carnot.reporting.evidence_audit_v671 import (
    audit_inventory,
    challenge_boundaries,
    reduce_decisions,
    reduce_feedback,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_custody_missing_and_failed_readiness(tmp_path):
    """SCENARIO-REPORT-7707-CUSTODY keeps absent and failed sources separate."""
    _write(
        tmp_path / "results/experiment_7705.json",
        {
            "honest_verdict": "complete_circular_positive_fixture_mechanics",
            "verdict_class": "circular_positive",
            "flagged_adversarial": False,
            "constraint_bank_ready_score": 0,
        },
    )
    plan = [
        (7705, "results/experiment_7705.json", "constraint_bank_ready_score"),
        (7706, "results/experiment_7706.json", "continuous_acquisition_complete_score"),
    ]
    rows, hashes, gates = audit_inventory(tmp_path, plan)
    assert [row["state"] for row in rows] == ["pre_gate", "missing"]
    assert "results/experiment_7705.json" in hashes["pre_gate_receipts"]
    assert hashes["missing_inputs"] == ["results/experiment_7706.json"]
    assert gates[0]["field"] == "constraint_bank_ready_score"
    assert gates[0]["expected"] == 1 and gates[0]["observed"] == 0
    assert gates[1]["field"] == "exists" and gates[1]["observed"] is False


def test_decision_reduction_recomputes_cost_and_brier():
    """SCENARIO-REPORT-7707-REDUCTION rejects a producer's false row loss."""
    row = {
        "unit_id": "family-a",
        "arm": "energy",
        "label": 1,
        "probability_error": 0.75,
        "typed_action": "reject",
        "brier": 0.0625,
        "decision_cost": 0.0,
        "excluded": False,
        "censored": False,
    }
    rows, summary = reduce_decisions([row])
    assert rows[0]["raw_metrics"]["brier"] == 0.0625
    assert summary["independent_families"] == 1
    assert summary["by_arm"]["energy"]["brier_mean"] == 0.0625
    with pytest.raises(ValueError, match="brier_mismatch"):
        reduce_decisions([{**row, "brier": 0.0}])
    with pytest.raises(ValueError, match="duplicate_unit_arm"):
        reduce_decisions([row, row])


def test_feedback_order_budget_and_later_use():
    """SCENARIO-CL-7707-FEEDBACK rejects future and duplicate release."""
    events = [
        {"event_id": "e1", "arm": "priority", "kind": "prediction", "tick": 1, "unit_id": "a"},
        {
            "event_id": "e2",
            "arm": "priority",
            "kind": "feedback",
            "tick": 2,
            "origin_tick": 1,
            "unit_id": "a",
        },
        {"event_id": "e3", "arm": "priority", "kind": "proposal", "tick": 3, "accepted": False},
        {"event_id": "e4", "arm": "priority", "kind": "prediction", "tick": 4, "unit_id": "b"},
    ]
    result = reduce_feedback(events, cap=6)
    assert result["priority"]["spent"] == 1
    assert result["priority"]["charged_rejections"] == 1
    assert result["priority"]["later_predictions"] == 1
    with pytest.raises(ValueError, match="future_feedback"):
        reduce_feedback([{**events[1], "origin_tick": 5}], cap=6)
    with pytest.raises(ValueError, match="duplicate_feedback"):
        reduce_feedback(events + [{**events[1], "event_id": "e5", "tick": 5}], cap=6)


def test_private_absence_and_corruption_are_distinct(tmp_path):
    """SCENARIO-REPORT-7707-TERMINAL distinguishes blocked from disqualified."""
    plan = [(7704, "results/static.json", "decision_measurement_complete_score")]
    _write(
        tmp_path / "results/static.json",
        {
            "honest_verdict": "complete_null_valid",
            "verdict_class": "null",
            "flagged_adversarial": False,
            "decision_measurement_complete_score": 1,
        },
    )
    rows, _, gates = audit_inventory(tmp_path, plan)
    assert rows[0]["state"] == "producer" and gates == []
    (tmp_path / "results/static.json").unlink()
    rows, _, gates = audit_inventory(tmp_path, plan)
    assert rows[0]["state"] == "missing" and gates[0]["check"] == "producer_exists"
    with pytest.raises(ValueError, match="brier_mismatch"):
        reduce_decisions(
            [
                {
                    "unit_id": "u",
                    "arm": "a",
                    "label": 0,
                    "probability_error": 0.5,
                    "typed_action": "accept",
                    "brier": 0.0,
                    "decision_cost": 0.0,
                }
            ]
        )


def test_fail_closed_operand_variants(tmp_path):
    """REQ-REPORT-7707 and REQ-CL-7707 reject malformed private controls."""
    plan = [(7707, "results/self.json", None), (7708, "results/future.json", None)]
    rows, _, gates = audit_inventory(tmp_path, plan)
    assert [r["state"] for r in rows] == ["planned_output", "missing"]
    assert gates == []
    bad = tmp_path / "results/bad.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{")
    rows, _, gates = audit_inventory(tmp_path, [(7706, "results/bad.json", "ready")])
    assert rows[0]["state"] == "pre_gate"
    assert gates[0]["field"] == "verdict_class"

    base = {
        "unit_id": "u",
        "arm": "a",
        "label": 0,
        "probability_error": 0.5,
        "typed_action": "escalate",
        "brier": 0.25,
        "decision_cost": 0.2,
    }
    with pytest.raises(ValueError, match="invalid_label_probability"):
        reduce_decisions([{**base, "label": 2}])
    with pytest.raises(ValueError, match="invalid_action"):
        reduce_decisions([{**base, "typed_action": "guess"}])
    with pytest.raises(ValueError, match="cost_mismatch"):
        reduce_decisions([{**base, "decision_cost": 0.0}])

    event = {"event_id": "p", "arm": "a", "kind": "prediction", "tick": 1, "unit_id": "u"}
    feedback = {
        "event_id": "f",
        "arm": "a",
        "kind": "feedback",
        "tick": 2,
        "origin_tick": 1,
        "unit_id": "u",
    }
    proposal = {"event_id": "q", "arm": "a", "kind": "proposal", "tick": 3, "accepted": False}
    cases = [
        ([event, event], "duplicate_event_id"),
        ([feedback], "feedback_without_prediction"),
        ([event, proposal], "credit_cap_exceeded"),
        ([{**event, "kind": "other"}], "event_kind_invalid"),
    ]
    for events, message in cases:
        with pytest.raises(ValueError, match=message):
            reduce_feedback(events, cap=0)


def test_private_boundary_challenges():
    """SCENARIO-REPORT-7707-REDUCTION keeps controls outside denominators."""
    assert all(challenge_boundaries().values())


def test_private_controls_detect_broken_reducers(monkeypatch):
    """SCENARIO-REPORT-7707-REDUCTION controls fail when a reducer accepts attacks."""
    import carnot.reporting.evidence_audit_v671 as audit

    monkeypatch.setattr(audit, "reduce_decisions", lambda rows: ([], {}))
    monkeypatch.setattr(audit, "reduce_feedback", lambda events, cap: {})
    outcome = audit.challenge_boundaries()
    assert outcome["permuted_label_rejected"] is False
    assert outcome["future_feedback_rejected"] is False
    assert outcome["duplicate_feedback_rejected"] is False
