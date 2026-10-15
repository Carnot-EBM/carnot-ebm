"""REQ-REPORT-8555, REQ-VERIFY-8555: independent primitive counter evidence."""

from copy import deepcopy
from pathlib import Path

import pytest

from carnot.reporting import canonical_update_8555 as e
from carnot.reporting.current_work_receipt import atomic_json


def test_counts_and_projection():
    """SCENARIO-REPORT-8555-COUNTERS: repeated assignments are distinct from changes."""
    rows = e.controls()
    reduced = e.reduce(rows)
    assert reduced["physical_coordinate_writes"] > reduced["changed_coordinate_count"]
    records, metrics, visibility, relations = e.canonical(rows)
    assert visibility["passed"]
    assert {r["name"]: r["value"] for r in records} == metrics == reduced
    assert all(r["unit"] and r["definition"] for r in records)
    assert all(r["independently_recomputed"] for r in relations)
    assert any(r["relation"] == "equal" for r in relations)
    assert any(r["relation"] == "different" for r in relations)
    assert e.ALIAS_MAP["label_updates"] == "applied_update_events"
    repeated = next(r for r in rows if r["event_id"] == "duplicate_assignment")
    assert len(repeated["assignment_indices"]) == 3
    assert e.reduce([repeated])["changed_coordinate_count"] == 1
    with pytest.raises(ValueError, match="duplicate_event"):
        e.reduce([repeated, repeated])
    with pytest.raises(ValueError, match="inactive_update"):
        e.reduce([dict(repeated, applied=False)])
    invalid = deepcopy(repeated)
    invalid["assignment_indices"] = [999]
    with pytest.raises(ValueError, match="coordinate_contract"):
        e.reduce([invalid])


def test_actual_original_transitions():
    """SCENARIO-REPORT-8555-COUNTERS: original costs and sealed heads agree independently."""
    rows, refs = e.history()
    assert rows and refs
    totals = e.reduce(rows)
    assert totals["applied_update_events"] == 1800
    assert totals["label_release_events"] > totals["applied_update_events"]
    assert any(not r["applied"] for r in rows)
    assert all(r["source_reference"] and r["label_released"] for r in rows)


def test_historical_authentication(tmp_path):
    """SCENARIO-REPORT-8555-PUBLICATION: source hashes and qualification remain independent."""
    value, refs, failures = e.authenticate(e.ROOT)
    assert value["historical_learning_validation_ready_score"] == 1
    assert refs and not failures
    _, _, failures = e.authenticate(tmp_path)
    assert failures[0]["observed"] is None
    assert all(e.GATE_FIELDS <= r.keys() for r in failures)
    target = tmp_path / "results" / (e.HISTORICAL_NAME + ".json")
    atomic_json(target, dict(value, historical_learning_validation_ready_score=0))
    _, _, failures = e.authenticate(tmp_path)
    assert failures


def test_finite_differences_and_state_controls(tmp_path):
    """SCENARIO-VERIFY-8555-CONTROLS: use real updates and independently computed gradients."""
    rows = e.numerical(tmp_path)
    assert rows and all(r["passed"] for r in rows)
    assert {"clipping", "pending_feedback", "frozen_arm", "rejected_feedback"} <= {
        r["name"] for r in rows
    }
    assert max(r.get("absolute_error", 0) for r in rows) <= 1e-7


def test_visibility_is_exact():
    """SCENARIO-REPORT-8555-PUBLICATION: omitted or stringified metrics cannot qualify."""
    records, metrics, _, _ = e.canonical(e.controls())
    assert not e.visibility(records, {})["passed"]
    assert not e.visibility(records, {k: str(v) for k, v in metrics.items()})["passed"]
    altered = dict(metrics)
    altered[records[0]["name"]] += 1
    assert not e.visibility(records, altered)["passed"]


def test_rehashed_original_and_authentication_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8555-PUBLICATION: independently sealed sources reject repaired drift."""
    monkeypatch.setattr(
        e.shipped, "historical_accounting", lambda: dict(reconciliation_passed=False)
    )
    with pytest.raises(ValueError, match="original_transition_reconciliation"):
        e.history()
    monkeypatch.setattr(
        e.shipped, "historical_accounting", lambda: dict(reconciliation_passed=True, rows=[])
    )
    monkeypatch.setattr(
        e,
        "read",
        lambda ref: dict(
            primitive_reference={},
            cells=[dict(configuration={}, steps=[{}], identity="foreign")],
            frozen=False,
            sealed_state=dict(pending=[dict(id="a")]),
            updates=[dict(feedback_id="a")],
        ),
    )
    with pytest.raises(ValueError, match="missing_applied_transition"):
        e.history()
    value = dict(
        source_artifact_hashes=[dict(path=str(tmp_path / "foreign"), sha256="bad")],
        code_config_hashes=[],
    )
    monkeypatch.setattr(e, "read", lambda ref: value)
    monkeypatch.setattr(e, "reference", lambda p: dict(path=str(p), sha256="different"))
    _, _, failures = e.authenticate(tmp_path)
    assert failures[0]["observed"] == "historical_source_test_hash"
    value["source_artifact_hashes"] = []
    value["terminal_validation_sidecar_path"] = str(tmp_path / "terminal")
    value["publication"] = dict(sidecar_path=str(tmp_path / "bound"))
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
    _, _, failures = e.authenticate(tmp_path)
    assert failures[0]["observed"] == "historical_bound_replay"


def test_accepted_future_feedback_is_failed_control(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8555-CONTROLS: a deliberately broken release cannot certify causality."""
    adapter = e.derivatives.adapter
    release = adapter.Store.release

    def broken_release(store, block, labels, *args):
        return release(
            store, block, {k: v for k, v in labels.items() if k != "future_label"}, *args
        )

    monkeypatch.setattr(adapter.Store, "release", broken_release)
    rows = e.numerical(tmp_path)
    assert not next(r for r in rows if r["name"] == "rejected_feedback")["passed"]
