"""REQ-REPORT-8566, REQ-VERIFY-8566: freeze proof and old-audit behavior."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import count_semantics_8566 as e


def event(identity="a"):
    """Keep operands distinct so each count is actually recomputed."""
    return dict(
        namespace="control",
        event_id=identity,
        admitted=True,
        label_released=True,
        applied=True,
        before=dict(parameters=[0.0, 0.0], decay_scale=1.0),
        after=dict(parameters=[1.0, 0.0], decay_scale=0.9),
        assignment_indices=[0],
    )


def test_independent_reductions_and_corrupt_events():
    """SCENARIO-REPORT-8566-CONTROLS: assignment multiplicity differs from change."""
    rows = [event()]
    rows[0]["assignment_indices"] = [0, 0, 1]
    totals = e.reduce(rows)
    assert totals == dict(
        completed_count=1,
        intended_count=1,
        admission_events=1,
        label_release_events=1,
        applied_update_events=1,
        decay_scale_changes=1,
        changed_coordinate_count=1,
        physical_coordinate_writes=3,
        effective_coefficient_changes=1,
        unique_changed_coordinates=1,
    )
    for changed in [
        [event(), event()],
        [],
        [dict(event(), admitted=1)],
        [dict(event(), assignment_indices=[False])],
    ]:
        with pytest.raises(ValueError):
            e.reduce(changed)
    for path, value in [
        ("parameters", [float("nan"), 0.0]),
        ("parameters", [0.0]),
        ("decay_scale", True),
    ]:
        changed = [event()]
        changed[0]["before"][path] = value
        with pytest.raises(ValueError):
            e.reduce(changed)


def test_frozen_corpus_and_expected_old_failures():
    """REQ-VERIFY-8566: old false positives stay asserted in the green suite."""
    corpus = e.corpus()
    assert corpus["version"] == "count-controls-v1"
    observed = e.classify(corpus)
    assert len(observed) >= 16 and all(r["expectation_met"] for r in observed)
    equal = [r for r in observed if r["expected"] == "valid_discrete_identity"]
    assert {r["value"] for r in equal} == {2400, 1800, 47472}
    assert all(r["old_critical"] for r in equal)
    assert all(r["old_critical"] for r in observed if r["expected"] == "critical_continuous")
    assert e.classify(corpus) == observed


def test_contract_rejects_self_declaration_and_rehashed_table():
    """REQ-REPORT-8566: labels cannot substitute for sealed integer operands."""
    corpus = e.corpus()
    good = corpus["cases"][0]
    assert e.prove(good["left"], corpus["tables"]) == 2400
    for mutation in [
        dict(value=True),
        dict(value=2400.0),
        dict(value=float("inf")),
        dict(unit="seconds"),
        dict(value=2399),
        dict(table="missing"),
        dict(operation="eval"),
        dict(table_sha256="wrong"),
    ]:
        changed = dict(good["left"], **mutation)
        with pytest.raises(ValueError):
            e.prove(changed, corpus["tables"])
    for change in ["duplicate", "missing", "oversized"]:
        changed = deepcopy(corpus["tables"])
        table = changed[good["left"]["table"]]
        table["rows"] = (
            [event(), event()]
            if change == "duplicate"
            else []
            if change == "missing"
            else [event(str(i)) for i in range(50001)]
        )
        claim = dict(good["left"], table_sha256=e.digest(table))
        with pytest.raises(ValueError):
            e.prove(claim, changed)


def test_source_pins_and_all_seven_flags(tmp_path):
    """REQ-REPORT-8566: original bytes and every controller defect remain visible."""
    measured = e.measure(e.ROOT, tmp_path)
    assert len(measured["reproduced_findings"]) == 7
    assert all(r["proven"] for r in measured["cardinality_reductions"])
    assert len(measured["owned_controller_failures"]) >= 4
    assert measured["historical_verdict"] == "disqualified"
    assert len(measured["historical_controls"]) == 2
    for row in measured["historical_controls"]:
        assert row["current_critical_findings"]
    assert e.authenticate(e.ROOT)["experiment_id"] == 8555
    with pytest.raises((ValueError, FileNotFoundError)):
        e.authenticate(tmp_path)
    assert e.measure(tmp_path, tmp_path / "missing")["failures"]


def test_evidence_rejections_are_independent(tmp_path, monkeypatch):
    """REQ-VERIFY-8566: copied totals and historical labels cannot certify themselves."""
    from carnot.reporting import count_semantics_evidence_8566 as evidence
    from carnot.reporting.current_work_receipt import atomic_json

    path = tmp_path / "bad.json"
    atomic_json(path, dict(x=1))
    with pytest.raises(ValueError, match="source_hash"):
        evidence.read(dict(path=str(path), sha256="wrong"))
    original = evidence.authenticate(e.ROOT)
    for change in [dict(experiment_id=0), dict(verdict_class="null"), dict(task_id="wrong")]:
        with monkeypatch.context() as patch:
            patch.setattr(evidence, "read", lambda ref: dict(original, **change))
            with pytest.raises(ValueError, match="historical_identity"):
                evidence.authenticate(e.ROOT)
    real_read = evidence.read
    for mutation in ["seal", "roster", "flags", "controller", "history"]:
        with monkeypatch.context() as patch:
            changed = deepcopy(original)
            if mutation == "roster":
                changed["completed_count"] = 1
            if mutation == "flags":
                changed["adversarial_findings"] = []
            if mutation == "controller":
                changed["gate_check_summary"][0]["sha256"] = "wrong"
            patch.setattr(evidence, "authenticate", lambda root: changed)
            if mutation == "seal":

                def corrupt(ref):
                    value = real_read(ref)
                    if ref == original["event_rows_reference"]:
                        value[0]["label_release_record"]["before_head_hash"] = "wrong"
                    return value

                patch.setattr(evidence, "read", corrupt)
            if mutation == "history":
                patch.setattr(
                    e,
                    "critical",
                    lambda d: (
                        original["adversarial_findings"] if d.get("experiment_id") == 8555 else []
                    ),
                )
            assert evidence.measure(e.ROOT, tmp_path / mutation)["failures"]


def test_reduction_work_budget_and_inactive_corruption():
    """REQ-REPORT-8566: finite bounds and inactive events cannot hide corrupt evidence."""
    with pytest.raises(ValueError, match="inactive_update"):
        e.reduce([dict(event(), applied=False)])
    rows = [event(str(i)) for i in range(21)]
    for row in rows:
        row["before"]["parameters"] = [0.0] * 50000
        row["after"]["parameters"] = [1.0] * 50000
    with pytest.raises(ValueError, match="work_budget"):
        e.reduce(rows)
