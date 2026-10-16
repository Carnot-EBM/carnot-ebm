"""REQ-REPORT-8568, REQ-VERIFY-8568: fresh accounting keeps original authority."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import causal_accounting_8568 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.discrete_count_proof import proven_counts


def test_actual_accounting_and_proof():
    """SCENARIO-REPORT-8568-COUNTERS: original integers and aliases remain unchanged."""
    rows, _ = e.old.history()
    metrics, proof = e.proof(rows)
    assert metrics == e.old.reduce(rows)
    assert metrics["admission_events"] == metrics["label_release_events"] == 2400
    assert metrics["applied_update_events"] == metrics["decay_scale_changes"] == 1800
    value = dict(
        metrics, intended_count=len(rows), completed_count=len(rows), discrete_count_proof=proof
    )
    assert len(proven_counts(value)) == 10
    flags = e.validator.findings(value, e.validator.av.check_tautology)
    assert flags and all(f["severity"] == "warn" for f in flags)
    changed = deepcopy(value)
    changed["discrete_count_proof"]["claims"]["admission_events"]["value"] -= 1
    assert "admission_events" not in proven_counts(changed)
    controls = e.old.controls()
    assert (
        e.proof(controls)[0]["physical_coordinate_writes"]
        != e.proof(controls)[0]["changed_coordinate_count"]
    )
    with pytest.raises(ValueError):
        e.proof([controls[0], controls[0]])


def test_authentication_and_saved_controller_diagnoses(tmp_path):
    """SCENARIO-REPORT-8568-PUBLICATION: current hash drift cannot reuse qualification."""
    historical, refs, failures = e.authority(e.ROOT)
    assert historical["historical_learning_validation_ready_score"] == 1
    assert refs and all(e.old.GATE_FIELDS <= f.keys() for f in failures)
    assert any(f["field"] == "validator_revision_hash" for f in failures)
    _, _, missing = e.authority(tmp_path)
    assert missing and missing[0]["observed"] is None
    rows = e.diagnostics(e.ROOT)
    assert len(rows) == 3 and all(r["historical_exit"] == 1 for r in rows)
    assert [r["regression"] for r in rows] == [
        "live_parent_save",
        "live_parent_save",
        "implicit_configuration",
    ]
    assert all(r["source"]["sha256"] for r in rows)


def test_build_gate_classes_and_immutable_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8568-REPLAY: repaired envelopes cannot replace original events."""
    from carnot.reporting import causal_accounting_runner_8568 as runner

    work = runner.prepare(tmp_path / "raw", tmp_path / "scratch", private=True)
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, e.build(work, path))
    assert e.replay(path)
    assert not e.replay(tmp_path / "absent")
    value = json.loads(path.read_bytes())
    value["metrics"]["physical_coordinate_writes"] += 1
    value["reproducibility_checksum"] = e.old.checksum(value)
    atomic_json(path, value)
    assert not e.replay(path)
    work["owned_ready"] = True
    work["receipts"] = []
    work["failures"] = []
    atomic_json(Path(work["raw"]) / "final.json", work)
    ready = e.build(work, path)
    assert ready["verdict_class"] == "null" and ready["online_adapter_ready_score"] == 1
    work["failures"] = [e.old.gate(tmp_path / "absent", "upstream", True, None)]
    atomic_json(Path(work["raw"]) / "final.json", work)
    assert e.build(work, path)["verdict_class"] == "blocked"
    work["receipts"] = [dict(name="owned", passed=False)]
    atomic_json(Path(work["raw"]) / "final.json", work)
    assert e.build(work, path)["verdict_class"] == "disqualified"
    monkeypatch.setattr(e, "authority", lambda root: ({}, [], []))
    assert not e.replay(path)


def test_rejected_authority_and_primitive_repairs(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8568-REPLAY: hash and operand repairs cannot replace original sources."""
    from carnot.reporting import causal_accounting_runner_8568 as runner

    work = runner.prepare(tmp_path / "raw", tmp_path / "scratch", private=True)
    path = tmp_path / (e.NAME + ".json")
    work["private"] = False
    monkeypatch.setattr(e.old, "history", lambda: (e.old.controls(), []))
    atomic_json(Path(work["raw"]) / "final.json", work)
    atomic_json(path, e.build(work, path))
    assert e.replay(path)
    value = json.loads(path.read_bytes())
    value["reproducibility_checksum"] = "wrong"
    atomic_json(path, value)
    assert not e.replay(path)
    value = e.build(work, path)
    value["code_config_hashes"][0]["sha256"] = "wrong"
    value["reproducibility_checksum"] = e.old.checksum(value)
    atomic_json(path, value)
    assert not e.replay(path)
    work["code"][0] = e.old.reference(Path(work["code"][0]["path"]))
    changed = deepcopy(work["receipts"][0])
    work["receipts"][0]["passed"] = False
    atomic_json(Path(work["raw"]) / "final.json", work)
    atomic_json(path, e.build(work, path))
    assert not e.replay(path)
    work["receipts"][0] = changed
    work["receipts"][0]["stdout_sha256"] = "wrong"
    atomic_json(Path(changed["stdout_path"]).with_suffix(".receipt.json"), work["receipts"][0])
    atomic_json(Path(work["raw"]) / "final.json", work)
    atomic_json(path, e.build(work, path))
    assert not e.replay(path)
    work["receipts"] = []
    rows = e.old.controls()
    rows[0]["after"]["parameters"][0] = 3.0
    atomic_json(Path(work["events"]["path"]), rows)
    work["events"] = e.old.reference(Path(work["events"]["path"]))
    atomic_json(Path(work["raw"]) / "final.json", work)
    atomic_json(path, e.build(work, path))
    assert not e.replay(path)
    monkeypatch.setattr(
        e.sealed, "reduce", lambda rows: dict(e.old.reduce(rows), admission_events=999)
    )
    with pytest.raises(ValueError, match="independent_count_reduction"):
        e.proof(e.old.controls())
    monkeypatch.setattr(e.old, "authenticate", lambda root: ({}, [], []))
    monkeypatch.setattr(e.old, "reference", lambda path: dict(path=str(path), sha256="fixture"))
    monkeypatch.setattr(
        e.old,
        "read",
        lambda ref: dict(
            count_validator_ready_score=0,
            terminal_validation_sidecar_path="unused",
            publication=dict(sidecar_path="unused"),
        ),
    )
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *args: dict(report=dict(passed=True)))
    _, _, failures = e.authority(e.ROOT)
    assert failures[0]["observed"] == "validator_qualification"


def test_corrupt_historical_controller_log(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8568-PUBLICATION: diagnostic source drift remains a failure."""
    path = tmp_path / "log"
    path.write_text("unknown diagnostic")
    failure = dict(path=str(path), sha256="wrong", observed=1, owned_required_check=True)
    monkeypatch.setattr(e, "failed_original", lambda root: dict(gate_check_summary=[failure]))
    with pytest.raises(ValueError, match="controller_log_hash"):
        e.diagnostics(tmp_path)
    failure["sha256"] = e.old.reference(path)["sha256"]
    with pytest.raises(ValueError, match="unknown_controller_failure"):
        e.diagnostics(tmp_path)
