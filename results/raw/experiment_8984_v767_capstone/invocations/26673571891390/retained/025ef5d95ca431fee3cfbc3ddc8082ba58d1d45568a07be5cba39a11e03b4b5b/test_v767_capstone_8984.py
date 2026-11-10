"""REQ-REPORT-8984 / SCENARIO-REPORT-8984-V767: truthful terminal accounting."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v767_capstone as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_census_and_science(tmp_path):
    """REQ-REPORT-8984: scientific nulls and external blocks retain independent findings."""
    work = m.collect(tmp_path / "raw", "20261110")
    assert len(work["slots"]) == 13
    assert work["tasks"][-1]["id"] == m.TASK
    science = m.reduce(work)
    assert science["shared_feature"]["collection_permitted"] is False
    assert len(science["shared_feature"]["admission_rows"]) == 3
    assert science["natural_evidence"]["joined_pairs"] == 0
    assert science["v766"]["answers"] == 1024
    assert science["v766"]["errors"] == 719
    assert all(
        not r["passed"]
        for r in science["v766"]["detector"]["paired_intervals"]["contrasts"].values()
    )
    assert all(
        not r["passed"]
        for r in science["v766"]["learning"]["acceptance_gate_rows"]
        if r["split"] == "online"
    )
    assert all(
        r["passed"]
        for r in science["v766"]["learning"]["acceptance_gate_rows"]
        if r["split"] == "retention"
    )
    validation = dict(passed=True, coverage=dict(complete=True), receipts=[])
    atomic_json(tmp_path / "raw/work.json", work)
    value = m.build(work, validation)
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert len(value["slot_rows"]) == 14
    assert value["completed_count"] == 14
    assert not any(value["model_invocation_counts"].values())
    assert value["G2"] == work["publication_gate"]["gates"]["G2"]
    assert m.build(work, dict(validation, passed=False))["verdict_class"] == "disqualified"
    atomic_json(tmp_path / "candidate.json", value)
    assert m.replay(tmp_path / "candidate.json")["passed"]
    value["scientific_reductions"]["natural_evidence"]["joined_pairs"] = 800
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(tmp_path / "candidate.json", value)
    with pytest.raises(ValueError, match="reduction_drift"):
        m.replay(tmp_path / "candidate.json")


def test_slot_controls(tmp_path):
    """SCENARIO-REPORT-8984-V767: missing fields, changed bytes and quarantine cannot qualify."""
    task = dict(
        id="exp8973-shared-feature-methods",
        deliverable="results/experiment_8973_expected.json",
        gated_on=[],
    )
    producer = dict(
        experiment_id=8973,
        task_id=task["id"],
        milestone=m.MILESTONE,
        honest_verdict="complete_null_test",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        owned_statement_coverage=dict(complete=True),
        rows=[],
    )
    path = tmp_path / "expected.json"
    atomic_json(path, producer)
    ref = m.retain(path, tmp_path / "retained", m.sha256_file(path))
    assert m.slot(task, producer, ref, [], {})["disposition"] == "qualified_primary"
    for field, value in [
        ("flagged_adversarial", True),
        ("required_checks_passed", False),
        ("task_id", "exp9999-wrong"),
        ("verdict_class", "disqualified"),
    ]:
        changed = dict(producer, **{field: value})
        assert m.slot(task, changed, ref, [], {})["disposition"] == "quarantined_primary"
    assert m.slot(task, {}, ref, [], {})["disposition"] == "quarantined_primary"
    assert (
        m.slot(
            task, dict(producer, verifier_is_oracle=True, verdict_class="positive"), ref, [], {}
        )["disposition"]
        == "quarantined_primary"
    )
    assert m.slot(task, {}, None, [], {})["disposition"] == "missing_primary"
    assert (
        m.slot(task, {}, None, ["GATE_BLOCK"], {})["disposition"]
        == "conductor_skip_without_primary"
    )
    skip = dict(
        experiment=8974,
        blocked_at_layer="conductor_pre_gate",
        failed_upstream=task["id"],
        failed_field="shared_methods_ready_score",
        failed_operator="==",
        failed_expected=1,
        failed_observed=0,
        failed_evidence_sha256=ref["sha256"],
        gate_check_summary="exact skip",
    )
    child_task = dict(
        id="exp8974-feature-online-methods",
        gated_on=[
            dict(upstream=task["id"], artifact_field="shared_methods_ready_score", op="==", value=1)
        ],
    )
    values = {task["id"]: dict(value=dict(producer, shared_methods_ready_score=0), reference=ref)}
    assert (
        m.slot(child_task, skip, ref, [], values)["disposition"] == "authenticated_scheduler_skip"
    )
    skip["failed_evidence_sha256"] = "sha256:changed"
    assert m.slot(child_task, skip, ref, [], values)["disposition"] == "quarantined_primary"
    del values[task["id"]]["value"]["shared_methods_ready_score"]
    assert m.edge(child_task["gated_on"][0], values)["observed"] == "missing_field"
    assert m.edge(child_task["gated_on"][0], {})["observed"] == "missing_file"


def test_resources_and_empty_reduction(tmp_path, monkeypatch):
    """REQ-REPORT-8984: resource failure blocks science; optional hardware never blocks CPU."""
    work = m.collect(tmp_path / "raw", "20261110", sources=tmp_path / "absent")
    assert all(
        s["disposition"] in {"missing_primary", "conductor_skip_without_primary"}
        for s in work["slots"]
    )
    assert m.reduce(work)["shared_feature"]["status"] == "unavailable"
    atomic_json(tmp_path / "raw/work.json", work)
    validation = dict(passed=True, coverage=dict(complete=True), receipts=[])
    work["resource_failures"] = [dict(check="required_resource", observed=0)]
    assert m.build(work, validation)["verdict_class"] == "blocked"
    work["tasks"].reverse()
    with pytest.raises(ValueError, match="task_authority"):
        m.build(work, validation)


def test_cli_publication_replay_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8984-V767: E2E-018/021 actual children publish and refuse forged aggregates."""
    output = tmp_path / (m.NAME + ".json")
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    child = subprocess.run(
        [*command, "--date", "20261110", "--output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    child = subprocess.run(
        [*command, "--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    value["completed_count"] -= 1
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(output, value)
    child = subprocess.run(
        [*command, "--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert child.returncode == 1
    assert "reduction_drift" in child.stdout


def test_checksum_refusal_and_cli_errors(tmp_path):
    """REQ-REPORT-8984: malformed evidence and invalid dates fail visibly."""
    path = tmp_path / "bad.json"
    atomic_json(path, dict(reproducibility_checksum="bad"))
    with pytest.raises(ValueError, match="checksum"):
        m.replay(path)
    assert m.main(["--cold-replay", str(path)]) == 1
    with pytest.raises(SystemExit):
        m.main(["--date", "wrong"])
    assert canonical_hash([]) != canonical_hash([{}])
    assert deepcopy(m.OWNED) == m.OWNED


def test_changed_producer_and_terminal_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8984: rejected graph bytes preserve independent natural access evidence."""
    monkeypatch.setitem(m.PINS, 8973, "sha256:changed")
    work = m.collect(tmp_path / "raw", "20261110")
    assert (
        next(s for s in work["slots"] if s["experiment_id"] == 8973)["disposition"]
        == "quarantined_primary"
    )
    reduced = m.reduce(work)
    assert reduced["shared_feature"]["status"] == "unavailable"
    assert reduced["natural_evidence"]["intended_pairs"] == 800
    validation = dict(passed=False, coverage=dict(complete=False), receipts=[])
    receipts = tmp_path / "receipts.json"
    atomic_json(receipts, validation)
    monkeypatch.setattr(m, "terminal", lambda *args: dict(passed=False))
    assert (
        m.main(["--output", str(tmp_path / (m.NAME + ".json")), "--receipts", str(receipts)]) == 1
    )


def test_validation_requires_actual_measurement(tmp_path, monkeypatch):
    """REQ-REPORT-8984: percentages and failed receipts cannot invent executed statements."""
    receipts = tmp_path / "validation.json"
    validation = dict(passed=True, coverage=dict(complete=True), receipts=[])
    atomic_json(receipts, validation)
    args = ["--output", str(tmp_path / (m.NAME + ".json")), "--receipts", str(receipts)]
    assert m.main(args) == 1
    monkeypatch.setattr(m.authority, "verify_measured", lambda *args: True)
    validation["receipts"] = [dict(passed=False)]
    atomic_json(receipts, validation)
    assert m.main(args) == 1
    validation["receipts"] = [dict(passed=True)]
    atomic_json(receipts, validation)
    monkeypatch.setattr(m, "terminal", lambda *args: dict(passed=False))
    assert m.main(args) == 1
