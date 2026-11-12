"""REQ-REPORT-9012-ACCOUNTING / SCENARIO-REPORT-9012-CUSTODY: retain real outcomes."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from carnot.reporting import v769_capstone as m
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Snapshot real local observations into private storage; fixtures never supply science."""
    value = m.collect(tmp_path_factory.mktemp("capstone9012"), "20261111")
    atomic_json(Path(value["raw"]) / "work.json", value)
    return value


def test_complete_accounting_keeps_claims_separate(work, tmp_path):
    """REQ-REPORT-9012: complete custody can be ready over blocked external science."""
    validation = dict(passed=True, coverage=dict(complete=True), receipts=[])
    value = m.build(work, validation)
    assert value["verdict_class"] == "blocked"
    assert value["reconciliation_ready_score"] == 1
    assert value["completed_count"] == len(value["slot_rows"]) == 14
    assert [s["experiment_id"] for s in value["slot_rows"]] == list(range(8999, 9013))
    slots = {s["experiment_id"]: s for s in value["slot_rows"]}
    assert slots[8999]["qualified"]
    assert slots[9000]["disposition"] == slots[9001]["disposition"] == "quarantined_primary"
    assert slots[9002]["disposition"] == "authenticated_scheduler_skip"
    assert slots[9002]["primary_status"] == "missing_file"
    assert slots[9004]["honest_verdict"] is slots[9004]["verdict_class"] is None
    assert slots[9004]["disposition"] == "conductor_skip_without_primary"
    science = value["v769_science_summary"]
    for field, count in [
        ("graph_detector_bounds", 7),
        ("causal_learning_bounds", 10),
        ("natural_label_bounds", 4),
    ]:
        assert len(science[field]) == count
        assert all(
            r["observed"] == "missing_file" and r["denominator"] == 0 for r in science[field]
        )
    assert science["natural_source_audit"]["source_group_count"] == 1823
    assert science["natural_source_audit"]["unseen_source_group_count"] == 0
    assert science["historical_generation_calls_counted_once"] == 512
    assert not science["scientific_benefit"]
    assert not any(value["model_invocation_counts"].values())
    assert value["MODEL_SPECS"] == value["model_specs"] == value["trained_head_specs"] == []
    assert value["independent_generalization_score"] == value["independent_count"] == 0
    assert value["G2"] == work["publication_gate"]["gates"]["G2"]
    assert any(
        r["prior_experiment_id"] == "exp8990-graph-corpus-a"
        and r["actual_prior_verdict"] == "complete_disqualified_graph_corpus_a"
        for r in value["retirement_scope_rows"]
    )
    assert m.build(work, dict(validation, passed=False))["reconciliation_ready_score"] == 0
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert m.replay(candidate)["passed"]
    value["v769_science_summary"]["natural_source_audit"]["unseen_source_group_count"] = 600
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        m.replay(candidate)


def test_missing_and_changed_producers(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9012-CUSTODY: absence and rejected bytes never become scientific nulls."""
    absent = m.collect(tmp_path / "absent_raw", "20261111", tmp_path / "missing")
    assert all(s["primary_status"] == "missing_file" for s in absent["slots"])
    assert m.reduce(absent)["natural_source_audit"]["status"] == "missing_file"
    monkeypatch.setitem(m.PINS, 9000, "sha256:changed")
    changed = m.collect(tmp_path / "changed_raw", "20261111")
    assert any(r["check"] == "input_bytes" and not r["passed"] for r in changed["failures"])
    assert not next(s for s in changed["slots"] if s["experiment_id"] == 9000)["qualified"]
    gate = dict(upstream="absent", artifact_field="score", op="==", value=1)
    assert m.edge(gate, {})["observed"] == "missing_file"
    values = {"absent": dict(value={}, reference={})}
    assert m.edge(gate, values)["observed"] == "missing_field"
    values["absent"]["value"]["score"] = None
    assert m.edge(gate, values)["operand_status"] == "scientific_null"


def test_authority_and_source_tamper(work, tmp_path):
    """REQ-REPORT-9012: E2E-018/021 reject changed authority and primitive custody."""
    changed = deepcopy(work)
    changed["tasks"][0]["prompt"] += "changed"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(changed, dict(passed=False, coverage={}, receipts=[]))
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, dict(reproducibility_checksum="wrong"))
    assert m.main(["--cold-replay", str(candidate)]) == 1
    with pytest.raises(SystemExit):
        m.main(["--date", "wrong"])
    changed = deepcopy(work)
    ref = changed["references"][-1]
    ref["sha256"] = "sha256:changed"
    changed_path = tmp_path / "work.json"
    atomic_json(changed_path, changed)
    value = m.build(changed, dict(passed=True, coverage=dict(complete=True), receipts=[]))
    value["work_reference"] = m.reference(changed_path)
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="operand_drift"):
        m.replay(candidate)


def test_receipts_and_primary_boundary(work, tmp_path, monkeypatch):
    """REQ-REPORT-9012: require byte-bound executed checks before primary replacement."""
    observed = deepcopy(work)

    def collect_control(raw, *args):
        copied = deepcopy(observed)
        shutil.copytree(Path(observed["raw"]) / "authority", raw / "authority")
        copied.update(raw=str(raw), contract=m.authority.authority(raw / "authority"))
        return copied

    monkeypatch.setattr(m, "collect", collect_control)
    output = tmp_path / (m.NAME + ".json")
    receipt = tmp_path / "receipts.json"
    args = ["--output", str(output), "--receipts", str(receipt)]
    assert m.main(args) == 1
    validation = dict(passed=True, coverage=dict(complete=True), receipts=[])
    atomic_json(receipt, validation)
    assert m.main(args) == 1
    monkeypatch.setattr(m, "verify_measured", lambda *a: True)
    validation["receipts"] = [dict(passed=False)]
    atomic_json(receipt, validation)
    assert m.main(args) == 1
    log = tmp_path / "validation.log"
    log.write_text("private receipt boundary control\n")
    row = dict(
        name="private_control",
        passed=True,
        exit_code=0,
        log_path=str(log),
        log_sha256=m.sha256_file(log),
        command_argv=["private_control"],
    )
    plan = tmp_path / "plan.json"
    atomic_json(plan, dict(argv=[["different"]]))
    validation.update(receipts=[row], plan_reference=m.reference(plan))
    atomic_json(receipt, validation)
    assert m.main(args) == 1
    atomic_json(
        plan,
        dict(
            argv=[row["command_argv"]],
            code_hashes={p: m.sha256_file(m.ROOT / p) for p in [*m.OWNED, m.TEST]},
        ),
    )
    validation["plan_reference"] = m.reference(plan)
    atomic_json(receipt, validation)
    assert m.main(args) == 1
    monkeypatch.setattr(m, "CHECKS", {"private_control"})
    monkeypatch.setattr(m, "terminal", lambda *a: dict(passed=False))
    assert m.main(args) == 1
    monkeypatch.setattr(m, "terminal", lambda *a: dict(passed=True, receipts=[]))
    assert m.main(args) == 0
    assert m.replay(output)["passed"]
    validation.update(passed=False, owned_passed=True, repository_health=[dict(passed=False)])
    atomic_json(receipt, validation)
    assert m.main(args) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["reconciliation_ready_score"] == 0
    observed["started_ns"] -= 4000 * 10**9
    assert m.main(args) == 1


def test_real_foreign_cli_and_terminal(work, tmp_path):
    """SCENARIO-REPORT-9012-CUSTODY: real E2E-018/021 CLI refuses, replays and detects tamper."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    output = tmp_path / (m.NAME + ".json")
    print("[test9012] before_cli completed=0 pending=3", flush=True)
    child = subprocess.run(
        [*command, "--output", str(output), "--sources", str(tmp_path / "absent")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 1, child.stdout + child.stderr
    assert "owned_validation_missing_or_failed" in child.stdout
    assert not output.exists()
    candidate = next((tmp_path / "raw" / m.NAME / "invocations").glob("*/candidate.json"))
    child = subprocess.run(
        [*command, "--cold-replay", str(candidate)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(candidate.read_bytes())
    value["completed_count"] = 15
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(candidate, value)
    child = subprocess.run(
        [*command, "--cold-replay", str(candidate)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 1 and "reduction_drift" in child.stdout
    print("[test9012] after_cli completed=3 pending=0", flush=True)
    atomic_json(
        candidate, m.build(work, dict(passed=True, coverage=dict(complete=True), receipts=[]))
    )
    assert m.terminal(candidate, tmp_path / "validators")["passed"]
