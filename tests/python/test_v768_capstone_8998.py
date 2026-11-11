"""REQ-REPORT-8998 / SCENARIO-REPORT-8998-CUSTODY: preserve absent science."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v768_capstone as m
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Retain real upstream observations once in private storage, never fabricate science."""
    value = m.collect(tmp_path_factory.mktemp("capstone"), "20261111")
    atomic_json(Path(value["raw"]) / "work.json", value)
    return value


def test_accounting_and_reductions(work, tmp_path):
    """REQ-REPORT-8998: method success and unavailable science have different denominators."""
    science = m.reduce(work)
    assert science["development"]["collection_permitted"]
    assert len(science["development"]["admission_rows"]) == 4
    assert len(science["detector_bounds"]) == 7
    assert len(science["learning_bounds"]) == 10
    assert all(r["observed"] == "missing_file" for r in science["detector_bounds"])
    assert science["natural"]["independent_human_groups"] == 0
    assert science["natural"]["response_rows"] == 18492
    assert science["scientific_benefit"] is False
    validation = dict(passed=True, coverage=dict(complete=True), receipts=[])
    value = m.build(work, validation)
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert len(value["slot_rows"]) == value["completed_count"] == 14
    assert not any(value["model_invocation_counts"].values())
    assert value["MODEL_SPECS"] == value["model_specs"] == value["trained_head_specs"] == []
    assert value["independent_generalization_score"] == 0
    assert value["G2"] == work["publication_gate"]["gates"]["G2"]
    assert m.build(work, dict(validation, passed=False))["verdict_class"] == "disqualified"
    slots = {s["experiment_id"]: s for s in value["slot_rows"]}
    assert slots[8990]["disposition"] == "quarantined_primary"
    assert slots[8991]["disposition"] == "missing_primary"
    assert slots[8992]["disposition"] == "authenticated_scheduler_skip"
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert m.replay(path)["passed"]
    value["v768_science_summary"]["natural"]["independent_human_groups"] = 200
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(path, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        m.replay(path)


def test_missing_and_changed_sources(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8998-CUSTODY: retain rejection and never turn absence into zero."""
    work = m.collect(tmp_path / "raw", "20261111", tmp_path / "absent")
    science = m.reduce(work)
    assert science["development"]["status"] == "missing_file"
    assert science["natural"]["status"] == "missing_file"
    monkeypatch.setitem(m.PINS, 8987, "sha256:changed")
    work = m.collect(tmp_path / "changed", "20261111")
    assert not next(s for s in work["slots"] if s["experiment_id"] == 8987)["qualified"]
    assert any(r["check"] == "input_bytes" and not r["passed"] for r in work["failures"])
    assert m.reduce(work)["development"]["status"] == "missing_file"
    gate = dict(upstream="exp8991-graph-corpus-b", artifact_field="score", op="==", value=1)
    assert m.edge(gate, {})["observed"] == "missing_file"
    values = {gate["upstream"]: dict(value={}, reference={})}
    assert m.edge(gate, values)["observed"] == "missing_field"
    values[gate["upstream"]]["value"]["score"] = 0
    assert m.edge(gate, values)["operand_status"] == "observed_zero"
    values[gate["upstream"]]["value"]["score"] = None
    assert m.edge(gate, values)["operand_status"] == "scientific_null"


def test_tamper_and_validation_refusal(work, tmp_path, monkeypatch):
    """REQ-REPORT-8998: E2E-018/021 reject authority drift and unexecuted validation."""
    changed = deepcopy(work)
    changed["tasks"].reverse()
    with pytest.raises(ValueError, match="task_authority"):
        m.build(changed, dict(passed=False, coverage={}, receipts=[]))
    path = tmp_path / "bad.json"
    atomic_json(path, dict(reproducibility_checksum="bad"))
    assert m.main(["--cold-replay", str(path)]) == 1
    with pytest.raises(SystemExit):
        m.main(["--date", "wrong"])
    monkeypatch.setattr(m, "collect", lambda *a: deepcopy(work))
    receipt = tmp_path / "validation.json"
    atomic_json(receipt, dict(passed=True, coverage=dict(complete=True), receipts=[]))
    assert m.main(["--output", str(tmp_path / (m.NAME + ".json")), "--receipts", str(receipt)]) == 1


def test_owned_receipts_and_primary(work, tmp_path, monkeypatch):
    """REQ-REPORT-8998: receipt checks precede primary replacement, including failures."""
    observed = deepcopy(work)
    monkeypatch.setattr(m, "collect", lambda raw, *a: dict(deepcopy(observed), raw=str(raw)))
    validation = dict(passed=True, coverage=dict(complete=True), receipts=[dict(passed=False)])
    receipt = tmp_path / "receipts.json"
    output = tmp_path / (m.NAME + ".json")
    args = ["--output", str(output), "--receipts", str(receipt)]
    monkeypatch.setattr(m.authority.prior, "verify_measured", lambda *a: True)
    atomic_json(receipt, validation)
    assert m.main(args) == 1
    log = tmp_path / "validation.log"
    log.write_text("real private validation boundary\n")
    row = dict(
        name="private_control",
        passed=True,
        exit_code=0,
        log_path=str(log),
        log_sha256=m.sha256_file(log),
        command_argv=["private_control"],
    )
    plan = tmp_path / "argv.json"
    atomic_json(plan, dict(argv=[["different"]]))
    validation.update(receipts=[row], plan_reference=m.reference(plan))
    atomic_json(receipt, validation)
    assert m.main(args) == 1
    monkeypatch.setattr(m, "CHECKS", {"private_control"})
    atomic_json(
        plan,
        dict(
            argv=[row["command_argv"]],
            code_hashes={p: m.sha256_file(m.ROOT / p) for p in [*m.OWNED, m.TEST]},
        ),
    )
    validation["plan_reference"] = m.reference(plan)
    atomic_json(receipt, validation)
    monkeypatch.setattr(m, "terminal", lambda *a: dict(passed=False))
    assert m.main(args) == 1
    monkeypatch.setattr(m, "terminal", lambda *a: dict(passed=True, receipts=[]))
    assert m.main(args) == 0
    value = json.loads(output.read_bytes())
    assert value["reconciliation_ready_score"] == 1
    assert m.replay(output)["passed"]
    observed["started_ns"] -= 4000 * 10**9
    assert m.main(args) == 1
    del value["v768_science_summary"]["natural"]["reference"]
    atomic_json(output, value)
    with pytest.raises(ValueError, match="checksum_drift"):
        m.replay(output)


def test_real_foreign_cli_and_terminal(work, tmp_path):
    """SCENARIO-REPORT-8998-CUSTODY: E2E-018/021 real CLI refuses and replays private bytes."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    output = tmp_path / (m.NAME + ".json")
    print("[test8998] before_cli completed=0 pending=3", flush=True)
    child = subprocess.run(
        [*command, "--output", str(output), "--sources", str(tmp_path / "absent")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
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
        timeout=180,
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
        timeout=180,
    )
    assert child.returncode == 1
    assert "reduction_drift" in child.stdout
    print("[test8998] after_cli completed=3 pending=0", flush=True)
    validation = dict(passed=True, coverage=dict(complete=True), receipts=[])
    atomic_json(candidate, m.build(work, validation))
    assert m.terminal(candidate, tmp_path / "validators")["passed"]
