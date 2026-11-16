"""REQ-REPORT-9123-V777: complete accounting never invents unavailable science."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v777_capstone as m
from carnot.reporting import v777_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Authenticate actual archived evidence without rerunning its producers."""
    return m.collect(tmp_path_factory.mktemp("capstone9123"), "20261116")


def test_accounting_scope_and_bounds(work, monkeypatch):
    """SCENARIO-REPORT-9123-V777: all slots survive the external gate cascade."""
    monkeypatch.setattr(m.legacy.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    value = m.build(work)
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert len(value["full_task_objects"]) == len(value["task_outcome_rows"]) == 14
    slots = {r["experiment_id"]: r for r in value["task_outcome_rows"]}
    assert list(slots) == list(range(9110, 9124))
    assert slots[9111]["verdict_class"] == "disqualified"
    assert slots[9112]["primary_status"] == "missing_file"
    assert slots[9112]["selected_path"] == "results/experiment_9112_masked_runtime.json"
    assert slots[9112]["scheduler_alternative"]["sha256"]
    assert slots[9114]["scheduler_alternative"] is None
    assert slots[9122]["readiness"]["cost_inventory_ready_score"] == 1
    assert [
        len(value[k]) for k in ["static_bound_rows", "online_bound_rows", "energy_bound_rows"]
    ] == [5, 6, 2]
    assert all(
        r["value"] is None and not r["passed"]
        for k in ["static_bound_rows", "online_bound_rows", "energy_bound_rows"]
        for r in value[k]
    )
    assert value["human_summary"]["question_count"] == 300
    assert value["judge_summary"]["qualified"] is False
    assert value["susceptibility_summary"]["counts"]["completed"] == 32
    assert value["original_exp8435_null"]["honest_verdict"].startswith("complete_null_")
    assert value["MODEL_SPECS"] == [] and value["current_llm_invocations"] == 0
    assert value["generated_tokens"] == value["independent_generalization_score"] == 0
    assert not value["activation_authorized"] and not value["external_publication_authorized"]
    assert len(value["cost_summary"]["rows"]) == 120
    assert value["historical_code_resolutions"]
    assert all(r["passed"] for r in work["checks"])


def test_failed_owned_checks_and_missing_inputs(work, tmp_path):
    """REQ-REPORT-9123-V777: unfinished validation disqualifies; external absence blocks."""
    assert m.build(work)["reconciliation_ready_score"] == 0
    blocked = m.collect(tmp_path / "raw", "20261116", tmp_path / "absent")
    value = m.build(blocked)
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert len(value["task_outcome_rows"]) == 14
    assert any(r["observed"] == "missing_file" for r in value["gate_check_summary"])
    changed = deepcopy(work)
    changed["tasks"][0]["prompt"] += "tampered"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(changed)


def test_gate_operands():
    """REQ-REPORT-9123-V777: missing is never silently converted to zero."""
    gate = dict(upstream="exp1-test", artifact_field="ready", op="==", value=1)
    assert e.edge(gate, {})["observed"] == "missing_file"
    assert e.edge(gate, {"exp1-test": dict(value={}, reference={})})["observed"] == "missing_field"
    assert e.edge(gate, {"exp1-test": dict(value=dict(ready=0), reference={})})["observed"] == 0


def test_replay_tamper(work, tmp_path):
    """SCENARIO-REPORT-9123-V777: a new checksum cannot authorize invented totals."""
    path = tmp_path / "candidate.json"
    value = m.build(work)
    atomic_json(path, value)
    assert m.replay(path)["passed"]
    for key in ["reconciliation_ready_score", "current_llm_invocations"]:
        changed = deepcopy(value)
        changed[key] = 1
        changed["reproducibility_checksum"] = m.checksum(changed)
        atomic_json(path, changed)
        assert not m.replay(path)["passed"]


def test_private_cli_success_refusal_tamper(work, tmp_path):
    """SCENARIO-REPORT-9123-V777: E2E-018/021 use the actual foreign CLI."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]

    def call(*args):
        print("[test9123] child_before", args, flush=True)
        result = subprocess.run(
            [*argv, *args], cwd=tmp_path, env=env, capture_output=True, timeout=180
        )
        print("[test9123] child_after", result.returncode, flush=True)
        return result

    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / "valid" / (m.NAME + ".json")
    result = call("--from-work", str(source), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert call("--cold-replay", str(output)).returncode == 0
    assert call("--date", "20261115").returncode == 2
    assert call("--from-work", str(source)).returncode == 2
    changed = json.loads(output.read_bytes())
    changed["energy_bound_rows"][0]["value"] = 1
    changed["reproducibility_checksum"] = m.checksum(changed)
    atomic_json(output, changed)
    assert call("--cold-replay", str(output)).returncode == 1


def test_owned_main_success_and_refusal(work, tmp_path, monkeypatch):
    """REQ-REPORT-9123-V777: a failed owned check prevents visible publication."""

    def collect(raw, date, root):
        current = deepcopy(work)
        current["raw"] = str(raw)
        return current

    monkeypatch.setattr(m, "collect", collect)
    monkeypatch.setattr(m.legacy, "validate", lambda _: dict(receipts=[], coverage={}))
    monkeypatch.setattr(m.runner, "terminal", lambda *_: dict(passed=True, receipts=[]))
    output = tmp_path / (m.NAME + ".json")
    assert m.main(["--output", str(output)]) == 1
    assert not output.exists()
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    assert m.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["reconciliation_ready_score"] == 1


def test_binding_tamper_and_missing_authority(work, tmp_path):
    """REQ-REPORT-9123-V777: changed primitive bytes cannot repair historical science."""
    ref = next(iter(work["operands"].values()))
    changed = deepcopy(work)
    changed["operands"]["fixture"] = dict(ref, sha256="sha256:changed")
    with pytest.raises(ValueError, match="primitive_identity"):
        e.primitive(changed, dict(path="fixture", sha256=ref["sha256"]))
    blocked = e.authenticate(tmp_path / "missing", tmp_path / "raw")
    assert not blocked["tasks"]
    assert len(e.outcomes(blocked)) == 13
