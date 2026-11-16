"""REQ-REPORT-9137-V778: complete accounting cannot create scientific evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v778_capstone as m
from carnot.reporting import v778_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Use actual immutable upstream bytes without rerunning their producers."""
    return m.collect(tmp_path_factory.mktemp("capstone9137"), "20261116")


def test_fourteen_outcomes_and_independent_bounds(work, monkeypatch):
    """SCENARIO-REPORT-9137-V778: all scientific gates remain distinct."""
    monkeypatch.setattr(m.legacy.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    value = m.build(work)
    assert all(r["passed"] for r in work["checks"])
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert len(value["full_task_objects"]) == len(value["task_outcome_rows"]) == 14
    slots = {r["experiment_id"]: r for r in value["task_outcome_rows"]}
    assert list(slots) == list(range(9124, 9138))
    assert slots[9125]["verdict_class"] == "disqualified"
    assert slots[9127]["primary_status"] == "missing_file"
    assert slots[9127]["selected_path"] == "results/experiment_9127_logit_runtime.json"
    assert slots[9127]["scheduler_alternative"]["sha256"]
    assert slots[9128]["scheduler_alternative"] is None
    assert slots[9136]["readiness"]["cost_inventory_ready_score"] == 1
    assert len(value["two_bound_rows"]) == 2
    assert len(value["seven_bound_rows"]) == 7
    assert all(
        r["value"] is None and r["status"] == "unmeasured"
        for r in value["two_bound_rows"] + value["seven_bound_rows"]
    )
    assert value["donor_changed_label_fraction"] == pytest.approx(0.4066666666666667)
    assert value["gauge_control_rows"]
    assert value["MODEL_SPECS"] == []
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert value["independent_generalization_score"] == 0
    assert not value["current_science_paper_ready"]
    assert not value["activation_authorized"] and not value["external_publication_authorized"]
    assert value["cost_summary"]["scientific_request_counts"]["completed"] == 0
    assert value["hardware_limits"]["KV260"]
    assert value["retirement_recommendations"]
    assert all(r["manifest_mutated"] is False for r in value["retirement_recommendations"])


def test_blocked_inputs_and_owned_failure(work, tmp_path):
    """REQ-REPORT-9137-V778: external absence blocks; owned failure disqualifies."""
    assert m.build(work)["verdict_class"] == "disqualified"
    blocked = m.collect(tmp_path / "raw", "20261116", tmp_path / "missing")
    value = m.build(blocked)
    assert value["verdict_class"] == "blocked"
    assert len(value["task_outcome_rows"]) == 14
    assert any(r["observed"] == "missing_file" for r in value["gate_check_summary"])
    altered = deepcopy(work)
    altered["tasks"][0]["prompt"] += "changed"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(altered)


def test_exact_gate_operands(tmp_path):
    """REQ-REPORT-9137-V778: no zero, bool or absence substitution."""
    gate = dict(upstream="exp1-test", artifact_field="ready", op="==", value=1)
    assert e.edge(gate, {})["observed"] == "missing_file"
    entry = dict(value={}, reference={})
    assert e.edge(gate, {"exp1-test": entry})["observed"] == "missing_field"
    entry["value"]["ready"] = 0
    assert e.edge(gate, {"exp1-test": entry})["observed"] == 0
    entry["value"]["ready"] = True
    assert not e.edge(gate, {"exp1-test": entry})["passed"]
    assert not e.check(tmp_path, "ready", 1, True, "strict")["passed"]


def test_replay_and_retained_byte_tamper(work, tmp_path):
    """SCENARIO-REPORT-9137-V778: a fresh checksum cannot bless invented rows."""
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
    changed = deepcopy(work)
    ref = next(iter(changed["operands"].values()))
    ref["sha256"] = "sha256:changed"
    atomic_json(Path(changed["raw"]) / "tampered_work.json", changed)
    atomic_json(
        path, dict(value, work_reference=m.reference(Path(changed["raw"]) / "tampered_work.json"))
    )
    assert not m.replay(path)["passed"]


def test_private_actual_cli(work, tmp_path):
    """SCENARIO-REPORT-9137-V778: E2E-018/021 real success/refusal/tamper/replay."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]

    def call(*args):
        print("[test9137] child_before", args, flush=True)
        result = subprocess.run(
            [*argv, *args], cwd=tmp_path, env=env, capture_output=True, timeout=180
        )
        print("[test9137] child_after", result.returncode, flush=True)
        return result

    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / (m.NAME + ".json")
    result = call("--from-work", str(source), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert call("--cold-replay", str(output)).returncode == 0
    assert call("--date", "20261115").returncode == 2
    assert call("--from-work", str(source)).returncode == 2
    changed = json.loads(output.read_bytes())
    changed["two_bound_rows"][0]["value"] = 1
    changed["reproducibility_checksum"] = m.checksum(changed)
    atomic_json(output, changed)
    assert call("--cold-replay", str(output)).returncode == 1
