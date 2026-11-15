"""REQ-REPORT-9109-V776: account for blocked science without inventing benefit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v776_capstone as m
from carnot.reporting import v776_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Read actual immutable inputs; never run an upstream producer."""
    return m.collect(tmp_path_factory.mktemp("capstone9109"), "20261115")


def test_accounting_and_scope(work, monkeypatch):
    """SCENARIO-REPORT-9109-V776-ACCOUNTING: availability grants no benefit."""
    monkeypatch.setattr(m.legacy.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    assert "rows" not in work["values"]["prior:8435"]["value"]
    value = m.build(work)
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert [r["experiment_id"] for r in value["task_outcome_rows"]] == list(range(9096, 9110))
    slots = {r["experiment_id"]: r for r in value["task_outcome_rows"]}
    assert slots[9097]["readiness"]["source_ready_score"] == 1
    assert slots[9098]["readiness"]["embeddings_ready_score"] == 0
    assert slots[9098]["readiness"]["logits_ready_score"] == 1
    assert slots[9098]["failed_gate_value"] == 0
    assert slots[9102]["honest_verdict"] is None
    assert slots[9100]["primary_status"] == "missing_file"
    assert slots[9100]["scheduler_alternative"]["sha256"]
    assert slots[9102]["verdict_class"] == "blocked"
    assert slots[9107]["verdict_class"] == "disqualified"
    assert value["human_summary"]["question_count"] == 300
    assert value["human_summary"]["role_counts"] == {"TRAIN": 450, "CAL": 150, "TEST": 300}
    assert value["human_summary"]["annotator2_missing_count"] == 600
    assert value["judge_summary"]["qualified"] is False
    assert value["judge_summary"]["metrics"]["Brier"] == pytest.approx(0.23546374784425125)
    assert value["historical_reduction_reference"]["sha256"]
    assert value["susceptibility_summary"]["counts"]["completed"] == 32
    assert len(value["static_six_arm_outcomes"]) == 6
    assert all(r["absolute_metric"] is None for r in value["static_six_arm_outcomes"])
    assert len(value["seven_bound_rows"]) == 7 and len(value["ten_bound_rows"]) == 10
    assert value["sample_size_budget"] == {"administrative_slots": 14}
    assert value["MODEL_SPECS"] == [] and value["current_llm_invocations"] == 0
    assert value["independent_generalization_score"] == 0
    assert not value["current_science_paper_ready"]
    assert value["gate_failure_rows"]
    assert value["open_gaps"]["GAP-ORACLE-DISTINCT"] == "open"
    assert all(r["manifest_mutated"] is False for r in value["retirement_recommendations"])


def test_replay_tampering(work, tmp_path):
    """SCENARIO-REPORT-9109-V776-REPLAY: rehashing cannot authenticate false totals."""
    path = tmp_path / "candidate.json"
    value = m.build(work)
    atomic_json(path, value)
    assert m.replay(path)["passed"]
    value["human_summary"]["question_count"] += 1
    value["reproducibility_checksum"] = m.legacy.checksum(value)
    atomic_json(path, value)
    assert m.replay(path)["reason"] == "reduction_drift"


def test_missing_inputs_and_frozen_task_tamper(work, tmp_path):
    """REQ-REPORT-9109-V776: terminal external absence still accounts for fourteen slots."""
    blocked = m.collect(tmp_path / "blocked", "20261115", tmp_path / "absent")
    value = m.build(blocked)
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert len(value["task_outcome_rows"]) == 14
    assert any(r["observed"] == "missing_file" for r in value["gate_check_summary"])
    changed = deepcopy(work)
    changed["tasks"][0]["prompt"] += "tampered"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(changed)


def test_private_cli_success_block_and_tamper(work, tmp_path):
    """SCENARIO-REPORT-9109-V776-REPLAY: foreign actual CLI has no ambient imports."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]

    def call(*args):
        print("[test9109] before_child", args, flush=True)
        result = subprocess.run(
            [*argv, *args], cwd=tmp_path, env=env, capture_output=True, timeout=180
        )
        print("[test9109] after_child", result.returncode, flush=True)
        return result

    for label, current in [
        ("valid", work),
        ("blocked", m.collect(tmp_path / "missing", "20261115", tmp_path / "absent")),
    ]:
        source = tmp_path / (label + ".json")
        atomic_json(source, current)
        output = tmp_path / label / (m.NAME + ".json")
        result = call("--from-work", str(source), "--output", str(output))
        assert result.returncode == 0, result.stdout + result.stderr
        assert call("--cold-replay", str(output)).returncode == 0
        value = json.loads(output.read_bytes())
        value["current_llm_invocations"] = 1
        value["reproducibility_checksum"] = m.legacy.checksum(value)
        atomic_json(output, value)
        assert call("--cold-replay", str(output)).returncode == 1


def test_gate_null_zero_and_prior_scope():
    """REQ-REPORT-9109-V776: row operands and prior scope stay explicit."""
    gate = dict(upstream="exp1-test", artifact_field="ready", op="==", value=1)
    assert e.edge(gate, {})["observed"] == "missing_file"
    assert e.edge(gate, {"exp1-test": dict(value={}, reference={})})["observed"] == "missing_field"
    assert e.edge(gate, {"exp1-test": dict(value=dict(ready=0), reference={})})["observed"] == 0


def test_changed_producer_bindings(tmp_path, monkeypatch):
    """REQ-REPORT-9109-V776: code drift is context; missing consumed data blocks."""
    root = tmp_path / "inputs"
    path = root / "results/experiment_9097_fixture.json"
    code = tmp_path / "producer.py"
    code.write_text("original = True\n")
    atomic_json(
        path,
        dict(
            experiment_id=9097,
            task_id=e.TASKS[1],
            code_config_hashes=[{}, dict(path=str(code), sha256="sha256:original")],
            raw_shard_hashes=[dict(path=str(tmp_path / "absent"), sha256="sha256:absent")],
        ),
    )
    monkeypatch.setattr(e, "PINS", {str(path.relative_to(root)): m.reference(path)["sha256"]})
    result = e.authenticate(root, tmp_path / "raw")
    assert result["values"][e.TASKS[1]]["current_code_drift"]
    assert any(r["check"] == "producer_binding" and not r["passed"] for r in result["checks"])
