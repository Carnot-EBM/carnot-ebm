"""REQ-REPORT-9165-V780: administrative closure cannot supply missing science."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v780_capstone as m
from carnot.reporting import v780_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Use the original pinned producers; do not rerun historical measurements."""
    return m.collect(tmp_path_factory.mktemp("capstone9165"), "20261116")


def test_fourteen_slots_and_science(work, monkeypatch):
    """SCENARIO-REPORT-9165-V780: scheduler closure supplies no scientific primary."""
    monkeypatch.setattr(m.legacy.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    value = m.build(work)
    assert all(r["passed"] for r in work["checks"])
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert len(value["task_outcome_rows"]) == len(value["full_task_objects"]) == 14
    slots = {r["experiment_id"]: r for r in value["task_outcome_rows"]}
    assert list(slots) == list(range(9152, 9166))
    assert slots[9153]["readiness"]["schedule_ready_score"] == 0
    assert slots[9154]["primary_status"] == "missing_file"
    assert slots[9154]["scheduler_alternative"]["sha256"]
    assert slots[9155]["scheduler_alternative"] is None
    assert slots[9164]["primary_status"] == "missing_file"
    assert value["validation_obligations"]["completed"] == 32
    assert value["validation_obligations"]["censored"] == 642
    assert value["validation_obligations"]["intended"] == 674
    assert value["collection_partition"]["answer_count"] == 900
    assert value["collection_partition"]["qid_counts"] == [100, 100, 100]
    assert value["collection_partition"]["union_equals_original"]
    assert len(value["two_bound_rows"]) == 2 and len(value["seven_bound_rows"]) == 7
    assert all(
        r["absolute_metric"] is None and r["status"] == "unmeasured"
        for r in value["two_bound_rows"] + value["seven_bound_rows"]
    )
    human = value["science_decisions"]["human_labels"]
    assert human["group_counts"] == dict(all_correct=100, all_wrong=76, mixed=124)
    assert human["observed_pairs"] == 300 and human["disagreements"] == 29
    assert human["missing_pairs"] == 600
    assert value["science_decisions"]["cost"]["scientific_request_counts"]["completed"] == 0
    assert value["hardware_limits"]["KV260"]
    assert value["MODEL_SPECS"] == []
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert all(x == 0 for x in value["model_invocation_counts"].values())
    assert value["claim_scope"]["independent_generalization_score"] == 0
    assert value["independent_generalization_score"] == 0
    assert not value["publication_gate"]["current_science_paper_ready"]
    assert not value["activation_authorized"] and not value["external_publication_authorized"]
    assert value["sample_size_budget"]["completed"] == 14
    assert any(
        r["decision"] == "retire_exact_schedule_scope" for r in value["continuation_decisions"]
    )
    assert all(r["manifest_mutated"] is False for r in value["retirement_recommendations"])


def test_absence_owned_failure_and_task_drift(work, tmp_path):
    """REQ-REPORT-9165-V780: absent prerequisites block, owned failures disqualify."""
    assert m.build(work)["verdict_class"] == "disqualified"
    blocked = m.collect(tmp_path / "raw", "20261116", tmp_path / "absent")
    value = m.build(blocked)
    assert value["verdict_class"] == "blocked"
    assert len(value["task_outcome_rows"]) == 14
    assert any(r["observed"] == "missing_file" for r in value["gate_check_summary"])
    changed = deepcopy(work)
    changed["tasks"][0]["prompt"] += " changed"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(changed)


def test_gate_contract(tmp_path):
    """REQ-REPORT-9165-V780: preserve absence, observed zero and strict operand types."""
    gate = dict(upstream="exp1-test", artifact_field="ready", op="==", value=1)
    assert e.edge(gate, {})["observed"] == "missing_file"
    entry = dict(value={}, reference={})
    assert e.edge(gate, {"exp1-test": entry})["observed"] == "missing_field"
    entry["value"]["ready"] = 0
    assert e.edge(gate, {"exp1-test": entry})["observed"] == 0
    entry["value"]["ready"] = True
    assert not e.edge(gate, {"exp1-test": entry})["passed"]
    assert not e.check(tmp_path, "ready", 1, True, "strict")["passed"]


def test_replay_and_input_tamper(work, tmp_path):
    """SCENARIO-REPORT-9165-V780: rehashing does not authenticate invented outcomes."""
    path = tmp_path / "candidate.json"
    value = m.build(work)
    atomic_json(path, value)
    assert m.replay(path)["passed"]
    for key in ["current_llm_invocations", "reconciliation_ready_score", "science_decisions"]:
        changed = deepcopy(value)
        changed[key] = 1
        changed["reproducibility_checksum"] = m.checksum(changed)
        atomic_json(path, changed)
        assert not m.replay(path)["passed"]
    altered = deepcopy(work)
    ref = next(iter(altered["operands"].values()))
    ref["sha256"] = "sha256:changed"
    source = tmp_path / "tampered_work.json"
    atomic_json(source, altered)
    changed = dict(value, work_reference=m.reference(source))
    changed["reproducibility_checksum"] = m.checksum(changed)
    atomic_json(path, changed)
    assert not m.replay(path)["passed"]
    changed = deepcopy(work)
    changed["values"]["prior:9151"]["value"]["human_label_rows"][0]["a1_error_count"] += 1
    with pytest.raises(ValueError, match="human_rows"):
        m.build(changed)


def test_private_real_cli(work, tmp_path):
    """SCENARIO-REPORT-9165-V780: E2E-018/021 success, refusal, tamper and cold replay."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]

    def call(*args):
        print("[test9165] child_before", args, flush=True)
        result = subprocess.run(
            [*argv, *args], cwd=tmp_path, env=env, capture_output=True, timeout=180
        )
        print("[test9165] child_after", result.returncode, flush=True)
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
    changed["two_bound_rows"][0]["absolute_metric"] = 1
    changed["reproducibility_checksum"] = m.checksum(changed)
    atomic_json(output, changed)
    assert call("--cold-replay", str(output)).returncode == 1
