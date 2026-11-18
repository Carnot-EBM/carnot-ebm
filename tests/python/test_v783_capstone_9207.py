"""REQ-REPORT-9207-V783: administrative closure cannot invent scientific evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v783_capstone as m
from carnot.reporting import v783_capstone_evidence as e
from carnot.reporting.historical_contract_inputs import original_root
from carnot.reporting.current_work_receipt import atomic_json, validate_current_work_receipt


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Authenticate real retained producers once without loading a model."""
    return m.collect(
        tmp_path_factory.mktemp("capstone9207"),
        "20261117",
        original_root(tmp_path_factory.mktemp("inputs9207"), 783, suite="capstone"),
    )


def test_fourteen_terminal_outcomes(work, monkeypatch):
    """SCENARIO-REPORT-9207-V783: terminal skips remain visible beside failed science."""
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    value = m.build(work)
    assert all(r["passed"] for r in work["checks"]), [r for r in work["checks"] if not r["passed"]]
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"] == "complete_blocked_declared_gate"
    assert value["full_task_objects"] == work["tasks"]
    assert value["full_task_digest"] == e.TASK_PIN
    slots = {r["experiment_id"]: r for r in value["outcome_rows"]}
    assert list(slots) == list(range(9194, 9208))
    assert slots[9194]["verdict_class"] == "disqualified"
    assert slots[9195]["scheduler_alternative"]["sha256"]
    assert slots[9197]["scheduler_alternative"]["sha256"]
    assert slots[9195]["gate_operands"][0]["observed"] == 0
    assert slots[9196]["gate_operands"][0]["observed"] == "missing_field"
    assert slots[9198]["gate_operands"][0]["observed"] == "missing_file"
    assert slots[9196]["scheduler_events"]
    assert all(slots[i]["primary_status"] == "missing_file" for i in range(9195, 9207))
    assert value["previous_milestone_accounting"] == dict(
        executed=4, skipped=10, scientific_success_inferred=False
    )
    assert value["current_milestone_accounting"]["executed"] == 2
    assert value["current_milestone_accounting"]["skipped"] == 12
    assert value["validation_obligations"]["union_count"] == 1040
    assert value["validation_obligations"]["current_qualification_credit"] == 0
    assert value["collection_partition"]["answer_count"] == 900
    assert value["collection_partition"]["qid_counts"] == [100] * 3
    assert len(value["two_bound_rows"]) == 2 and len(value["seven_bound_rows"]) == 7
    assert len(value["ces_summary"]["ranking_bound_rows"]) == 3
    for r in (
        value["two_bound_rows"]
        + value["seven_bound_rows"]
        + value["ces_summary"]["ranking_bound_rows"]
    ):
        assert r["absolute_metric"] is None and r["denominator"] == 0
    assert not value["ces_summary"]["primary_gate_rescue"]
    assert set(value["label_ambiguity_summary"]["branches"]) == {
        "observed_disagreement",
        "missing_A2_envelope",
    }
    assert value["label_ambiguity_summary"]["method_status"] == "blocked"
    assert value["corpus_scan_summary"]["qualified_checker_correction"] is False
    assert value["arc_summary"]["verdict_class"] == "disqualified"
    assert not value["independent_human_confirmation"]["warranted_now"]
    assert value["MODEL_SPECS"] == [] and value["random_seed"] == 783207
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert value["claim_scope"]["independent_generalization_score"] == 0
    assert not value["publication_gates"]["current_science_paper_ready"]
    assert all(r["next_reopen_condition"] for r in value["continuation_decisions"])
    assert (
        validate_current_work_receipt(
            json.loads(Path(work["current_work_reference"]["path"]).read_bytes())
        )
        == []
    )
    retired = [r for r in value["retirement_recommendations"] if r["condition_met"]]
    assert any(r["task_id"] == "exp9198-logit-runtime" for r in retired)
    assert not any(
        r["task_id"]
        in {
            "exp9195-corpus-prefilter",
            "exp9196-forward-qualification",
            "exp9197-label-ambiguity",
            "exp9207-capstone",
        }
        for r in retired
    )
    assert all(not r["hypothesis_retired"] and not r["manifest_mutated"] for r in retired)
    assert all(r["prior_reference"]["sha256"] for r in retired)


def test_missing_and_owned_failure(work, tmp_path):
    """REQ-REPORT-9207-V783: external absence blocks; affected failures disqualify."""
    assert m.build(work)["verdict_class"] == "disqualified"
    blocked = m.collect(tmp_path / "raw", "20261117", tmp_path / "absent")
    value = m.build(blocked)
    assert value["verdict_class"] == "blocked"
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert len(value["outcome_rows"]) == 14
    assert any(r["observed"] == "missing_file" for r in value["gate_check_summary"])
    changed = deepcopy(work)
    changed["tasks"][0]["prompt"] += " changed"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(changed)


def test_exact_operands(tmp_path):
    """REQ-REPORT-9207-V783: boolean, zero and the two missing states differ."""
    gate = dict(upstream=e.TASKS[0], artifact_field="ready", op="==", value=1)
    assert e.edge(gate, {})["observed"] == "missing_file"
    entry = dict(value={}, reference={})
    assert e.edge(gate, {e.TASKS[0]: entry})["observed"] == "missing_field"
    entry["value"]["ready"] = 0
    assert e.edge(gate, {e.TASKS[0]: entry})["observed"] == 0
    assert not e.check(tmp_path, "ready", 1, True, "strict")["passed"]


def test_replay_controls(work, tmp_path):
    """SCENARIO-REPORT-9207-V783-CONTROLS: rehashing cannot bless invented outcomes."""
    path = tmp_path / "candidate.json"
    value = m.build(work)
    atomic_json(path, value)
    report = m.replay(path)
    assert report["passed"], report
    retained = Path(work["snapshots"][e.PROTOCOL["primary_paths"][0]]["path"])
    original = retained.read_bytes()
    try:
        retained.write_bytes(original + b" ")
        assert not m.replay(path)["passed"]
    finally:
        retained.write_bytes(original)
    for key in ["outcome_rows", "label_ambiguity_summary", "current_llm_invocations"]:
        changed = dict(value, **{key: 1})
        changed["reproducibility_checksum"] = m.checksum(changed)
        atomic_json(path, changed)
        assert not m.replay(path)["passed"]
    changed = deepcopy(work)
    changed["values"][e.TASKS[0]]["value"]["source_ready_score"] = 1
    atomic_json(tmp_path / "work.json", changed)
    value["work_reference"] = m.reference(tmp_path / "work.json")
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(path, value)
    assert not m.replay(path)["passed"]


def test_private_real_cli(work, tmp_path):
    """REQ-REPORT-9207-V783: E2E-018/021 run outside checkout without PYTHONPATH."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]

    def call(*args):
        print("[test9207] child_before", args, flush=True)
        result = subprocess.run(
            [*argv, *args], cwd=tmp_path, env=env, capture_output=True, timeout=180
        )
        print("[test9207] child_after", result.returncode, flush=True)
        return result

    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / (m.NAME + ".json")
    result = call("--from-work", str(source), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert call("--cold-replay", str(output)).returncode == 0
    assert call("--date", "20261116").returncode == 2
    assert call("--from-work", str(source)).returncode == 2
    changed = json.loads(output.read_bytes())
    changed["two_bound_rows"][0]["absolute_metric"] = 1
    changed["reproducibility_checksum"] = m.checksum(changed)
    atomic_json(output, changed)
    assert call("--cold-replay", str(output)).returncode == 1


def test_orchestration(work, tmp_path, monkeypatch):
    """REQ-REPORT-9207-V783: orchestration retains validation and shared deadlines."""
    monkeypatch.setattr(m, "collect", lambda raw, *_: dict(deepcopy(work), raw=str(raw)))
    monkeypatch.setattr(m.legacy, "validate", lambda _: dict(receipts=[], coverage={}))
    monkeypatch.setattr(m.runner, "terminal", lambda *_: dict(passed=True, receipts=[]))
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    output = tmp_path / (m.NAME + ".json")
    assert m.main(["--output", str(output)]) == 0
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: False)
    assert m.main(["--output", str(output)]) == 1
    assert m.main(["--from-work", str(tmp_path / "missing"), "--output", str(output)]) == 1
    monkeypatch.setenv("CARNOT_V783_CAPSTONE_DEADLINE", "0")
    monkeypatch.setattr(m, "_main", lambda _: m.runner.child("expired", [], tmp_path))
    with pytest.raises(ValueError, match="shared_deadline"):
        m.main([])
