"""REQ-REPORT-9221-V784: administrative closure cannot invent scientific evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v784_capstone as m
from carnot.reporting import v784_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, validate_current_work_receipt


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Authenticate real retained producers once without loading a model."""
    root = e.ROOT
    published = e.ROOT / "results" / (m.NAME + ".json")
    if (
        published.is_file()
        and m.reference(e.ROOT / e.PROTOCOL["active"])["sha256"] != e.PINS[e.PROTOCOL["active"]]
    ):
        saved = json.loads(
            Path(json.loads(published.read_bytes())["work_reference"]["path"]).read_bytes()
        )
        root = tmp_path_factory.mktemp("inputs9221")
        for name, ref in saved["snapshots"].items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(ref["path"]).read_bytes())
    return m.collect(tmp_path_factory.mktemp("capstone9221"), "20261118", root)


def test_fourteen_terminal_outcomes(work, monkeypatch):
    """SCENARIO-REPORT-9221-V784: terminal skips and qualified label methods stay distinct."""
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
    assert list(slots) == list(range(9208, 9222))
    assert slots[9208]["verdict_class"] == "disqualified"
    assert slots[9209]["scheduler_alternative"]["sha256"]
    assert slots[9209]["gate_operands"][0]["observed"] == 0
    assert slots[9209]["gate_operands"][1]["observed"] == 0
    assert slots[9210]["gate_operands"][0]["observed"] == "missing_field"
    assert slots[9210]["scheduler_events"]
    assert slots[9211]["primary_status"] == "present"
    assert slots[9211]["verdict_class"] == "null" and slots[9211]["qualified"] is True
    assert slots[9212]["gate_operands"][0]["observed"] == "missing_file"
    assert slots[9212]["gate_operands"][1]["observed"] == 0
    assert all(
        slots[i]["primary_status"] == "missing_file" for i in [9209, 9210, *range(9212, 9221)]
    )
    assert value["preserved_v782_accounting"] == dict(
        executed=4, skipped=10, scientific_success_inferred=False
    )
    assert value["previous_milestone_accounting"] == dict(
        executed=2,
        skipped=12,
        scheduler_artifact_skips=2,
        scheduler_log_skips=10,
        scientific_success_inferred=False,
    )
    assert value["current_milestone_accounting"] == dict(
        executed=3,
        skipped=11,
        scheduler_artifact_skips=1,
        scheduler_log_skips=10,
        scientific_success_inferred=False,
    )
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
    summary = value["label_ambiguity_summary"]
    assert summary["method_status"] == "qualified"
    assert (
        summary["label_ambiguity_ready_score"] == summary["target_identifiability_ready_score"] == 1
    )
    assert summary["target_identifiability_summary"]["identifiable_factual_error_rows"] == 900
    assert summary["target_identifiability_summary"]["identifiable_si_cf_rows"] == 0
    assert set(summary["branches"]) == {"observed_disagreement", "missing_A2_envelope"}
    assert all(
        b["status"] == "blocked" and b["denominator"] == 0 for b in summary["branches"].values()
    )
    assert value["replay_authority_summary"]["replay_authority_ready_score"] == 0
    assert value["replay_authority_summary"]["preserved_v783_readiness_scores"] == [0] * 4
    assert value["corpus_scan_summary"]["qualified_checker_correction"] is False
    assert value["arc_summary"]["verdict_class"] == "disqualified"
    assert not value["arc_summary"]["current_ARC_experiment_scheduled"]
    assert not value["independent_human_confirmation"]["warranted_now"]
    assert value["MODEL_SPECS"] == [] and value["random_seed"] == 784221
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
    assert any(r["task_id"] == "exp9212-logit-runtime" for r in retired)
    assert not any(
        r["task_id"]
        in {"exp9210-forward-qualification", "exp9211-label-ambiguity", "exp9221-capstone"}
        for r in retired
    )
    assert all(not r["hypothesis_retired"] and not r["manifest_mutated"] for r in retired)
    assert all(r["prior_reference"]["sha256"] for r in retired)


def test_missing_and_owned_failure(work, tmp_path):
    """REQ-REPORT-9221-V784: external absence blocks; affected failures disqualify."""
    assert m.build(work)["verdict_class"] == "disqualified"
    blocked = m.collect(tmp_path / "raw", "20261118", tmp_path / "absent")
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
    """REQ-REPORT-9221-V784: boolean, zero and the two missing states differ."""
    gate = dict(upstream=e.TASKS[0], artifact_field="ready", op="==", value=1)
    assert e.edge(gate, {})["observed"] == "missing_file"
    entry = dict(value={}, reference={})
    assert e.edge(gate, {e.TASKS[0]: entry})["observed"] == "missing_field"
    entry["value"]["ready"] = 0
    assert e.edge(gate, {e.TASKS[0]: entry})["observed"] == 0
    assert not e.check(tmp_path, "ready", 1, True, "strict")["passed"]


def test_replay_controls(work, tmp_path):
    """SCENARIO-REPORT-9221-V784-CONTROLS: rehashing cannot bless invented outcomes."""
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
    """REQ-REPORT-9221-V784: E2E-018/021 run outside checkout without PYTHONPATH."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]

    def call(*args):
        print("[test9221] child_before", args, flush=True)
        result = subprocess.run(
            [*argv, *args], cwd=tmp_path, env=env, capture_output=True, timeout=180
        )
        print("[test9221] child_after", result.returncode, flush=True)
        return result

    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / (m.NAME + ".json")
    result = call("--from-work", str(source), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert call("--cold-replay", str(output)).returncode == 0
    assert call("--date", "20261117").returncode == 2
    assert call("--from-work", str(source)).returncode == 2
    changed = json.loads(output.read_bytes())
    changed["two_bound_rows"][0]["absolute_metric"] = 1
    changed["reproducibility_checksum"] = m.checksum(changed)
    atomic_json(output, changed)
    assert call("--cold-replay", str(output)).returncode == 1


def test_orchestration(work, tmp_path, monkeypatch):
    """REQ-REPORT-9221-V784: orchestration retains validation and shared deadlines."""
    monkeypatch.setattr(m, "collect", lambda raw, *_: dict(deepcopy(work), raw=str(raw)))
    monkeypatch.setattr(m.legacy, "validate", lambda _: dict(receipts=[], coverage={}))
    monkeypatch.setattr(m.runner, "terminal", lambda *_: dict(passed=True, receipts=[]))
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    output = tmp_path / (m.NAME + ".json")
    assert m.main(["--date", "20261118", "--output", str(output)]) == 0
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: False)
    assert m.main(["--output", str(output)]) == 1
    assert m.main(["--from-work", str(tmp_path / "missing"), "--output", str(output)]) == 1
    monkeypatch.setenv("CARNOT_V784_CAPSTONE_DEADLINE", "0")
    monkeypatch.setattr(m, "_main", lambda _: m.runner.child("expired", [], tmp_path))
    with pytest.raises(ValueError, match="shared_deadline"):
        m.main([])
