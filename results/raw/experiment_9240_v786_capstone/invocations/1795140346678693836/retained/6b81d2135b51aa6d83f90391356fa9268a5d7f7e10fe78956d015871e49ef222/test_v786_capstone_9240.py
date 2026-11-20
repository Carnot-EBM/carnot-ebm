"""REQ-REPORT-9240-V786: administrative closure cannot invent scientific evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v786_capstone as m
from carnot.reporting import v786_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, validate_current_work_receipt


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Authenticate real retained producers once without loading a model."""
    return m.collect(
        tmp_path_factory.mktemp("capstone9240"),
        "20261120",
    )


def test_fourteen_terminal_outcomes(work, monkeypatch):
    """SCENARIO-REPORT-9240-V786: fourteen V786 outcomes and preserved V785 history stay distinct."""
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
    assert list(slots) == list(range(9227, 9241))
    assert slots[9227]["primary_status"] == "missing_file"
    assert slots[9227]["disposition"] == "authenticated_scheduler_pretest_skip"
    assert slots[9227]["scheduler_honest_verdict"] == "blocked_pretest_self_heal_failed"
    assert len(slots[9227]["scheduler_events"]) == 3
    assert slots[9228]["gate_operands"][0]["observed"] == "missing_file"
    assert slots[9228]["gate_operands"][1]["observed"] == "missing_file"
    assert slots[9229]["gate_operands"][0]["observed"] == "missing_file"
    assert slots[9230]["primary_status"] == "missing_file"
    assert slots[9230]["disposition"] == "authenticated_scheduler_pretest_skip"
    assert slots[9230]["scheduler_honest_verdict"] == "blocked_pretest_self_heal_failed"
    assert len(slots[9230]["scheduler_events"]) == 3
    assert all(slots[i]["primary_status"] == "missing_file" for i in range(9227, 9240))
    assert all(
        slots[i]["scheduler_honest_verdict"] == "blocked_upstream_retired"
        for i in [9228, 9229, *range(9231, 9240)]
    )
    assert slots[9240]["primary_status"] == "self_slot"
    assert value["preserved_v782_accounting"] == dict(
        executed=4, skipped=10, scientific_success_inferred=False
    )
    assert value["preserved_v783_accounting"] == dict(
        executed=2,
        skipped=12,
        scheduler_artifact_skips=2,
        scheduler_log_skips=10,
        scientific_success_inferred=False,
    )
    assert value["preserved_v784_accounting"] == dict(
        executed=3,
        skipped=11,
        scheduler_artifact_skips=1,
        scheduler_log_skips=10,
        scientific_success_inferred=False,
    )
    expected_v785 = dict(
        executed=2,
        skipped=3,
        scheduler_artifact_skips=1,
        scheduler_log_skips=2,
        omitted_slots=9,
        scientific_success_inferred=False,
    )
    assert value["preserved_v785_accounting"] == expected_v785
    assert value["previous_milestone_accounting"] == expected_v785
    assert value["current_milestone_accounting"] == dict(
        executed=1,
        skipped=13,
        scheduler_artifact_skips=0,
        scheduler_log_skips=13,
        scheduler_pretest_log_skips=2,
        scheduler_upstream_retired_log_skips=11,
        scientific_success_inferred=False,
    )
    v785_rows = value["preserved_v785_outcomes"]
    assert len(v785_rows) == 14
    assert v785_rows[0]["experiment_id"] == 9222 and v785_rows[0]["primary_status"] == "present"
    assert v785_rows[0]["readiness"]["replay_authority_ready_score"] == 1
    assert v785_rows[0]["readiness"]["contract_ready_score"] == 0
    assert v785_rows[1]["experiment_id"] == 9223 and v785_rows[1]["scheduler_alternative"]["sha256"]
    assert (
        v785_rows[2]["experiment_id"] == 9224 and v785_rows[2]["primary_status"] == "missing_file"
    )
    assert v785_rows[3]["experiment_id"] == 9225 and v785_rows[3]["primary_status"] == "present"
    assert v785_rows[3]["readiness"]["selective_deferral_ready_score"] == 1
    assert v785_rows[3]["readiness"]["spline_support_diagnostic_ready_score"] == 1
    assert (
        v785_rows[4]["experiment_id"] == 9226 and v785_rows[4]["primary_status"] == "missing_file"
    )
    assert all(r["primary_status"] == "missing_field" for r in v785_rows[5:])
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
    ambiguity = value["label_ambiguity_summary"]
    assert ambiguity["method_status"] == "qualified"
    assert (
        ambiguity["label_ambiguity_ready_score"]
        == ambiguity["target_identifiability_ready_score"]
        == 1
    )
    assert all(
        b["status"] == "blocked" and b["denominator"] == 0 for b in ambiguity["branches"].values()
    )
    deferral = value["selective_deferral_summary"]
    assert deferral["method_status"] == "qualified"
    assert deferral["selective_deferral_ready_score"] == 1
    assert all(
        b["status"] == "blocked" and b["denominator"] == 0 for b in deferral["branches"].values()
    )
    spline = value["spline_support_summary"]
    assert spline["method_status"] == "qualified"
    assert spline["spline_support_diagnostic_ready_score"] == 1
    assert all(
        b["status"] == "blocked" and b["denominator"] == 0 for b in spline["branches"].values()
    )
    conformal = value["conformal_risk_summary"]
    assert conformal["method_status"] == "blocked"
    assert conformal["conformal_risk_ready_score"] == "missing_file"
    assert conformal["mycelium_provenance_ready_score"] == "missing_file"
    assert all(
        b["status"] == "blocked" and b["denominator"] == 0 for b in conformal["branches"].values()
    )
    assert value["replay_authority_summary"]["replay_authority_ready_score"] == "missing_file"
    assert value["replay_authority_summary"]["contract_ready_score"] == "missing_file"
    assert value["replay_authority_summary"]["v785_replay_authority_ready_score"] == 1
    assert value["replay_authority_summary"]["v785_contract_ready_score"] == 0
    assert value["corpus_scan_summary"]["qualified_checker_correction"] is False
    assert value["arc_summary"]["verdict_class"] == "disqualified"
    assert not value["arc_summary"]["current_ARC_experiment_scheduled"]
    assert not value["independent_human_confirmation"]["warranted_now"]
    assert value["MODEL_SPECS"] == [] and value["random_seed"] == 786240
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
    assert any(
        r["task_id"] == "exp9229-forward-qualification"
        and r["prior_task_id"] == "exp9224-forward-qualification"
        for r in retired
    )
    assert any(
        r["task_id"] == "exp9231-logit-runtime" and r["prior_task_id"] == "exp9226-logit-runtime"
        for r in retired
    )
    assert not any(
        r["task_id"]
        in {
            "exp9227-contract",
            "exp9228-corpus-prefilter",
            "exp9230-conformal-risk-protocol",
            "exp9240-capstone",
        }
        for r in retired
    )
    assert all(not r["hypothesis_retired"] and not r["manifest_mutated"] for r in retired)
    assert all(r["prior_reference"]["sha256"] for r in retired)


def test_missing_and_owned_failure(work, tmp_path):
    """REQ-REPORT-9240-V786: external absence blocks; affected failures disqualify."""
    assert m.build(work)["verdict_class"] == "disqualified"
    blocked = m.collect(tmp_path / "raw", "20261120", tmp_path / "absent")
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
    """REQ-REPORT-9240-V786: boolean, zero and the two missing states differ."""
    gate = dict(upstream=e.TASKS[0], artifact_field="ready", op="==", value=1)
    assert e.edge(gate, {})["observed"] == "missing_file"
    entry = dict(value={}, reference={})
    assert e.edge(gate, {e.TASKS[0]: entry})["observed"] == "missing_field"
    entry["value"]["ready"] = 0
    assert e.edge(gate, {e.TASKS[0]: entry})["observed"] == 0
    assert not e.check(tmp_path, "ready", 1, True, "strict")["passed"]


def test_replay_controls(work, tmp_path):
    """SCENARIO-REPORT-9240-V786-CONTROLS: rehashing cannot bless invented outcomes."""
    path = tmp_path / "candidate.json"
    value = m.build(work)
    atomic_json(path, value)
    report = m.replay(path)
    assert report["passed"], report
    retained = Path(work["snapshots"]["results/experiment_9222_v785_contract.json"]["path"])
    original = retained.read_bytes()
    try:
        retained.write_bytes(original + b" ")
        assert not m.replay(path)["passed"]
    finally:
        retained.write_bytes(original)
    for key in ["outcome_rows", "selective_deferral_summary", "current_llm_invocations"]:
        changed = dict(value, **{key: 1})
        changed["reproducibility_checksum"] = m.checksum(changed)
        atomic_json(path, changed)
        assert not m.replay(path)["passed"]
    changed = deepcopy(work)
    changed["values"]["prior:9222"]["value"]["contract_ready_score"] = 1
    atomic_json(tmp_path / "work.json", changed)
    value["work_reference"] = m.reference(tmp_path / "work.json")
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(path, value)
    assert not m.replay(path)["passed"]


def test_private_real_cli(work, tmp_path):
    """REQ-REPORT-9240-V786: E2E-018/021 run outside checkout without PYTHONPATH."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]

    def call(*args):
        print("[test9240] child_before", args, flush=True)
        result = subprocess.run(
            [*argv, *args], cwd=tmp_path, env=env, capture_output=True, timeout=180
        )
        print("[test9240] child_after", result.returncode, flush=True)
        return result

    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / (m.NAME + ".json")
    result = call("--from-work", str(source), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert call("--cold-replay", str(output)).returncode == 0
    assert call("--date", "20261119").returncode == 2
    assert call("--from-work", str(source)).returncode == 2
    changed = json.loads(output.read_bytes())
    changed["two_bound_rows"][0]["absolute_metric"] = 1
    changed["reproducibility_checksum"] = m.checksum(changed)
    atomic_json(output, changed)
    assert call("--cold-replay", str(output)).returncode == 1


def test_orchestration(work, tmp_path, monkeypatch):
    """REQ-REPORT-9240-V786: orchestration retains validation and shared deadlines."""
    monkeypatch.setattr(m, "collect", lambda raw, *_: dict(deepcopy(work), raw=str(raw)))
    monkeypatch.setattr(m.legacy, "validate", lambda _: dict(receipts=[], coverage={}))
    monkeypatch.setattr(m.runner, "terminal", lambda *_: dict(passed=True, receipts=[]))
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    output = tmp_path / (m.NAME + ".json")
    assert m.main(["--date", "20261120", "--output", str(output)]) == 0
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: False)
    assert m.main(["--output", str(output)]) == 1
    assert m.main(["--from-work", str(tmp_path / "missing"), "--output", str(output)]) == 1
    monkeypatch.setenv("CARNOT_V786_CAPSTONE_DEADLINE", "0")
    monkeypatch.setattr(m, "_main", lambda _: m.runner.child("expired", [], tmp_path))
    with pytest.raises(ValueError, match="shared_deadline"):
        m.main([])
