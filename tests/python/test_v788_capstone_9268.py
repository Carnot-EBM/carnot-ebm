"""REQ-REPORT-9268-V788: administrative closure cannot invent scientific evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from carnot.reporting import v788_capstone as m
from carnot.reporting import v788_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, validate_current_work_receipt
from carnot.reporting.historical_contract_inputs import original_root


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Authenticate real retained producers once without loading a model."""
    return m.collect(
        tmp_path_factory.mktemp("capstone9268"),
        "20261120",
        original_root(tmp_path_factory.mktemp("inputs9268"), 788, suite="capstone"),
    )


def test_fourteen_terminal_outcomes(work, monkeypatch):
    """SCENARIO-REPORT-9268-V788: fourteen V788 outcomes and preserved V787 history stay distinct."""
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
    assert list(slots) == list(range(9255, 9269))
    assert slots[9255]["primary_status"] == "present"
    assert slots[9255]["verdict_class"] == "null" and slots[9255]["qualified"] is True
    assert slots[9255]["readiness"]["contract_ready_score"] == 1
    assert slots[9255]["readiness"]["replay_authority_ready_score"] == 1
    assert slots[9255]["readiness"]["source_ready_score"] == 1
    assert slots[9255]["readiness"]["static_methods_ready_score"] == 1
    assert slots[9255]["readiness"]["online_methods_ready_score"] == 1
    assert slots[9256]["primary_status"] == "present"
    assert slots[9256]["verdict_class"] == "disqualified" and slots[9256]["qualified"] is False
    assert slots[9256]["readiness"]["corpus_scan_ready_score"] == 0
    assert slots[9256]["gate_operands"][0]["observed"] == 1
    assert slots[9256]["gate_operands"][1]["observed"] == 1
    assert slots[9257]["primary_status"] == "missing_file"
    assert slots[9257]["scheduler_alternative"]["sha256"]
    assert slots[9257]["disposition"] == "authenticated_scheduler_skip"
    assert slots[9257]["scheduler_honest_verdict"] == "blocked_gate_check_failed"
    assert slots[9257]["gate_operands"][0]["observed"] == 0
    assert slots[9258]["primary_status"] == "present"
    assert slots[9258]["verdict_class"] == "blocked" and slots[9258]["qualified"] is True
    assert slots[9258]["honest_verdict"] == "complete_blocked_arena_sim_wheel_missing"
    assert slots[9258]["readiness"]["victris_harness_ready_score"] == 1
    assert slots[9259]["gate_operands"][0]["observed"] == "missing_field"
    assert slots[9259]["gate_operands"][1]["observed"] == 1
    assert all(slots[i]["primary_status"] == "missing_file" for i in [9257, *range(9259, 9268)])
    assert all(
        slots[i]["scheduler_honest_verdict"] == "blocked_upstream_retired"
        for i in range(9259, 9268)
    )
    assert slots[9268]["primary_status"] == "self_slot"
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
    assert value["preserved_v785_accounting"] == dict(
        executed=2,
        skipped=3,
        scheduler_artifact_skips=1,
        scheduler_log_skips=2,
        omitted_slots=9,
        scientific_success_inferred=False,
    )
    assert value["preserved_v786_accounting"] == dict(
        executed=1,
        skipped=13,
        scheduler_artifact_skips=0,
        scheduler_log_skips=13,
        scheduler_pretest_log_skips=2,
        scheduler_upstream_retired_log_skips=11,
        scientific_success_inferred=False,
    )
    expected_v787 = dict(
        executed=2,
        skipped=5,
        scheduler_artifact_skips=1,
        scheduler_log_skips=4,
        omitted_slots=7,
        scientific_success_inferred=False,
    )
    assert value["preserved_v787_accounting"] == expected_v787
    assert value["previous_milestone_accounting"] == expected_v787
    assert value["current_milestone_accounting"] == dict(
        executed=4,
        skipped=10,
        scheduler_artifact_skips=1,
        scheduler_log_skips=9,
        scientific_success_inferred=False,
    )
    v787_rows = value["preserved_v787_outcomes"]
    assert len(v787_rows) == 14
    assert v787_rows[0]["experiment_id"] == 9241 and v787_rows[0]["primary_status"] == "present"
    assert v787_rows[0]["readiness"]["contract_ready_score"] == 0
    assert v787_rows[0]["readiness"]["replay_authority_ready_score"] == 0
    assert v787_rows[1]["experiment_id"] == 9242 and v787_rows[1]["scheduler_alternative"]["sha256"]
    assert (
        v787_rows[2]["experiment_id"] == 9243 and v787_rows[2]["primary_status"] == "missing_file"
    )
    assert v787_rows[3]["experiment_id"] == 9244 and v787_rows[3]["primary_status"] == "present"
    assert v787_rows[3]["readiness"]["conformal_risk_ready_score"] == 1
    assert v787_rows[3]["readiness"]["ca_ocrc_ready_score"] == 1
    assert v787_rows[3]["readiness"]["provenance_audit_ready_score"] == 1
    assert all(v787_rows[i]["primary_status"] == "missing_file" for i in range(4, 7))
    assert all(r["primary_status"] == "missing_field" for r in v787_rows[7:])
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
    assert conformal["method_status"] == "qualified"
    assert conformal["conformal_risk_ready_score"] == 1
    assert conformal["ca_ocrc_ready_score"] == 1
    assert conformal["provenance_audit_ready_score"] == 1
    assert conformal["mycelium_provenance_ready_score"] == 1
    assert all(
        b["status"] == "blocked" and b["denominator"] == 0 for b in conformal["branches"].values()
    )
    ca_ocrc = value["ca_ocrc_summary"]
    assert ca_ocrc["method_status"] == "qualified" and ca_ocrc["ca_ocrc_ready_score"] == 1
    assert all(
        b["status"] == "blocked" and b["denominator"] == 0 for b in ca_ocrc["branches"].values()
    )
    victris = value["victris_arena_summary"]
    assert victris["status"] == "blocked" and victris["harness_status"] == "qualified"
    assert victris["victris_harness_ready_score"] == 1
    assert victris["arena_sim_installed"] is False and victris["heldout_matches_played"] == 0
    assert len(victris["archive_versions"]) == 6
    assert value["replay_authority_summary"]["replay_authority_ready_score"] == 1
    assert value["replay_authority_summary"]["contract_ready_score"] == 1
    assert value["replay_authority_summary"]["v787_replay_authority_ready_score"] == 0
    assert value["replay_authority_summary"]["v787_contract_ready_score"] == 0
    assert value["replay_authority_summary"]["v785_replay_authority_ready_score"] == 1
    assert value["replay_authority_summary"]["v785_contract_ready_score"] == 0
    assert value["corpus_scan_summary"]["corpus_scan_ready_score"] == 0
    assert value["corpus_scan_summary"]["qualified_checker_correction"] is False
    assert value["corpus_scan_summary"]["full_corpus_traversal_complete"] is True
    assert (
        value["forward_qualification_summary"]["forward_validation_ready_score"] == "missing_field"
    )
    assert value["forward_qualification_summary"]["failed_upstream_gate_observed"] == 0
    assert value["arc_summary"]["verdict_class"] == "disqualified"
    assert not value["arc_summary"]["current_ARC_experiment_scheduled"]
    assert not value["independent_human_confirmation"]["warranted_now"]
    assert value["MODEL_SPECS"] == [] and value["random_seed"] == 788268
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
        r["task_id"] == "exp9257-forward-qualification"
        and r["prior_task_id"] == "exp9182-forward-qualification"
        for r in retired
    )
    assert any(
        r["task_id"] == "exp9259-logit-runtime" and r["prior_task_id"] == "exp9245-logit-runtime"
        for r in retired
    )
    assert not any(
        r["task_id"]
        in {
            "exp9255-contract",
            "exp9256-corpus-prefilter",
            "exp9258-victris-energy-bot",
            "exp9268-capstone",
        }
        for r in retired
    )
    assert all(not r["hypothesis_retired"] and not r["manifest_mutated"] for r in retired)
    assert all(r["prior_reference"]["sha256"] for r in retired)


def test_missing_and_owned_failure(work, tmp_path):
    """REQ-REPORT-9268-V788: external absence blocks; affected failures disqualify."""
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
    """REQ-REPORT-9268-V788: boolean, zero and the two missing states differ."""
    gate = dict(upstream=e.TASKS[0], artifact_field="ready", op="==", value=1)
    assert e.edge(gate, {})["observed"] == "missing_file"
    entry = dict(value={}, reference={})
    assert e.edge(gate, {e.TASKS[0]: entry})["observed"] == "missing_field"
    entry["value"]["ready"] = 0
    assert e.edge(gate, {e.TASKS[0]: entry})["observed"] == 0
    assert not e.check(tmp_path, "ready", 1, True, "strict")["passed"]


def test_import_projection(tmp_path):
    """SCENARIO-REPORT-9268-V788-CONTROLS: full bytes retain authority without duplicate metadata."""
    value = dict(
        experiment_id=9222,
        contract_ready_score=0,
        replay_authority_ready_score=1,
        consumer_shard_rows=[dict(passed=True)],
        validation_obligations=dict(union_count=1040),
        gate_check_summary=[dict(passed=True), dict(passed=False, observed=0)],
        work_reference=dict(path="producer", sha256="original"),
        corpus_manifest_reference=dict(total_files=10, ordered_paths=[dict(path="a.json")]),
        validation_scope_manifest=dict(unused="duplicate"),
        historical_model_provenance=[dict(unused="not current work")],
    )
    projected = e.project(value)
    assert "validation_scope_manifest" not in projected
    assert "historical_model_provenance" not in projected
    assert "ordered_paths" not in projected["corpus_manifest_reference"]
    assert projected["corpus_manifest_reference"]["total_files"] == 10
    assert projected["gate_check_summary"] == [dict(passed=False, observed=0)]
    assert projected["validation_obligations"] == value["validation_obligations"]
    assert projected["consumer_shard_rows"] == value["consumer_shard_rows"]
    assert projected["contract_ready_score"] == 0
    assert projected["replay_authority_ready_score"] == 1
    assert projected["work_reference"] == value["work_reference"]
    assert len(value["gate_check_summary"]) == 2
    legacy = dict(gate_check_summary=dict(failed_observed=0))
    assert e.project(legacy)["gate_check_summary"] == legacy["gate_check_summary"]
    legacy = dict(gate_check_summary=["missing_inputs"])
    assert e.project(legacy)["gate_check_summary"] == legacy["gate_check_summary"]
    older = dict(value, experiment_id=9221)
    assert "validation_obligations" not in e.project(older)
    older["full_task_objects"] = [dict(id="historical")]
    assert "full_task_objects" not in e.project(older)
    sample = tmp_path / "sample.json"
    atomic_json(sample, value)
    assert e.load_projected(sample) == projected
    assert e.load_projected(sample) == projected


def test_replay_controls(work, tmp_path):
    """SCENARIO-REPORT-9268-V788-CONTROLS: rehashing cannot bless invented outcomes."""
    path = tmp_path / "candidate.json"
    value = m.build(work)
    atomic_json(path, value)
    report = m.replay(path)
    assert report["passed"], report
    retained = Path(work["snapshots"]["results/experiment_9242_corpus_prefilter.json"]["path"])
    original = retained.read_bytes()
    try:
        retained.write_bytes(original + b" ")
        assert not m.replay(path)["passed"]
    finally:
        retained.write_bytes(original)
    for key in ["outcome_rows", "conformal_risk_summary", "current_llm_invocations"]:
        changed = dict(value, **{key: 1})
        changed["reproducibility_checksum"] = m.checksum(changed)
        atomic_json(path, changed)
        assert not m.replay(path)["passed"]
    changed = deepcopy(work)
    changed["values"]["prior:9241"]["value"]["contract_ready_score"] = 1
    atomic_json(tmp_path / "work.json", changed)
    value["work_reference"] = m.reference(tmp_path / "work.json")
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(path, value)
    assert not m.replay(path)["passed"]


def test_private_real_cli(work, tmp_path):
    """REQ-REPORT-9268-V788: E2E-018/021 run outside checkout without PYTHONPATH."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]

    def call(*args):
        print("[test9268] child_before", args, flush=True)
        result = subprocess.run(
            [*argv, *args], cwd=tmp_path, env=env, capture_output=True, timeout=180
        )
        print("[test9268] child_after", result.returncode, flush=True)
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


def test_orchestration(tmp_path, monkeypatch):
    """REQ-REPORT-9268-V788: orchestration retains validation and shared deadlines."""
    work = dict(started_ns=time.monotonic_ns(), checks=[], validation_scope_manifest={})
    monkeypatch.setattr(m, "collect", lambda raw, *_: dict(work, raw=str(raw)))
    monkeypatch.setattr(
        m,
        "build",
        lambda current: dict(
            experiment_id=9268,
            task_id=m.TASK,
            honest_verdict="complete_blocked_orchestration_fixture",
            verdict_class="blocked",
            required_checks_passed=m.runner.receipts_pass(current),
        ),
    )
    monkeypatch.setattr(m.legacy, "validate", lambda _: dict(receipts=[], coverage={}))
    monkeypatch.setattr(m.runner, "terminal", lambda *_: dict(passed=True, receipts=[]))
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    output = tmp_path / (m.NAME + ".json")
    assert m.main(["--date", "20261120", "--output", str(output)]) == 0
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: False)
    assert m.main(["--output", str(output)]) == 1
    assert m.main(["--from-work", str(tmp_path / "missing"), "--output", str(output)]) == 1
    monkeypatch.setenv("CARNOT_V788_CAPSTONE_DEADLINE", "0")
    monkeypatch.setattr(m, "_main", lambda _: m.runner.child("expired", [], tmp_path))
    with pytest.raises(ValueError, match="shared_deadline"):
        m.main([])
