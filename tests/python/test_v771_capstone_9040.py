"""REQ-REPORT-9040-V771: private custody controls grant no scientific credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from carnot.reporting import v771_capstone as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Retain actual upstream bytes privately; never replace missing measurements."""
    value = m.collect(tmp_path_factory.mktemp("capstone9040"), "20261112")
    atomic_json(Path(value["raw"]) / "work.json", value)
    return value


def receipts(work, tmp_path):
    """Synthetic administrative receipts exercise refusal rules only inside private tests."""
    value = deepcopy(work)
    value["validation"] = dict(receipts=[], coverage=dict(complete=True))
    for c in value["validation_scope_manifest"]["commands"]:
        path = tmp_path / (c["name"] + ".log")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private receipt control; no science credit\n")
        value["validation"]["receipts"].append(
            dict(
                name=c["name"],
                passed=True,
                exit_code=0,
                normal_exit=True,
                timed_out=False,
                started_monotonic_ns=value["frozen_ns"] + 1,
                ended_monotonic_ns=value["frozen_ns"] + 2,
                scope_sha256=canonical_hash(value["validation_scope_manifest"]),
                command_argv=c["argv"],
                log_path=str(path),
                log_sha256=m.reference(path)["sha256"],
            )
        )
    return value


def child(tmp_path, *args):
    """Exercise the actual entry point with no checkout-dependent import environment."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9040] before_child completed=0 pending=1", args, flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print(
        "[test9040] after_child completed=1 pending=0",
        result.returncode,
        result.stdout[-600:],
        flush=True,
    )
    return result


def test_full_accounting_and_independent_science(work, tmp_path, monkeypatch):
    """REQ-REPORT-9040-V771: completed accounting leaves every missing science gate open."""
    w = receipts(work, tmp_path)
    monkeypatch.setattr(m, "verify_measured", lambda *_: True)
    value = m.build(w)
    assert value["reconciliation_ready_score"] == 1 and value["required_checks_passed"]
    assert value["verdict_class"] == "blocked" and value["honest_verdict"].startswith(
        "complete_blocked_"
    )
    assert value["completed_count"] == len(value["rows"]) == len(value["slot_rows"]) == 14
    slots = {s["experiment_id"]: s for s in value["slot_rows"]}
    assert list(slots) == list(range(9027, 9041))
    assert slots[9027]["qualified"] and slots[9029]["qualified"]
    assert slots[9028]["qualified"] and slots[9030]["qualified"] and slots[9031]["qualified"]
    assert slots[9032]["primary_status"] == "missing_file"
    assert slots[9032]["disposition"] == "authenticated_scheduler_skip"
    assert slots[9034]["honest_verdict"] is slots[9034]["verdict_class"] is None
    assert slots[9034]["disposition"] == "conductor_skip_without_primary"
    assert all("exp900" not in line for s in slots.values() for line in s.get("conductor_rows", []))
    science = value["v771_science_summary"]
    for name, count in [
        ("graph_detector_bounds", 7),
        ("causal_learning_bounds", 10),
        ("human_disagreement_bounds", 4),
    ]:
        assert len(science[name]) == count
        assert all(
            r["observed"] == "missing_file" and r["denominator"] == 0 and not r["passed"]
            for r in science[name]
        )
    assert science["human_source_audit"]["source_group_count"] == 795
    assert science["human_source_audit"]["source_identity_ready_score"] == 0
    assert science["human_source_audit"]["original_annotation_count"] == 0
    assert science["historical_generation_calls_counted_once"] == 512
    assert science["historical_generation_verdict"] == "complete_disqualified_graph_corpus_a"
    custody = {r["experiment_id"]: r for r in science["source_custody_rows"]}
    assert custody[9030]["intended_count"] == custody[9030]["completed_count"] == 512
    assert custody[9030]["censored_count"] == 0
    assert custody[9031]["intended_count"] == custody[9031]["censored_count"] == 512
    assert custody[9031]["completed_count"] == 0
    assert custody[9030]["numerical_ready"] == custody[9031]["numerical_ready"] == 0
    assert not science["scientific_benefit"] and not science["arc"]["ready"]
    assert len(science["complete_costs"]) == 4
    assert not value["repository_health_passed"] and len(value["repository_health"]["rows"]) == 3
    assert not any(value["model_invocation_counts"].values())
    assert value["MODEL_SPECS"] == value["model_specs"] == value["trained_head_specs"] == []
    assert value["independent_count"] == value["independent_generalization_score"] == 0
    assert value["G1"] == work["publication_gate"]["gates"]["G1"]
    assert any(
        r["prior_experiment_id"] == "exp9000-invocation-provenance"
        and r["actual_prior_verdict"] == "complete_disqualified_invocation_provenance"
        for r in value["retirement_scope_rows"]
    )
    assert slots[9040]["honest_verdict"] == value["honest_verdict"]
    self_prior = next(
        r
        for r in value["retirement_scope_rows"]
        if r["task_id"] == m.TASK and r["prior_experiment_id"] == "exp9026-capstone"
    )
    assert self_prior["actual_current_verdict"] == value["honest_verdict"]
    assert self_prior["unchanged_verdict"]
    assert "python/carnot/reporting/graph_corpus_b_9031.py" in m.REUSED
    assert all("op" in g and "observed" in g for g in value["gate_check_summary"])
    assert value["validation_scope_manifest"]["excluded_statements"] == 0
    assert "full_python_suite" not in m.CHECKS
    assert any(r["original_path"].endswith("coverage.ini") for r in value["code_config_hashes"])
    assert value["validation_scope_manifest"]["coverage_config_reference"]["sha256"]
    coverage_command = next(
        c["argv"]
        for c in value["validation_scope_manifest"]["commands"]
        if c["name"] == "coverage_report"
    )
    assert "--include=" + ",".join(str(m.ROOT / p) for p in m.OWNED) in coverage_command
    for ref in value["repository_health"].get("current_observations", []):
        observed = m.load(ref)
        assert (
            not observed["passed"]
            and observed["scope"] == "repository_health_only_not_current_acceptance"
        )


@pytest.mark.parametrize(
    "mutation",
    ["valid", "missing", "stale", "failed", "timeout", "log", "command", "duplicate", "code"],
)
def test_current_receipt_controls_and_foreign_cli(work, tmp_path, mutation):
    """SCENARIO-REPORT-9040-V771: actual CLI refuses invalid current evidence."""
    w = receipts(work, tmp_path)
    m.runner.mutate_receipt(w, mutation)
    source = tmp_path / "receipt_work.json"
    atomic_json(source, w)
    assert m.receipts_pass(w) == (mutation == "valid")
    result = child(tmp_path, "--check-receipts", source)
    assert result.returncode == int(mutation != "valid"), result.stdout + result.stderr


def test_missing_changed_and_scheduler_tamper(tmp_path, monkeypatch):
    """REQ-REPORT-9040-V771: preserve rejected bytes and keep missing operands explicit."""
    absent = m.collect(tmp_path / "absent_raw", "20261112", tmp_path / "missing")
    assert all(s["primary_status"] == "missing_file" for s in absent["slots"])
    assert m.reduce(absent)["human_source_audit"]["status"] == "missing_file"
    assert m.build(absent)["reconciliation_ready_score"] == 0
    monkeypatch.setitem(m.PINS, 9028, "sha256:changed")
    rejected = m.collect(tmp_path / "changed_raw", "20261112")
    assert any(r["check"] == "input_bytes" and not r["passed"] for r in rejected["checks"])
    assert any(
        r.get("original_path", "").endswith(
            "experiment_9028_v771_historical_provenance_qualification.json"
        )
        for r in rejected["references"]
    )
    gate = dict(upstream="absent", artifact_field="score", op="==", value=1)
    assert m.edge(gate, {})["observed"] == "missing_file"
    assert m.edge(gate, {"absent": dict(value={}, reference={})})["observed"] == "missing_field"
    assert (
        m.edge(gate, {"absent": dict(value=dict(score=None), reference={})})["operand_status"]
        == "scientific_null"
    )
    changed = deepcopy(rejected)
    changed["tasks"][0]["prompt"] += "changed"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(changed)
    source = tmp_path / "scheduler"
    source.mkdir()
    name = m.ALTERNATES[9032]
    original = json.loads((m.ROOT / "results" / name).read_bytes())
    original["failed_evidence_sha256"] = "sha256:changed"
    atomic_json(source / name, original)
    producer = m.ROOT / "results/experiment_9028_v771_historical_provenance_qualification.json"
    rejected_value = json.loads(producer.read_bytes())
    log = tmp_path / "failed_producer.log"
    log.write_text("Private rejected-log retention control; no measurement credit.\n")
    rejected_value["validation_receipts"] = [
        dict(passed=False, log_path=str(log), log_sha256=m.reference(log)["sha256"])
    ]
    atomic_json(source / producer.name, rejected_value)
    alternate = m.collect(tmp_path / "alternate_raw", "20261112", source)
    assert not alternate["slots"][5]["qualified"]
    assert alternate["slots"][5]["primary_status"] == "missing_file"
    assert not alternate["slots"][1]["qualified"]
    assert any(r["original_path"] == str(log) for r in alternate["references"])


def test_primitive_replay_and_rehashed_tamper(work, tmp_path):
    """SCENARIO-REPORT-9040-V771: replay recomputes independent counts and full objects."""
    path = tmp_path / "candidate.json"
    value = m.build(work)
    atomic_json(path, value)
    assert m.replay(path)["passed"]
    assert child(tmp_path, "--cold-replay", path).returncode == 0
    shutil.rmtree(work["private"])
    assert child(tmp_path, "--cold-replay", path).returncode == 0
    value["v771_science_summary"]["human_source_audit"]["eligible_source_group_count"] = 400
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(path, value)
    assert not m.replay(path)["passed"]

    changed = deepcopy(work)
    changed["scheduler_lines"] = []
    changed["slots"] = m.slots(changed)
    source = tmp_path / "scheduler_work.json"
    atomic_json(source, changed)
    value = m.build(changed)
    value["work_reference"] = m.reference(source)
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(path, value)
    assert not m.replay(path)["passed"]
    assert child(tmp_path, "--cold-replay", path).returncode == 1
    value["reproducibility_checksum"] = "changed"
    atomic_json(path, value)
    assert not m.replay(path)["passed"]
    assert child(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1
    changed = deepcopy(work)
    changed["references"][-1]["sha256"] = "sha256:changed"
    source = tmp_path / "changed_work.json"
    atomic_json(source, changed)
    value = m.build(changed)
    value["work_reference"] = m.reference(source)
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(path, value)
    assert not m.replay(path)["passed"]


def test_authority_and_numerical_refusals(work, tmp_path):
    """REQ-REPORT-9040-V771: full-object and primitive mutations cannot invent support."""
    w = deepcopy(work)
    w["contract"]["tasks"][1]["prompt"] += "changed"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(w)
    w = deepcopy(work)
    ref = w["contract"]["design_reference"]
    changed = tmp_path / "design.md"
    changed.write_bytes(Path(ref["path"]).read_bytes().replace(b"14 tasks", b"15 tasks", 1))
    ref.update(m.reference(changed))
    assert not m.contract(w["contract"])["ready"]
    w = deepcopy(work)
    w["slots"][7]["qualified"] = True
    with pytest.raises(ValueError, match="registered_raw"):
        m.reduce(w)
    w = deepcopy(work)
    w["values"]["exp9029-source-identity-resolution"]["original"]["source_group_rows"] = []
    with pytest.raises(ValueError, match="human_primitive"):
        m.reduce(w)


def test_private_cli_missing_checks_and_dates(work, tmp_path):
    """SCENARIO-REPORT-9040-V771: failed owned work remains private and disqualified."""
    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / "published" / (m.NAME + ".json")
    result = child(tmp_path, "--date", "20261112", "--from-work", source, "--output", output)
    assert result.returncode == 1 and "owned_validation" in result.stdout
    assert not output.exists()
    assert child(tmp_path, "--date", "wrong").returncode == 2
    assert child(tmp_path, "--from-work", source).returncode == 2
    w = deepcopy(work)
    w["validation"]["receipts"] = [dict(name="malformed")]
    assert not m.receipts_pass(w)


def test_validation_orchestration_and_terminal(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9040-V771: frozen commands own actual bounded subprocess checks."""
    w = deepcopy(work)
    w["raw"], w["private"] = str(tmp_path / "raw"), str(tmp_path / "private")
    private = Path(w["private"])
    private.mkdir()
    (private / "coverage.ini").write_text("[run]\n")
    spec = m.CommandSpec(
        "private_check", (sys.executable, "-c", "print('actual normal exit')"), "test", 30
    )
    w["validation_scope_manifest"]["commands"] = [
        dict(name=spec.name, argv=list(spec.argv), timeout_s=spec.timeout_s)
    ]
    monkeypatch.setattr(m.runner, "validation_plan", lambda _: [spec])
    result = m.validate(w)
    assert result["receipts"][0]["normal_exit"] and result["receipts"][0]["passed"]
    assert not result["coverage"]["complete"]
    (private / ".coverage").write_bytes(b"private orchestration fixture")
    monkeypatch.setattr(m, "measured", lambda *_: dict(complete=False, files=[], rows=[]))
    assert not m.validate(w)["coverage"]["complete"]
    w["validation_scope_manifest"]["commands"] = []
    with pytest.raises(ValueError, match="frozen_commands"):
        m.validate(w)
    candidate = tmp_path / "rejected.json"
    atomic_json(candidate, m.build(work))
    report = m.terminal(candidate, tmp_path / "terminal", work["validation_scope_manifest"])
    assert report["passed"] and len(report["receipts"]) == 3
    assert all(r["normal_exit"] for r in report["receipts"])
    value = m.build(work)
    value["completed_count"] = 15
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(candidate, value)
    assert not m.terminal(
        candidate, tmp_path / "tamper_terminal", work["validation_scope_manifest"]
    )["passed"]


def test_main_validation_publication_and_deadlines(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9040-V771: only successful owned checks reach atomic publication."""
    observed = receipts(work, tmp_path / "logs")

    def collect(raw, *args):
        value = deepcopy(observed)
        value["raw"] = str(raw)
        return value

    monkeypatch.setattr(m, "collect", collect)
    monkeypatch.setattr(m, "validate", lambda _: deepcopy(observed["validation"]))
    monkeypatch.setattr(m, "verify_measured", lambda *_: True)
    monkeypatch.setattr(m, "terminal", lambda *_: dict(passed=False, receipts=[]))
    output = tmp_path / (m.NAME + ".json")
    args = ["--output", str(output)]
    assert m.main(args) == 1
    assert not output.exists()
    monkeypatch.setattr(m, "terminal", lambda *_: dict(passed=True, receipts=[]))
    assert m.main(args) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["reconciliation_ready_score"] == 1
    observed["started_ns"] -= 3301 * 10**9
    assert m.main(args) == 1
    observed["started_ns"] -= 1500 * 10**9
    source = tmp_path / "expired_work.json"
    atomic_json(source, observed)
    assert m.main([*args, "--from-work", str(source)]) == 1


def test_private_historical_consumers(monkeypatch):
    """SCENARIO-REPORT-9040-V771: historical tests receive preserved bytes, unchanged assertions."""
    original = m.historical.DESIGN

    def run(argv, plugins):
        assert argv == ["private-node"]
        assert len(plugins) == 1
        return 0

    monkeypatch.setattr(pytest, "main", run)
    assert m.main(["--direct-consumers", "private-node"]) == 0
    assert m.historical.DESIGN == original


def test_coverage_child_budget(tmp_path):
    """REQ-REPORT-9040-V771: bounded coverage children must finish all original assertions."""
    plan = {c.name: c for c in m.validation_plan(tmp_path)}
    assert plan["owned_tests"].timeout_s == 600
    assert set(plan) == set(m.CHECKS)
