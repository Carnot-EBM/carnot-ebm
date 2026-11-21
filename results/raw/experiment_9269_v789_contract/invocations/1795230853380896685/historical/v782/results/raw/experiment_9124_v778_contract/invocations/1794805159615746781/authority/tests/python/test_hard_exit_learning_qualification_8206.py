"""REQ-VERIFY-8206 / REQ-REPORT-8206: measure real exits without a profiler."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import coverage
import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import hard_exit_learning_qualification_8206 as e


def test_real_child_two_crashes_and_no_crash(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8206-CHILD: real exit73 retains all pending state."""
    rows, labels = e.legacy.fixture("learnable")
    from carnot.reporting import v709_execution

    monkeypatch.setattr(v709_execution, "ROOT", tmp_path)
    atomic_json(tmp_path / "restart-input.json", dict(rows=rows, labels=labels, seed=101))
    specs = e.restart_specs(tmp_path, tmp_path)
    config = tmp_path / "child_coverage/coverage.ini"
    assert "patch = _exit" in config.read_text()
    assert "parallel = true" in config.read_text()
    assert all(str(e.ROOT / e.legacy.CLI) in s["argv"] for s in specs)
    assert all(str(e.ROOT / e.CLI) not in s["argv"] for s in specs)
    before = sys.getprofile()
    receipts = [e.run_check(e.ROOT, s, tmp_path, tmp_path / "logs") for s in specs]
    assert sys.getprofile() is before
    assert [r["actual_exit"] for r in receipts] == [0, 73, 0, 73, 0]
    for r in receipts:
        assert r["passed"] and r["stdout_sha256"] and r["stderr_sha256"]
        assert r["ended_monotonic_ns"] >= r["started_monotonic_ns"]
    measured = e.child_coverage(tmp_path)
    assert measured["passed"] and measured["child_shards"] == 5
    assert {289, 290, 291, 292} <= set(measured["executed_lines"])
    for row in measured["restart_state_hashes"]:
        assert row["baseline_sha256"] == row["resumed_sha256"]
        assert row["pending_count"] > 0
    baseline = json.loads((tmp_path / "uninterrupted/final.json").read_text())
    assert e.legacy.fixture_summary(baseline, rows)["passed"]
    cov = coverage.Coverage(data_file=measured["data_path"], config_file=False)
    cov.load()
    assert 292 in cov.get_data().lines(str(e.ROOT / e.legacy.MODULE))
    outer = tmp_path / "outer"
    outer.mkdir()
    monkeypatch.setenv("CARNOT_8206_COVERAGE_PARENT", str(outer))
    assert e.child_coverage(tmp_path)["passed"]
    assert len(list(outer.glob(".coverage.child-*"))) == 5
    changed = dict(baseline)
    changed["cursor"] += 1
    atomic_json(tmp_path / "restart/final.json", changed)
    assert not e.child_coverage(tmp_path)["passed"]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """SCENARIO-VERIFY-8206-CUSTODY: private fixtures preserve old artifacts."""
    raw = tmp_path_factory.mktemp("8206-evidence")
    from carnot.reporting.local_consumer_qualification_8347 import (
        freeze_historical,
        historical_operands,
    )

    closure = freeze_historical(tmp_path_factory.mktemp("8206-historical-operands"))
    with historical_operands(closure):
        measured = e.measure(e.ROOT, raw, fixture=True)
    return measured, raw


def test_readiness_and_failed_owned_check(work):
    """REQ-VERIFY-8206: stream authentication is independent of fit success."""
    measured, raw = work
    value = e.build(measured, raw, [dict(passed=True, normal_exit=True)], fixture=True)
    assert value["experiment_id"] == 8206
    assert value["calibrated_memory_ready_score"] == value["stream_input_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert value["independent_count"] == value["independent_generalization_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert value["numerical_protocol_sha256"] == e.legacy.PROTOCOL_HASH
    assert value["prior_8193_failure"]["verdict_class"] == "disqualified"
    assert value["prior_8193_failure"]["missing_lines"] == [54, 55, 56, 57]
    assert [r["actual_exit"] for r in value["child_exit_rows"]] == [0, 73, 0, 73, 0]
    assert all(r["passed"] for r in value["fixture_rows"] if r["case"] == "learnable")
    failed = e.build(measured, raw, [dict(passed=False)], fixture=True)
    assert failed["verdict_class"] == "disqualified"
    assert failed["calibrated_memory_ready_score"] == 0
    assert failed["stream_input_ready_score"] == 1
    changed = deepcopy(measured)
    changed["measured_child_coverage"]["passed"] = False
    assert (
        e.build(changed, raw, [dict(passed=True)], fixture=True)["verdict_class"] == "disqualified"
    )


def test_missing_operand_stops_measurement(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8206-CUSTODY: absent external bytes do not become zero."""
    preflight = tmp_path / "preflight.json"
    atomic_json(preflight, dict(command="private input probe", exit_code=0))
    monkeypatch.setenv("CARNOT_8206_PREFLIGHT_RECEIPT", str(preflight))
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    value = e.build(work, tmp_path / "raw", [dict(passed=True)], fixture=True)
    assert value["verdict_class"] == "blocked" and not value["fixture_states"]
    assert value["gate_check_summary"][-1]["observed"] is False
    assert value["stream_input_ready_score"] == value["calibrated_memory_ready_score"] == 0
    upstream = json.loads((e.ROOT / e.legacy.UPSTREAM).read_text())
    upstream["learning_audit_ready_score"] = 0
    atomic_json(tmp_path / e.legacy.UPSTREAM, upstream)
    assert not e.measure(tmp_path, tmp_path / "mutated", fixture=True)["input_ready"]


def test_manifest_frozen_and_static_paths(tmp_path):
    """REQ-REPORT-8206: static tools receive files and all owned/E2E routes."""
    from carnot.reporting import hard_exit_learning_execution_8206 as runner

    specs = runner.manifest(tmp_path, tmp_path / "candidate.json")
    assert "patch = _exit" in (tmp_path / "coverage.ini").read_text()
    for spec in specs["commands"]:
        if spec["name"].startswith(("ruff", "strict_mypy")):
            assert all("::" not in arg for arg in spec["argv"])
    assert "--files" in next(s for s in specs["commands"] if s["name"] == "spec_coverage")["argv"]
    assert any(s["name"] == "consumer_and_E2E015_019" for s in specs["commands"])
    assert specs["repository_health"]["argv"][-2:] == ["tests/python", "-q"]


def test_real_cli_replay_blocked_worker_and_guard(work, tmp_path):
    """SCENARIO-REPORT-8206-CLI: direct paths work without ambient imports."""
    measured, raw = work
    value = e.build(measured, raw, [dict(passed=True)], fixture=True)
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    cli = [
        "/usr/bin/env",
        "-u",
        "PYTHONPATH",
        str(e.ROOT / ".venv/bin/python"),
        "-u",
        str(e.ROOT / e.CLI),
    ]
    config = os.environ.get("COVERAGE_RCFILE")
    if config:
        cli.insert(3, "COVERAGE_PROCESS_START=" + config)

    def child(name, args, expected=0):
        from carnot.reporting import v709_execution

        monkey = pytest.MonkeyPatch()
        monkey.setattr(v709_execution, "ROOT", tmp_path)
        receipt = e.run_check(
            e.ROOT,
            dict(name=name, argv=cli + args, expected_exit=expected, deadline_s=300),
            tmp_path,
            tmp_path / "logs",
        )
        monkey.undo()
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()

    child("replay", ["--cold-replay", str(path)])
    for operand in ["raw_shard_hashes", "child_exit_rows"]:
        bad = deepcopy(value)
        field = "sha256" if operand == "raw_shard_hashes" else "stderr_sha256"
        bad[operand][0][field] = "sha256:wrong"
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path)
    for field in [
        "completed_count",
        "stream_input_ready_score",
        "child_exit_rows",
        "restart_state_hashes",
        "numerical_protocol_sha256",
    ]:
        bad = deepcopy(value)
        bad[field] = [] if isinstance(bad[field], list) else "tampered"
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path)
    bad = dict(value, qualification_replay_sha256="sha256:wrong")
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(path, bad)
    assert not e.replay(path)
    atomic_json(path, dict(value, experiment_id=0))
    child("tamper", ["--cold-replay", str(path)], 1)
    assert not e.replay(tmp_path / "absent.json")
    child(
        "blocked",
        [
            "--fixture-output",
            str(tmp_path / "blocked" / path.name),
            "--root",
            str(tmp_path / "absent"),
        ],
    )
    child("private_success", ["--fixture-output", str(tmp_path / "success" / path.name)])
    child("guard", ["--fixture-output", str(e.ROOT / "results/forbidden.json")], 2)
    child("bad_date", ["--date", "20261005"], 2)
    child(
        "worker_blocked",
        [
            "--worker-output",
            str(tmp_path / "worker/measurement.json"),
            "--root",
            str(tmp_path / "absent"),
        ],
    )


def test_coverage_summary_is_measured(work, tmp_path):
    """REQ-VERIFY-8206: real coverage report bytes provide statement counts."""
    measured, raw = work
    report = measured["measured_child_coverage"]["report_path"]
    receipt = dict(
        name="coverage_json",
        passed=True,
        argv=[
            "--rcfile=" + str(raw / "child_coverage/coverage.ini"),
            "-o",
            report,
            "--data-file=x",
        ],
    )
    value = e.build(measured, raw, [receipt], fixture=True)
    assert value["coverage_statement_counts"]
    assert set(value["field_principles"]) >= set(value) - {"reproducibility_checksum"}
    from carnot.reporting import hard_exit_learning_execution_8206 as runner

    monkey = pytest.MonkeyPatch()
    monkey.setattr(runner.previous, "main", lambda args: 0)
    assert runner.main([]) == 0
    monkey.undo()
