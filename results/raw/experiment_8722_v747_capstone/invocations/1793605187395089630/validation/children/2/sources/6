"""REQ-REPORT-8722 / REQ-VERIFY-8722: preserve every scheduling and science boundary."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v747_capstone as runner
from carnot.reporting import v747_capstone_inputs as e
from carnot.reporting import v747_capstone_report as report


def cli(*args):
    """SCENARIO-VERIFY-8722-REPLAY: execute the real entry point outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """SCENARIO-VERIFY-8722-REPLAY: create independently sealed private final bytes."""
    path = tmp_path_factory.mktemp("v747-capstone") / (e.NAME + ".json")
    result = cli("--output", path, "--private-control", "--date", "20261102")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_accounting_and_scientific_reconstruction(private):
    """SCENARIO-REPORT-8722-ACCOUNTING: skips never become scientific observations."""
    path, value = private
    assert cli("--cold-replay", path).returncode == 0
    assert value["completed_count"] == value["intended_count"] == len(value["rows"]) == 14
    rows = {r["experiment_id"]: r for r in value["rows"]}
    assert rows[8709]["verdict_class"] == "disqualified"
    assert not rows[8709]["producer_qualified"]
    assert sum(not r["producer_executed"] for r in rows.values()) == 5
    assert all(not rows[n]["scientific_result_measured"] for n in [8710, 8711, 8713, 8714, 8718])
    assert value["canonical_tasks_sha256"] == e.contract_source.PIN
    assert len(value["full_task_objects"]) == len(value["next_decision_rows"]) == 14
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert value["MODEL_SPECS"] == [] and value["current_language_model_calls"] == 0
    assert value["reconciliation_ready_score"] == value["independent_count"] == 0
    science = value["scientific_reconstruction"]
    assert science["relational"]["collision_count"] > 0
    assert science["relational"]["missing_probability_count"] == 480
    assert science["verifier"]["baseline_detected"] == 120
    assert science["verifier"]["extended_detected"] == 130
    assert science["batch"]["batching_benefit_score"] == 1
    assert science["batch"]["same_batch_10x_score"] == 0
    assert science["compiled"]["integer_state_mismatches"] == 0
    assert science["arc"]["new_event_count"] == 0
    assert (
        value["physical_hardware_ready_score"] == value["generalized_learning_benefit_score"] == 0
    )
    assert set(value) <= set(value["field_principles"])
    assert value["paper_ready"] == all(value[g]["pass"] for g in ["G1", "G2", "G3", "G4"])
    assert all(type(v) is int and value[k] == v for k, v in value["branch_gate_values"].items())


@pytest.mark.parametrize("mode", ["rows", "receipts", "primitive", "missing"])
def test_rehashed_rejection(private, tmp_path, mode):
    """SCENARIO-VERIFY-8722-REPLAY: a candidate cannot replace its independent seal."""
    _, original = private
    value = deepcopy(original)
    if mode == "rows":
        value["rows"][0]["producer_qualified"] = True
    elif mode == "receipts":
        value["validation_receipts"][0]["passed"] = False
    else:
        work = e.load(value["primitive_reference"])
        work["duration_s"] += 1
        target = tmp_path / "measurement.json"
        if mode == "primitive":
            e.atomic_json(target, work)
        value["primitive_reference"] = e.reference(target)
    value["reproducibility_checksum"] = e.digest(value)
    target = tmp_path / "candidate.json"
    e.atomic_json(target, value)
    assert cli("--cold-replay", target).returncode == 1


def test_private_guards_and_failed_child(private, tmp_path):
    """REQ-VERIFY-8722: real failed subprocesses disqualify without publishing readiness."""
    for args in [
        ["--date", "123"],
        ["--date", "20260230"],
        ["--private-control"],
        ["--cold-replay", tmp_path / "absent"],
    ]:
        assert cli(*args).returncode == 1
    alias = tmp_path / "alias"
    alias.symlink_to(e.ROOT, target_is_directory=True)
    assert (
        cli("--private-control", "--output", alias / "results" / (e.NAME + ".json")).returncode == 1
    )
    path = tmp_path / "failed" / (e.NAME + ".json")
    result = cli("--private-control", "--failed-child", "--output", path)
    assert result.returncode == 1, result.stdout + result.stderr
    value = json.loads(path.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["failed_count"] == 1
    assert value["validation_receipts"][-1]["actual_exit"] == 1


def test_report_qualification(private):
    """REQ-REPORT-8722: local accounting qualifies independently of missing upstream science."""
    path, value = private
    work = e.load(value["primitive_reference"])
    work["private_control"] = False
    checked = report.build(work, [dict(name="owned", passed=True)], path)
    assert checked["reconciliation_ready_score"] == 1
    assert checked["verdict_class"] == "blocked"
    assert report.build(work, [], path)["verdict_class"] == "disqualified"
    work["contract"]["ready"] = False
    assert report.build(work, [dict(passed=True)], path)["reconciliation_ready_score"] == 0


def test_owned_failure_paths(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8722: failed publication, deadlines and exact-path checks fail closed."""
    from carnot.reporting import v747_capstone_measure as measurement
    from carnot.reporting.v747_capstone_reduce import same

    _, value = private
    work = e.load(value["primitive_reference"])
    row = deepcopy(work["slots"][0])
    row["primary_reference"]["path"] = "different_declared_path"
    monkeypatch.setattr(e.previous, "resolve", lambda *a: deepcopy(row))
    rejected = e.slot(dict(id=row["task_id"]), tmp_path, [])
    assert not rejected["producer_qualified"]
    assert "exact_declared_versioned_primary" in rejected["authentication_errors"]
    monkeypatch.undo()
    with pytest.raises(ValueError, match="scientific_reconstruction"):
        same(0, 1, "private_tamper")
    monkeypatch.setattr(e.contract_source, "STAGED", "research-roadmap.yaml")
    monkeypatch.setattr(measurement, "child", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="publication_gate_execution"):
        measurement.measure(tmp_path / "publication", "20261102", True)
    monkeypatch.setattr(measurement, "child", lambda *a, **k: work["publication_receipt"])
    ticks = iter([0, 3600_000_000_000])
    monkeypatch.setattr(measurement.time, "monotonic_ns", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        measurement.measure(tmp_path / "deadline", "20261102", True)
    assert runner.runtime.e is e and runner.runtime.report is report
