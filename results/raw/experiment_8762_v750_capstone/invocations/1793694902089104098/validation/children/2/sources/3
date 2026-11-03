"""REQ-REPORT-8762 / REQ-VERIFY-8762: bind terminal reconciliation to real evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v750_capstone as c


def cli(*args):
    """SCENARIO-VERIFY-8762-REPLAY: exercise the public entry point outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(c.e.ROOT / c.e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """SCENARIO-REPORT-8762-ACCOUNTING: private artifacts never grant production readiness."""
    path = tmp_path_factory.mktemp("v750-capstone") / (c.e.NAME + ".json")
    result = cli("--output", path, "--private-control", "--date", "20261103")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_accounting(private):
    """SCENARIO-REPORT-8762-ACCOUNTING: every intended slot has one honest disposition."""
    path, value = private
    assert cli("--cold-replay", path).returncode == 0
    assert value["experiment_id"] == 8762 and value["milestone"] == "2026.11.750"
    assert value["intended_count"] == value["completed_count"] == len(value["rows"]) == 14
    assert len(value["full_task_objects"]) == len(value["next_decision_rows"]) == 14
    assert value["canonical_tasks_sha256"] == c.binding.PIN
    assert value["producer_executed_count"] + value["skipped_study_count"] == 14
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert value["MODEL_SPECS"] == [] and value["current_language_model_calls"] == 0
    assert value["scientific_reconstruction"]["learning"]["future_brier"] is None
    assert value["same_batch_10x_score"] == value["physical_hardware_ready_score"] == 0
    assert value["paper_ready"] == all(value[g]["pass"] for g in ["G1", "G2", "G3", "G4"])
    assert set(value) <= set(value["field_principles"])
    assert all(r["retire_if_same_verdict"] for r in value["retirement_candidate_rows"])


@pytest.mark.parametrize("mode", ["rows", "receipts", "primitive", "missing"])
def test_rehashed_rejection(private, tmp_path, mode):
    """SCENARIO-VERIFY-8762-REPLAY: an internally rehashed claim owns no authority."""
    _, original = private
    value = deepcopy(original)
    if mode == "rows":
        value["rows"][0]["producer_qualified"] = not value["rows"][0]["producer_qualified"]
    elif mode == "receipts":
        value["validation_receipts"][0]["passed"] = False
    else:
        work = c.e.load(value["primitive_reference"])
        work["duration_s"] += 1
        target = tmp_path / "measurement.json"
        if mode == "primitive":
            c.e.atomic_json(target, work)
        value["primitive_reference"] = c.e.reference(target)
    value["reproducibility_checksum"] = c.e.digest(value)
    target = tmp_path / "candidate.json"
    c.e.atomic_json(target, value)
    assert cli("--cold-replay", target).returncode == 1


def test_failure_paths(private, tmp_path):
    """REQ-VERIFY-8762: malformed dates and failed children cannot qualify execution."""
    for args in [["--date", "123"], ["--date", "20260230"], ["--private-control"]]:
        assert cli(*args).returncode == 1
    target = tmp_path / "failed" / (c.e.NAME + ".json")
    result = cli("--private-control", "--failed-child", "--output", target)
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(target.read_bytes())["verdict_class"] == "disqualified"
    path, value = private
    work = c.e.load(value["primitive_reference"])
    work["private_control"] = False
    assert c.build(work, [dict(passed=True)], path)["reconciliation_ready_score"] == 1
    assert c.build(work, [], path)["verdict_class"] == "disqualified"


def test_full_object_controls():
    """SCENARIO-VERIFY-8762-REPLAY: prompts, routing, gates and order remain bound."""
    assert all(r["passed"] for r in c.controls())


def test_reconstruction(private):
    """REQ-REPORT-8762: scientific null and complete process benefit stay separate."""
    _, value = private
    science = value["scientific_reconstruction"]
    assert science["verifier_transfer"]["paired_world_intervals"]["new"]["lower95"] == 0
    assert science["surface_sensitivity"]["decision_disagreements"] == 0
    assert science["feature_reuse_cost"]["feature_reuse_benefit_score"] == 1
    assert science["feature_reuse_cost"]["same_batch_10x_score"] == 0
    assert not science["bounded_durable_cost"]["measured"]
    assert value["repository_health_reference"]["full_suite_pass"] is False


def test_rejected_inputs(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8762: independent primary pins reject coherent substitutions."""
    _, value = private
    work = c.e.load(value["primitive_reference"])
    row = deepcopy(work["slots"][0])
    row["declared_reference"]["sha256"] = "sha256:forged"
    monkeypatch.setattr(c.previous, "resolve", lambda *args: deepcopy(row))
    rejected = c.slot(work["contract"]["tasks"][0], tmp_path, [])
    assert not rejected["producer_qualified"]
    assert rejected["authentication_errors"][-1] == "independent_v750_primary_pin"
    for r in work["slots"]:
        r["producer_qualified"] = False
    assert c.reduce(work)["feature_reuse_cost"]["rows"] == []


def test_scoped_health_separation(private):
    """REQ-REPORT-8762: an authenticated health-only receipt is not an owned failure."""
    _, value = private
    row = next(r for r in value["rows"] if r["experiment_id"] == 8756)
    assert row["producer_qualified"]
    assert row["repository_health_receipts"]
    assert not row["repository_health_receipts"][0]["passed"]


def test_methods_and_scoped_metric(tmp_path):
    """REQ-REPORT-8762: methods custody is separate from measured scientific outcomes."""
    tasks = c.authority(c.e.ROOT)["tasks"]
    methods = c.slot(tasks[5], tmp_path, [])
    assert not methods["scientific_result_measured"]
    qualified = c.slot(tasks[7], tmp_path, [])
    assert qualified["producer_qualified"] and qualified["absolute_metric"] == 1
