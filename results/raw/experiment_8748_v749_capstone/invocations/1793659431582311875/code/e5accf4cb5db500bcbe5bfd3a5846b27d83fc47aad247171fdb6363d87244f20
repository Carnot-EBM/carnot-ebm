"""REQ-REPORT-8748 / REQ-VERIFY-8748: bind terminal reconciliation to real evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v749_capstone as c


def cli(*args):
    """SCENARIO-VERIFY-8748-REPLAY: exercise the public entry point outside the checkout."""
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
    """SCENARIO-REPORT-8748-ACCOUNTING: private artifacts never grant production readiness."""
    path = tmp_path_factory.mktemp("v749-capstone") / (c.e.NAME + ".json")
    result = cli("--output", path, "--private-control", "--date", "20261102")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_accounting(private):
    """SCENARIO-REPORT-8748-ACCOUNTING: every intended slot has one honest disposition."""
    path, value = private
    assert cli("--cold-replay", path).returncode == 0
    assert value["experiment_id"] == 8748 and value["milestone"] == "2026.11.749"
    assert value["intended_count"] == value["completed_count"] == len(value["rows"]) == 12
    assert len(value["full_task_objects"]) == len(value["next_decision_rows"]) == 12
    assert value["canonical_tasks_sha256"] == c.binding.PIN
    assert value["producer_executed_count"] + value["skipped_study_count"] == 12
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert value["MODEL_SPECS"] == [] and value["current_language_model_calls"] == 0
    assert value["scientific_reconstruction"]["learning"]["future_brier"] is None
    assert value["same_batch_10x_score"] == value["physical_hardware_ready_score"] == 0
    assert value["paper_ready"] == all(value[g]["pass"] for g in ["G1", "G2", "G3", "G4"])
    assert set(value) <= set(value["field_principles"])
    assert all(r["retire_if_same_verdict"] for r in value["retirement_candidate_rows"])


@pytest.mark.parametrize("mode", ["rows", "receipts", "primitive", "missing"])
def test_rehashed_rejection(private, tmp_path, mode):
    """SCENARIO-VERIFY-8748-REPLAY: an internally rehashed claim owns no authority."""
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
    """REQ-VERIFY-8748: malformed dates and failed children cannot qualify execution."""
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
    """SCENARIO-VERIFY-8748-REPLAY: prompts, routing, gates and order remain bound."""
    assert all(r["passed"] for r in c.binding.mutations())


def test_producer_source_rejection(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8748: exact primary identity cannot hide changed executable bytes."""
    _, value = private
    work = c.e.load(value["primitive_reference"])
    row = deepcopy(work["slots"][0])
    task = work["contract"]["tasks"][0]
    source = tmp_path / "source.py"
    source.write_text("changed source")
    monkeypatch.setattr(c.previous, "resolve", lambda *args: deepcopy(row))
    with monkeypatch.context() as patch:
        patch.setattr(
            c.e,
            "load",
            lambda ref: dict(
                code_config_hashes={
                    str(source): "sha256:wrong",
                    str(source.with_suffix(".txt")): "ignored",
                }
            ),
        )
        rejected = c.slot(task, tmp_path, [])
        assert not rejected["producer_qualified"]
        assert rejected["source_identity_failures"][0]["check"] == "unchanged_executable_identity"
    row["declared_reference"]["sha256"] = "sha256:forged"
    rejected = c.slot(task, tmp_path, [])
    assert rejected["source_identity_failures"][0]["check"] == "exact_declared_primary"


def test_missing_science_and_changed_prerequisite(private):
    """REQ-REPORT-8748: an empty ARC delta or disqualified rows cannot supply benefit."""
    _, value = private
    work = c.e.load(value["primitive_reference"])
    for row in work["slots"]:
        row["producer_qualified"] = row["producer_executed"] = False
    reduced = c.reduce(work)
    assert not reduced["arc"]["measured"] and reduced["arc"]["new_event_count"] is None
    assert reduced["disqualified_diagnostics"] == []
    assert reduced["learning"]["future_brier"] is None
    next(r for r in work["slots"] if r["experiment_id"] == 8740)["producer_qualified"] = True
    with pytest.raises(ValueError, match="changed_prerequisite"):
        c.reduce(work)


def test_exclusion_and_full_suite_receipt(monkeypatch, tmp_path):
    """REQ-VERIFY-8748: preserve a failed suite receipt and honor retirement without edits."""
    with monkeypatch.context() as patch:
        patch.setattr(
            c.binding.yaml, "safe_load", lambda raw: dict(retired=[dict(experiment_id=8748)])
        )
        assert not c.methods()["gate_check_summary"][-1]["passed"]
    actual = c.e.load(
        c.e.reference(
            c.e.ROOT / "results/raw" / c.e.NAME / "startup/full_python_suite.receipt.json"
        )
    )
    monkeypatch.setattr(c, "BASE_VALIDATE", lambda raw: [])
    assert c.validate(tmp_path) == [actual]
