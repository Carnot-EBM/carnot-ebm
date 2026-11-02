"""REQ-REPORT-8736 / REQ-VERIFY-8736: exercise current binding and sealed evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v748_capstone as c


def cli(*args):
    """SCENARIO-VERIFY-8736-REPLAY: the real CLI must work outside the checkout."""
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
    """SCENARIO-VERIFY-8736-REPLAY: private evidence cannot grant public readiness."""
    path = tmp_path_factory.mktemp("v748-capstone") / (c.e.NAME + ".json")
    result = cli("--output", path, "--private-control", "--date", "20261102")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_accounting(private):
    """SCENARIO-REPORT-8736-ACCOUNTING: execution and scientific benefit stay separate."""
    path, value = private
    assert cli("--cold-replay", path).returncode == 0
    assert value["experiment_id"] == 8736 and value["milestone"] == "2026.11.748"
    assert value["intended_count"] == value["completed_count"] == len(value["rows"]) == 14
    assert value["producer_executed_count"] == 10 and value["skipped_study_count"] == 4
    rows = {r["experiment_id"]: r for r in value["rows"]}
    assert rows[8724]["verdict_class"] == "disqualified"
    assert rows[8725]["disposition"] == "conductor_pre_gate"
    assert all(not rows[n]["scientific_result_measured"] for n in [8725, 8727, 8728, 8730])
    assert value["canonical_tasks_sha256"] == c.binding.m.PIN
    assert len(value["full_task_objects"]) == len(value["next_decision_rows"]) == 14
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert value["MODEL_SPECS"] == [] and value["current_language_model_calls"] == 0
    science = value["scientific_reconstruction"]
    assert science["learning"]["future_brier"] is None
    assert science["qwen"]["support_scores"] is None
    arms = science["verifier"]["arm_rows"]
    assert [r["defects_detected"] for r in arms] == [120, 130]
    assert [r["evaluation_invocations"] for r in arms] == [900, 900]
    assert science["verifier"]["independent_generalization"] is False
    assert science["compiled"]["update_rows"] == 82240
    assert science["resident_cost"]["projected_minimum_measurement_s"] > 3600
    assert science["resident_cost"]["speed_ratio"] is None
    assert science["installed"]["ready"] and science["installed"]["complete"]
    assert science["arc"]["new_event_count"] == 0
    assert value["same_batch_10x_score"] == value["physical_hardware_ready_score"] == 0
    assert value["paper_ready"] == all(value[g]["pass"] for g in ["G1", "G2", "G3", "G4"])
    assert set(value) <= set(value["field_principles"])


@pytest.mark.parametrize("mode", ["rows", "receipts", "primitive", "missing"])
def test_rehashed_rejection(private, tmp_path, mode):
    """SCENARIO-VERIFY-8736-REPLAY: consistently rehashed claims lack independent authority."""
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


def test_failure_paths(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8736: failed children and substituted producers cannot qualify."""
    for args in [["--date", "123"], ["--date", "20260230"], ["--private-control"]]:
        assert cli(*args).returncode == 1
    path = tmp_path / "failed" / (c.e.NAME + ".json")
    result = cli("--private-control", "--failed-child", "--output", path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(path.read_bytes())["verdict_class"] == "disqualified"
    source, value = private
    work = c.e.load(value["primitive_reference"])
    work["private_control"] = False
    assert c.build(work, [dict(passed=True)], source)["reconciliation_ready_score"] == 1
    assert c.build(work, [], source)["verdict_class"] == "disqualified"
    row = deepcopy(work["slots"][0])
    row["primary_reference"]["path"] = "substituted"
    monkeypatch.setattr(c.previous, "resolve", lambda *a: deepcopy(row))
    assert not c.slot(dict(id=row["task_id"]), tmp_path, [])["producer_qualified"]


def test_absent_science(private):
    """REQ-REPORT-8736: no admitted producers leaves scientific quantities absent."""
    _, value = private
    work = c.e.load(value["primitive_reference"])
    for row in work["slots"]:
        row["producer_qualified"] = False
    reduced = c.reduce(work)
    assert reduced["learning"]["future_brier"] is None
    assert reduced["resident_cost"]["speed_ratio"] is None
    assert not reduced["installed"]["measured"]
