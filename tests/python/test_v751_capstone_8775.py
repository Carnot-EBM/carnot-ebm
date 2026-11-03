"""REQ-REPORT-8775 / REQ-VERIFY-8775: source truth survives terminal reconciliation."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v751_capstone as c


def cli(*args):
    """SCENARIO-VERIFY-8775-REPLAY: real children run outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(c.e.ROOT / c.e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """SCENARIO-REPORT-8775-ACCOUNTING: no fixture publishes production evidence."""
    path = tmp_path_factory.mktemp("v751-capstone") / (c.e.NAME + ".json")
    result = cli("--output", path, "--private-control", "--date", "20261103")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_accounting(private):
    """SCENARIO-REPORT-8775-ACCOUNTING: all thirteen slots preserve their boundaries."""
    path, v = private
    assert cli("--cold-replay", path).returncode == 0
    assert v["intended_count"] == v["completed_count"] == len(v["rows"]) == 13
    assert len(v["full_task_objects"]) == len(v["next_decision_rows"]) == 13
    assert v["producer_executed_count"] == 8 and v["skipped_study_count"] == 5
    assert v["verdict_class"] == "blocked" and v["required_checks_passed"]
    assert v["MODEL_SPECS"] == [] and v["current_language_model_calls"] == 0
    assert v["canonical_tasks_sha256"] == c.binding.PIN
    assert set(v) <= set(v["field_principles"])
    assert v["paper_ready"] == all(v[g]["pass"] for g in ("G1", "G2", "G3", "G4"))
    assert all(r["disposition"] != "missing_disposition" for r in v["rows"])
    assert all(not r["automatic_rerun_authorized"] for r in v["next_decision_rows"])


def test_reconstruction(private):
    """REQ-REPORT-8775: reconstructed rejected rows never gain eligibility."""
    _, v = private
    s = v["scientific_reconstruction"]
    assert s["learning"]["future_brier"] is None
    assert (
        s["historical_v750"]["verifier_transfer"]["paired_world_intervals"]["new"]["lower95"] == 0
    )
    assert s["historical_v750"]["feature_reuse_cost"]["same_batch_10x_score"] == 0
    assert s["evidence_set_methods"]["reconstructed"]
    assert not s["evidence_set_methods"]["producer_eligible"]
    assert s["optimized_residual_cost"]["reconstructed"]
    durable = s["bounded_durable_cost"]
    assert durable["reconstructed"] and not durable["producer_eligible"]
    assert durable["producer_verdict_class"] == "null"
    assert durable["completed_window_count"] == 90 and durable["completed_update_count"] == 2880
    assert durable["natural_dense_event_count"] == 0


@pytest.mark.parametrize("mode", ["rows", "receipts", "primitive", "missing", "authority"])
def test_rehashed_rejection(private, tmp_path, mode):
    """SCENARIO-VERIFY-8775-REPLAY: coordinated envelope repairs grant no authority."""
    _, original = private
    v = deepcopy(original)
    if mode == "rows":
        v["rows"][0]["producer_qualified"] = True
    elif mode == "receipts":
        v["validation_receipts"][0]["passed"] = False
    elif mode == "authority":
        v["full_task_objects"][0]["prompt"] += "forged"
    else:
        work = c.e.load(v["primitive_reference"])
        work["duration_s"] += 1
        target = tmp_path / "measurement.json"
        if mode == "primitive":
            c.e.atomic_json(target, work)
        v["primitive_reference"] = c.e.reference(target)
    v["reproducibility_checksum"] = c.e.digest(v)
    target = tmp_path / "candidate.json"
    c.e.atomic_json(target, v)
    assert cli("--cold-replay", target).returncode == 1


def test_failure_and_readiness(private, tmp_path):
    """REQ-VERIFY-8775: failed owned checks suppress readiness and terminal acceptance."""
    for args in [["--date", "123"], ["--date", "20260230"], ["--private-control"]]:
        assert cli(*args).returncode == 1
    target = tmp_path / (c.e.NAME + ".json")
    result = cli("--private-control", "--failed-child", "--output", target)
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(target.read_bytes())["verdict_class"] == "disqualified"
    path, v = private
    work = c.e.load(v["primitive_reference"])
    work["private_control"] = False
    assert c.build(work, [dict(passed=True)], path)["reconciliation_ready_score"] == 1
    assert c.build(work, [], path)["verdict_class"] == "disqualified"


def test_controls_and_saved_authority(private, tmp_path):
    """SCENARIO-VERIFY-8775-REPLAY: replay reads immutable milestone authority."""
    _, v = private
    work = c.e.load(v["primitive_reference"])
    assert all(r["passed"] for r in work["contract_mutation_rows"])
    assert c.authority(Path(work["authority_root"]))["ready"]
    moving = {str(c.e.ROOT / p) for p in [c.e.DESIGN, c.binding.ACTIVE, c.binding.STAGED]}
    assert not any(r["path"] in moving for r in work["source_refs"])


def test_pin_rejection(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8775: pin checks reject internally consistent substituted producers."""
    _, v = private
    work = c.e.load(v["primitive_reference"])
    row = deepcopy(work["slots"][0])
    row["declared_reference"]["sha256"] = "sha256:forged"
    monkeypatch.setattr(c.retained.previous, "resolve", lambda *args: deepcopy(row))
    rejected = c.slot(work["contract"]["tasks"][0], tmp_path, [])
    assert not rejected["producer_qualified"]
    assert "independent_v750_primary_pin" in rejected["authentication_errors"]


def test_conductor_and_identity_checks(tmp_path):
    """REQ-REPORT-8775: conductor comparisons and rejected task identities remain explicit."""
    tasks = c.authority(c.e.ROOT)["tasks"]
    skipped = c.slot(tasks[2], tmp_path / "skipped", [])
    assert len(skipped["conductor_pre_gate_checks"]) == 2
    assert all(r["passed"] for r in skipped["conductor_pre_gate_checks"])
    rejected = c.slot(tasks[9], tmp_path / "mismatch", [])
    check = next(r for r in rejected["source_identity_failures"] if r["field"] == "task_id")
    assert check["expected"] == tasks[9]["id"]
    assert check["observed"] == "exp8772-ordinary-update-cost"
    assert rejected["producer_terminal_authenticated"] and not rejected["producer_qualified"]


def test_owned_supervision_preserves_scientific_limits(tmp_path, monkeypatch):
    """REQ-VERIFY-8775: instrumented validation cannot change scientific deadlines."""
    monkeypatch.setattr(c, "base_child", lambda name, argv, logs, **limits: limits)
    assert c.supervised_child("tests_0", [], tmp_path, deadline=480)["deadline"] == 1200
    assert c.supervised_child("scientific_control", [], tmp_path, deadline=3000)["deadline"] == 3000
