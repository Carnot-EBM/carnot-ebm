"""REQ-REPORT-8390 / REQ-VERIFY-8390: qualify current direct state honestly."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

from carnot.reporting import direct_state_qualification_8390 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-VERIFY-8390: share one real three-seed crash measurement across tests."""
    if os.environ.get("CARNOT8390_WORK"):
        return json.loads(Path(os.environ["CARNOT8390_WORK"]).read_bytes())
    private = tmp_path_factory.mktemp("v723-state")
    with e.bindings():
        return e.measure(e.ROOT, private / "raw", private)


def test_crashes_and_current_authority(work):
    """REQ-VERIFY-8390: every real killed child recovers the uninterrupted state."""
    assert len(work["rows"]) == 33
    assert all(r["passed"] for r in work["rows"])
    assert work["current_authority"]["activated"]
    assert work["original_controls"]["acceptance_gates"]["exact_recovery"]
    assert all(r["actual_exit"] == -9 for r in work["rows"] if r["barrier"] != "none")
    value = e.build(work, [dict(passed=True)])
    assert value["direct_state_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["power_loss_certified"] is False
    assert value["intended_count"] == value["completed_count"]


def test_replay_and_failures(work, tmp_path):
    """SCENARIO-REPORT-8390-REPLAY: repaired summaries cannot replace primitives."""
    candidate = tmp_path / (e.NAME + ".json")
    value = e.build(work, [dict(passed=True)])
    atomic_json(candidate, value)
    assert e.replay(candidate)
    assert not e.replay(tmp_path / "absent")
    assert e.build(work, [dict(passed=False)])["verdict_class"] == "disqualified"
    blocked = deepcopy(work)
    blocked["gate_check_summary"].append(e.authority.failure(tmp_path / "absent", "input", 1, None))
    assert e.build(blocked, [dict(passed=True)])["verdict_class"] == "blocked"
    for field in ("completed_count", "direct_state_ready_score"):
        bad = deepcopy(value)
        bad[field] += 1
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(candidate, bad)
        assert not e.replay(candidate)
    atomic_json(candidate, value)
    for name, args, expected in [
        ("valid", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("error", ["--deliberate-error"], 1),
        ("date", ["--date", "20261009"], 2),
    ]:
        receipt = child(
            name,
            [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
            tmp_path / "logs",
            expected=expected,
        )
        assert receipt["passed"]


def test_fresh_state_controls(work, tmp_path):
    """SCENARIO-VERIFY-8390-CONTROLS: real children reject each invalid state."""
    receipts = e.state_controls(work, tmp_path)
    assert all(r["passed"] for r in receipts)
    assert len(receipts) == len(e.PROBES)


def test_missing_current_authority(tmp_path):
    """REQ-REPORT-8390: external absence blocks without inventing measurements."""
    with e.bindings():
        missing = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    value = e.build(missing, [dict(passed=True)])
    assert value["direct_state_ready_score"] == 0
    assert value["verdict_class"] == "blocked"
    assert value["censored_count"] == 33
    assert any(g["observed"] is None for g in value["gate_check_summary"])


def test_frozen_plan(tmp_path, monkeypatch):
    """REQ-VERIFY-8390: scoped coverage and named E2E commands precede measurement."""
    monkeypatch.setenv("COVERAGE_RCFILE", os.environ.get("COVERAGE_RCFILE", ""))
    monkeypatch.setenv("COVERAGE_FILE", os.environ.get("COVERAGE_FILE", ""))
    plan = e.plan(tmp_path)
    assert all(c["scope"] == "owned" for c in plan)
    assert any(c["name"] == "private_E2E020_direct_consumers" for c in plan)
    assert any("--fail-under=100" in c["argv"] for c in plan)


def test_real_entry_publication(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8390-REPLAY: publish through the actual owned entry flow."""
    original = e.build
    monkeypatch.setattr(e, "measure", lambda root, raw, private: deepcopy(work))
    monkeypatch.setattr(e, "plan", lambda private: [])
    output = tmp_path / (e.NAME + ".json")
    assert e.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"]
    assert original(work, [dict(passed=True)])["direct_state_ready_score"] == 1


def test_missing_probe_argument():
    """SCENARIO-VERIFY-8390-CONTROLS: malformed real probe argv fails explicitly."""
    with pytest.raises(SystemExit):
        e.main(["--probe"])


def test_authenticated_operand_failures(tmp_path):
    """REQ-REPORT-8390: changed original control bytes cannot grant readiness."""
    original = tmp_path / e.ORIGINAL
    atomic_json(original, dict(acceptance_gates=dict(exact_recovery=False, owned_validation=False)))
    with e.bindings():
        work = e.measure(tmp_path, tmp_path / "raw", tmp_path)
    checks = {g["check"] for g in work["gate_check_summary"]}
    assert "input_hash" in checks
    assert "original_state_controls" in checks


def test_identity_checksum_and_authority_controls(work, tmp_path):
    """SCENARIO-REPORT-8390-REPLAY: fresh authority defeats a repaired work summary."""
    candidate = tmp_path / "candidate.json"
    valid = e.build(work, [dict(passed=True)])
    atomic_json(candidate, dict(valid, experiment_id=8376))
    assert not e.replay(candidate)
    atomic_json(candidate, dict(valid, reproducibility_checksum="wrong"))
    assert not e.replay(candidate)
    changed = deepcopy(work)
    changed["current_authority"]["canonical_tasks_sha256"] = "changed"
    changed["work_path"] = str(tmp_path / "changed-work.json")
    atomic_json(Path(changed["work_path"]), changed)
    atomic_json(candidate, e.build(changed, [dict(passed=True)]))
    assert not e.replay(candidate)
