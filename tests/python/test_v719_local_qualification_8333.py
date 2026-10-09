"""REQ-REPORT-8333 and REQ-VERIFY-8333: primitive evidence governs readiness."""

from copy import deepcopy
import json
from pathlib import Path
import os
import sys

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import local_update_isolation_8306 as k
from carnot.verify import local_qualification_8333 as n
from carnot.reporting import local_qualification_8333 as e
from carnot.reporting import local_qualification_execution_8333 as runner


def primitives(count=48):
    """Construct controls from the frozen events, without replacing source observations."""
    protocol = k.manifest()
    protocol["trajectories"] = protocol["trajectories"][:count]
    states = []
    for trajectory in protocol["trajectories"]:
        arms = {}
        for arm in k.ARMS:
            state = k.initial(trajectory, arm)
            for slot in range(72):
                k.issue(state, slot)
                k.release(trajectory, state, slot, arm)
            arms[arm] = state
        states.append(dict(trajectory_id=trajectory["id"], arms=arms))
    return dict(states=states), protocol


def test_dense_reference_and_false_zero():
    """SCENARIO-VERIFY-8333-NUMERIC: each primitive update has an independent oracle."""
    work, protocol = primitives()
    proof = n.audit(work, protocol)
    assert proof["passed"] and proof["recomputed"]
    assert proof["release_count"] == 48 * 64 * 2
    assert proof["probability_error_max"] <= 1e-10
    assert n.controls(work, protocol)["deliberate_error_rejected"]
    bad = deepcopy(work)
    bad["states"][0]["arms"]["indexed"]["releases"][0]["coefficients"][2] += 0.125
    assert not n.audit(bad, protocol)["passed"]
    assert not n.audit(bad, protocol, claimed_zero=True)["recomputed"]
    assert n.derivatives()["passed"]
    assert n.feedback_control()["passed"]


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Preserve one measured private trajectory for replay and finding controls."""
    raw = tmp_path_factory.mktemp("local8333") / "raw"
    return e.measure(e.ROOT, raw, fixture=True), raw


def test_authentication_and_finding_consumer(measured, tmp_path):
    """SCENARIO-VERIFY-8333-FINDINGS: raw exit one is qualified against exact bytes."""
    work, raw = measured
    assert not work["failures"]
    assert work["original"]["verdict_class"] == "disqualified"
    assert work["consumer_controls"]["passed"]
    assert work["original_finding"]["passed"]
    assert any(f["kind"] == "IMPLAUSIBLE_PERFECT" for f in work["original_finding"]["findings"])
    blocked = e.measure(tmp_path / "absent", tmp_path / "blocked", fixture=True)
    value = e.build(blocked, tmp_path / "blocked", [dict(passed=True)])
    assert value["verdict_class"] == "blocked" and value["local_kernel_ready_score"] == 0
    assert not e.replay(tmp_path / "absent.json")
    assert (raw / "protocol.json").is_file()


def test_replay_and_readiness(measured, tmp_path):
    """SCENARIO-REPORT-8333-CLI: rehashing cannot repair false causal claims."""
    work, raw = measured
    value = e.build(work, raw, [dict(passed=True)])
    assert value["local_kernel_ready_score"] == 0
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    for field in ["local_kernel_ready_score", "completed_count", "dense_sparse_error_max"]:
        bad = deepcopy(value)
        bad[field] = 999
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(candidate, bad)
        assert not e.replay(candidate)
    atomic_json(candidate, {})
    assert not e.replay(candidate)
    assert e.build(work, raw, [dict(passed=False)])["verdict_class"] == "disqualified"


def test_actual_cli_recovery_and_errors(tmp_path):
    """SCENARIO-VERIFY-8333-RECOVERY: actual killed children save measured coverage."""
    protocol = tmp_path / "protocol.json"
    atomic_json(protocol, k.manifest())
    checkpoint = tmp_path / "checkpoint"
    prefix = [sys.executable]
    if os.environ.get("COVERAGE_RCFILE"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + os.environ["COVERAGE_RCFILE"]]
    base = [
        *prefix,
        str(e.ROOT / e.CLI),
        "--worker",
        str(protocol),
        "--checkpoint",
        str(checkpoint),
        "--cohort-count",
        "1",
    ]
    from carnot.reporting.v709_execution import child

    for crash, expected in [(31, -9), (63, -9), (-1, 0)]:
        receipt = child(
            "crash_" + str(crash),
            base + ["--crash", str(crash)],
            tmp_path / "logs",
            deadline=60,
            expected=expected,
            heartbeat=20,
        )
        assert receipt["passed"]
    trajectory = k.manifest()["trajectories"][0]
    state = k.load(trajectory, checkpoint / (trajectory["id"] + ".json"))
    expected, _ = primitives(1)
    assert n.durable(state) == n.durable(expected["states"][0]["arms"]["indexed"])
    with pytest.raises(SystemExit):
        runner.main(["--worker", str(protocol)])
    with pytest.raises(SystemExit):
        runner.main(["--date", "20000101"])
    assert runner.main(["--cold-replay", str(tmp_path / "missing")]) == 1


def test_rehashed_primitive_faults(measured, tmp_path):
    """SCENARIO-REPORT-8333-CLI: refreshed hashes never authenticate false arithmetic."""
    work, source = measured
    for mutation in [
        "state",
        "recovered",
        "numeric",
        "derivative",
        "receipt",
        "finding",
        "protocol",
        "checksum",
        "input_hash",
    ]:
        raw = tmp_path / mutation
        raw.mkdir()
        changed = deepcopy(work)
        plan = json.loads((source / "protocol.json").read_bytes())
        if mutation == "state":
            changed["states"][0]["arms"]["indexed"]["coefficients"][2] += 0.125
        elif mutation == "recovered":
            changed["recovered_states"][0]["index"]["2"] = []
        elif mutation == "numeric":
            changed["numeric"]["release_count"] = 0
        elif mutation == "derivative":
            changed["derivatives"]["temperature_derivative_error_max"] = 2
        elif mutation == "receipt":
            changed["crashes"][0]["stderr_sha256"] = "wrong"
        elif mutation == "finding":
            changed["original_finding"]["passed"] = False
        elif mutation == "protocol":
            plan["trajectories"][0]["events"][0]["y"] = 7
        atomic_json(raw / "protocol.json", plan)
        changed["protocol_reference"] = e.reference(raw / "protocol.json")
        atomic_json(raw / "measurement.json", changed)
        value = e.build(changed, raw, [dict(passed=True)])
        if mutation == "input_hash":
            value["source_artifact_hashes"][0]["sha256"] = "wrong"
            value.pop("reproducibility_checksum")
            value["reproducibility_checksum"] = canonical_hash(value)
        if mutation == "checksum":
            value["completed_count"] += 1
        candidate = raw / "candidate.json"
        atomic_json(candidate, value)
        assert not e.replay(candidate), mutation


def test_private_publication_and_worker_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8333-CLI: actual private publication binds every check."""
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--root", str(tmp_path / "absent"), "--fixture-output", str(output)]) == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    worker_output = tmp_path / "worker" / "measurement.json"
    assert (
        runner.main(["--root", str(tmp_path / "absent"), "--worker-output", str(worker_output)])
        == 0
    )
    with pytest.raises(SystemExit):
        runner.main(["--fixture-output", str(e.ROOT / "results" / (e.NAME + ".json"))])
    original_manifest = runner.manifest

    def bounded(private, candidate):
        plan = original_manifest(private, candidate)
        plan["commands"] = [
            dict(
                name="actual_owned_child_control",
                argv=["/bin/true"],
                deadline_s=10,
                expected_exit=0,
            )
        ]
        atomic_json(private / "coverage.json", dict(totals=dict(percent_covered=0), files={}))
        return plan

    monkeypatch.setattr(runner, "manifest", bounded)
    second = tmp_path / "normal" / (e.NAME + ".json")
    assert runner.main(["--root", str(tmp_path / "absent"), "--output", str(second)]) == 0
    value = json.loads(second.read_bytes())
    assert value["verdict_class"] == "blocked"
    raw = Path(value["measurement_reference"]["path"]).parent
    actual_check = runner.check
    calls = []

    def fail_once(spec, logs):
        receipt = actual_check(spec, logs)
        if not calls:
            receipt["passed"] = False
        calls.append(spec["name"])
        return receipt

    monkeypatch.setattr(runner, "check", fail_once)
    runner.publish(value, second, raw)
    assert json.loads(second.read_bytes())["verdict_class"] == "disqualified"
    conflicting = dict(value, experiment_id=99)
    with pytest.raises(ValueError, match="producer_identity"):
        runner.publish(conflicting, second, raw)


def test_coverage_gate_and_causal_issue_control(measured, tmp_path):
    """REQ-VERIFY-8333: zero coverage and issued-coefficient drift cannot pass."""
    work, raw = measured
    changed = deepcopy(work)
    coverage = tmp_path / "coverage.json"
    atomic_json(
        coverage, dict(totals=dict(percent_covered=100), files={path: {} for path in e.OWNED})
    )
    changed["owned_coverage_reference"] = e.reference(coverage)
    qualified = e.build(changed, raw, [dict(passed=True)])
    assert qualified["local_kernel_ready_score"] == 1
    assert qualified["flagged_adversarial"] is False
    assert qualified["adversarial_findings"][0]["kind"] == "IMPLAUSIBLE_PERFECT"
    from scripts.conductor_gates import _is_quarantined

    assert not _is_quarantined(qualified)
    plan = json.loads((raw / "protocol.json").read_bytes())
    changed["states"][0]["arms"]["indexed"]["issues"][0]["coefficients"][2] += 0.125
    assert not n.audit(changed, plan)["passed"]
    event = dict(id="zero", x=[0.0] * 5, y=0.5)
    assert n.step([0.0] * 34, event, []) == [0.0] * 34


def test_inactive_authority_blocks(measured, tmp_path, monkeypatch):
    """REQ-REPORT-8333: mismatched activation is an exact external authority block."""
    work, _ = measured
    authority = dict(work["authority"], activated=False)
    monkeypatch.setattr(e.current, "authority", lambda *args: authority)
    monkeypatch.setattr(e.custody, "authenticate", lambda *args: {})
    blocked = e.measure(e.ROOT, tmp_path / "inactive", fixture=True)
    assert any(row["artifact_field"] == "current_task_authority" for row in blocked["failures"])
    assert (
        e.build(blocked, tmp_path / "inactive", [dict(passed=True)])["verdict_class"] == "blocked"
    )
