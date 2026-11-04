"""REQ-REPORT-8809 / REQ-VERIFY-8809: real overflow and causal numerical learning."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import delayed_feedback_capacity_8809 as r


def test_roster_and_overflow_controls(tmp_path):
    """SCENARIO-REPORT-8809-OVERFLOW: real eviction precedes interpretation."""
    plan = r.manifest()
    assert (plan["streams"], plan["queries"], len(plan["seeds"])) == (30, 64, 3)
    assert len(r.census(plan, "pending")) == 86400
    head = r.s.heads(json.loads((r.ROOT / r.UPSTREAM).read_bytes()), [11])[0][0]
    cells = [r.cell(0, 11, d, c, p, head, 64) for d, c, p in r.arms()]
    rows = [q for c in cells for q in c["per_query_rows"]]
    assert len(rows) == 960
    controls = r.controls(cells)
    assert all(controls.values()), controls
    for cell in cells:
        assert cell["queue_cost_rows"][0]["pending_after_drain"] == 0
        assert len(cell["admission_rows"]) == 64
        assert len(cell["release_rows"]) == 64
        assert all(e["time"] >= e["issued_time"] + cell["delay"] for e in cell["release_rows"])
        dropped = {e["event_id"] for e in cell["feedback_loss_rows"]}
        assert dropped.isdisjoint(e["event_id"] for e in cell["applied_rows"])
        assert all(
            q["paid_checks"] == 5 and q["sealed_ns"] <= q["scored_ns"]
            for q in cell["per_query_rows"]
        )
    report = r.summarize(cells)
    assert len(report["simultaneous_intervals"]) == 6
    assert all(
        i["confidence"] == pytest.approx(1 - 0.05 / 6) for i in report["simultaneous_intervals"]
    )
    assert not report["useful_pending_feedback_policy_score"]
    replicated = [dict(c, cluster_id=f"stream-{i:02}") for i in range(30) for c in cells]
    assert all(
        q["independent_count"] == 30 for q in r.summarize(replicated)["simultaneous_intervals"]
    )


def test_stream_cluster_intervals_and_empty_overflow():
    """SCENARIO-REPORT-8809-INFERENCE: seeds add no independent observations."""
    interval = r.interval([-0.1] * 30, 0.05 / 6)
    assert interval["lower"] == interval["upper"] == pytest.approx(-0.1)
    assert interval["independent_count"] == 30
    assert r.interval([0.0], 0.05 / 6)["lower"] is None
    assert not any(r.controls([]).values())


def test_blocked_census_and_external_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8809-CUSTODY: absence yields a terminal block with no work."""
    monkeypatch.setattr(r, "UPSTREAM", str(tmp_path / "absent.json"))
    value = r.run(tmp_path / "raw", "20261104")
    assert value["verdict_class"] == "blocked"
    assert value["censored_count"] == value["intended_count"] == 86400
    assert not value["completed_count"] and not value["numerical_training_calls"]
    path = tmp_path / "candidate.json"
    r.atomic_json(path, value)
    assert r.replay(path)["passed"]
    drifted = deepcopy(value)
    drifted["code_config_hashes"][0]["sha256"] = "sha256:changed"
    r.atomic_json(path, drifted)
    assert r.replay(path)["reason"] == "source_hash_drift"
    r.atomic_json(path, value)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    covdir = Path(env.get("CARNOT_8804_COVERAGE_DIRECTORY", str(tmp_path)))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=/dev/null",
            "--parallel-mode",
            "--data-file=" + str(covdir / ".coverage.child"),
            str(r.ROOT / r.CLI),
            "--replay",
            str(path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value["capacity_rows_ready_score"] = 1
    r.atomic_json(path, value)
    assert not r.replay(path)["passed"]
    assert r.main(["--replay", str(path)]) == 1
    path.write_text("{}")
    assert not r.replay(path)["passed"]


def test_private_measurement_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8809-CAUSAL-QUEUE: replay rebuilds operands, not reported losses."""
    value = r.run(tmp_path / "raw", "20261104", plan=r.manifest(1, 12, [11]))
    assert value["completed_count"] == 180
    path = tmp_path / "candidate.json"
    r.atomic_json(path, value)
    assert r.replay(path)["passed"]
    from coverage import Coverage

    cov = Coverage(data_file=str(tmp_path / "incomplete.coverage"), config_file=False)
    cov.start()
    r.interval([0.0], 0.05)
    cov.stop()
    cov.save()
    claim = deepcopy(value)
    claim["owned_statement_coverage"] = r.base.coverage_summary(
        tmp_path / "incomplete.coverage", r.OWNED
    )
    claim["owned_statement_coverage"]["complete"] = True
    claim["acceptance_gates"] = dict.fromkeys(claim["acceptance_gates"], True)
    claim["capacity_rows_ready_score"] = 1
    claim["required_checks_passed"] = True
    stdout = tmp_path / "check.stdout"
    stderr = tmp_path / "check.stderr"
    stdout.write_text("private receipt fixture")
    stderr.write_text("")
    receipt = dict(
        name="owned_pytest",
        passed=True,
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        stdout_sha256=r.sha256_file(stdout),
        stderr_sha256=r.sha256_file(stderr),
    )
    r.atomic_json(stdout.with_suffix(".receipt.json"), receipt)
    claim["validation_receipts"] = [receipt]
    r.atomic_json(path, claim)
    assert not r.replay(path)["passed"]
    for key, changed in [
        ("completed_count", -1),
        ("numerical_training_calls", -1),
        ("simultaneous_intervals", []),
        ("independent_count", 3),
    ]:
        altered = deepcopy(value)
        altered[key] = changed
        r.atomic_json(path, altered)
        assert not r.replay(path)["passed"]
    primitive = Path(value["raw_shard_hashes"][0]["path"])
    work = json.loads(primitive.read_bytes())
    work["cells"][0]["per_query_rows"][0]["metric"] += 0.1
    r.atomic_json(primitive, work)
    value["raw_shard_hashes"][0] = r.reference(primitive)
    value["reproducibility_checksum"] = r.canonical_hash(work)
    value["per_query_rows"][0]["metric"] += 0.1
    value["rows"][0]["metric"] += 0.1
    r.atomic_json(path, value)
    assert not r.replay(path)["passed"]


def test_owned_failures_and_cli(tmp_path, monkeypatch):
    """REQ-VERIFY-8809: failed validation suppresses readiness before publication."""
    with pytest.raises(SystemExit):
        r.main(["--date", "20261103"])
    monkeypatch.setattr(
        r, "validate", lambda raw: dict(passed=False, receipts=[], coverage={"complete": False})
    )
    monkeypatch.setattr(r, "publish_primary", lambda out, val, validator: {"passed": True})
    value = r.run(tmp_path / "raw", "20261104", plan=r.manifest(1, 4, [11]))
    r.qualify(value, tmp_path / "raw", tmp_path / "out")
    assert value["verdict_class"] == "disqualified" and value["capacity_rows_ready_score"] == 0
    value["verdict_class"] = "blocked"
    r.qualify(value, tmp_path / "raw", tmp_path / "out")
    assert value["verdict_class"] == "disqualified"
    monkeypatch.setattr(
        r, "validate", lambda raw: dict(passed=True, receipts=[], coverage={"complete": True})
    )
    r.qualify(value, tmp_path / "raw", tmp_path / "out")
    assert value["verdict_class"] == "null" and value["capacity_rows_ready_score"] == 1
    value["useful_pending_feedback_policy_score"] = 1
    r.qualify(value, tmp_path / "raw", tmp_path / "out")
    assert value["verdict_class"] == "circular_positive"
    value["acceptance_gates"]["preconditions"] = False
    r.qualify(value, tmp_path / "raw", tmp_path / "out")
    assert value["verdict_class"] == "blocked"

    def reject(*args):
        raise ValueError("candidate_rejected")

    monkeypatch.setattr(r, "publish_primary", reject)
    with pytest.raises(ValueError, match="candidate_rejected"):
        r.qualify(value, tmp_path / "raw", tmp_path / "out")
    assert (tmp_path / "raw/rejected_candidate.json").is_file()
    monkeypatch.setattr(r, "run", lambda *a: value)
    monkeypatch.setattr(r, "qualify", lambda *a: None)
    assert r.main(["--raw", str(tmp_path / "other")]) == 0
    assert r.main([]) == 0
