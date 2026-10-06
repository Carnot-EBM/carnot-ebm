"""REQ-VERIFY-8193 / REQ-REPORT-8193: real children qualify mechanics only."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import coverage
import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import learning_qualification_8193 as e


def test_real_covered_child_exit_resume_and_combine(tmp_path):
    """SCENARIO-VERIFY-8193-CHILD: cleanup cannot stand in for active.save()."""
    private = tmp_path / "private"
    private.mkdir()
    rows, labels = e.legacy.fixture("learnable")
    atomic_json(private / "restart-input.json", dict(rows=rows, labels=labels, seed=101))
    specs = e.restart_specs(tmp_path, private)
    receipts = []
    for spec in specs:
        receipts.append(e.run_check(e.ROOT, spec, private, tmp_path / "logs", heartbeat_s=20))
    assert [r["actual_exit"] for r in receipts] == [73, 0]
    measured = e.child_coverage(tmp_path)
    assert measured["passed"] and {291, 292} <= set(measured["executed_lines"])
    restored = json.loads((tmp_path / "restart/final.json").read_text())
    baseline = e.legacy.run(rows, labels, 101)
    assert e.legacy.stable(restored) == e.legacy.stable(baseline)
    cov = coverage.Coverage(data_file=measured["data_path"], config_file=False)
    cov.load()
    assert {291, 292} <= set(cov.get_data().lines(str(e.ROOT / e.legacy.MODULE)))


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8193: fixtures never write into historical results."""
    raw = tmp_path_factory.mktemp("8193-evidence")
    return e.measure(e.ROOT, raw, fixture=True), raw


def test_qualification_and_cold_replay(work, tmp_path):
    """SCENARIO-VERIFY-8193-REPLAY: independent reduction rejects new hashes."""
    measured, raw = work
    value = e.build(measured, raw, [dict(passed=True, normal_exit=True)], fixture=True)
    assert value["verdict_class"] == "circular_positive"
    assert value["calibrated_memory_ready_score"] == value["stream_input_ready_score"] == 1
    assert value["measured_child_coverage"]["passed"]
    assert value["prior_failure"]["actual_exit"] == 2
    assert value["prior_failure"]["verdict_class"] == "disqualified"
    assert value["protocol_sha256"] == e.legacy.PROTOCOL_HASH
    assert value["MODEL_SPECS"] == value["call_ledger"] == []
    assert value["independent_count"] == value["generalized_learning_benefit_score"] == 0
    assert all(r["passed"] for r in value["fixture_rows"] if r["case"] == "learnable")
    path = tmp_path / "value.json"
    atomic_json(path, value)
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    if env.get("COVERAGE_RCFILE"):
        env["COVERAGE_PROCESS_START"] = env["COVERAGE_RCFILE"]
    print("[test8193] before_positive_cold_child completed=0 pending=1", flush=True)
    result = subprocess.run(
        [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
        cwd=tmp_path,
        env=env,
        timeout=240,
    )
    print("[test8193] after_positive_cold_child completed=1 pending=0", flush=True)
    assert result.returncode == 0
    assert not e.replay(tmp_path / "absent.json")
    for field in ["completed_count", "calibrated_memory_ready_score", "measured_child_coverage"]:
        changed = dict(value)
        if field == "measured_child_coverage":
            changed[field] = dict(value[field])
            changed[field]["executed_lines"] = []
        else:
            changed[field] += 1
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(path, changed)
        assert not e.replay(path)
    bad = e.build(measured, raw, [dict(passed=False, normal_exit=True)], fixture=True)
    assert bad["verdict_class"] == "disqualified" and bad["calibrated_memory_ready_score"] == 0


def test_external_missing_and_tamper(tmp_path):
    """SCENARIO-REPORT-8193-CLI: failed operands must retain actual values."""
    work = e.measure(tmp_path, tmp_path / "missing", fixture=True)
    value = e.build(work, tmp_path, [dict(passed=True, normal_exit=True)], fixture=True)
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert value["gate_check_summary"][-1]["observed"] is False
    upstream = json.loads((e.ROOT / e.legacy.UPSTREAM).read_text())
    upstream["learning_audit_ready_score"] = 0
    atomic_json(tmp_path / e.legacy.UPSTREAM, upstream)
    assert not e.measure(tmp_path, tmp_path / "tamper", fixture=True)["input_ready"]


def test_coverage_receipt_and_forged_child_evidence(work, tmp_path):
    """REQ-VERIFY-8193: sealed measurements outrank rehashed coverage claims."""
    from carnot.reporting.current_work_receipt import sha256_file

    measured, raw = work
    receipt = dict(
        name="coverage_json",
        passed=True,
        normal_exit=True,
        argv=["-o", measured["measured_child_coverage"]["report_path"], "--data-file=x"],
    )
    value = e.build(measured, raw, [receipt], fixture=True)
    assert value["coverage_statement_counts"]
    path = tmp_path / "forged.json"

    def rejected(changed):
        changed.pop("reproducibility_checksum", None)
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(path, changed)
        assert not e.replay(path)

    bad = dict(value)
    bad["legacy_replay_sha256"] = "sha256:wrong"
    rejected(bad)
    bad = dict(value)
    bad["experiment_id"] = 0
    rejected(bad)
    bad = dict(value)
    bad["raw_shard_hashes"] = deepcopy(value["raw_shard_hashes"])
    bad["raw_shard_hashes"][0]["sha256"] = "sha256:wrong"
    rejected(bad)
    old = json.loads(Path(value["legacy_replay_path"]).read_text())
    old["measured_child_coverage"]["executed_lines"] = []
    old.pop("reproducibility_checksum")
    old["reproducibility_checksum"] = canonical_hash(old)
    forged_legacy = tmp_path / "legacy.json"
    atomic_json(forged_legacy, old)
    bad = dict(value)
    bad["legacy_replay_path"] = str(forged_legacy)
    bad["legacy_replay_sha256"] = sha256_file(forged_legacy)
    bad["measured_child_coverage"] = dict(value["measured_child_coverage"], executed_lines=[])
    rejected(bad)


def test_wait_counts_and_single_health_custody(tmp_path, monkeypatch, capsys):
    """REQ-REPORT-8193: recovery preserves one normally completed health receipt."""
    import time

    from carnot.reporting import learning_qualification_execution_8193 as runner
    from carnot.reporting.current_work_receipt import sha256_file

    for phase in ["before_subprocess", "subprocess_outstanding", "after_subprocess"]:
        runner.child_progress("real_check", phase, time.monotonic())
    printed = capsys.readouterr().out
    assert "completed=0 pending=1" in printed and "completed=1 pending=0" in printed
    log = tmp_path / "health.log"
    log.write_text("normally exited repository diagnostic\n")
    receipt = dict(
        argv=["pytest", "tests/python", "-q"],
        actual_exit=1,
        passed=False,
        log_path=str(log),
        log_sha256=sha256_file(log),
    )
    work = tmp_path / "measurement.json"
    atomic_json(work, dict(global_health=receipt))
    monkeypatch.setenv("CARNOT_8193_HEALTH_WORK", str(work))
    spec = dict(name="repository_full_suite", argv=receipt["argv"])
    assert runner.check(e.ROOT, spec, tmp_path, tmp_path)["reused"]
    atomic_json(work, {})
    monkeypatch.setattr(
        runner.time, "sleep", lambda _: atomic_json(work, dict(global_health=receipt))
    )
    assert runner.check(e.ROOT, spec, tmp_path, tmp_path)["reused"]
    log.write_text("changed receipt\n")
    with pytest.raises(ValueError, match="health_custody"):
        runner.check(e.ROOT, spec, tmp_path, tmp_path)
    monkeypatch.setattr(runner, "BASE_CHECK", lambda *args, **kwargs: dict(passed=True))
    assert runner.check(e.ROOT, dict(name="other"), tmp_path, tmp_path)["passed"]


def test_private_cli_and_manifest(tmp_path, work):
    """SCENARIO-REPORT-8193-CLI: direct E2E success/block/tamper/replay."""
    from carnot.reporting import learning_qualification_execution_8193 as runner

    specs = runner.manifest(tmp_path, tmp_path / "candidate.json")
    static = [s for s in specs["commands"] if s["name"].startswith(("ruff", "strict_mypy"))]
    assert all(not any("::" in arg for arg in s["argv"]) for s in static)
    assert any(s["name"] == "E2E016_success" for s in specs["commands"])
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    if env.get("COVERAGE_RCFILE"):
        env["COVERAGE_PROCESS_START"] = env["COVERAGE_RCFILE"]
    cli = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]

    def child(args, expected):
        print("[test8193] before_subprocess", args, flush=True)
        with subprocess.Popen(cli + args, cwd=tmp_path, env=env) as process:
            for _ in range(8):
                try:
                    actual = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print("[test8193] child_wait completed=0 pending=1", flush=True)
            else:
                process.kill()
                actual = process.wait(timeout=30)
                pytest.fail("private child exceeded240 seconds")
        print("[test8193] after_subprocess", actual, flush=True)
        assert actual == expected

    path = tmp_path / (e.NAME + ".json")
    measured, raw = work
    atomic_json(path, e.build(measured, raw, [dict(passed=True, normal_exit=True)], fixture=True))
    child(["--cold-replay", str(path)], 0)
    value = json.loads(path.read_text())
    value["rows"][0]["numerator"] += 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    child(["--cold-replay", str(path)], 1)
    child(
        [
            "--fixture-output",
            str(tmp_path / "blocked" / path.name),
            "--root",
            str(tmp_path / "absent"),
        ],
        0,
    )
    child(["--fixture-output", str(e.ROOT / "results/forbidden.json")], 2)
    child(["--worker-output", str(tmp_path / "success/measurement.json")], 0)
    child(
        [
            "--worker-output",
            str(tmp_path / "worker/measurement.json"),
            "--root",
            str(tmp_path / "absent"),
        ],
        0,
    )
