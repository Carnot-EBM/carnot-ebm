"""REQ-REPORT-8804 / REQ-VERIFY-8804: private custody and terminal publication."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from carnot.reporting import causal_template_streams_8804 as r
from carnot.reporting.current_work_receipt import atomic_json


def test_gate_absence_zero_rejection_and_preflight(tmp_path: Path):
    """SCENARIO-VERIFY-8804-QUALIFICATION: absence cannot become a measured zero."""
    path = tmp_path / "upstream.json"
    assert r.upstream_gate(path)["observed"] == "missing_file"
    atomic_json(path, {"experiment_id": 8802})
    assert r.upstream_gate(path)["observed"] == "missing_field"
    atomic_json(path, {"experiment_id": 8802, "memory_execution_ready_score": 0})
    assert r.upstream_gate(path)["observed"] == 0
    atomic_json(
        path,
        {"experiment_id": 8802, "memory_execution_ready_score": 1, "verdict_class": "disqualified"},
    )
    assert r.upstream_gate(path)["observed"] == "rejected_producer"
    path.write_text("invalid")
    assert r.upstream_gate(path)["observed"] == "rejected_producer"
    pre = r.preflight(tmp_path / "pre", intended_count=340)
    assert pre["passed"], {
        k: pre[k] for k in ["checked", "measured_cheap_unit_s", "projected_work_s"]
    }
    assert pre["validation_reserve_s"] == 900
    assert pre["sources"] and pre["measured_cheap_unit_s"] > 0
    assert pre["parallel_workers"] == 30
    assert pre["measured_parallel_wave_s"] > 0
    assert pre["projected_work_s"] == pre["measured_parallel_wave_s"] * 340 / 30


def test_blocked_census_and_cold_cli(tmp_path: Path, monkeypatch):
    """SCENARIO-REPORT-8804-REPLAY: terminal blocks retain all slots and replay cold."""
    monkeypatch.setattr(r, "UPSTREAM", str(tmp_path / "missing.json"))
    value = r.run(tmp_path / "raw", "20261104")
    assert value["verdict_class"] == "blocked"
    assert value["completed_count"] == 0 and value["censored_count"] == 30600
    assert len(value["per_query_rows"]) == 14400 and len(value["retention_rows"]) == 16200
    assert not value["numerical_training_calls"]
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    coverage_dir = Path(env.get("CARNOT_8804_COVERAGE_DIRECTORY", str(tmp_path)))
    argv = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--rcfile=/dev/null",
        "--parallel-mode",
        "--data-file=" + str(coverage_dir / ".coverage.child"),
        str(r.ROOT / r.CLI),
        "--replay",
        str(candidate),
    ]
    child = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0 and '"passed": true' in child.stdout, child.stdout + child.stderr
    value["memory_rows_ready_score"] = 1
    atomic_json(candidate, value)
    assert not r.replay(candidate)["passed"]
    assert r.main(["--replay", str(candidate)]) == 1
    atomic_json(candidate, {})
    assert not r.replay(candidate)["passed"]
    candidate.write_text("broken")
    assert not r.replay(candidate)["passed"]


def test_measured_replay_and_rehashed_tampering(tmp_path: Path):
    """REQ-REPORT-8804: independent seal operands reject changed loss and source claims."""
    frozen = r.s.manifest()
    frozen["streams"] = frozen["streams"][:1]
    frozen["seeds"] = [11]
    value = r.run(tmp_path / "raw", "20261104", frozen=frozen)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]
    changed = deepcopy(value)
    changed["protocol_scope"] = "production"
    atomic_json(candidate, changed)
    assert not r.replay(candidate)["passed"]
    from coverage import Coverage

    cov = Coverage(data_file=str(tmp_path / "incomplete.coverage"), config_file=False)
    cov.start()
    r.upstream_gate(tmp_path / "not_present.json")
    cov.stop()
    cov.save()
    changed = deepcopy(value)
    changed["owned_statement_coverage"] = r.coverage_summary(
        tmp_path / "incomplete.coverage", r.OWNED
    )
    changed["owned_statement_coverage"]["complete"] = True
    changed["acceptance_gates"] = {k: True for k in changed["acceptance_gates"]}
    changed["memory_rows_ready_score"] = changed["learning_cost_ready_score"] = 1
    changed["required_checks_passed"] = True
    changed["validation_receipts"] = [dict(passed=True)]
    atomic_json(candidate, changed)
    assert not r.replay(candidate)["passed"]
    for field, supplied in [
        ("completed_count", -1),
        ("numerical_training_calls", -1),
        ("learning_cost_ready_score", 1),
    ]:
        changed = deepcopy(value)
        changed[field] = supplied
        atomic_json(candidate, changed)
        assert not r.replay(candidate)["passed"]
    work_path = Path(value["raw_shard_hashes"][0]["path"])
    work = json.loads(work_path.read_bytes())
    work["per_query_rows"][0]["metric"] += 0.1
    atomic_json(work_path, work)
    value["raw_shard_hashes"][0] = r.reference(work_path)
    value["per_query_rows"] = work["per_query_rows"]
    value["rows"][0] = work["per_query_rows"][0]
    value["reproducibility_checksum"] = r.canonical_hash(work)
    atomic_json(candidate, value)
    assert not r.replay(candidate)["passed"]


def test_validation_plan_and_failure_readiness(tmp_path: Path, monkeypatch):
    """REQ-VERIFY-8804: actual commands are planned; failed owned checks disqualify."""
    calls = []

    def child(name, argv, logs, **kwargs):
        calls.append((name, argv))
        if name == "owned_pytest":
            directory = Path(argv[1].split("=", 1)[1])
            (directory / ".coverage.child.fixture").write_bytes(b"private SQLite copy fixture")
        return {"name": name, "passed": True}

    monkeypatch.setattr(r, "child", child)
    monkeypatch.setattr(
        r, "Coverage", lambda **kw: SimpleNamespace(combine=lambda **kw: None, save=lambda: None)
    )
    monkeypatch.setattr(r, "coverage_summary", lambda *a: {"complete": True, "percent": 100})
    checked = r.validate(tmp_path / "validation")
    assert checked["passed"]
    pytest_argv = next(argv for name, argv in calls if name == "owned_pytest")
    assert Path(pytest_argv[1].split("=", 1)[1]).is_absolute()
    assert {
        "owned_pytest",
        "private_E2E_018_021",
        "ruff_check",
        "ruff_format",
        "strict_mypy",
        "spec_coverage",
    } <= {name for name, _ in calls}
    assert r.terminal_checks(tmp_path / "candidate", tmp_path)["passed"]
    monkeypatch.setattr(r, "validate", lambda raw: checked)
    monkeypatch.setattr(r, "publish_primary", lambda out, val, validator: validator(out))
    value = {
        "acceptance_gates": {
            "preconditions": True,
            "rows": True,
            "cost": True,
            "owned_validation": False,
        },
        "verdict_class": "disqualified",
        "phase_spans": [],
        "duration_s": 0,
    }
    r.qualify_and_publish(value, tmp_path / "publish", tmp_path / "out")
    assert value["memory_rows_ready_score"] == value["learning_cost_ready_score"] == 1
    assert value["verdict_class"] == "null"
    checked["passed"] = False
    r.qualify_and_publish(value, tmp_path / "failed", tmp_path / "out")
    assert value["verdict_class"] == "disqualified"
    assert value["memory_rows_ready_score"] == value["learning_cost_ready_score"] == 0
    value["verdict_class"] = "blocked"
    value["acceptance_gates"]["preconditions"] = False
    r.qualify_and_publish(value, tmp_path / "blocked", tmp_path / "out")
    assert value["verdict_class"] == "blocked"

    def reject(*args):
        raise ValueError("candidate_rejected")

    monkeypatch.setattr(r, "publish_primary", reject)
    with pytest.raises(ValueError, match="candidate_rejected"):
        r.qualify_and_publish(value, tmp_path / "rejected", tmp_path / "out")
    assert (tmp_path / "rejected/rejected_candidate.json").is_file()


def test_main_dispatch_and_date(tmp_path: Path, monkeypatch):
    """REQ-REPORT-8804: CLI limits dates and delegates all scientific work."""
    with pytest.raises(SystemExit):
        r.main(["--date", "20261103"])
    monkeypatch.setattr(r, "run", lambda *a: {})
    monkeypatch.setattr(r, "qualify_and_publish", lambda *a: None)
    assert r.main(["--raw", str(tmp_path / "raw"), "--output", str(tmp_path / "out")]) == 0
    assert r.main([]) == 0
