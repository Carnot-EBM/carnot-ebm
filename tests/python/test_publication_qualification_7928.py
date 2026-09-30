"""Qualify fixture, cold CLI and terminal reduction for REQ-REPORT-7928."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import publication_qualification_7928 as qualification
from carnot.reporting.current_work_receipt import atomic_json
from scripts.experiments import experiment_7928_v688_primary_publication as cli


def test_matrix_cold_reduction_and_tampering(tmp_path):
    """SCENARIO-REPORT-7928-1: every mutation retains its expected outcome."""
    result = qualification.fixture(tmp_path / "fixture.json")
    assert result["passed"] and len(result["rows"]) >= 10
    assert qualification.replay(tmp_path / "fixture.json")["passed"]
    result["rows"][0]["passed"] = False
    atomic_json(tmp_path / "fixture.json", result)
    assert not qualification.replay(tmp_path / "fixture.json")["passed"]


def test_real_cli_fixture_cold_and_failure(tmp_path):
    """SCENARIO-REPORT-7928-2: child interpreters execute actual script paths."""
    script = qualification.ROOT / qualification.CLI
    target = tmp_path / "fixture.json"
    prefix = [sys.executable]
    if os.environ.get("CARNOT7928_COVERAGE") == "1":
        prefix += [
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--data-file=" + os.environ["COVERAGE_FILE"],
            "--include=" + qualification.INCLUDES,
        ]
    argv = [*prefix, str(script), "--date", "20260930"]
    for mode in ["--fixture-e2e", "--cold-replay", "--terminal-recheck"]:
        result = subprocess.run(
            [*argv, mode, str(target)], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(target.read_text())
    value["rows"][0]["passed"] = False
    atomic_json(target, value)
    result = subprocess.run(
        [*argv, "--cold-replay", str(target)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 1


def test_main_routes_and_wrong_date(monkeypatch, tmp_path):
    """REQ-REPORT-7928: default execution and invalid dates stay explicit."""
    monkeypatch.setattr(cli, "qualify", lambda date: 0)
    assert cli.main(["--date", "20260930"]) == 0
    with pytest.raises(ValueError, match="run_date_mismatch"):
        cli.main(["--date", "20260929"])
    assert cli.main(["--fixture-e2e", str(tmp_path / "fixture.json")]) == 0
    assert cli.main(["--cold-replay", str(tmp_path / "fixture.json")]) == 0
    assert cli.main(["--terminal-recheck", str(tmp_path / "fixture.json")]) == 0


def test_history_and_frozen_validation(tmp_path):
    """SCENARIO-REPORT-7928-2: fixed custody includes preserved historical failures."""
    sources, history, failures = qualification.history(qualification.ROOT / "results")
    assert len(sources) == 5 and len(history) == 2 and failures
    manifest, commands = qualification.freeze(tmp_path, tmp_path / "raw")
    value = json.loads(manifest.read_text())
    assert value["coverage_includes"] == qualification.INCLUDES
    assert all(row["expected_exit"] == 0 for row in value["commands"])
    dated = [row for row in commands if row.name.startswith("e2e_016")]
    assert len(dated) == 2 and all("20260929" in row.argv for row in dated)
    with pytest.raises(FileNotFoundError):
        qualification.history(tmp_path / "missing")


def test_qualify_success_owned_failure_and_external_block(monkeypatch, tmp_path):
    """SCENARIO-REPORT-7928-3: owned failures differ from external incompleteness."""
    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    monkeypatch.setattr(qualification, "dependency_hashes", lambda: {"code": "sha256:fixture"})
    source = {"path": "historical", "sha256": "sha256:fixture", "role": "historical"}
    monkeypatch.setattr(qualification, "history", lambda p: ([source], [], []))
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, {})
    monkeypatch.setattr(qualification, "freeze", lambda p, r: (manifest, []))
    monkeypatch.setattr(qualification, "run_checks", lambda c, r, p: [])
    monkeypatch.setattr(
        qualification,
        "coverage_counts",
        lambda p: {
            name: {"num_statements": 1, "covered_lines": 1, "missing_lines": 0}
            for name in qualification.OWNED
        },
    )
    monkeypatch.setattr(
        qualification,
        "terminal_checks",
        lambda p, r: {"passed": True, "flagged_adversarial": False, "receipts": []},
    )
    assert qualification.qualify("20260930") == 0
    output = tmp_path / "results/experiment_7928_v688_primary_publication.json"
    result = json.loads(output.read_text())
    assert result["artifact_resolution_ready_score"] == 1
    assert qualification.replay(output)["passed"]
    monkeypatch.setattr(
        qualification,
        "run_checks",
        lambda c, r, p: [{"passed": False, "name": "owned", "exit_code": 1}],
    )
    assert qualification.qualify("20260930") == 1
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(
        qualification, "history", lambda p: (_ for _ in ()).throw(FileNotFoundError("upstream"))
    )
    assert qualification.qualify("20260930") == 1
    assert json.loads(output.read_text())["verdict_class"] == "blocked"


def test_coverage_missing_and_terminal_flags(monkeypatch, tmp_path):
    """REQ-REPORT-7928: missing measurement cannot qualify and flags are read."""
    assert qualification.coverage_counts(tmp_path) == {}
    atomic_json(
        tmp_path / "coverage.json", {"files": {"example.py": {"summary": {"num_statements": 3}}}}
    )
    assert qualification.coverage_counts(tmp_path)["example.py"]["num_statements"] == 3

    def run(root, commands, **kwargs):
        log = tmp_path / "adversarial.log"
        log.write_text('{"flagged_count": 1}')
        return [{"name": "adversarial", "passed": False, "exit_code": 1, "log_path": str(log)}]

    monkeypatch.setattr(qualification, "run_commands", run)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, {"rows": []})
    assert qualification.terminal_checks(candidate, tmp_path)["flagged_adversarial"]


def test_real_bounded_child_log_sealing(tmp_path):
    """SCENARIO-REPORT-7928-2: seal child bytes only after a real exit."""
    spec = qualification.CommandSpec(
        "real_child", (sys.executable, "-c", "print('finished')"), "owned", 30
    )
    rows = qualification.run_checks([spec], tmp_path / "raw", tmp_path / "private")
    assert rows[0]["passed"] and Path(rows[0]["log_path"]).is_file()
    assert rows[0]["argv"] == list(spec.argv)


def test_relative_log_resolution(monkeypatch, tmp_path):
    """REQ-REPORT-7928: repository-relative receipts retain their actual bytes."""
    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    (tmp_path / "child.log").write_text("exited")
    monkeypatch.setattr(
        qualification,
        "run_commands",
        lambda *a, **kw: [{"name": "relative", "log_path": "child.log", "exit_code": 0}],
    )
    spec = qualification.CommandSpec("relative", ("unused",), "owned", 30)
    assert qualification.run_checks([spec], tmp_path / "raw", tmp_path)[0]["actual_exit"] == 0


def test_terminal_failure_disqualifies_then_rechecks(monkeypatch, tmp_path):
    """SCENARIO-REPORT-7928-3: required terminal failure cannot open readiness."""
    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    monkeypatch.setattr(qualification, "dependency_hashes", lambda: {})
    monkeypatch.setattr(qualification, "history", lambda p: ([], [], []))
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, {})
    monkeypatch.setattr(qualification, "freeze", lambda p, r: (manifest, []))
    monkeypatch.setattr(qualification, "run_checks", lambda c, r, p: [])
    monkeypatch.setattr(qualification, "coverage_counts", lambda p: {})
    calls = []

    def terminal(path, raw):
        calls.append(path.read_bytes())
        return {"passed": len(calls) > 1, "flagged_adversarial": False, "receipts": []}

    monkeypatch.setattr(qualification, "terminal_checks", terminal)
    assert qualification.qualify("20260930") == 1
    assert len(calls) == 2 and calls[0] != calls[1]


def test_historical_log_drift_is_external_block(monkeypatch):
    """SCENARIO-REPORT-7928-3: historical failed logs require current custody."""
    original = qualification.sha256_file
    monkeypatch.setattr(
        qualification,
        "sha256_file",
        lambda p: "sha256:changed" if p.suffix == ".log" else original(p),
    )
    with pytest.raises(FileNotFoundError, match="historical_log_hash_mismatch"):
        qualification.history(qualification.ROOT / "results")
