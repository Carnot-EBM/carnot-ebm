"""REQ-VERIFY-8664: real private CLI paths and corrected adapter boundaries."""

import json
import os
from pathlib import Path
import subprocess
import sys

from carnot.reporting import arc_reset_runner_8664 as runner
from carnot.reporting import arc_reset_transfer_8664 as e
from carnot.reporting.current_work_receipt import atomic_json


def test_private_cli(tmp_path):
    """SCENARIO-VERIFY-8664-REPLAY: execute from private scratch without PYTHONPATH."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)

    def run(*args):
        return subprocess.run(
            [sys.executable, str(e.ROOT / e.CLI), *map(str, args)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=150,
        )

    assert run("--date", "invalid").returncode == 2
    assert run("--cold-replay", tmp_path / "missing").returncode == 1
    assert run("--private-control", "--root", tmp_path).returncode == 2
    for failed in (False, True):
        output = tmp_path / str(failed) / (e.NAME + ".json")
        args = [
            "--date",
            "20261019",
            "--private-control",
            "--root",
            tmp_path / "absent",
            "--output",
            output,
        ]
        if failed:
            args.append("--owned-failure")
        result = run(*args)
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert run("--cold-replay", output).returncode == 0
        value["completed_count"] = 999
        atomic_json(output, value)
        assert run("--cold-replay", output).returncode == 1


def test_current_adapter_fixture(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8664-TRANSFER: real wrapper fixtures retain callback identities."""
    from carnot.reporting.arc_observer_fixtures_8624 import frame

    class Env:
        def reset(self):
            return frame("navigation", 1, "cutoff")

        def step(self, *args, **kwargs):
            return self.reset()

    class SDK:
        def make(self, *args, **kwargs):
            return Env()

        def open_scorecard(self):
            return "private_fixture"

    unit = dict(e.units()[1], game="fixture-unknown", max_actions=3)
    value = runner.collect(unit, SDK(), tmp_path)
    assert value["recorder_errors"] == value["recorder_error_metadata"] == []
    assert value["reset_provenance"]["callback_sequence_rows"]
    assert value["llm_calls"] == 0
    from carnot.agentic import arc_reset_provenance as reset

    def reject(value):
        raise TypeError("bounded recorder diagnostic")

    monkeypatch.setattr(reset, "public_hash", reject)
    failed = runner.collect(unit, SDK(), tmp_path / "recorder_failure")
    assert failed["recorder_error_metadata"]
    assert all(
        r["error"] == "TypeError" and r["site"]["function"] == "reject"
        for r in failed["recorder_error_metadata"]
    )


def test_entry_orchestration(tmp_path, monkeypatch):
    """REQ-VERIFY-8664: production entry binds only this panel's validation."""
    monkeypatch.setattr(runner.prior, "main", lambda argv: int("--date" not in argv))
    assert runner.main([]) == 0
    assert runner.main(["--date", "20261019"]) == 0
    monkeypatch.setattr(sys, "argv", [e.CLI, "--date", "20261019"])
    assert runner.main() == 0


def test_episode_corrected_binding(tmp_path, monkeypatch):
    """REQ-VERIFY-8664: the episode entry actually uses the corrected collector."""
    from carnot.reporting import arc_supervisor_live_panel_8384 as sdk
    from carnot.reporting.arc_observer_fixtures_8624 import frame

    class Env:
        def reset(self):
            return frame("navigation", 1, "cutoff")

        def step(self, *args, **kwargs):
            return self.reset()

    class SDK:
        def make(self, *args, **kwargs):
            return Env()

        def open_scorecard(self):
            return "private_fixture"

    monkeypatch.setattr(sdk, "sdk", SDK)
    unit = dict(e.units()[1], game="fixture-unknown", max_actions=3, directory=str(tmp_path))
    path = tmp_path / "unit.json"
    atomic_json(path, unit)
    assert runner.episode(path) == 0
    value = json.loads((tmp_path / "transfer_stream.json").read_bytes())
    assert value["recorder_error_metadata"] == []
    assert value["reset_observations"][0]["public_observation"]["pixels"]
    assert value["sdk_identity"]["official"] is False


def test_measure_success_binding(tmp_path, monkeypatch):
    """REQ-VERIFY-8664: the measurement preserves exact upstream checks and identity."""
    monkeypatch.setattr(
        runner, "BASE_MEASURE", lambda *a: dict(failures=[], source_artifact_hashes={})
    )
    work = runner.measure(e.ROOT, tmp_path / "raw", tmp_path / "private", "20261019")
    assert all(row["passed"] for row in work["upstream_gate_checks"])
    assert work["adapter_binding"]["original_observer_sha256"] == e.OBSERVER_PIN


def test_validation_roles_and_raw_custody(tmp_path, monkeypatch):
    """REQ-VERIFY-8664: historical consumers cannot contribute producer coverage."""
    import sqlite3

    calls = []
    private = tmp_path / "private"

    def child(name, argv, logs, **kwargs):
        calls.append((name, argv))
        if name.startswith("historical_"):
            with sqlite3.connect(private / name / ".coverage.child") as db:
                db.execute("create table receipt (value integer)")
        return dict(name=name, passed=True)

    def validate(raw, scratch):
        return [
            runner.prior.inherited.child(n, [sys.executable, "-m", "pytest"], raw / "validation")
            for n in ("historical_0_0", "owned_tests", "ruff_check")
        ]

    monkeypatch.setattr(runner.prior.inherited, "child", child)
    monkeypatch.setattr(runner, "BASE_VALIDATE", validate)
    receipts = runner.validate_checks(tmp_path / "raw", private)
    assert receipts[0]["artifact_hashes"]
    assert "coverage" in calls[0][1] and any("junitxml" in arg for arg in calls[0][1])
    assert any("junitxml" in arg for arg in calls[1][1])
    assert receipts[2]["passed"]
