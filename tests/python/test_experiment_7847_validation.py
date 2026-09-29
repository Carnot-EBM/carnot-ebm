"""Validation custody tests for REQ-REPORT-7847 and SCENARIO-REPORT-7847-CLI."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time

import pytest

from scripts.experiments import experiment_7847_v681_hardware_evidence as cli
from carnot.reporting.experiment_7847_v681_hardware_evidence import read_evidence
from test_experiment_7847_v681_hardware_evidence import fixture_root

ROOT = Path(__file__).resolve().parents[2]


def test_manifest_is_frozen_and_worktree_scoped(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7847-CLI: no broad test is a required command."""
    commands = cli.manifest(ROOT, tmp_path, tmp_path / "candidate.json")
    assert {x["name"] for x in commands} == set(cli.REQUIRED) | {
        "coverage_combine",
        "repository_health_180s",
    }
    assert (
        next(x for x in commands if x["name"] == "repository_health_180s")["classification"]
        == "diagnostic"
    )
    pytest_cmd = next(x for x in commands if x["name"] == "affected_pytest")
    assert "tests/python" not in pytest_cmd["argv"]
    assert all(x["deadline_s"] <= 180 for x in commands)


def test_owned_child_seals_closed_output(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7847-CLI: log identity is recorded only after exit."""
    spec = {
        "name": "small",
        "argv": [sys.executable, "-c", "print('done')"],
        "deadline_s": 5,
        "classification": "required",
    }
    receipt = cli.run_child(ROOT, spec, tmp_path / "logs", time.monotonic(), 0)
    assert receipt["passed"] is True
    assert Path(receipt["log_path"]).read_text().strip() == "done"
    assert cli.sha256_file(Path(receipt["log_path"])) == receipt["log_sha256"]
    bad = {**spec, "name": "worktree_imports"}
    invalid = cli.run_child(ROOT, bad, tmp_path / "logs", time.monotonic(), 1)
    assert invalid["passed"] is False
    assert invalid["resolved_imports"] == {}
    valid = {
        **bad,
        "argv": [
            sys.executable,
            "-c",
            "import json; print(json.dumps({'resolved_imports': {'x': '/worktree/x.py'}}))",
        ],
    }
    imported = cli.run_child(ROOT, valid, tmp_path / "logs", time.monotonic(), 2)
    assert imported["passed"] is True
    assert imported["resolved_imports"]["x"] == "/worktree/x.py"


def test_owned_child_deadline_kills_only_that_child(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7847-CLI: a stuck child has a bounded terminal receipt."""
    spec = {
        "name": "slow",
        "argv": [
            sys.executable,
            "-u",
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(60)",
        ],
        "deadline_s": 0.3,
        "classification": "required",
    }
    receipt = cli.run_child(ROOT, spec, tmp_path / "logs", time.monotonic(), 0)
    assert receipt["passed"] is False
    assert receipt["timed_out"] is True
    assert receipt["exit_code"] != 0
    assert "ready" in Path(receipt["log_path"]).read_text()


def test_direct_main_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7847-CLI: fresh reduction runs without child dispatch."""
    root = fixture_root(tmp_path / "source")
    candidate = tmp_path / "candidate.json"
    cli.atomic_json(candidate, read_evidence(root, "20260929"))
    assert cli.main(["--root", str(root), "--cold-replay", str(candidate)]) == 0


@pytest.mark.parametrize("failed_name", [None, "ruff_check", "bad_adversarial"])
def test_full_reducer_keeps_required_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_name: str | None,
) -> None:
    """SCENARIO-REPORT-7847-CLI: failed required checks disqualify readiness."""
    root = fixture_root(tmp_path / "source")
    for name in (*cli.CODE, *cli.DEPENDENCIES, *cli.TESTS):
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    monkeypatch.setattr(cli, "ROOT", root)

    def fake_child(_root: Path, spec: dict, logs: Path, _started: float, _units: int) -> dict:
        log = logs / spec["name"] / "closed.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "malformed"
            if spec["name"] == "adversarial_verify" and failed_name == "bad_adversarial"
            else json.dumps({"flagged_count": 0})
            if spec["name"] == "adversarial_verify"
            else "closed"
        )
        return {
            **spec,
            "log_path": str(log),
            "log_sha256": cli.sha256_file(log),
            "exit_code": 1 if spec["name"] == failed_name else 0,
            "passed": spec["name"] != failed_name,
            "timed_out": False,
            "duration_s": 0.001,
        }

    monkeypatch.setattr(cli, "run_child", fake_child)
    output = tmp_path / "terminal.json"
    assert cli.main(["--root", str(root), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["experiment_id"] == 7847
    assert result["hardware_inventory_ready_score"] == 1
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert result["historical_failures"]["exp7834_required_coverage_passed"] is False
    if failed_name is None:
        assert result["verdict_class"] == "blocked"
        assert result["validation_receipts"]["required_checks_passed"] is True
        monkeypatch.setattr(
            cli, "run_child", lambda *_args: pytest.fail("checkpoint was not resumed")
        )
        assert cli.main(["--root", str(root), "--output", str(output)]) == 0
        checkpoint = Path(result["validation_command_manifest_path"]).with_name(
            "completed_units.json"
        )
        saved = json.loads(checkpoint.read_text())
        saved["checks"][0]["log_sha256"] = "sha256:wrong"
        cli.atomic_json(checkpoint, saved)
        with pytest.raises(ValueError, match="checkpoint_receipt_changed"):
            cli.main(["--root", str(root), "--output", str(output)])
    else:
        assert result["verdict_class"] == "disqualified"
        assert result["honest_verdict"] == "complete_disqualified_required_checks"


def test_full_runner_rejects_foreign_root(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7847-CLI: private E2E output cannot dispatch checks."""
    root = fixture_root(tmp_path)
    with pytest.raises(ValueError, match="worktree root"):
        cli.main(["--root", str(root)])
