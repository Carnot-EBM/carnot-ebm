"""REQ-REPORT-7862: dated board custody and private CLI regression tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.experiment_7862_v682_hardware_evidence import (
    EXP7847,
    INVENTORY,
    PRIOR,
    SERVICE,
    _service_fit,
    cold_reduce,
    read_evidence,
)
from scripts.experiments import experiment_7862_v682_hardware_evidence as cli

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/experiments/experiment_7862_v682_hardware_evidence.py"


def test_historical_exdev_is_retained() -> None:
    """SCENARIO-REPORT-7862-CLI: the old failed check stays a failed check."""
    old = json.loads((ROOT / "results/experiment_7847_v681_hardware_evidence.json").read_text())
    receipt = next(
        x for x in old["validation_receipts"]["checks"] if x["name"] == "affected_pytest"
    )
    assert receipt["passed"] is False
    assert sha256_file(Path(receipt["log_path"])) == receipt["log_sha256"]
    log = Path(receipt["log_path"]).read_text()
    assert "Invalid cross-device link" in log
    assert "/tmp/carnot-pytest-artifacts-" in log


def test_board_limits_and_missing_service() -> None:
    """SCENARIO-REPORT-7862-CUSTODY: old reachability remains dated and narrow."""
    result = read_evidence(ROOT, "20260929")
    assert result["experiment_id"] == 7862
    assert result["task_id"] == "exp7862-hardware-evidence"
    assert result["milestone"] == "2026.09.682"
    assert result["hardware_evidence_ready_score"] == 1
    assert result["new_device_execution_count"] == 0
    assert result["verdict_class"] == "null"
    assert result["service_fit_rows"][0]["applicability"] == "unknown"
    assert [r["board"] for r in result["board_rows"]] == ["KV260", "PolarFire", "GateMate"]
    assert result["board_rows"][0]["k_max"] == 5
    assert result["board_rows"][1]["processor_class"] == "linux_cpu"
    assert result["board_rows"][2]["blocker"] == "0xffffffff"
    assert all(not r["current_hardware_execution"] for r in result["board_rows"])
    assert result["historical_failures"]["exp7834_required_coverage_passed"] is False
    assert result["historical_failures"]["exp7847_affected_pytest_passed"] is False
    assert {r["name"] for r in result["historical_failures"]["exp7847_required_failures"]} == {
        "affected_pytest",
        "unit_coverage",
    }
    assert result["acceptance_gate_results"]["decision_benefit"] is None
    assert cold_reduce(ROOT, result)["rows_checksum"] == result["rows_checksum"]


def test_missing_source_is_terminal_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7862-CUSTODY: absence gives an exact missing operand."""
    result = read_evidence(tmp_path, "20260929")
    assert result["verdict_class"] == "blocked"
    assert result["honest_verdict"].startswith("complete_")
    assert result["hardware_evidence_ready_score"] == 0
    assert result["gate_check_summary"]
    assert result["gate_check_summary"][0]["observed"] is None
    assert result["gate_check_summary"][0]["artifact_path"]


def test_malformed_source_is_terminal_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7862-CUSTODY: malformed bytes cannot crash custody."""
    for name in (INVENTORY, PRIOR, EXP7847):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    (tmp_path / INVENTORY).write_text("{")
    result = read_evidence(tmp_path, "20260929")
    assert result["verdict_class"] == "blocked"
    assert any(c["artifact_field"] == "schema_json" for c in result["gate_check_summary"])
    (tmp_path / INVENTORY).write_text(json.dumps({"board_rows": [None, None, None]}))
    result = read_evidence(tmp_path, "20260929")
    assert result["verdict_class"] == "blocked"


def test_real_cli_success_missing_and_replay() -> None:
    """SCENARIO-REPORT-7862-CLI: exercise both CLI routes under the active guard."""
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}"}
    with tempfile.TemporaryDirectory(prefix="exp7862-e2e-", dir="/tmp") as folder:
        private = Path(folder)
        good = private / "good.json"
        bad = private / "bad.json"
        commands = [
            [
                sys.executable,
                "-u",
                str(SCRIPT),
                "--date",
                "20260929",
                "--root",
                str(ROOT),
                "--output",
                str(good),
                "--evidence-only",
            ],
            [
                sys.executable,
                "-u",
                str(SCRIPT),
                "--date",
                "20260929",
                "--root",
                str(private),
                "--output",
                str(bad),
                "--evidence-only",
            ],
            [sys.executable, "-u", str(SCRIPT), "--root", str(ROOT), "--cold-replay", str(good)],
        ]
        for command in commands:
            completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
            assert completed.returncode == 0, completed.stderr
            assert "completed_units=" in completed.stdout
        assert json.loads(good.read_text())["hardware_evidence_ready_score"] == 1
        assert json.loads(bad.read_text())["verdict_class"] == "blocked"
        assert cold_reduce(ROOT, json.loads(good.read_text()))["row_count"] == 3


def test_service_fit_is_only_a_candidate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7862-CUSTODY: matching k does not claim device work."""
    path = tmp_path / SERVICE
    path.parent.mkdir(parents=True)
    boards = [{"board": name} for name in ("KV260", "PolarFire", "GateMate")]
    path.write_text(
        json.dumps(
            {
                "experiment_id": 7861,
                "verdict_class": "null",
                "flagged_adversarial": False,
                "service_evidence_ready_score": 1,
                "component_requirements": {"model_class": "quadratic_ising", "k": 5},
            }
        )
    )
    rows, status = _service_fit(tmp_path, boards)
    assert status["qualified"] is True
    assert rows[0]["fabric_fit"] is True
    assert rows[0]["applicability"] == "candidate_only"
    assert all(not row["hardware_execution_measured"] for row in rows)
    assert rows[1]["fabric_fit"] is False
    value = json.loads(path.read_text())
    value["flagged_adversarial"] = True
    path.write_text(json.dumps(value))
    rows, status = _service_fit(tmp_path, boards)
    assert status["status"] == "disqualified"
    assert rows[0]["fabric_fit"] is False


def test_cold_replay_rejects_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7862-CLI: replay must rehash rows, gates, and logs."""
    result = read_evidence(ROOT, "20260929")
    changed = json.loads(json.dumps(result))
    changed["rows_checksum"] = "sha256:wrong"
    with pytest.raises(ValueError, match="rows_changed"):
        cold_reduce(ROOT, changed)
    changed = json.loads(json.dumps(result))
    changed["gate_check_summary"] = [{"wrong": True}]
    with pytest.raises(ValueError, match="gate_operands_changed"):
        cold_reduce(ROOT, changed)
    changed = json.loads(json.dumps(result))
    changed["validation_receipts"] = {
        "checks": [{"log_path": str(tmp_path / "gone"), "log_sha256": "sha256:wrong"}]
    }
    with pytest.raises(ValueError, match="sealed_log_changed"):
        cold_reduce(ROOT, changed)
    candidate = tmp_path / "candidate.json"
    cli.atomic_json(candidate, result)
    assert cli.main(["--root", str(ROOT), "--cold-replay", str(candidate)]) == 0


def test_manifest_and_owned_child(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7862-VALIDATION: commands and log seals are explicit."""
    commands = cli.manifest(tmp_path, tmp_path / "pending.json")
    assert {x["name"] for x in commands} == set(cli.REQUIRED) | {"repository_health_180s"}
    assert (
        next(x for x in commands if x["name"] == "repository_health_180s")["classification"]
        == "diagnostic"
    )
    assert all(
        "/tmp" in str(x["argv"])
        for x in commands
        if x["name"] in {"affected_pytest", "cli_success"}
    )
    spec = {
        "name": "worktree_imports",
        "argv": [sys.executable, "-c", "print('invalid')"],
        "deadline_s": 5,
        "classification": "required",
    }
    bad = cli.run_child(spec, tmp_path, time.monotonic(), 0)
    assert bad["resolved_imports"] == {}
    assert bad["passed"] is False
    spec["argv"] = [
        sys.executable,
        "-c",
        "import json; print(json.dumps({'resolved_imports': {'x':'/tmp/x.py'}}))",
    ]
    good = cli.run_child(spec, tmp_path, time.monotonic(), 1)
    assert good["passed"] is True
    assert sha256_file(Path(good["log_path"])) == good["log_sha256"]
    slow = {
        "name": "slow",
        "argv": [
            sys.executable,
            "-u",
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(10)",
        ],
        "deadline_s": 0.2,
        "classification": "required",
    }
    expired = cli.run_child(slow, tmp_path, time.monotonic(), 2)
    assert expired["timed_out"] is True
    assert expired["passed"] is False


def test_full_reducer_and_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7862-VALIDATION: current failures and old failures stay separate."""
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path))

    def fake_child(spec: dict, private: Path, _started: float, _units: int) -> dict:
        log = private / "fake" / f"{spec['name']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            json.dumps({"flagged_count": 0})
            if spec["name"] == "adversarial_verify"
            else json.dumps({"resolved_imports": {"x": str(ROOT / "python/x.py")}})
            if spec["name"] == "worktree_imports"
            else "ok"
        )
        return {
            **spec,
            "log_path": str(log),
            "log_sha256": sha256_file(log),
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "duration_s": 0.001,
            "resolved_imports": {"x": str(ROOT / "python/x.py")},
        }

    monkeypatch.setattr(cli, "run_child", fake_child)
    output = tmp_path / "terminal.json"
    assert cli.main(["--root", str(ROOT), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["validation_receipts"]["required_checks_passed"] is True
    assert result["verdict_class"] == "null"
    assert result["historical_failures"]["exp7847_affected_pytest_passed"] is False
    monkeypatch.setattr(cli, "run_child", lambda *_args: pytest.fail("checkpoint not resumed"))
    assert cli.main(["--root", str(ROOT), "--output", str(output)]) == 0
    checkpoint = Path(result["validation_command_manifest_path"]).with_name("completed_units.json")
    saved = json.loads(checkpoint.read_text())
    saved["checks"][0]["log_sha256"] = "sha256:wrong"
    cli.atomic_json(checkpoint, saved)
    with pytest.raises(ValueError, match="checkpoint_receipt_changed"):
        cli.main(["--root", str(ROOT), "--output", str(output)])
    shutil.rmtree(checkpoint.parent)

    def failed_child(spec: dict, private: Path, _started: float, _units: int) -> dict:
        receipt = fake_child(spec, private, _started, _units)
        if spec["name"] == "adversarial_verify":
            log = Path(receipt["log_path"])
            log.write_text("malformed")
            receipt["log_sha256"] = sha256_file(log)
        if spec["name"] == "ruff_check":
            receipt["passed"] = False
            receipt["exit_code"] = 1
        return receipt

    monkeypatch.setattr(cli, "run_child", failed_child)
    assert cli.main(["--root", str(ROOT), "--output", str(output)]) == 0
    failed = json.loads(output.read_text())
    assert failed["verdict_class"] == "disqualified"
    assert failed["flagged_adversarial"] is True
    assert failed["acceptance_gate_results"]["readiness"] == 0
    with pytest.raises(ValueError, match="worktree root"):
        cli.main(["--root", str(tmp_path)])
