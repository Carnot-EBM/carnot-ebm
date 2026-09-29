"""REQ-REPORT-7876: board custody and optional CPU workload attachment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.experiment_7876_v683_hardware_evidence import (
    _load,
    _prior_checks,
    _workload,
    cold_reduce,
    read_evidence,
)
from scripts.experiments import experiment_7876_v683_hardware_evidence as cli

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/experiments/experiment_7876_v683_hardware_evidence.py"
PRIOR = "results/experiment_7862_v682_hardware_evidence.json"
WORKLOAD = "results/experiment_7875_v683_service_cost.json"


def _fixture(tmp_path: Path) -> Path:
    """Copy the exact prior closure so mutations cannot change shared history."""
    prior = json.loads((ROOT / PRIOR).read_text())
    for name in [PRIOR, *prior["source_artifact_hashes"]]:
        source = ROOT / name
        if source.is_file():
            destination = tmp_path / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    return tmp_path


def test_scenario_report_7876_custody_and_replay() -> None:
    """SCENARIO-REPORT-7876-CUSTODY: old evidence stays dated and narrow."""
    result = read_evidence(ROOT, "20260929")
    assert result["experiment_id"] == 7876
    assert result["task_id"] == "exp7876-hardware-evidence"
    assert result["milestone"] == "2026.09.683"
    assert result["honest_verdict"] == "complete_null_historical_board_scope"
    assert result["hardware_evidence_ready_score"] == 1
    assert result["workload_attachment_available"] is False
    assert result["current_device_execution_count"] == 0
    assert result["hardware_speedup_claimed"] is False
    assert result["inference_substrate_class"] == "no_model_load"
    assert result["MODEL_SPECS"] == []
    assert result["target_model"] == "none (no pretrained model)"
    assert [row["board"] for row in result["hardware_rows"]] == ["KV260", "PolarFire", "GateMate"]
    assert result["hardware_rows"][0]["k_max"] == 5
    assert result["hardware_rows"][1]["processor_class"] == "linux_cpu"
    assert result["hardware_rows"][2]["blocker"] == "0xffffffff"
    assert all(row["claim_class"] == "historical" for row in result["hardware_rows"])
    assert result["historical_failures"]["exp7847_affected_pytest_passed"] is False
    assert cold_reduce(ROOT, result)["rows_checksum"] == result["rows_checksum"]


def test_scenario_report_7876_missing_and_changed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7876-CUSTODY: absence and byte drift have exact operands."""
    missing = read_evidence(tmp_path, "20260929")
    assert missing["verdict_class"] == "blocked"
    assert missing["gate_check_summary"][0]["artifact_field"] == "exists"
    root = _fixture(tmp_path)
    prior = json.loads((root / PRIOR).read_text())
    raw = root / "results/raw/experiment_7231/polarfire_dispatch.json"
    raw.write_bytes(raw.read_bytes() + b" ")
    changed = read_evidence(root, "20260929")
    assert changed["hardware_evidence_ready_score"] == 0
    assert any(
        c["artifact_path"].endswith("polarfire_dispatch.json")
        and c["artifact_field"] == "sha256"
        and c["observed"] == sha256_file(raw)
        for c in changed["gate_check_summary"]
    )
    assert prior["source_artifact_hashes"]["results/raw/experiment_7231/polarfire_dispatch.json"][
        "sha256"
    ] != sha256_file(raw)


def test_scenario_report_7876_stale_and_inflated(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7876-CUSTODY: later dates and new-device aliases fail."""
    root = _fixture(tmp_path)
    path = root / PRIOR
    prior = json.loads(path.read_text())
    prior["board_rows"][0]["last_authenticated_evidence_date"] = "20260930"
    prior["current_device_execution_count"] = 1
    prior["board_rows"][1]["current_hardware_execution"] = True
    path.write_text(json.dumps(prior))
    result = read_evidence(root, "20260929")
    assert result["verdict_class"] == "blocked"
    fields = {c["artifact_field"] for c in result["gate_check_summary"]}
    assert "board_rows.KV260.last_authenticated_evidence_date" in fields
    assert "current_device_execution_count" in fields
    assert "board_rows.PolarFire.current_hardware_execution" in fields


def test_scenario_report_7876_workload_requires_qualified_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7876-WORKLOAD: CPU rows attach only from a valid producer."""
    root = _fixture(tmp_path)
    path = root / WORKLOAD
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": 7875,
        "task_id": "exp7875-service-cost",
        "run_date": "20260929",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "service_cost_ready_score": 1,
        "validation_receipts": {"required_checks_passed": True},
        "workload_shapes": [{"operation": "verify", "memory_bytes": 64, "duration_ms": 2.0}],
    }
    path.write_text(json.dumps(payload))
    valid = read_evidence(root, "20260929")
    assert valid["workload_attachment_available"] is True
    assert valid["workload_feasibility_rows"]
    assert all(not row["hardware_execution_measured"] for row in valid["workload_feasibility_rows"])
    payload["flagged_adversarial"] = True
    path.write_text(json.dumps(payload))
    invalid = read_evidence(root, "20260929")
    assert invalid["workload_attachment_available"] is False
    assert invalid["hardware_evidence_ready_score"] == 1


def test_scenario_report_7876_cli_success_failure_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7876-CLI: exercise real script paths in private output."""
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}"}
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    for root, output in ((ROOT, good), (tmp_path, bad)):
        completed = subprocess.run(
            [
                sys.executable,
                "-u",
                str(SCRIPT),
                "--date",
                "20260929",
                "--root",
                str(root),
                "--output",
                str(output),
                "--evidence-only",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "completed_units=" in completed.stdout
    assert json.loads(good.read_text())["verdict_class"] == "null"
    assert json.loads(bad.read_text())["verdict_class"] == "blocked"
    replay = subprocess.run(
        [sys.executable, "-u", str(SCRIPT), "--root", str(ROOT), "--cold-replay", str(good)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert replay.returncode == 0, replay.stderr
    changed = json.loads(good.read_text())
    changed["hardware_rows"][0]["k_max"] = 6
    with pytest.raises(ValueError, match="rows_changed"):
        cold_reduce(ROOT, changed)


def test_scenario_report_7876_malformed_prior_and_replay_guards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7876-CUSTODY: malformed data and altered logs fail closed."""
    root = _fixture(tmp_path)
    path = root / PRIOR
    path.write_text("{")
    value, digest = _load(root, PRIOR)
    assert value is None and digest
    checks = _prior_checks(root, value, digest, "20260929")
    assert any(c["artifact_field"] == "schema_json" and not c["passed"] for c in checks)
    result = read_evidence(root, "20260929")
    assert result["verdict_class"] == "blocked"
    path.write_text((ROOT / PRIOR).read_text())
    result = read_evidence(root, "20260929")
    changed = json.loads(json.dumps(result))
    changed["gate_check_summary"] = [{"wrong": True}]
    with pytest.raises(ValueError, match="gate_operands_changed"):
        cold_reduce(root, changed)
    changed = json.loads(json.dumps(result))
    changed["terminal_receipt_hashes"][result["hardware_rows"][0]["source_path"]] = "bad"
    with pytest.raises(ValueError, match="terminal_receipt_changed"):
        cold_reduce(root, changed)
    changed = json.loads(json.dumps(result))
    changed["validation_receipts"] = {
        "checks": [{"log_path": str(tmp_path / "gone.log"), "log_sha256": "bad"}]
    }
    with pytest.raises(ValueError, match="sealed_log_changed"):
        cold_reduce(root, changed)


def test_scenario_report_7876_bad_dates_and_workload_schema(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7876-WORKLOAD: malformed optional rows cannot attach."""
    prior = json.loads((ROOT / PRIOR).read_text())
    prior["board_rows"][0]["last_authenticated_evidence_date"] = "bad"
    prior["validation_receipts"]["checks"].append(None)
    checks = _prior_checks(ROOT, prior, "sha256:fixture", "20260929")
    assert any(c["artifact_field"].endswith("last_authenticated_evidence_date") and not c["passed"] for c in checks)
    assert _workload(tmp_path, "20260929")[1] == []
    path = tmp_path / WORKLOAD
    path.parent.mkdir(parents=True)
    path.write_text("[]")
    assert _workload(tmp_path, "20260929")[0]["exposure_status"] == "disqualified"


def test_scenario_report_7876_owned_child_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7876-CLI: owned exits and sealed logs keep actual status."""
    commands = cli.manifest(tmp_path, tmp_path / "candidate.json")
    assert {x["name"] for x in commands} >= {
        "full_pytest", "affected_pytest", "changed_coverage", "adversarial_verify", "strict_rows"
    }
    spec = {"name": "worktree_imports", "argv": [sys.executable, "-c", "print('bad')"],
            "deadline_s": 5, "classification": "required"}
    bad = cli.run_child(spec, tmp_path, 0.0, 0)
    assert bad["passed"] is False and bad["resolved_imports"] == {}
    spec["argv"] = [sys.executable, "-c", "import json;print(json.dumps({'resolved_imports':{'x':'/tmp/x.py'}}))"]
    good = cli.run_child(spec, tmp_path, 0.0, 1)
    assert good["passed"] is True
    assert sha256_file(Path(good["log_path"])) == good["log_sha256"]
    spec = {"name": "slow", "argv": [sys.executable, "-c", "import time;time.sleep(5)"],
            "deadline_s": 0.1, "classification": "required"}
    expired = cli.run_child(spec, tmp_path, 0.0, 2)
    assert expired["timed_out"] is True and expired["passed"] is False


def test_scenario_report_7876_full_driver_preserves_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7876-CLI: an inherited required timeout still disqualifies."""
    prior = json.loads((ROOT / cli.OUTPUT).read_text())
    old_full = next(x for x in prior["validation_receipts"]["checks"] if x["name"] == "full_pytest")
    assert old_full["classification"] == "required" and old_full["timed_out"] is True
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path))

    def fake_child(spec: dict, private: Path, _started: float, _units: int) -> dict:
        log = private / "fixture" / f"{spec['name']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            json.dumps({"flagged_count": 0}) if spec["name"] == "adversarial_verify"
            else json.dumps({"resolved_imports": {"carnot.reporting.experiment_7876_v683_hardware_evidence": str(ROOT / cli.CODE[0])}})
            if spec["name"] == "worktree_imports" else "ok"
        )
        return {**spec, "exit_code": 0, "timed_out": False, "passed": True,
                "duration_s": 0.001, "log_path": str(log), "log_sha256": sha256_file(log),
                "resolved_imports": {"carnot.reporting.experiment_7876_v683_hardware_evidence": str(ROOT / cli.CODE[0])}}

    monkeypatch.setattr(cli, "run_child", fake_child)
    output = tmp_path / "terminal.json"
    assert cli.main(["--root", str(ROOT), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    full = next(x for x in result["validation_receipts"]["checks"] if x["name"] == "full_pytest")
    assert full["passed"] is False and full["log_sha256"] == old_full["log_sha256"]
    assert result["verdict_class"] == "disqualified"
    assert result["hardware_evidence_ready_score"] == 0
    assert result["acceptance_gate_results"]["readiness"] == 0
    monkeypatch.setattr(cli, "run_child", lambda *_args: pytest.fail("checkpoint should resume"))
    assert cli.main(["--root", str(ROOT), "--output", str(output)]) == 0
    checkpoint = Path(result["validation_command_manifest_path"]).with_name("completed_units.json")
    saved = json.loads(checkpoint.read_text())
    saved["checks"][0]["log_sha256"] = "sha256:wrong"
    cli.atomic_json(checkpoint, saved)
    with pytest.raises(ValueError, match="checkpoint_receipt_changed"):
        cli.main(["--root", str(ROOT), "--output", str(output)])
    with pytest.raises(ValueError, match="worktree root"):
        cli.main(["--root", str(tmp_path), "--output", str(output)])
