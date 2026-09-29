"""Direct evidence and CLI checks for REQ-REPORT-7847."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

from carnot.reporting.experiment_7847_v681_hardware_evidence import (
    cold_reduce,
    read_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = "results/experiment_7820_v679_hardware_evidence.json"
PRIOR = "results/experiment_7834_v680_hardware_evidence.json"
SERVICE = "results/experiment_7846_v681_service_cost.json"


def fixture_root(tmp_path: Path) -> Path:
    """Copy only qualified old bytes so a test can mutate private evidence."""
    old = json.loads((ROOT / INVENTORY).read_text())
    paths = [INVENTORY, PRIOR]
    paths.extend(row["source_path"] for row in old["board_rows"])
    paths.extend(
        [
            "results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json",
            "results/raw/experiment_7231/polarfire_dispatch.json",
            "results/raw/experiment_7751_v674_hardware_continuity/rows.json",
            "results/raw/experiment_7779_v676_hardware_evidence/rows.json",
        ]
    )
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


def test_missing_service_is_terminal_and_preserves_boards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7847-BLOCKED: absence is a complete external block."""
    result = read_evidence(fixture_root(tmp_path), "20260929")
    assert result["experiment_id"] == 7847
    assert result["task_id"] == "exp7847-hardware-evidence"
    assert result["honest_verdict"] == "complete_blocked_missing_service_evidence"
    assert result["verdict_class"] == "blocked"
    assert result["hardware_inventory_ready_score"] == 1
    assert [row["board"] for row in result["board_rows"]] == ["KV260", "PolarFire", "GateMate"]
    assert result["gate_check_summary"][0]["artifact_field"] == "service_evidence_ready_score"
    assert result["gate_check_summary"][0]["observed"] is None
    assert all(
        row["whole_service_upper_bound"] is None for row in result["accelerator_opportunity_rows"]
    )
    assert result["historical_failures"]["exp7834_required_coverage_passed"] is False
    assert result["MODEL_SPECS"] == []
    assert cold_reduce(tmp_path, result)["rows_checksum"] == result["rows_checksum"]


def test_raw_transcript_mutation_blocks_board_readiness(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7847-CUSTODY: raw bytes must match their old receipt."""
    root = fixture_root(tmp_path)
    path = root / "results/raw/experiment_7231/polarfire_dispatch.json"
    path.write_bytes(path.read_bytes() + b"\n")
    result = read_evidence(root, "20260929")
    assert result["hardware_inventory_ready_score"] == 0
    assert result["verdict_class"] == "blocked"
    assert any(x["artifact_field"] == "raw_sha256" for x in result["gate_check_summary"])


def test_wrong_service_identity_cannot_open_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7847-SERVICE: an integer ID is distinct from a task slug."""
    root = fixture_root(tmp_path)
    path = root / SERVICE
    path.write_text(
        json.dumps(
            {
                "experiment_id": "exp7846-service-cost",
                "task_id": "exp7846-service-cost",
                "service_evidence_ready_score": 1,
                "verdict_class": "null",
                "flagged_adversarial": False,
                "stage_times_ms": {"whole_service": 10, "host_stage": 2},
            }
        )
    )
    result = read_evidence(root, "20260929")
    assert result["service_check"]["qualified"] is False
    assert any(x["artifact_field"] == "experiment_id" for x in result["gate_check_summary"])
    assert result["accelerator_opportunity_rows"][0]["whole_service_upper_bound"] is None


def test_real_cli_private_output_and_cold_replay() -> None:
    """SCENARIO-REPORT-7847-CLI: execute the real entrypoint in a private root."""
    with tempfile.TemporaryDirectory(prefix="exp7847-cli-") as private:
        base = Path(private)
        root = fixture_root(base / "sources")
        output = base / "private" / "candidate.json"
        script = ROOT / "scripts/experiments/experiment_7847_v681_hardware_evidence.py"
        env = {**os.environ, "PYTHONPATH": str(ROOT / "python") + ":" + str(ROOT)}
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                str(script),
                "--date",
                "20260929",
                "--root",
                str(root),
                "--output",
                str(output),
                "--evidence-only",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "phase=preconditions" in result.stdout
        assert output.is_file()
        replay = subprocess.run(
            [sys.executable, "-u", str(script), "--root", str(root), "--cold-replay", str(output)],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert replay.returncode == 0, replay.stderr


def test_qualified_current_service_has_only_optimistic_bound(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7847-SERVICE: current host time alone opens a bound."""
    root = fixture_root(tmp_path)
    (root / SERVICE).write_text(
        json.dumps(
            {
                "experiment_id": 7846,
                "task_id": "exp7846-service-cost",
                "milestone": "2026.09.681",
                "run_date": "20260929",
                "service_evidence_ready_score": 1,
                "verdict_class": "null",
                "flagged_adversarial": False,
                "stage_times_ms": {"whole_service": 10.0, "host_stage": 2.0},
                "transfer_bytes": {"feature": 128},
                "setup_ms": {"feature": 0.5},
                "update_bytes": {"feature": 16},
            }
        )
    )
    result = read_evidence(root, "20260929")
    assert result["service_check"]["qualified"] is True
    assert result["gate_check_summary"] == []
    assert result["accelerator_opportunity_rows"][0]["whole_service_upper_bound"] == 1.25
    assert result["accelerator_opportunity_rows"][0]["transfer_bytes"] == 128
    assert result["accelerator_opportunity_rows"][-1]["whole_service_upper_bound"] is None
    assert result["verdict_class"] == "null"


def test_missing_board_and_bad_timing_are_visible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7847-CUSTODY: malformed inventory and time fail closed."""
    root = fixture_root(tmp_path)
    inventory = root / INVENTORY
    value = json.loads(inventory.read_text())
    value["board_rows"].pop()
    inventory.write_text(json.dumps(value))
    result = read_evidence(root, "20260929")
    assert result["hardware_inventory_ready_score"] == 0
    assert any(x["artifact_field"] == "board_names" for x in result["gate_check_summary"])
    inventory.unlink()
    with pytest.raises(ValueError, match="missing_board_inventory"):
        read_evidence(root, "20260929")
    root = fixture_root(tmp_path)
    (root / SERVICE).write_text(
        json.dumps(
            {
                "experiment_id": 7846,
                "task_id": "exp7846-service-cost",
                "milestone": "2026.09.681",
                "service_evidence_ready_score": 1,
                "verdict_class": "null",
                "flagged_adversarial": False,
                "stage_times_ms": {"whole_service": 0, "host_stage": 2},
            }
        )
    )
    result = read_evidence(root, "20260929")
    assert any(x["artifact_field"] == "stage_times_ms.valid" for x in result["gate_check_summary"])


def test_cold_replay_detects_rows_gates_and_log_mutation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7847-CLI: replay rehashes source rows and sealed logs."""
    root = fixture_root(tmp_path)
    result = read_evidence(root, "20260929")
    changed = json.loads(json.dumps(result))
    changed["rows_checksum"] = "sha256:wrong"
    with pytest.raises(ValueError, match="rows_changed"):
        cold_reduce(root, changed)
    changed = json.loads(json.dumps(result))
    changed["gate_check_summary"] = []
    with pytest.raises(ValueError, match="gate_operands_changed"):
        cold_reduce(root, changed)
    log = tmp_path / "sealed.log"
    log.write_text("before")
    result["validation_receipts"] = {
        "checks": [{"log_path": str(log), "log_sha256": "sha256:wrong"}]
    }
    with pytest.raises(ValueError, match="sealed_log_changed"):
        cold_reduce(root, result)
