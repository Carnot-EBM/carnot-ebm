"""REQ-REPORT-7901-V685: dated board custody and optional service fit."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import os

import pytest
from coverage import CoverageData

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.experiment_7901_v685_hardware_evidence import cold_reduce, read_evidence
from scripts.experiments.experiment_7889_v684_hardware_evidence import (
    MODULE as OLD_MODULE,
    check_coverage_shards,
)
from scripts.experiments import experiment_7901_v685_hardware_evidence as cli

ROOT = Path(__file__).resolve().parents[2]
PRIOR = "results/experiment_7889_v684_hardware_evidence.json"
SERVICE = "results/experiment_7900_v685_service_cost.json"


def private_root(tmp_path: Path) -> Path:
    """Copy dated inputs so a damaged fixture cannot alter historical bytes."""
    prior = json.loads((ROOT / PRIOR).read_text())
    for name in [PRIOR, *prior["source_artifact_hashes"]]:
        source = ROOT / name
        if source.is_file():
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return tmp_path


def test_scenario_report_7901_coverage_existing_empty(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7901-COVERAGE: reach the old zero-line rejection."""
    path = tmp_path / "empty.coverage"
    data = CoverageData(basename=str(path))
    data.add_lines({str(ROOT / OLD_MODULE): set()})
    data.write()
    assert path.is_file()
    with pytest.raises(ValueError, match="empty_coverage_shard"):
        check_coverage_shards([path])
    with pytest.raises(ValueError, match="empty_coverage_shard"):
        check_coverage_shards([tmp_path / "missing.coverage"])
    foreign = tmp_path / "foreign.coverage"
    data = CoverageData(basename=str(foreign))
    data.add_lines({str(tmp_path / "unrelated.py"): {1}})
    data.write()
    with pytest.raises(ValueError, match="empty_coverage_shard"):
        check_coverage_shards([foreign])


def test_scenario_report_7901_custody(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7901-CUSTODY: old board facts stay dated and narrow."""
    root = private_root(tmp_path)
    result = read_evidence(root, "20260929")
    assert result["experiment_id"] == 7901
    assert result["task_id"] == "exp7901-hardware-evidence"
    assert result["verdict_class"] == "null"
    assert result["hardware_evidence_ready_score"] == 1
    assert result["workload_attachment_available"] is False
    assert [r["board"] for r in result["board_rows"]] == ["KV260", "PolarFire", "GateMate"]
    assert result["board_rows"][0]["k_max"] == 5
    assert result["board_rows"][1]["processor_class"] == "linux_cpu"
    assert result["board_rows"][2]["blocker"] == "0xffffffff"
    assert all(r["receipt_date"] and r["receipt_hash"] for r in result["board_rows"])
    assert all(r["next_operator_or_device_change"] for r in result["board_rows"])
    assert result["historical_required_failures"]
    assert result["MODEL_SPECS"] == []
    assert result["model_invocation_counts"]["generation_calls"] == 0
    assert cold_reduce(root, result)["row_count"] == 3


def test_scenario_report_7901_missing_changed_and_malformed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7901-CUSTODY: wrong bytes fail with exact operands."""
    empty = read_evidence(tmp_path, "20260929")
    assert empty["verdict_class"] == "blocked"
    assert empty["honest_verdict"].startswith("complete_blocked_")
    assert all("artifact_field" in row and "observed" in row for row in empty["gate_check_summary"])
    root = private_root(tmp_path)
    source = root / "results/experiment_7231_v636_board_continuity.json"
    source.write_bytes(source.read_bytes() + b" ")
    changed = read_evidence(root, "20260929")
    assert changed["verdict_class"] == "blocked"
    assert any(row["observed"] == sha256_file(source) for row in changed["gate_check_summary"])
    with pytest.raises(ValueError, match="gate_operands_changed"):
        cold_reduce(root, read_evidence(ROOT, "20260929"))
    clean = private_root(tmp_path)
    row = read_evidence(clean, "20260929")
    row["board_rows"][0]["k_max"] = 6
    with pytest.raises(ValueError, match="rows_changed"):
        cold_reduce(clean, row)
    row = read_evidence(clean, "20260929")
    row["terminal_receipt_hashes"]["changed"] = "sha256:wrong"
    with pytest.raises(ValueError, match="terminal_receipt_changed"):
        cold_reduce(clean, row)


def test_scenario_report_7901_workload_refusal(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7901-WORKLOAD: a flagged producer cannot attach."""
    root = private_root(tmp_path)
    path = root / SERVICE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": 7900,
                "task_id": "exp7900-service-cost",
                "run_date": "20260929",
                "verdict_class": "null",
                "flagged_adversarial": True,
                "service_measurement_ready_score": 1,
                "rows": [{"operation": "verify", "transfer_bytes": 64}],
            }
        )
    )
    result = read_evidence(root, "20260929")
    assert result["workload_attachment_available"] is False
    assert any(
        row["artifact_field"] == "flagged_adversarial"
        for row in result["workload_attachment_operands"]
    )
    assert result["hardware_evidence_ready_score"] == 1
    path.unlink()
    with pytest.raises(ValueError, match="workload_rows_changed"):
        cold_reduce(root, result)


def test_scenario_report_7901_workload_attachment(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7901-WORKLOAD: eligible operations retain cost limits."""
    root = private_root(tmp_path)
    path = root / SERVICE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": 7900,
                "task_id": "exp7900-service-cost",
                "run_date": "20260929",
                "verdict_class": "null",
                "flagged_adversarial": False,
                "service_measurement_ready_score": 1,
                "rows": [
                    {
                        "operation": "verify",
                        "host_fraction": 0.6,
                        "transport_fraction": 0.2,
                        "transfer_bytes": 64,
                        "connection_cost_ms": 2,
                        "coupling_update_cost_ms": 3,
                    }
                ],
            }
        )
    )
    result = read_evidence(root, "20260929")
    assert result["workload_attachment_available"] is True
    assert len(result["workload_feasibility_rows"]) == 3
    assert result["workload_feasibility_rows"][0]["connection_cost_ms"] == 2
    assert result["workload_feasibility_rows"][1]["fabric_boundary"] == "Linux CPU only"
    assert cold_reduce(root, result)["row_count"] == 3


def test_scenario_report_7901_cli_private(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7901-CLI: real success, missing, and replay paths."""
    root = private_root(tmp_path / "root")
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}"}
    for source, name, verdict in ((root, "good", "null"), (tmp_path / "absent", "bad", "blocked")):
        output = tmp_path / f"{name}.json"
        done = subprocess.run(
            [
                sys.executable,
                "-u",
                str(ROOT / cli.SCRIPT),
                "--date",
                "20260929",
                "--root",
                str(source),
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
        assert done.returncode == 0, done.stderr
        assert "completed_units=" in done.stdout
        assert json.loads(output.read_text())["verdict_class"] == verdict
    replay = subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / cli.SCRIPT),
            "--root",
            str(root),
            "--cold-replay",
            str(tmp_path / "good.json"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert replay.returncode == 0, replay.stderr
    broken = json.loads((tmp_path / "good.json").read_text())
    broken["board_rows"].pop()
    (tmp_path / "broken.json").write_text(json.dumps(broken))
    rejected = subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / cli.SCRIPT),
            "--root",
            str(root),
            "--cold-replay",
            str(tmp_path / "broken.json"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert rejected.returncode != 0 and "rows_changed" in rejected.stderr


def test_scenario_report_7901_manifest_and_shards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7901-COVERAGE: freeze owned includes and reject empty data."""
    commands = cli.manifest(tmp_path)
    names = [row["name"] for row in commands]
    assert names.index("coverage_shards") < names.index("coverage_combine")
    assert "full_pytest" not in names
    assert "e2e_016_fixture" in names and "e2e_016_replay" in names
    assert all(row["deadline_s"] <= 180 for row in commands)
    missing = tmp_path / "absent.coverage"
    with pytest.raises(ValueError, match="empty_coverage_shard"):
        cli.check_coverage_shards([missing])
    measured = tmp_path / "measured.coverage"
    data = CoverageData(basename=str(measured))
    data.add_lines({str(ROOT / cli.MODULE): {1}})
    data.write()
    cli.check_coverage_shards([measured])
    unrelated = tmp_path / "unrelated.coverage"
    data = CoverageData(basename=str(unrelated))
    data.add_lines({str(tmp_path / "other.py"): {1}})
    data.write()
    with pytest.raises(ValueError, match="empty_coverage_shard"):
        cli.check_coverage_shards([unrelated])


def test_scenario_report_7901_terminal_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7901-CLI: current checks and old health stay separate."""
    root = private_root(tmp_path / "isolated")
    for name in (cli.MODULE, cli.SCRIPT, cli.TEST, *cli.CONSUMERS, *cli.LIBRARIES):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    monkeypatch.setattr(cli, "ROOT", root)
    mode = {"value": "passing"}

    def fake_child(spec: dict, private: Path, _started: float, _units: int) -> dict:
        if spec["name"] == "cli_success":
            cli.atomic_json(private / "success.json", read_evidence(root, "20260929"))
        path = private / f"{spec['name']}.log"
        path.write_text(
            "rows_changed"
            if spec["name"] == "negative_replay"
            else "malformed"
            if spec["name"] == "adversarial_verify" and mode["value"] == "invalid_report"
            else '{"flagged_count": 0}'
            if spec["name"] == "adversarial_verify"
            else "ok"
        )
        exit_code = (
            1
            if spec["name"] == "negative_replay"
            else 2
            if spec["name"] == "changed_coverage" and mode["value"] == "failed_required"
            else 0
        )
        return {
            **spec,
            "passed": exit_code == 0
            and not (spec["name"] == "strict_rows" and mode["value"] == "invalid_report"),
            "exit_code": exit_code,
            "timed_out": False,
            "log_path": str(path),
            "log_sha256": sha256_file(path),
        }

    monkeypatch.setattr(cli, "run_child", fake_child)
    output = tmp_path / "terminal.json"
    args = ["--date", "20260929", "--root", str(root), "--output", str(output)]
    assert cli.main(args) == 0
    result = json.loads(output.read_text())
    assert result["validation_receipts"]["required_checks_passed"] is True
    assert result["repository_health"]["status"] == "degraded_open"
    assert result["hardware_evidence_ready_score"] == 1
    assert cli.main(args) == 0
    checkpoint = Path(result["validation_command_manifest_path"]).with_name("completed_units.json")
    saved = json.loads(checkpoint.read_text())
    saved["checks"][0]["log_sha256"] = "sha256:wrong"
    cli.atomic_json(checkpoint, saved)
    with pytest.raises(ValueError, match="checkpoint_receipt_changed"):
        cli.main(args)
    with pytest.raises(ValueError, match="worktree root"):
        cli.main(["--root", str(tmp_path / "absent"), "--output", str(output)])
    with (root / cli.TEST).open("a") as stream:
        stream.write("\n# isolated closure change\n")
    mode["value"] = "failed_required"
    failed = tmp_path / "failed.json"
    assert cli.main(["--root", str(root), "--output", str(failed)]) == 0
    bad = json.loads(failed.read_text())
    assert bad["verdict_class"] == "disqualified"
    assert bad["hardware_evidence_ready_score"] == 0
    with (root / cli.TEST).open("a") as stream:
        stream.write("\n# second isolated closure change\n")
    mode["value"] = "invalid_report"
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="terminal_validation_failed"):
        cli.main(["--root", str(root), "--output", str(invalid)])
    assert not invalid.exists()
