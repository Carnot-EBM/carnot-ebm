"""REQ-REPORT-7889: current custody without a new board execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.experiment_7889_v684_hardware_evidence import (
    cold_reduce,
    read_evidence,
)
from scripts.experiments import experiment_7889_v684_hardware_evidence as cli

ROOT = Path(__file__).resolve().parents[2]
PRIOR = "results/experiment_7876_v683_hardware_evidence.json"
SERVICE = "results/experiment_7888_v684_service_cost.json"


def fixture_root(tmp_path: Path) -> Path:
    """Copy source bytes so a mutation does not alter checked-in history."""
    prior = json.loads((ROOT / PRIOR).read_text())
    for name in [PRIOR, *prior["source_artifact_hashes"]]:
        source = ROOT / name
        if source.is_file():
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return tmp_path


def test_scenario_report_7889_custody() -> None:
    """SCENARIO-REPORT-7889-CUSTODY: old validity is separate from old QA failure."""
    result = read_evidence(ROOT, "20260929")
    assert result["experiment_id"] == 7889
    assert result["task_id"] == "exp7889-hardware-evidence"
    assert result["milestone"] == "2026.09.684"
    assert result["verdict_class"] == "null"
    assert result["honest_verdict"].startswith("complete_")
    assert result["hardware_evidence_ready_score"] == 1
    assert result["workload_attachment_available"] is False
    assert result["current_device_execution_count"] == 0
    assert result["hardware_speedup_claimed"] is False
    assert result["inference_substrate_class"] == "no_model_load"
    assert result["execution_venue"] == "host"
    assert result["MODEL_SPECS"] == result["model_specs"] == []
    assert result["target_model"] == "none (no pretrained model)"
    assert [r["board"] for r in result["board_rows"]] == ["KV260", "PolarFire", "GateMate"]
    assert result["board_rows"][0]["k_max"] == 5
    assert result["board_rows"][1]["processor_class"] == "linux_cpu"
    assert result["board_rows"][2]["blocker"] == "0xffffffff"
    assert all(r["terminal_criterion_met"] is False for r in result["board_rows"])
    assert result["historical_required_failures"]
    assert result["source_artifact_hashes"][PRIOR]["sha256"] == sha256_file(ROOT / PRIOR)
    assert cold_reduce(ROOT, result)["rows_checksum"] == result["rows_checksum"]


def test_scenario_report_7889_missing_and_changed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7889-CUSTODY: failed source operands are exact."""
    missing = read_evidence(tmp_path, "20260929")
    assert missing["verdict_class"] == "blocked"
    assert missing["gate_check_summary"]
    assert all(
        set(
            (
                "upstream_id",
                "artifact_path",
                "artifact_hash",
                "artifact_field",
                "op",
                "expected",
                "observed",
            )
        )
        <= row.keys()
        for row in missing["gate_check_summary"]
    )
    root = fixture_root(tmp_path)
    raw = root / "results/raw/experiment_7231/polarfire_dispatch.json"
    raw.write_bytes(raw.read_bytes() + b" ")
    changed = read_evidence(root, "20260929")
    assert changed["verdict_class"] == "blocked"
    assert any(
        row["artifact_path"].endswith("polarfire_dispatch.json")
        and row["observed"] == sha256_file(raw)
        for row in changed["gate_check_summary"]
    )
    with pytest.raises(ValueError, match="gate_operands_changed"):
        cold_reduce(root, read_evidence(ROOT, "20260929"))


def test_scenario_report_7889_workload(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7889-WORKLOAD: only current qualified primitives attach."""
    root = fixture_root(tmp_path)
    path = root / SERVICE
    path.parent.mkdir(parents=True, exist_ok=True)
    service = {
        "experiment_id": 7888,
        "task_id": "exp7888-service-cost",
        "run_date": "20260929",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "service_cost_ready_score": 1,
        "rows": [{"operation": "verify", "traffic_bytes": 64, "family": "service", "seed": 0}],
    }
    path.write_text(json.dumps(service))
    ready = read_evidence(root, "20260929")
    assert ready["workload_attachment_available"] is True
    assert ready["workload_feasibility_rows"][0]["operation"] == "verify"
    assert ready["workload_feasibility_rows"][0]["source_row_hash"]
    service["flagged_adversarial"] = True
    path.write_text(json.dumps(service))
    rejected = read_evidence(root, "20260929")
    assert rejected["workload_attachment_available"] is False
    assert any(
        x["artifact_field"] == "flagged_adversarial"
        for x in rejected["workload_attachment_operands"]
    )


def test_scenario_report_7889_replay_rejects_rows() -> None:
    """SCENARIO-REPORT-7889-CLI: cold replay recomputes all primitive rows."""
    result = read_evidence(ROOT, "20260929")
    result["rows"][0]["k_max"] = 99
    with pytest.raises(ValueError, match="rows_changed"):
        cold_reduce(ROOT, result)
    result = read_evidence(ROOT, "20260929")
    result["terminal_receipt_hashes"]["changed"] = "bad"
    with pytest.raises(ValueError, match="terminal_receipt_changed"):
        cold_reduce(ROOT, result)
    result = read_evidence(ROOT, "20260929")
    result["workload_feasibility_rows"].append({"unearned": True})
    with pytest.raises(ValueError, match="workload_rows_changed"):
        cold_reduce(ROOT, result)


def test_scenario_report_7889_cli_private_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7889-CLI: the actual script handles both input states."""
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}"}
    script = ROOT / cli.SCRIPT
    for root, name, expected in ((ROOT, "good", "null"), (tmp_path, "absent", "blocked")):
        output = tmp_path / f"{name}.json"
        done = subprocess.run(
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
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert "completed_units=" in done.stdout
        assert json.loads(output.read_text())["verdict_class"] == expected
    replay = subprocess.run(
        [
            sys.executable,
            "-u",
            str(script),
            "--root",
            str(ROOT),
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


def test_scenario_report_7889_manifest_and_child(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7889-CLI: exact commands and closed logs are observable."""
    commands = cli.manifest(tmp_path)
    names = [row["name"] for row in commands]
    assert names == [
        "worktree_imports",
        "affected_pytest",
        "unit_coverage",
        "cli_success",
        "cli_missing_input",
        "cold_replay",
        "coverage_shards",
        "coverage_combine",
        "changed_coverage",
        "ruff_check",
        "ruff_format",
        "mypy",
        "scoped_spec",
    ]
    assert all(row["classification"] == "required" for row in commands)
    assert all(
        str(ROOT / cli.SCRIPT) in row["argv"]
        or row["name"] not in {"cli_success", "cli_missing_input", "cold_replay"}
        for row in commands
    )
    spec = {
        "name": "probe",
        "argv": [sys.executable, "-c", "print('ok')"],
        "deadline_s": 5,
        "classification": "required",
    }
    receipt = cli.run_child(spec, tmp_path, 0.0, 0)
    assert receipt["passed"] is True
    assert sha256_file(Path(receipt["log_path"])) == receipt["log_sha256"]


def test_scenario_report_7889_terminal_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7889-CLI: diagnostic debt does not become a new gate."""
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path))

    def fake_child(spec: dict, private: Path, _started: float, _units: int) -> dict:
        path = private / f"{spec['name']}.log"
        path.write_text('{"flagged_count": 0}' if spec["name"] == "adversarial_verify" else "ok")
        return {
            **spec,
            "passed": spec["name"] != "full_pytest",
            "exit_code": 0,
            "timed_out": False,
            "log_path": str(path),
            "log_sha256": sha256_file(path),
            "resolved_imports": {
                "carnot.reporting.experiment_7889_v684_hardware_evidence": str(ROOT / cli.MODULE)
            },
        }

    monkeypatch.setattr(cli, "run_child", fake_child)
    output = tmp_path / "terminal.json"
    assert cli.main(["--date", "20260929", "--root", str(ROOT), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["validation_receipts"]["required_checks_passed"] is True
    assert result["repository_health"]["status"] == "degraded_open"
    assert result["hardware_evidence_ready_score"] == 1
    assert result["validation_command_manifest_path"]
    assert cli.main(["--date", "20260929", "--root", str(ROOT), "--output", str(output)]) == 0
    checkpoint = Path(result["validation_command_manifest_path"]).with_name("completed_units.json")
    saved = json.loads(checkpoint.read_text())
    saved["checks"][0]["log_sha256"] = "sha256:wrong"
    cli.atomic_json(checkpoint, saved)
    with pytest.raises(ValueError, match="checkpoint_receipt_changed"):
        cli.main(["--date", "20260929", "--root", str(ROOT), "--output", str(output)])
    with pytest.raises(ValueError, match="worktree root"):
        cli.main(["--root", str(tmp_path), "--output", str(output)])


def test_scenario_report_7889_required_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7889-CLI: an owned failure disqualifies current readiness."""
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path))

    def failed_child(spec: dict, private: Path, _started: float, _units: int) -> dict:
        path = private / f"{spec['name']}.log"
        path.write_text('{"flagged_count": 0}' if spec["name"] == "adversarial_verify" else "ok")
        passed = spec["name"] != "changed_coverage"
        return {
            **spec,
            "passed": passed,
            "exit_code": 0 if passed else 2,
            "timed_out": False,
            "log_path": str(path),
            "log_sha256": sha256_file(path),
        }

    monkeypatch.setattr(cli, "run_child", failed_child)
    output = tmp_path / "failed.json"
    assert cli.main(["--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["verdict_class"] == "disqualified"
    assert result["hardware_evidence_ready_score"] == 0
    assert result["acceptance_gate_results"]["readiness"] == 0


def test_scenario_report_7889_terminal_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7889-CLI: malformed validator output cannot publish."""
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path))

    def malformed_child(spec: dict, private: Path, _started: float, _units: int) -> dict:
        path = private / f"{spec['name']}.log"
        path.write_text("malformed" if spec["name"] == "adversarial_verify" else "ok")
        return {
            **spec,
            "passed": spec["name"] != "strict_rows",
            "exit_code": 0,
            "timed_out": False,
            "log_path": str(path),
            "log_sha256": sha256_file(path),
        }

    monkeypatch.setattr(cli, "run_child", malformed_child)
    output = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="terminal_validation_failed"):
        cli.main(["--output", str(output)])
    assert not output.exists()


def test_scenario_report_7889_coverage_shards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7889-COVERAGE: reject an empty completed shard."""
    from coverage import CoverageData

    paths = [tmp_path / f"{name}.coverage" for name in ("unit", "success", "missing", "replay")]
    for path in paths:
        data = CoverageData(basename=str(path))
        data.add_lines({str(ROOT / cli.MODULE): {1}})
        data.write()
    cli.check_coverage_shards(paths)
    empty = CoverageData(basename=str(paths[2]))
    empty.erase()
    empty.write()
    with pytest.raises(ValueError, match="empty_coverage_shard"):
        cli.check_coverage_shards(paths)
    with pytest.raises(ValueError, match="empty_coverage_shard"):
        cli.check_coverage_shards([tmp_path / "absent.coverage"])
    foreign = CoverageData(basename=str(paths[2]))
    foreign.erase()
    foreign.add_lines({str(tmp_path / "unowned.py"): {1}})
    foreign.write()
    with pytest.raises(ValueError, match="empty_coverage_shard"):
        cli.check_coverage_shards(paths)


def test_scenario_report_7889_coverage_manifest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7889-COVERAGE: freeze checked shards, keep health separate."""
    commands = cli.manifest(tmp_path)
    names = [row["name"] for row in commands]
    assert names.index("coverage_shards") < names.index("coverage_combine")
    assert "full_pytest" not in names


def test_scenario_report_7889_sealed_repository_health(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7889-COVERAGE: preserve a prior diagnostic by log hash."""
    log = tmp_path / "full.log"
    log.write_text("timed out after a failed test")
    path = tmp_path / "earlier.json"
    path.write_text(
        json.dumps(
            {
                "repository_health": {
                    "full_pytest": {
                        "name": "full_pytest",
                        "classification": "diagnostic",
                        "passed": False,
                        "log_path": str(log),
                        "log_sha256": sha256_file(log),
                    }
                }
            }
        )
    )
    health = cli.repository_health_from_artifact(path)
    assert health["status"] == "degraded_open"
    assert health["affects_required_checks"] is False
    assert health["source_artifact_hash"] == sha256_file(path)
    log.write_text("changed")
    with pytest.raises(ValueError, match="diagnostic_log_changed"):
        cli.repository_health_from_artifact(path)
