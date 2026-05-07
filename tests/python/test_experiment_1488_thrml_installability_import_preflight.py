"""Tests for Exp 1488 THRML installability/import terminal preflight.

Spec traces: REQ-SAMPLE-043, SCENARIO-SAMPLE-071.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import carnot.samplers.thrml_installability_preflight as exp1488
from carnot.samplers.thrml_installability_preflight import (
    REQUIRED_ARTIFACT_FIELDS,
    CommandResult,
    build_artifact,
    probe_installability,
    probe_thrml_import,
    run_command,
    run_preflight,
    validate_artifact,
    write_in_progress_artifact,
)


def _result(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
) -> CommandResult:
    return CommandResult(
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        duration_s=0.01,
    )


def test_req_sample_043_spec_anchor_exists() -> None:
    """REQ-SAMPLE-043, SCENARIO-SAMPLE-071: Exp1488 is spec-anchored."""

    spec = (
        exp1488.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md"
    ).read_text(encoding="utf-8")

    assert "REQ-SAMPLE-043" in spec
    assert "SCENARIO-SAMPLE-071" in spec
    assert "experiment_1488_thrml_installability_import_preflight.json" in spec


def test_req_sample_043_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-SAMPLE-043: the deliverable starts with an in-progress marker."""

    output = tmp_path / "experiment_1488_thrml_installability_import_preflight.json"

    marker = write_in_progress_artifact(output)

    assert REQUIRED_ARTIFACT_FIELDS <= set(marker)
    assert marker["status"] == "in_progress"
    assert marker["thrml_preflight_complete"] is False
    assert marker["thrml_import_ready"] is False
    assert marker["hardware_claim_allowed"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == marker


def test_req_sample_043_import_success_records_version_and_skips_install_probe() -> None:
    """REQ-SAMPLE-043: import success is the only path to thrml_import_ready=true."""

    def _runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        if command[-1] == "import thrml":
            return _result(command)
        return _result(command, stdout="0.2.0\n")

    import_probe = probe_thrml_import(runner=_runner)
    install_probe = probe_installability(import_probe, runner=_runner)
    artifact = build_artifact(
        import_probe=import_probe,
        install_probe=install_probe,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260507",
    )

    validate_artifact(artifact)
    assert artifact["thrml_import_ready"] is True
    assert artifact["thrml_version"] == "0.2.0"
    assert artifact["import_error"] is None
    assert artifact["install_probe_attempted"] is False
    assert artifact["install_probe_result"]["status"] == "skipped_import_ready"
    assert artifact["simulator_lane_allowed"] is True
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["next_task_gate_value"] == "thrml_import_ready_simulator_only"


def test_req_sample_043_import_failure_attempts_bounded_non_mutating_probe() -> None:
    """REQ-SAMPLE-043: missing THRML triggers only a dry-run/no-deps install probe."""

    commands: list[list[str]] = []

    def _runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        commands.append(command)
        if command[-1] == "import thrml":
            return _result(
                command,
                returncode=1,
                stderr="ModuleNotFoundError: No module named 'thrml'\n",
            )
        return _result(command, stdout="Would install thrml-0.2.0\n")

    import_probe = probe_thrml_import(runner=_runner)
    install_probe = probe_installability(import_probe, runner=_runner)
    artifact = build_artifact(
        import_probe=import_probe,
        install_probe=install_probe,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260507",
    )

    validate_artifact(artifact)
    assert commands[0] == ["python3", "-c", "import thrml"]
    assert "--dry-run" in commands[1]
    assert "--no-deps" in commands[1]
    assert "install" in commands[1]
    assert artifact["thrml_import_ready"] is False
    assert "ModuleNotFoundError" in artifact["import_error"]
    assert artifact["install_probe_attempted"] is True
    assert artifact["install_probe_result"]["status"] == "dry_run_installable"
    assert artifact["next_task_gate_value"] == "thrml_import_blocked_simulator_only"
    assert artifact["honest_verdict"] == (
        "thrml_not_importable_bounded_install_probe_installable_simulator_only"
    )


def test_req_sample_043_failed_install_probe_remains_simulator_only() -> None:
    """SCENARIO-SAMPLE-071: install-probe failure still disallows hardware claims."""

    import_probe = exp1488.ThrmlImportProbe(
        import_ready=False,
        version=None,
        import_error="ModuleNotFoundError: No module named 'thrml'",
        command_result={"returncode": 1},
        version_command_result=None,
    )
    install_probe = exp1488.InstallProbe(
        attempted=True,
        result={
            "status": "dry_run_failed",
            "returncode": 1,
            "stderr": "No matching distribution found for thrml",
        },
    )

    artifact = build_artifact(
        import_probe=import_probe,
        install_probe=install_probe,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260507",
    )

    validate_artifact(artifact)
    assert artifact["thrml_import_ready"] is False
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["simulator_lane_allowed"] is True
    assert artifact["honest_verdict"] == (
        "thrml_not_importable_bounded_install_probe_blocked_simulator_only"
    )


def test_req_sample_043_probe_handles_empty_import_error_and_probe_failures() -> None:
    """REQ-SAMPLE-043: empty command output still records a concrete import error."""

    def _empty_import_failure(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        return _result(command, returncode=9)

    import_probe = probe_thrml_import(runner=_empty_import_failure)
    failed_install = probe_installability(import_probe, runner=_empty_import_failure)

    def _timeout_install(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        return _result(command, returncode=-1, stderr="timed out", timed_out=True)

    timed_out_install = probe_installability(import_probe, runner=_timeout_install)

    assert import_probe.import_error == "thrml import returned non-zero exit code 9"
    assert failed_install.result["status"] == "dry_run_failed"
    assert timed_out_install.result["status"] == "dry_run_timeout"


def test_scenario_sample_071_run_preflight_writes_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-071: runner writes the validated terminal JSON artifact."""

    output = tmp_path / "experiment_1488_thrml_installability_import_preflight.json"

    def _runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        if command[-1] == "import thrml":
            return _result(command, returncode=1, stderr="No module named 'thrml'\n")
        return _result(command, stdout="Would install thrml-0.2.0\n")

    artifact = run_preflight(
        output_path=output,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260507",
        runner=_runner,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == artifact
    assert REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["status"] == "complete"
    assert payload["thrml_preflight_complete"] is True
    assert payload["install_probe_attempted"] is True


def test_req_sample_043_validator_rejects_partial_or_dishonest_artifacts() -> None:
    """REQ-SAMPLE-043: validation catches unsafe hardware and import flags."""

    valid = {
        "status": "complete",
        "thrml_preflight_complete": True,
        "thrml_import_ready": False,
        "thrml_version": None,
        "import_error": "No module named 'thrml'",
        "install_probe_attempted": True,
        "install_probe_result": {"status": "dry_run_failed"},
        "simulator_lane_allowed": True,
        "hardware_claim_allowed": False,
        "next_task_gate_value": "thrml_import_blocked_simulator_only",
        "honest_verdict": "thrml_not_importable_bounded_install_probe_blocked_simulator_only",
    }
    validate_artifact(valid)

    missing = dict(valid)
    missing.pop("install_probe_result")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        validate_artifact(missing)

    dishonest = dict(valid)
    dishonest["hardware_claim_allowed"] = True
    with pytest.raises(ValueError, match="hardware_claim_allowed"):
        validate_artifact(dishonest)

    inconsistent = dict(valid)
    inconsistent["thrml_import_ready"] = True
    with pytest.raises(ValueError, match="import_ready"):
        validate_artifact(inconsistent)

    incomplete = dict(valid)
    incomplete["thrml_preflight_complete"] = False
    with pytest.raises(ValueError, match="thrml_preflight_complete"):
        validate_artifact(incomplete)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "failed", "invalid status"),
        ("simulator_lane_allowed", False, "simulator_lane_allowed"),
        ("install_probe_result", None, "install_probe_result"),
        ("honest_verdict", "claimed_tsu_hardware", "honest_verdict"),
        ("thrml_import_ready", None, "thrml_import_ready"),
        ("import_error", "", "import_error"),
        ("install_probe_attempted", False, "install_probe_attempted"),
    ],
)
def test_req_sample_043_validator_rejects_inconsistent_blocked_artifacts(
    field: str,
    value: Any,
    message: str,
) -> None:
    """REQ-SAMPLE-043: blocked artifacts must remain complete and actionable."""

    artifact = {
        "status": "complete",
        "thrml_preflight_complete": True,
        "thrml_import_ready": False,
        "thrml_version": None,
        "import_error": "No module named 'thrml'",
        "install_probe_attempted": True,
        "install_probe_result": {"status": "dry_run_failed"},
        "simulator_lane_allowed": True,
        "hardware_claim_allowed": False,
        "next_task_gate_value": "thrml_import_blocked_simulator_only",
        "honest_verdict": "thrml_not_importable_bounded_install_probe_blocked_simulator_only",
    }
    artifact[field] = value

    with pytest.raises(ValueError, match=message):
        validate_artifact(artifact)


def test_req_sample_043_validator_rejects_import_ready_with_install_probe() -> None:
    """REQ-SAMPLE-043: import-ready artifacts must not also run install probes."""

    artifact = {
        "status": "complete",
        "thrml_preflight_complete": True,
        "thrml_import_ready": True,
        "thrml_version": "0.2.0",
        "import_error": None,
        "install_probe_attempted": True,
        "install_probe_result": {"status": "dry_run_installable"},
        "simulator_lane_allowed": True,
        "hardware_claim_allowed": False,
        "next_task_gate_value": "thrml_import_ready_simulator_only",
        "honest_verdict": "thrml_import_ready_simulator_lane_only_no_hardware_claim",
    }

    with pytest.raises(ValueError, match="installability probe"):
        validate_artifact(artifact)


def test_req_sample_043_run_command_captures_success_and_timeout() -> None:
    """REQ-SAMPLE-043: subprocess results preserve stdout, return code, and timeout."""

    success = run_command(["python3", "-c", "print('ok')"], timeout_s=5)
    timed_out = run_command(["python3", "-c", "import time; time.sleep(2)"], timeout_s=0.1)

    assert success.returncode == 0
    assert success.stdout.strip() == "ok"
    assert success.timed_out is False
    assert timed_out.returncode == -1
    assert timed_out.timed_out is True
    assert "timed out" in timed_out.stderr


def test_req_sample_043_compacts_long_command_text() -> None:
    """REQ-SAMPLE-043: long pip output is compacted for stable JSON artifacts."""

    assert exp1488._compact_text("x" * 20, limit=8) == "xxxxx..."
    assert exp1488._command_payload(_result(["python3"], stdout="ok"))["stdout"] == "ok"


def test_deliverable_json_has_required_fields_when_complete() -> None:
    """SCENARIO-SAMPLE-071: generated deliverable satisfies the roadmap contract."""

    if not exp1488.DELIVERABLE_PATH.exists():
        pytest.skip("artifact not yet generated")
    payload = json.loads(exp1488.DELIVERABLE_PATH.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        pytest.skip("artifact not yet complete")

    validate_artifact(payload)
    assert REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["hardware_claim_allowed"] is False
    assert payload["metadata"]["run_date"] == "20260507"
