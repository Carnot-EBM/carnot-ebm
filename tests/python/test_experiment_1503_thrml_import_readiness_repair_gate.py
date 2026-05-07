"""Tests for Exp 1503 THRML import readiness repair gate.

Spec traces: REQ-SAMPLE-044, SCENARIO-SAMPLE-072.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import carnot.samplers.thrml_import_readiness_repair_gate as exp1503
from carnot.samplers.thrml_import_readiness_repair_gate import (
    REQUIRED_ARTIFACT_FIELDS,
    CommandResult,
    build_artifact,
    probe_compatibility,
    probe_import_details,
    repair_thrml_if_safe,
    run_readiness_gate,
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


def _details_stdout(version: str = "0.1.3", path: str = "/venv/site/thrml/__init__.py") -> str:
    return json.dumps(
        {
            "metadata_version": version,
            "module_name": "thrml",
            "path": path,
            "version": version,
        },
        sort_keys=True,
    )


def _compat_stdout(*, passed: bool = True, public_surface_count: int = 3) -> str:
    return json.dumps(
        {
            "models_importable": True,
            "passed": passed,
            "public_surface_count": public_surface_count,
            "public_surfaces": ["Block", "SpinNode", "models"][:public_surface_count],
        },
        sort_keys=True,
    )


def test_req_sample_044_spec_anchor_exists() -> None:
    """REQ-SAMPLE-044, SCENARIO-SAMPLE-072: Exp1503 is spec-anchored."""

    spec = (
        exp1503.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md"
    ).read_text(encoding="utf-8")

    assert "REQ-SAMPLE-044" in spec
    assert "SCENARIO-SAMPLE-072" in spec
    assert "experiment_1503_thrml_import_readiness_repair_gate.json" in spec


def test_req_sample_044_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-SAMPLE-044: the deliverable starts with an in-progress marker."""

    output = tmp_path / "experiment_1503_thrml_import_readiness_repair_gate.json"

    marker = write_in_progress_artifact(output)

    assert REQUIRED_ARTIFACT_FIELDS <= set(marker)
    assert marker["status"] == "in_progress"
    assert marker["thrml_import_ready"] is False
    assert marker["compatibility_probe_passed"] is False
    assert marker["parity_followup_allowed"] is False
    assert marker["hardware_claim_allowed"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == marker


def test_req_sample_044_import_ready_records_version_path_and_gate(tmp_path: Path) -> None:
    """REQ-SAMPLE-044: import success opens only the simulator parity follow-up gate."""

    commands: list[list[str]] = []

    def _runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        commands.append(command)
        if command == ["python3", "-c", "import thrml"]:
            return _result(command)
        if "importlib.metadata" in command[-1]:
            return _result(command, stdout=_details_stdout())
        return _result(command, stdout=_compat_stdout())

    python_executable = str(tmp_path / ".venv" / "bin" / "python")
    output = tmp_path / "experiment_1503_thrml_import_readiness_repair_gate.json"

    artifact = run_readiness_gate(
        output_path=output,
        project_root=tmp_path,
        python_executable=python_executable,
        runner=_runner,
    )

    validate_artifact(artifact)
    assert artifact["thrml_import_ready"] is True
    assert artifact["import_error"] is None
    assert artifact["repair_attempted"] is False
    assert artifact["repair_actions"] == []
    assert artifact["thrml_version"] == "0.1.3"
    assert artifact["thrml_import_path"] == "/venv/site/thrml/__init__.py"
    assert artifact["compatibility_probe_passed"] is True
    assert artifact["parity_followup_allowed"] is True
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["metadata"]["exp1488_reproduction"]["command"] == [
        "python3",
        "-c",
        "import thrml",
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert not any("pip" in part for command in commands for part in command)


def test_req_sample_044_missing_thrml_repairs_project_virtualenv(tmp_path: Path) -> None:
    """REQ-SAMPLE-044: missing THRML can be repaired only inside the project venv."""

    detail_calls = 0
    commands: list[list[str]] = []

    def _runner(command: list[str], timeout_s: float) -> CommandResult:
        nonlocal detail_calls
        del timeout_s
        commands.append(command)
        if command == ["python3", "-c", "import thrml"]:
            return _result(command, returncode=1, stderr="ModuleNotFoundError: No module named 'thrml'\n")
        if "importlib.metadata" in command[-1]:
            detail_calls += 1
            if detail_calls == 1:
                return _result(
                    command,
                    returncode=1,
                    stderr=(
                        "Traceback (most recent call last):\n"
                        "  File \"<string>\", line 1, in <module>\n"
                        "ModuleNotFoundError: No module named 'thrml'\n"
                    ),
                )
            return _result(command, stdout=_details_stdout(version="0.1.4"))
        if command[1:4] == ["-m", "pip", "--disable-pip-version-check"]:
            return _result(command, stdout="Successfully installed thrml-0.1.4\n")
        return _result(command, stdout=_compat_stdout())

    python_executable = str(tmp_path / ".venv" / "bin" / "python")
    artifact = run_readiness_gate(
        output_path=tmp_path / "experiment_1503_thrml_import_readiness_repair_gate.json",
        project_root=tmp_path,
        python_executable=python_executable,
        runner=_runner,
    )

    validate_artifact(artifact)
    assert artifact["thrml_import_ready"] is True
    assert artifact["repair_attempted"] is True
    assert artifact["repair_actions"][0]["status"] == "repair_install_succeeded"
    assert artifact["repair_actions"][0]["mutating_install_performed"] is True
    assert artifact["thrml_version"] == "0.1.4"
    assert any(command[:5] == [python_executable, "-m", "pip", "--disable-pip-version-check", "install"] for command in commands)


def test_req_sample_044_missing_thrml_outside_venv_is_terminal(tmp_path: Path) -> None:
    """REQ-SAMPLE-044: unsafe Python scopes are classified without repair."""

    def _runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        if command == ["python3", "-c", "import thrml"]:
            return _result(command, returncode=1, stderr="ModuleNotFoundError: No module named 'thrml'\n")
        return _result(command, returncode=1, stderr="ModuleNotFoundError: No module named 'thrml'\n")

    artifact = run_readiness_gate(
        output_path=tmp_path / "experiment_1503_thrml_import_readiness_repair_gate.json",
        project_root=tmp_path,
        python_executable="/usr/bin/python3",
        runner=_runner,
    )

    validate_artifact(artifact)
    assert artifact["thrml_import_ready"] is False
    assert artifact["repair_attempted"] is False
    assert artifact["parity_followup_allowed"] is False
    assert artifact["hardware_claim_allowed"] is False
    assert "ModuleNotFoundError" in artifact["import_error"]
    assert any("unsafe" in blocker for blocker in artifact["blockers"])
    assert artifact["honest_verdict"].startswith("complete_")


def test_req_sample_044_import_ready_with_failed_compatibility_keeps_gate_closed() -> None:
    """SCENARIO-SAMPLE-072: compatibility failure prevents parity follow-up."""

    import_probe = exp1503.ThrmlImportDetails(
        import_ready=True,
        version="0.1.3",
        import_path="/venv/thrml/__init__.py",
        import_error=None,
        command_result={"returncode": 0},
    )
    compatibility = exp1503.CompatibilityProbe(
        passed=False,
        result={"passed": False, "error": "public surface missing"},
        error="public surface missing",
    )

    artifact = build_artifact(
        exp1488_reproduction={"returncode": 0},
        initial_import=import_probe,
        terminal_import=import_probe,
        compatibility=compatibility,
        repair_attempted=False,
        repair_actions=[],
        blockers=["compatibility_probe_failed: public surface missing"],
        project_root="/repo",
        python_executable="/repo/.venv/bin/python",
    )

    validate_artifact(artifact)
    assert artifact["thrml_import_ready"] is False
    assert artifact["import_error"] is None
    assert artifact["compatibility_probe_passed"] is False
    assert artifact["parity_followup_allowed"] is False


def test_scenario_sample_072_runner_records_compatibility_blocker(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-072: the runner records compatibility blockers itself."""

    def _runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        if command == ["python3", "-c", "import thrml"]:
            return _result(command)
        if "importlib.metadata" in command[-1]:
            return _result(command, stdout=_details_stdout())
        return _result(command, stdout=_compat_stdout(passed=False, public_surface_count=0))

    artifact = run_readiness_gate(
        output_path=tmp_path / "experiment_1503_thrml_import_readiness_repair_gate.json",
        project_root=tmp_path,
        python_executable=str(tmp_path / ".venv" / "bin" / "python"),
        runner=_runner,
    )

    validate_artifact(artifact)
    assert artifact["thrml_import_ready"] is False
    assert artifact["import_error"] is None
    assert artifact["compatibility_probe_passed"] is False
    assert artifact["blockers"] == ["compatibility_probe_failed: compatibility probe failed"]


def test_req_sample_044_probe_helpers_handle_non_json_and_non_missing_errors(tmp_path: Path) -> None:
    """REQ-SAMPLE-044: malformed probes and non-THRML import errors are terminal data."""

    def _runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        if "importlib.metadata" in command[-1]:
            return _result(command, stdout="not json\n")
        return _result(command, stdout="also not json\n")

    import_details = probe_import_details(python_executable="python3", runner=_runner)
    compatibility = probe_compatibility(python_executable="python3", runner=_runner)
    repair_attempted, repair_actions = repair_thrml_if_safe(
        "ImportError: dependency failed",
        project_root=tmp_path,
        python_executable=str(tmp_path / ".venv" / "bin" / "python"),
        runner=_runner,
    )

    assert import_details.import_ready is False
    assert "failed to parse" in str(import_details.import_error)
    assert compatibility.passed is False
    assert "failed to parse" in str(compatibility.error)
    assert repair_attempted is False
    assert repair_actions[0]["status"] == "skipped_non_thrml_import_error"


def test_req_sample_044_probe_helpers_preserve_empty_error_fallback(tmp_path: Path) -> None:
    """REQ-SAMPLE-044: empty command output still records a concrete blocker."""

    def _runner(command: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        return _result(command, returncode=9)

    import_details = probe_import_details(python_executable="python3", runner=_runner)
    compatibility = probe_compatibility(python_executable="python3", runner=_runner)
    repair_attempted, repair_actions = repair_thrml_if_safe(
        "ModuleNotFoundError: No module named 'thrml'",
        project_root=tmp_path,
        python_executable=str(tmp_path / ".venv" / "bin" / "python"),
        runner=_runner,
    )

    assert import_details.import_error == "thrml import details returned non-zero exit code 9"
    assert compatibility.error == "thrml compatibility probe returned non-zero exit code 9"
    assert repair_attempted is True
    assert repair_actions[0]["status"] == "repair_install_failed"


def test_req_sample_044_validator_rejects_invalid_terminal_artifacts() -> None:
    """REQ-SAMPLE-044: validation catches schema and no-hardware-claim violations."""

    valid: dict[str, Any] = {
        "status": "complete",
        "thrml_import_ready": False,
        "import_error": "No module named 'thrml'",
        "repair_attempted": False,
        "repair_actions": [{"status": "skipped_unsafe_python"}],
        "thrml_version": None,
        "thrml_import_path": None,
        "compatibility_probe_passed": False,
        "parity_followup_allowed": False,
        "hardware_claim_allowed": False,
        "blockers": ["repair_skipped_unsafe_python"],
        "honest_verdict": "complete_thrml_import_not_ready_terminal_environment_blocker_no_hardware_claim",
    }
    validate_artifact(valid)

    missing = dict(valid)
    missing.pop("repair_actions")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        validate_artifact(missing)

    invalid = dict(valid)
    invalid["hardware_claim_allowed"] = True
    with pytest.raises(ValueError, match="invalid terminal artifact"):
        validate_artifact(invalid)

    bad_prefix = dict(valid)
    bad_prefix["honest_verdict"] = "blocked_thrml_missing"
    with pytest.raises(ValueError, match="invalid terminal artifact"):
        validate_artifact(bad_prefix)

    ready_missing_version = dict(valid)
    ready_missing_version.update(
        {
            "thrml_import_ready": True,
            "compatibility_probe_passed": True,
            "parity_followup_allowed": True,
            "blockers": [],
            "import_error": None,
        }
    )
    with pytest.raises(ValueError, match="invalid terminal artifact"):
        validate_artifact(ready_missing_version)
