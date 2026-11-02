"""Exp 1503 THRML import-readiness repair gate.

This module keeps the THRML follow-up decision narrow: it records the previous
Exp 1488-style import command, checks the project virtualenv that runs Carnot's
Python tests, optionally installs only the missing `thrml` package into that
virtualenv, and runs a metadata-only compatibility probe. It does not run
Carnot/THRML parity and it never treats THRML software importability as Extropic
TSU hardware evidence.

Spec traces: REQ-SAMPLE-044, SCENARIO-SAMPLE-072.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from carnot.samplers.thrml_installability_preflight import (
    CommandResult,
    _command_payload,
    run_command,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1503_thrml_import_readiness_repair_gate.json"
)

EXPERIMENT_ID = 1503
RUN_DATE = "20260507"
SCHEMA = "thrml_import_readiness_repair_gate_v1"
DEFAULT_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
EXP1488_REPRO_COMMAND = ["python3", "-c", "import thrml"]
TERMINAL_VERDICT_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "thrml_import_ready",
    "import_error",
    "repair_attempted",
    "repair_actions",
    "thrml_version",
    "thrml_import_path",
    "compatibility_probe_passed",
    "parity_followup_allowed",
    "hardware_claim_allowed",
    "blockers",
    "honest_verdict",
}

IMPORT_DETAILS_CODE = """\
import importlib.metadata as metadata
import json
import thrml

metadata_version = None
try:
    metadata_version = metadata.version("thrml")
except Exception:
    metadata_version = None

version = getattr(thrml, "__version__", None) or metadata_version or "unknown"
print(json.dumps({
    "metadata_version": metadata_version,
    "module_name": getattr(thrml, "__name__", "thrml"),
    "path": getattr(thrml, "__file__", None),
    "version": str(version),
}, sort_keys=True))
"""

COMPATIBILITY_PROBE_CODE = """\
import importlib
import json
import thrml

public_surfaces = sorted(name for name in dir(thrml) if not name.startswith("_"))
models_importable = False
models_error = None
try:
    importlib.import_module("thrml.models")
    models_importable = True
except Exception as exc:
    models_error = f"{exc.__class__.__name__}: {exc}"

print(json.dumps({
    "models_error": models_error,
    "models_importable": models_importable,
    "passed": bool(getattr(thrml, "__file__", None)) and len(public_surfaces) > 0,
    "public_surface_count": len(public_surfaces),
    "public_surfaces": public_surfaces[:12],
}, sort_keys=True))
"""


CommandRunner = Callable[[list[str], float], CommandResult]


@dataclass(frozen=True)
class ThrmlImportDetails:
    """Result of the active-environment THRML import details probe.

    Spec traces: REQ-SAMPLE-044.
    """

    import_ready: bool
    version: str | None
    import_path: str | None
    import_error: str | None
    command_result: dict[str, Any]


@dataclass(frozen=True)
class CompatibilityProbe:
    """Metadata-only compatibility result for an already-importable THRML.

    Spec traces: REQ-SAMPLE-044.
    """

    passed: bool
    result: dict[str, Any]
    error: str | None


def _json_from_stdout(stdout: str) -> dict[str, Any]:
    line = next((item.strip() for item in stdout.splitlines() if item.strip()), "")
    return json.loads(line)


def _error_text(result: CommandResult, fallback_label: str) -> str:
    text = result.stderr.strip() or result.stdout.strip()
    return text or f"{fallback_label} returned non-zero exit code {result.returncode}"


def _is_thrml_missing_error(import_error: str | None) -> bool:
    text = import_error or ""
    return "No module named 'thrml'" in text or "ModuleNotFoundError" in text and "thrml" in text


def _is_safe_project_venv_python(project_root: str | Path, python_executable: str) -> bool:
    root = Path(project_root).expanduser().absolute()
    candidate = Path(python_executable).expanduser()
    candidate = candidate if candidate.is_absolute() else root / candidate
    try:
        candidate.absolute().relative_to((root / ".venv").absolute())
    except ValueError:
        return False
    return True


def probe_import_details(
    *,
    python_executable: str = DEFAULT_PYTHON,
    runner: CommandRunner = run_command,
    timeout_s: float = 30.0,
) -> ThrmlImportDetails:
    """Import THRML in the active project Python and capture version/path data.

    Spec traces: REQ-SAMPLE-044.
    """

    result = runner([python_executable, "-c", IMPORT_DETAILS_CODE], timeout_s)
    payload = _command_payload(result)
    if result.returncode != 0:
        return ThrmlImportDetails(
            import_ready=False,
            version=None,
            import_path=None,
            import_error=_error_text(result, "thrml import details"),
            command_result=payload,
        )
    try:
        details = _json_from_stdout(result.stdout)
    except Exception as exc:
        return ThrmlImportDetails(
            import_ready=False,
            version=None,
            import_path=None,
            import_error=f"thrml import details failed to parse JSON: {exc}",
            command_result=payload,
        )
    return ThrmlImportDetails(
        import_ready=True,
        version=str(details.get("version") or details.get("metadata_version") or "unknown"),
        import_path=details.get("path"),
        import_error=None,
        command_result=payload | {"parsed": details},
    )


def probe_compatibility(
    *,
    python_executable: str = DEFAULT_PYTHON,
    runner: CommandRunner = run_command,
    timeout_s: float = 30.0,
) -> CompatibilityProbe:
    """Run the bounded compatibility probe without parity or hardware behavior.

    Spec traces: REQ-SAMPLE-044, SCENARIO-SAMPLE-072.
    """

    result = runner([python_executable, "-c", COMPATIBILITY_PROBE_CODE], timeout_s)
    payload = _command_payload(result)
    if result.returncode != 0:
        error = _error_text(result, "thrml compatibility probe")
        return CompatibilityProbe(passed=False, result=payload, error=error)
    try:
        parsed = _json_from_stdout(result.stdout)
    except Exception as exc:
        error = f"thrml compatibility probe failed to parse JSON: {exc}"
        return CompatibilityProbe(passed=False, result=payload, error=error)
    passed = bool(parsed.get("passed"))
    error = None if passed else str(parsed.get("models_error") or "compatibility probe failed")
    return CompatibilityProbe(passed=passed, result=payload | {"parsed": parsed}, error=error)


def repair_thrml_if_safe(
    import_error: str | None,
    *,
    project_root: str | Path = PROJECT_ROOT,
    python_executable: str = DEFAULT_PYTHON,
    runner: CommandRunner = run_command,
    timeout_s: float = 180.0,
) -> tuple[bool, list[dict[str, Any]]]:
    """Install only `thrml` into the project venv when that is the exact blocker.

    Spec traces: REQ-SAMPLE-044.
    """

    if not _is_thrml_missing_error(import_error):
        return False, [{"status": "skipped_non_thrml_import_error", "mutating_install_performed": False}]
    if not _is_safe_project_venv_python(project_root, python_executable):
        return False, [{"status": "skipped_unsafe_python", "mutating_install_performed": False}]

    command = [
        python_executable,
        "-m",
        "pip",
        "--disable-pip-version-check",
        "install",
        "thrml",
    ]
    result = runner(command, timeout_s)
    payload = _command_payload(result)
    payload.update(
        {
            "mutating_install_performed": True,
            "scope": "project_virtualenv",
            "status": "repair_install_succeeded" if result.returncode == 0 else "repair_install_failed",
        }
    )
    return True, [payload]


def write_in_progress_artifact(output_path: Path = DELIVERABLE_PATH) -> dict[str, Any]:
    """Create the bootstrap artifact before THRML probing starts.

    Spec traces: REQ-SAMPLE-044.
    """

    artifact: dict[str, Any] = {
        "status": "in_progress",
        "thrml_import_ready": False,
        "import_error": None,
        "repair_attempted": False,
        "repair_actions": [],
        "thrml_version": None,
        "thrml_import_path": None,
        "compatibility_probe_passed": False,
        "parity_followup_allowed": False,
        "hardware_claim_allowed": False,
        "blockers": ["in_progress_before_thrml_probe"],
        "honest_verdict": "complete_pending: bootstrap artifact written before THRML import probing",
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "project_root": str(PROJECT_ROOT),
            "run_date": RUN_DATE,
            "schema": SCHEMA,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _collect_blockers(
    terminal_import: ThrmlImportDetails,
    compatibility: CompatibilityProbe,
    repair_actions: list[dict[str, Any]],
) -> list[str]:
    if terminal_import.import_ready and compatibility.passed:
        return []
    blockers: list[str] = []
    if terminal_import.import_error:
        blockers.append(terminal_import.import_error)
    blockers.extend(
        str(action.get("status"))
        for action in repair_actions
        if action.get("status") != "repair_install_succeeded"
    )
    if terminal_import.import_ready and not compatibility.passed:
        blockers.append(f"compatibility_probe_failed: {compatibility.error}")
    return blockers if blockers else ["thrml_import_readiness_not_ready"]


def build_artifact(
    *,
    exp1488_reproduction: Mapping[str, Any],
    initial_import: ThrmlImportDetails,
    terminal_import: ThrmlImportDetails,
    compatibility: CompatibilityProbe,
    repair_attempted: bool,
    repair_actions: list[dict[str, Any]],
    blockers: list[str],
    project_root: str | Path = PROJECT_ROOT,
    python_executable: str = DEFAULT_PYTHON,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the terminal Exp1503 readiness artifact.

    Spec traces: REQ-SAMPLE-044, SCENARIO-SAMPLE-072.
    """

    compatibility_passed = bool(compatibility.passed)
    import_ready = bool(terminal_import.import_ready and compatibility_passed)
    honest_verdict = (
        "complete_thrml_import_ready_repair_gate_open_simulator_only_no_hardware_claim"
        if import_ready
        else "complete_thrml_import_not_ready_terminal_environment_blocker_no_hardware_claim"
    )
    return {
        "status": "complete",
        "thrml_import_ready": import_ready,
        "import_error": None if terminal_import.import_ready else terminal_import.import_error,
        "repair_attempted": repair_attempted,
        "repair_actions": repair_actions,
        "thrml_version": terminal_import.version if terminal_import.import_ready else None,
        "thrml_import_path": terminal_import.import_path if terminal_import.import_ready else None,
        "compatibility_probe_passed": compatibility_passed,
        "parity_followup_allowed": import_ready,
        "hardware_claim_allowed": False,
        "blockers": blockers,
        "honest_verdict": honest_verdict,
        "metadata": {
            "active_python_executable": python_executable,
            "compatibility_probe_result": compatibility.result,
            "experiment_id": EXPERIMENT_ID,
            "exp1488_reproduction": dict(exp1488_reproduction),
            "initial_import_command_result": initial_import.command_result,
            "project_root": str(project_root),
            "run_date": run_date,
            "schema": SCHEMA,
            "terminal_import_command_result": terminal_import.command_result,
        },
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal schema and hard no-hardware-claim gate.

    Spec traces: REQ-SAMPLE-044, SCENARIO-SAMPLE-072.
    """

    missing = REQUIRED_ARTIFACT_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    ready = artifact.get("thrml_import_ready")
    checks = [
        ("status", artifact.get("status") == "complete"),
        ("thrml_import_ready_boolean", isinstance(ready, bool)),
        ("repair_attempted_boolean", isinstance(artifact.get("repair_attempted"), bool)),
        ("repair_actions_list", isinstance(artifact.get("repair_actions"), list)),
        ("compatibility_probe_passed_boolean", isinstance(artifact.get("compatibility_probe_passed"), bool)),
        ("parity_followup_allowed_matches", artifact.get("parity_followup_allowed") is ready),
        ("hardware_claim_allowed_false", artifact.get("hardware_claim_allowed") is False),
        ("blockers_list", isinstance(artifact.get("blockers"), list)),
        ("honest_verdict_prefix", str(artifact.get("honest_verdict", "")).startswith(TERMINAL_VERDICT_PREFIXES)),
    ]
    if ready is True:
        checks.extend(
            [
                ("ready_has_version", bool(artifact.get("thrml_version"))),
                ("ready_has_path", bool(artifact.get("thrml_import_path"))),
                ("ready_no_import_error", artifact.get("import_error") is None),
                ("ready_compatibility_passed", artifact.get("compatibility_probe_passed") is True),
                ("ready_no_blockers", artifact.get("blockers") == []),
            ]
        )
    else:
        checks.extend(
            [
                ("not_ready_has_blockers", len(artifact.get("blockers", [])) > 0),
                ("not_ready_compatibility_false", artifact.get("compatibility_probe_passed") is False),
            ]
        )
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise ValueError(f"invalid terminal artifact: {failed}")


def write_artifact(output_path: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the validated terminal artifact as stable JSON."""

    validate_artifact(artifact)
    payload = dict(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_readiness_gate(
    *,
    output_path: Path = DELIVERABLE_PATH,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = RUN_DATE,
    python_executable: str = DEFAULT_PYTHON,
    runner: CommandRunner = run_command,
) -> dict[str, Any]:
    """Run Exp1503 and write the terminal readiness artifact."""

    write_in_progress_artifact(output_path)
    exp1488_reproduction = _command_payload(runner(EXP1488_REPRO_COMMAND, 30.0))
    initial_import = probe_import_details(python_executable=python_executable, runner=runner)
    repair_attempted = False
    repair_actions: list[dict[str, Any]] = []
    terminal_import = initial_import
    if not initial_import.import_ready:
        repair_attempted, repair_actions = repair_thrml_if_safe(
            initial_import.import_error,
            project_root=project_root,
            python_executable=python_executable,
            runner=runner,
        )
        if repair_attempted and repair_actions[0].get("status") == "repair_install_succeeded":
            terminal_import = probe_import_details(python_executable=python_executable, runner=runner)

    compatibility = (
        probe_compatibility(python_executable=python_executable, runner=runner)
        if terminal_import.import_ready
        else CompatibilityProbe(
            passed=False,
            result={"status": "skipped_import_not_ready"},
            error="THRML import is not ready",
        )
    )
    blockers = _collect_blockers(terminal_import, compatibility, repair_actions)
    artifact = build_artifact(
        exp1488_reproduction=exp1488_reproduction,
        initial_import=initial_import,
        terminal_import=terminal_import,
        compatibility=compatibility,
        repair_attempted=repair_attempted,
        repair_actions=repair_actions,
        blockers=blockers,
        project_root=project_root,
        python_executable=python_executable,
        run_date=run_date,
    )
    return write_artifact(output_path, artifact)
