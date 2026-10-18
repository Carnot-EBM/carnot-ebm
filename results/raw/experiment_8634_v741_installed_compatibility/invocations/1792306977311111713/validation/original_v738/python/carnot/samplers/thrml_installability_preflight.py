"""Exp 1488 THRML import/installability preflight for the simulator lane.

The preflight answers one narrow operational question: can the active terminal
Python import `thrml` today? If the answer is no, it performs only a bounded
pip dry-run/no-deps check so the artifact can distinguish "not installed" from
"not currently installable" without changing the project environment. THRML
software readiness is still simulator readiness only; it is never evidence of
Extropic TSU hardware access.

Spec traces: REQ-SAMPLE-043, SCENARIO-SAMPLE-071.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1488_thrml_installability_import_preflight.json"
)

EXPERIMENT_ID = 1488
RUN_DATE = "20260507"
SCHEMA = "thrml_installability_import_preflight_v1"
DEFAULT_PYTHON = "python3"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "thrml_preflight_complete",
    "thrml_import_ready",
    "thrml_version",
    "import_error",
    "install_probe_attempted",
    "install_probe_result",
    "simulator_lane_allowed",
    "hardware_claim_allowed",
    "next_task_gate_value",
    "honest_verdict",
}

HONEST_VERDICTS = {
    "in_progress",
    "thrml_import_ready_simulator_lane_only_no_hardware_claim",
    "thrml_not_importable_bounded_install_probe_installable_simulator_only",
    "thrml_not_importable_bounded_install_probe_blocked_simulator_only",
}

VERSION_PROBE_CODE = """\
import importlib.metadata as metadata
import thrml
version = getattr(thrml, "__version__", None)
if version is None:
    try:
        version = metadata.version("thrml")
    except Exception:
        version = "unknown"
print(version)
"""


@dataclass(frozen=True)
class CommandResult:
    """Captured terminal command result with enough detail for honest artifacts.

    Spec traces: REQ-SAMPLE-043.
    """

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float


@dataclass(frozen=True)
class ThrmlImportProbe:
    """Result of importing THRML through the active Python executable.

    Spec traces: REQ-SAMPLE-043.
    """

    import_ready: bool
    version: str | None
    import_error: str | None
    command_result: dict[str, Any]
    version_command_result: dict[str, Any] | None


@dataclass(frozen=True)
class InstallProbe:
    """Non-mutating installability probe result.

    Spec traces: REQ-SAMPLE-043.
    """

    attempted: bool
    result: dict[str, Any]


CommandRunner = Callable[[list[str], float], CommandResult]


def _compact_text(text: str, *, limit: int = 4000) -> str:
    """Keep subprocess text useful without letting pip output dominate JSON."""

    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _command_payload(result: CommandResult) -> dict[str, Any]:
    """Convert a command result into a stable JSON object."""

    return {
        "command": result.command,
        "returncode": result.returncode,
        "stdout": _compact_text(result.stdout),
        "stderr": _compact_text(result.stderr),
        "timed_out": result.timed_out,
        "duration_s": round(float(result.duration_s), 6),
    }


def run_command(command: list[str], timeout_s: float) -> CommandResult:
    """Run a bounded terminal command and preserve success or failure details.

    The caller decides how to interpret the return code. This helper does not
    hide import failures behind exceptions because the artifact must report the
    active environment exactly as the terminal observed it.

    Spec traces: REQ-SAMPLE-043.
    """

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        duration_s = time.perf_counter() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        timeout_text = f"command timed out after {timeout_s} seconds"
        stderr = f"{stderr}\n{timeout_text}".strip()
        return CommandResult(
            command=command,
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            duration_s=duration_s,
        )

    return CommandResult(
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
        duration_s=time.perf_counter() - started,
    )


def _first_nonempty_text(*values: str) -> str | None:
    """Return the first non-empty command-output string after stripping."""

    for value in values:
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def probe_thrml_import(
    *,
    python_executable: str = DEFAULT_PYTHON,
    runner: CommandRunner = run_command,
    timeout_s: float = 30.0,
) -> ThrmlImportProbe:
    """Probe `python3 -c "import thrml"` and optional version metadata.

    Spec traces: REQ-SAMPLE-043.
    """

    import_result = runner([python_executable, "-c", "import thrml"], timeout_s)
    import_payload = _command_payload(import_result)
    if import_result.returncode != 0:
        error = _first_nonempty_text(import_result.stderr, import_result.stdout)
        if error is None:
            error = f"thrml import returned non-zero exit code {import_result.returncode}"
        return ThrmlImportProbe(
            import_ready=False,
            version=None,
            import_error=error,
            command_result=import_payload,
            version_command_result=None,
        )

    version_result = runner([python_executable, "-c", VERSION_PROBE_CODE], timeout_s)
    version_payload = _command_payload(version_result)
    version = "unknown"
    if version_result.returncode == 0:
        version = _first_nonempty_text(version_result.stdout) or "unknown"
        version = version.splitlines()[-1].strip() or "unknown"
    return ThrmlImportProbe(
        import_ready=True,
        version=version,
        import_error=None,
        command_result=import_payload,
        version_command_result=version_payload,
    )


def probe_installability(
    import_probe: ThrmlImportProbe,
    *,
    python_executable: str = DEFAULT_PYTHON,
    runner: CommandRunner = run_command,
    timeout_s: float = 120.0,
) -> InstallProbe:
    """Run a bounded pip dry-run/no-deps probe only when THRML is absent.

    Spec traces: REQ-SAMPLE-043.
    """

    if import_probe.import_ready:
        return InstallProbe(
            attempted=False,
            result={
                "status": "skipped_import_ready",
                "reason": "active Python environment already imports thrml",
                "mutating_install_performed": False,
            },
        )

    command = [
        python_executable,
        "-m",
        "pip",
        "--disable-pip-version-check",
        "install",
        "--dry-run",
        "--no-deps",
        "thrml",
    ]
    result = runner(command, timeout_s)
    payload = _command_payload(result)
    if result.timed_out:
        status = "dry_run_timeout"
    elif result.returncode == 0:
        status = "dry_run_installable"
    else:
        status = "dry_run_failed"
    payload.update(
        {
            "status": status,
            "mutating_install_performed": False,
        }
    )
    return InstallProbe(attempted=True, result=payload)


def write_in_progress_artifact(output_path: Path = DELIVERABLE_PATH) -> dict[str, Any]:
    """Create the required bootstrap artifact before the terminal probes run.

    Spec traces: REQ-SAMPLE-043.
    """

    artifact: dict[str, Any] = {
        "status": "in_progress",
        "thrml_preflight_complete": False,
        "thrml_import_ready": False,
        "thrml_version": None,
        "import_error": None,
        "install_probe_attempted": False,
        "install_probe_result": {"status": "not_started"},
        "simulator_lane_allowed": False,
        "hardware_claim_allowed": False,
        "next_task_gate_value": "pending_preflight",
        "honest_verdict": "in_progress",
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _honest_verdict(import_probe: ThrmlImportProbe, install_probe: InstallProbe) -> str:
    """Classify the terminal preflight without converting simulation into hardware."""

    if import_probe.import_ready:
        return "thrml_import_ready_simulator_lane_only_no_hardware_claim"
    if install_probe.result.get("status") == "dry_run_installable":
        return "thrml_not_importable_bounded_install_probe_installable_simulator_only"
    return "thrml_not_importable_bounded_install_probe_blocked_simulator_only"


def build_artifact(
    *,
    import_probe: ThrmlImportProbe,
    install_probe: InstallProbe,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the terminal Exp1488 readiness artifact.

    Spec traces: REQ-SAMPLE-043, SCENARIO-SAMPLE-071.
    """

    next_task_gate_value = (
        "thrml_import_ready_simulator_only"
        if import_probe.import_ready
        else "thrml_import_blocked_simulator_only"
    )
    artifact: dict[str, Any] = {
        "status": "complete",
        "thrml_preflight_complete": True,
        "thrml_import_ready": import_probe.import_ready,
        "thrml_version": import_probe.version,
        "import_error": import_probe.import_error,
        "install_probe_attempted": install_probe.attempted,
        "install_probe_result": install_probe.result,
        "simulator_lane_allowed": True,
        "hardware_claim_allowed": False,
        "next_task_gate_value": next_task_gate_value,
        "honest_verdict": _honest_verdict(import_probe, install_probe),
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": run_date,
            "project_root": str(project_root),
            "import_command_result": import_probe.command_result,
            "version_command_result": import_probe.version_command_result,
        },
    }
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal artifact schema and no-hardware-claim gate.

    Spec traces: REQ-SAMPLE-043, SCENARIO-SAMPLE-071.
    """

    missing = REQUIRED_ARTIFACT_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError(f"invalid status: {artifact['status']!r}")
    if artifact["thrml_preflight_complete"] is not True:
        raise ValueError("thrml_preflight_complete must be true for terminal artifact")
    if artifact["hardware_claim_allowed"] is not False:
        raise ValueError("hardware_claim_allowed must remain false")
    if artifact["simulator_lane_allowed"] is not True:
        raise ValueError("simulator_lane_allowed must be true after preflight completion")
    if not isinstance(artifact["install_probe_result"], Mapping):
        raise ValueError("install_probe_result must be an object")
    if artifact["honest_verdict"] not in HONEST_VERDICTS - {"in_progress"}:
        raise ValueError(f"invalid honest_verdict: {artifact['honest_verdict']!r}")

    if artifact["thrml_import_ready"] is True:
        if artifact["import_error"] is not None or artifact["thrml_version"] is None:
            raise ValueError("import_ready requires a version and no import_error")
        if artifact["install_probe_attempted"] is not False:
            raise ValueError("import_ready must skip the installability probe")
        return

    if artifact["thrml_import_ready"] is not False:
        raise ValueError("thrml_import_ready must be a boolean")
    if not artifact["import_error"]:
        raise ValueError("import_error is required when THRML is not import-ready")
    if artifact["install_probe_attempted"] is not True:
        raise ValueError("install_probe_attempted must be true when THRML is not import-ready")


def write_artifact(output_path: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the validated artifact as stable, sorted JSON."""

    payload = dict(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_preflight(
    *,
    output_path: Path = DELIVERABLE_PATH,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = RUN_DATE,
    python_executable: str = DEFAULT_PYTHON,
    runner: CommandRunner = run_command,
) -> dict[str, Any]:
    """Run the full Exp1488 preflight and write the terminal artifact."""

    write_in_progress_artifact(output_path)
    import_probe = probe_thrml_import(python_executable=python_executable, runner=runner)
    install_probe = probe_installability(
        import_probe,
        python_executable=python_executable,
        runner=runner,
    )
    artifact = build_artifact(
        import_probe=import_probe,
        install_probe=install_probe,
        project_root=project_root,
        run_date=run_date,
    )
    validate_artifact(artifact)
    return write_artifact(output_path, artifact)
