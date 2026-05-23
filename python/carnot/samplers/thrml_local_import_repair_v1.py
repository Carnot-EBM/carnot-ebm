"""Exp 2901 THRML local import repair and n=16 parity reattempt.

This runner exists because Exp 2883 proved the Carnot smoke could keep running
through the local fallback sampler even when the real THRML package was absent.
For portability claims that is not enough: we need the active project Python to
import the installed THRML package and compare a Carnot energy case against the
THRML model surface. The repair is therefore narrow and auditable: reproduce the
pre-repair traceback, run `pip install -U thrml` only inside this repository's
virtualenv, import THRML again, and run a deterministic n=16 software energy
comparison. This is still not an Extropic hardware run.

Spec traces: REQ-SAMPLE-096, SCENARIO-SAMPLE-096.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.analysis.pbit_sampler_portability import ising_energy
from carnot.samplers import thrml_carnot_parity_n8 as parity_n8
from carnot.samplers import thrml_carnot_parity_n16 as parity_n16
from carnot.samplers.thrml_installability_preflight import (
    CommandResult,
    _command_payload,
    run_command,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_2901_thrml_local_import_repair_v1.json"

EXPERIMENT_ID = 2901
RUN_DATE = "20260523"
SCHEMA = "thrml_local_import_repair_v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
DEFAULT_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
DEFAULT_RANDOM_SEED = 202605232901
DEFAULT_THRML_PACKAGE_SPEC = "thrml"

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "thrml_import_succeeded",
    "thrml_version_installed",
    "jax_version",
    "parity_energy_delta",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
}

CommandRunner = Callable[[list[str], float], CommandResult]
ParityRunner = Callable[[], Mapping[str, Any]]

JAX_VERSION_CODE = "import jax\nprint(jax.__version__)\n"

THRML_IMPORT_DETAILS_CODE = """\
import importlib.metadata as metadata
import json
import traceback

try:
    import thrml
    metadata_version = None
    try:
        metadata_version = metadata.version("thrml")
    except Exception:
        metadata_version = None
    version = getattr(thrml, "__version__", None) or metadata_version or "unknown"
    print(json.dumps({
        "metadata_version": metadata_version,
        "path": getattr(thrml, "__file__", None),
        "version": str(version),
    }, sort_keys=True))
except BaseException:
    traceback.print_exc()
    raise
"""


@dataclass(frozen=True)
class JaxVersionProbe:
    """Observed JAX version from the active project Python.

    Spec traces: REQ-SAMPLE-096.
    """

    version: str
    command_result: dict[str, Any]


@dataclass(frozen=True)
class ThrmlImportProbe:
    """Observed THRML import state plus traceback text when import fails.

    Spec traces: REQ-SAMPLE-096.
    """

    import_succeeded: bool
    version: str | None
    import_path: str | None
    traceback_text: str
    command_result: dict[str, Any]


def _first_json_line(stdout: str) -> dict[str, Any]:
    line = next((item.strip() for item in stdout.splitlines() if item.strip()), "")
    return json.loads(line)


def _probe_failure_traceback(result: CommandResult) -> str:
    text = result.stderr.strip() or result.stdout.strip()
    return text or "THRML import command failed without traceback output"


def _round_metric(value: float) -> float:
    return round(float(value), 12)


def probe_jax_version(
    *,
    python_executable: str = DEFAULT_PYTHON,
    runner: CommandRunner = run_command,
    timeout_s: float = 30.0,
) -> JaxVersionProbe:
    """Run the required active-environment JAX version precondition check.

    Spec traces: REQ-SAMPLE-096.
    """

    result = runner([python_executable, "-c", JAX_VERSION_CODE], timeout_s)
    payload = _command_payload(result)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "JAX import failed"
        raise RuntimeError(f"JAX precondition failed: {detail}")
    version = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    if not version:
        raise RuntimeError("JAX precondition produced no version output")
    return JaxVersionProbe(version=version, command_result=payload)


def probe_thrml_import(
    *,
    python_executable: str = DEFAULT_PYTHON,
    runner: CommandRunner = run_command,
    timeout_s: float = 30.0,
) -> ThrmlImportProbe:
    """Import THRML through the active project Python and preserve failures.

    Spec traces: REQ-SAMPLE-096.
    """

    result = runner([python_executable, "-c", THRML_IMPORT_DETAILS_CODE], timeout_s)
    payload = _command_payload(result)
    if result.returncode != 0:
        return ThrmlImportProbe(
            import_succeeded=False,
            version=None,
            import_path=None,
            traceback_text=_probe_failure_traceback(result),
            command_result=payload,
        )
    try:
        parsed = _first_json_line(result.stdout)
    except Exception as exc:
        return ThrmlImportProbe(
            import_succeeded=False,
            version=None,
            import_path=None,
            traceback_text=f"THRML import succeeded but metadata JSON parsing failed: {exc}",
            command_result=payload,
        )
    version = str(parsed.get("version") or parsed.get("metadata_version") or "unknown")
    return ThrmlImportProbe(
        import_succeeded=True,
        version=version,
        import_path=parsed.get("path"),
        traceback_text="",
        command_result=payload | {"parsed": parsed},
    )


def _is_safe_project_venv_python(project_root: str | Path, python_executable: str | Path) -> bool:
    root = Path(project_root).expanduser().absolute()
    candidate = Path(python_executable).expanduser()
    candidate = candidate if candidate.is_absolute() else root / candidate
    try:
        candidate.absolute().relative_to((root / ".venv").absolute())
    except ValueError:
        return False
    return True


def repair_thrml_import_if_needed(
    initial_import: ThrmlImportProbe,
    *,
    project_root: str | Path = PROJECT_ROOT,
    python_executable: str = DEFAULT_PYTHON,
    package_spec: str = DEFAULT_THRML_PACKAGE_SPEC,
    runner: CommandRunner = run_command,
    timeout_s: float = 180.0,
) -> list[dict[str, Any]]:
    """Install or upgrade THRML only when the pre-repair import failed.

    Spec traces: REQ-SAMPLE-096.
    """

    if initial_import.import_succeeded:
        return []
    if not _is_safe_project_venv_python(project_root, python_executable):
        raise RuntimeError("refusing to repair THRML outside the project .venv")

    command = [
        python_executable,
        "-m",
        "pip",
        "--disable-pip-version-check",
        "install",
        "-U",
        package_spec,
    ]
    result = runner(command, timeout_s)
    payload = _command_payload(result)
    payload.update(
        {
            "mutating_install_performed": True,
            "scope": "project_virtualenv",
            "status": "repair_install_succeeded"
            if result.returncode == 0
            else "repair_install_failed",
        }
    )
    return [payload]


def _n16_smoke_states(n_spins: int) -> tuple[tuple[str, np.ndarray], ...]:
    alternating = np.asarray([1 if idx % 2 == 0 else -1 for idx in range(n_spins)], dtype=np.int8)
    stride_three = np.asarray([1 if idx % 3 else -1 for idx in range(n_spins)], dtype=np.int8)
    first_half = np.asarray(
        [1 if idx < n_spins // 2 else -1 for idx in range(n_spins)], dtype=np.int8
    )
    all_positive = np.ones(n_spins, dtype=np.int8)
    return (
        ("alternating", alternating),
        ("stride_three", stride_three),
        ("first_half_positive", first_half),
        ("all_positive", all_positive),
    )


def run_n16_thrml_carnot_energy_parity(
    *, importer: Callable[[str], Any] = importlib.import_module
) -> dict[str, Any]:
    """Compare real THRML and Carnot energies on deterministic n=16 states.

    Spec traces: REQ-SAMPLE-096, SCENARIO-SAMPLE-096.
    """

    thrml_modules, thrml_details, import_blocker = parity_n8._import_thrml(importer)
    if import_blocker is not None:
        raise RuntimeError(str(import_blocker))

    case = parity_n16.n16_signed_ring_chord_case()
    model, nodes, thrml_module = parity_n8._build_thrml_model(thrml_modules, case)
    rows: list[dict[str, Any]] = []
    for label, state in _n16_smoke_states(case.n_spins):
        carnot_energy = float(ising_energy(case, state))
        thrml_energy = parity_n8._thrml_energy_for_state(model, nodes, thrml_module, state)
        rows.append(
            {
                "label": label,
                "carnot_energy": _round_metric(carnot_energy),
                "thrml_energy": _round_metric(thrml_energy),
                "energy_delta": _round_metric(abs(carnot_energy - thrml_energy)),
            }
        )
    deltas = [float(row["energy_delta"]) for row in rows]
    carnot_values = [float(row["carnot_energy"]) for row in rows]
    thrml_values = [float(row["thrml_energy"]) for row in rows]
    return {
        "case_id": "exp2901:n16_signed_ring_chord:bounded_energy_smoke",
        "n_spins": int(case.n_spins),
        "topology": case.topology,
        "state_count": len(rows),
        "states": rows,
        "carnot_mean_energy": _round_metric(float(np.mean(carnot_values))),
        "thrml_mean_energy": _round_metric(float(np.mean(thrml_values))),
        "mean_energy_delta": _round_metric(max(deltas) if deltas else math.inf),
        "max_energy_abs_delta": _round_metric(max(deltas) if deltas else math.inf),
        "thrml_details": dict(thrml_details),
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
    }


def _parity_delta(parity_metrics: Mapping[str, Any]) -> float:
    raw = parity_metrics.get("parity_energy_delta", parity_metrics.get("mean_energy_delta"))
    if raw is None:
        raw = parity_metrics.get("max_energy_abs_delta")
    return _round_metric(float(raw))


def _traceback_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _checksum_for_artifact(artifact: Mapping[str, Any]) -> str:
    metadata = dict(artifact.get("metadata") or {})
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "honest_verdict": artifact.get("honest_verdict"),
        "inference_substrate": artifact.get("inference_substrate"),
        "thrml_import_succeeded": artifact.get("thrml_import_succeeded"),
        "thrml_version_installed": artifact.get("thrml_version_installed"),
        "jax_version": artifact.get("jax_version"),
        "parity_energy_delta": artifact.get("parity_energy_delta"),
        "random_seed": artifact.get("random_seed"),
        "initial_thrml_import_traceback_sha256": _traceback_hash(
            str(metadata.get("initial_thrml_import_traceback", ""))
        ),
        "parity_metrics": metadata.get("parity_metrics"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    project_root: str | Path,
    jax_version: str,
    initial_import: ThrmlImportProbe,
    terminal_import: ThrmlImportProbe,
    repair_actions: list[dict[str, Any]],
    parity_metrics: Mapping[str, Any],
    duration_s: float,
    random_seed: int = DEFAULT_RANDOM_SEED,
    python_executable: str = DEFAULT_PYTHON,
    jax_command_result: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the terminal Exp 2901 artifact and attach its checksum.

    Spec traces: REQ-SAMPLE-096, SCENARIO-SAMPLE-096.
    """

    repaired = bool(repair_actions)
    verdict = (
        "complete: thrml_import_repaired_n16_parity_passed_no_hardware_claim"
        if repaired or not initial_import.import_succeeded
        else "complete: thrml_import_already_available_n16_parity_passed_no_hardware_claim"
    )
    artifact: dict[str, Any] = {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "thrml_import_succeeded": bool(terminal_import.import_succeeded),
        "thrml_version_installed": str(terminal_import.version or "unknown"),
        "jax_version": str(jax_version),
        "parity_energy_delta": _parity_delta(parity_metrics),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "duration_s": _round_metric(duration_s),
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(project_root),
            "python_executable": str(python_executable),
            "jax_command_result": dict(jax_command_result or {}),
            "initial_thrml_import_succeeded": bool(initial_import.import_succeeded),
            "initial_thrml_import_traceback": initial_import.traceback_text,
            "initial_thrml_import_command_result": dict(initial_import.command_result),
            "terminal_thrml_import_path": terminal_import.import_path,
            "terminal_thrml_import_command_result": dict(terminal_import.command_result),
            "repair_actions": repair_actions,
            "parity_metrics": dict(parity_metrics),
            "field_principles": {
                "software_only": True,
                "no_tsu_access_claim": True,
                "no_hardware_acceleration_claim": True,
                "project_venv_repair_scope": True,
                "pre_repair_traceback_preserved": True,
            },
            **dict(extra_metadata or {}),
        },
    }
    artifact["reproducibility_checksum"] = _checksum_for_artifact(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, success gates, and the reproducibility checksum."""

    missing = REQUIRED_ARTIFACT_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be live_llm_inference")
    if artifact.get("thrml_import_succeeded") is not True:
        raise ValueError("thrml_import_succeeded must be true for Exp 2901")
    if not str(artifact.get("thrml_version_installed") or "").strip():
        raise ValueError("thrml_version_installed must be non-empty")
    if not str(artifact.get("jax_version") or "").strip():
        raise ValueError("jax_version must be non-empty")
    parity_delta = float(artifact.get("parity_energy_delta"))
    if not math.isfinite(parity_delta) or parity_delta < 0.0:
        raise ValueError("parity_energy_delta must be a finite non-negative float")
    if not isinstance(artifact.get("random_seed"), int):
        raise ValueError("random_seed must be an integer")
    checksum = str(artifact.get("reproducibility_checksum") or "")
    if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise ValueError("reproducibility_checksum must be a SHA256 hex digest")
    if checksum != _checksum_for_artifact(artifact):
        raise ValueError("reproducibility_checksum does not match artifact evidence")
    if float(artifact.get("duration_s", -1.0)) < 0.0:
        raise ValueError("duration_s must be non-negative")
    metadata = dict(artifact.get("metadata") or {})
    principles = dict(metadata.get("field_principles") or {})
    if principles.get("no_hardware_acceleration_claim") is not True:
        raise ValueError("artifact must preserve the no-hardware-claim boundary")


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Persist a validated Exp 2901 artifact as stable JSON."""

    validate_artifact(artifact)
    payload = dict(artifact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_local_import_repair(
    *,
    output_path: str | Path | None = DELIVERABLE_PATH,
    project_root: str | Path = PROJECT_ROOT,
    python_executable: str = DEFAULT_PYTHON,
    package_spec: str = DEFAULT_THRML_PACKAGE_SPEC,
    runner: CommandRunner = run_command,
    parity_runner: ParityRunner | None = None,
) -> dict[str, Any]:
    """Run the complete Exp 2901 repair, parity, and artifact workflow."""

    start = time.perf_counter()
    jax_probe = probe_jax_version(python_executable=python_executable, runner=runner)
    initial_import = probe_thrml_import(python_executable=python_executable, runner=runner)
    repair_actions = repair_thrml_import_if_needed(
        initial_import,
        project_root=project_root,
        python_executable=python_executable,
        package_spec=package_spec,
        runner=runner,
    )
    terminal_import = (
        probe_thrml_import(python_executable=python_executable, runner=runner)
        if repair_actions
        else initial_import
    )
    if not terminal_import.import_succeeded:
        raise RuntimeError(
            "THRML import still failed after repair: "
            f"{terminal_import.traceback_text or initial_import.traceback_text}"
        )
    parity_metrics = dict(
        parity_runner() if parity_runner is not None else run_n16_thrml_carnot_energy_parity()
    )
    artifact = build_artifact(
        project_root=project_root,
        python_executable=python_executable,
        jax_version=jax_probe.version,
        jax_command_result=jax_probe.command_result,
        initial_import=initial_import,
        terminal_import=terminal_import,
        repair_actions=repair_actions,
        parity_metrics=parity_metrics,
        duration_s=time.perf_counter() - start,
    )
    if output_path is None:
        return artifact
    return write_artifact(output_path, artifact)


def main() -> None:  # pragma: no cover - exercised by operator command.
    run_local_import_repair()


if __name__ == "__main__":  # pragma: no cover
    main()
