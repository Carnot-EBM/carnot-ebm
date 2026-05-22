"""Exp 2883 tiny THRML sampler portability smoke.

This module keeps Carnot's THRML path warm at the software boundary only. It
checks whether the current project Python can import JAX, THRML, and Carnot's
local fallback sampler, then runs a four-spin Ising sampler smoke. If THRML is
missing, the artifact records a dependency block and still runs the local
fallback when possible. If THRML is importable, both software lanes run with
fixed seeds and report shape, energy histogram, update-count, and runtime
sanity metrics. No package installation or Extropic TSU hardware claim is made.

Spec traces: REQ-SAMPLE-067, SCENARIO-SAMPLE-095.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from carnot.samplers.thrml_carnot_parity_independent_rng_audit import (
    AuditIsingCase,
    build_audit_cases,
    carnot_cpu_sampler,
    direct_thrml_sampler,
    _samples_to_energies,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_2883_thrml_sampler_portability_smoke_v2.json"
)

EXPERIMENT_ID = 2883
RUN_DATE = "20260522"
SCHEMA = "thrml_sampler_portability_smoke_v2"
DEFAULT_SAMPLE_COUNT = 32
DEFAULT_LOCAL_SEED = 20260522288301
DEFAULT_THRML_SEED = 20260522288399
DEFAULT_N_WARMUP = 4
DEFAULT_STEPS_PER_SAMPLE = 1
DEFAULT_ENERGY_BIN_COUNT = 4

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "thrml_portability_ready",
    "blocked_reason",
    "preconditions_checked",
    "thrml_import_available",
    "jax_devices",
    "local_fallback_ran",
    "problem_spec",
    "sample_count",
    "parity_metrics",
    "hardware_claim_made",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
}

SamplerFn = Callable[[AuditIsingCase], np.ndarray]
InjectedSamplerFn = Callable[..., np.ndarray]


def _exception_text(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{exc.__class__.__name__}: {text}" if text else exc.__class__.__name__


def _device_label(device: Any) -> str:
    platform_name = getattr(device, "platform", None)
    device_id = getattr(device, "id", None)
    if platform_name is not None and device_id is not None:
        return f"{platform_name}:{device_id}"
    return str(device)


def _round_metric(value: float) -> float:
    return round(float(value), 12)


def probe_preconditions(
    *,
    importer: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    """Check Python, JAX, THRML, and local fallback availability.

    Spec traces: REQ-SAMPLE-067.
    """

    payload: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "jax_available": False,
        "jax_version": None,
        "jax_devices": [],
        "jax_default_backend": None,
        "jax_error": None,
        "thrml_import_available": False,
        "thrml_import_error": None,
        "thrml_version": None,
        "thrml_import_path": None,
        "local_fallback_available": False,
        "local_fallback_error": None,
    }
    try:
        jax = importer("jax")
        payload["jax_available"] = True
        payload["jax_version"] = str(getattr(jax, "__version__", "unknown"))
        payload["jax_devices"] = [_device_label(device) for device in jax.devices()]
        payload["jax_default_backend"] = str(jax.default_backend())
    except BaseException as exc:
        payload["jax_error"] = _exception_text(exc)

    try:
        thrml = importer("thrml")
        payload["thrml_import_available"] = True
        payload["thrml_version"] = str(getattr(thrml, "__version__", "unknown"))
        payload["thrml_import_path"] = getattr(thrml, "__file__", None)
    except BaseException as exc:
        payload["thrml_import_error"] = _exception_text(exc)

    try:
        backend = importer("carnot.samplers.backend")
        payload["local_fallback_available"] = bool(getattr(backend, "CpuBackend", None))
    except BaseException as exc:
        payload["local_fallback_error"] = _exception_text(exc)

    return payload


def tiny_portability_case() -> AuditIsingCase:
    """Return the tiny Ising case used for the portability smoke."""

    return build_audit_cases(n_values=(4,), topologies=("signed_ring_chord",))[0]


def local_fallback_sampler(
    case: AuditIsingCase,
    *,
    seed: int,
    n_samples: int,
    schedule: dict[str, Any],
) -> np.ndarray:
    """Sample through Carnot's local CPU fallback backend."""

    return carnot_cpu_sampler(case, seed=seed, n_samples=n_samples, schedule=schedule)


def thrml_software_sampler(  # pragma: no cover - requires optional external THRML.
    case: AuditIsingCase,
    *,
    seed: int,
    n_samples: int,
    schedule: dict[str, Any],
) -> np.ndarray:
    """Sample through the installed THRML software API when it exists."""

    return direct_thrml_sampler(case, seed=seed, n_samples=n_samples, schedule=schedule)


def _problem_spec(case: AuditIsingCase) -> dict[str, Any]:
    upper = np.triu(np.asarray(case.j_matrix), k=1)
    edge_count = int(np.count_nonzero(upper))
    return {
        "name": case.name,
        "source": "carnot.samplers.thrml_carnot_parity_independent_rng_audit.build_audit_cases",
        "problem_family": "tiny_binary_ising_pgm",
        "topology": case.topology,
        "n_spins": int(case.n_spins),
        "edge_count": edge_count,
        "beta": _round_metric(case.beta),
        "bias": [_round_metric(value) for value in case.bias],
    }


def _histogram(
    energies: np.ndarray, *, bin_count: int = DEFAULT_ENERGY_BIN_COUNT
) -> dict[str, Any]:
    values = np.asarray(energies, dtype=np.float64)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if upper <= lower:
        lower -= 0.5
        upper += 0.5
    counts, edges = np.histogram(values, bins=np.linspace(lower, upper, int(bin_count) + 1))
    return {
        "energy_bin_count": int(len(counts)),
        "bin_edges": [_round_metric(edge) for edge in edges],
        "counts": [int(value) for value in counts],
        "nonempty_bins": int(np.count_nonzero(counts)),
        "total_count": int(np.sum(counts)),
    }


def _scheduled_spin_updates(
    case: AuditIsingCase, sample_count: int, schedule: Mapping[str, Any]
) -> int:
    sweeps = int(schedule["n_warmup"]) + int(sample_count) * int(schedule["steps_per_sample"])
    return int(case.n_spins) * sweeps


def _sampler_row(
    *,
    label: str,
    case: AuditIsingCase,
    sampler: InjectedSamplerFn,
    seed: int,
    sample_count: int,
    schedule: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        samples = np.asarray(
            sampler(case, seed=int(seed), n_samples=int(sample_count), schedule=dict(schedule)),
            dtype=bool,
        )
        if samples.shape != (int(sample_count), int(case.n_spins)):
            raise ValueError(
                "sampler returned shape "
                f"{tuple(samples.shape)}; expected {(int(sample_count), int(case.n_spins))}"
            )
        energies = _samples_to_energies(case, samples)
    except BaseException as exc:
        return {
            "label": label,
            "ran": False,
            "error": _exception_text(exc),
            "runtime_s": _round_metric(time.perf_counter() - start),
            "sample_shape": None,
            "mean_energy": None,
            "energy_histogram": None,
            "scheduled_spin_updates": _scheduled_spin_updates(case, sample_count, schedule),
            "acceptance_count_available": False,
            "acceptance_count": None,
        }
    return {
        "label": label,
        "ran": True,
        "error": None,
        "runtime_s": _round_metric(time.perf_counter() - start),
        "sample_shape": [int(value) for value in samples.shape],
        "mean_energy": _round_metric(float(np.mean(energies))),
        "energy_histogram": _histogram(energies),
        "scheduled_spin_updates": _scheduled_spin_updates(case, sample_count, schedule),
        "acceptance_count_available": False,
        "acceptance_count": None,
    }


def _compare_rows(local_row: Mapping[str, Any], thrml_row: Mapping[str, Any]) -> dict[str, Any]:
    local_ran = bool(local_row.get("ran"))
    thrml_ran = bool(thrml_row.get("ran"))
    shape_match = (
        local_row.get("sample_shape") == thrml_row.get("sample_shape")
        if local_ran and thrml_ran
        else None
    )
    local_hist = dict(local_row.get("energy_histogram") or {})
    thrml_hist = dict(thrml_row.get("energy_histogram") or {})
    local_hist_ok = bool(local_hist.get("total_count")) and bool(local_hist.get("nonempty_bins"))
    thrml_hist_ok = bool(thrml_hist.get("total_count")) and bool(thrml_hist.get("nonempty_bins"))
    histogram_sanity_passed = local_hist_ok and (thrml_hist_ok if thrml_ran else True)
    mean_delta = None
    if local_ran and thrml_ran:
        mean_delta = _round_metric(
            abs(float(local_row["mean_energy"]) - float(thrml_row["mean_energy"]))
        )
    return {
        "local": dict(local_row),
        "thrml": dict(thrml_row),
        "shape_match": shape_match,
        "histogram_sanity_passed": bool(histogram_sanity_passed),
        "mean_energy_delta_abs": mean_delta,
    }


def _blocked_reason(
    *,
    preconditions: Mapping[str, Any],
    local_row: Mapping[str, Any],
    thrml_row: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> str:
    if not bool(local_row.get("ran")):
        return "local_fallback_failed"
    if not bool(preconditions.get("thrml_import_available")):
        return "blocked_thrml_unavailable"
    if not bool(thrml_row.get("ran")):
        return "thrml_sampler_failed"
    if (
        comparison.get("shape_match") is not True
        or comparison.get("histogram_sanity_passed") is not True
    ):
        return "parity_sanity_failed"
    return "none"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp2883 schema and no-hardware-claim boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact.get("hardware_claim_made") is not False:
        raise ValueError("hardware_claim_made must remain false")
    if artifact.get("run_date") != RUN_DATE:
        raise ValueError(f"run_date must be {RUN_DATE}")
    if not isinstance(artifact.get("preconditions_checked"), list):
        raise ValueError("preconditions_checked must be a list")
    if not isinstance(artifact.get("jax_devices"), list):
        raise ValueError("jax_devices must be a list")
    if not isinstance(artifact.get("tests_run"), list):
        raise ValueError("tests_run must be a list")
    if float(artifact.get("duration_s", -1.0)) < 0.0:
        raise ValueError("duration_s must be non-negative")
    if artifact.get("thrml_portability_ready") is True:
        if artifact.get("thrml_import_available") is not True:
            raise ValueError("thrml_portability_ready requires THRML import")
        if artifact.get("local_fallback_ran") is not True:
            raise ValueError("thrml_portability_ready requires local fallback run")
        parity = dict(artifact.get("parity_metrics") or {})
        if dict(parity.get("thrml") or {}).get("ran") is not True:
            raise ValueError("thrml_portability_ready requires THRML sampler run")
        if parity.get("shape_match") is not True:
            raise ValueError("thrml_portability_ready requires matching sample shape")
        if parity.get("histogram_sanity_passed") is not True:
            raise ValueError("thrml_portability_ready requires histogram sanity")


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Persist a validated Exp2883 artifact as stable JSON."""

    validate_artifact(artifact)
    payload = dict(artifact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_sampler_portability_smoke(
    *,
    output_path: str | Path | None = DELIVERABLE_PATH,
    preconditions: Mapping[str, Any] | None = None,
    local_sampler: InjectedSamplerFn = local_fallback_sampler,
    thrml_sampler: InjectedSamplerFn = thrml_software_sampler,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    tests_run: Sequence[str] = (),
) -> dict[str, Any]:
    """Run the tiny THRML portability smoke and optionally write the artifact."""

    start = time.perf_counter()
    checked = [
        "python_version",
        "jax_import",
        "jax_device_inventory",
        "thrml_import",
        "local_fallback_sampler",
    ]
    pre = dict(preconditions or probe_preconditions())
    case = tiny_portability_case()
    schedule = {
        "beta": float(case.beta),
        "n_warmup": DEFAULT_N_WARMUP,
        "steps_per_sample": DEFAULT_STEPS_PER_SAMPLE,
        "use_checkerboard": True,
    }
    local_row = (
        _sampler_row(
            label="local_fallback",
            case=case,
            sampler=local_sampler,
            seed=DEFAULT_LOCAL_SEED,
            sample_count=int(sample_count),
            schedule=schedule,
        )
        if bool(pre.get("local_fallback_available", True))
        else {
            "label": "local_fallback",
            "ran": False,
            "error": str(pre.get("local_fallback_error") or "local fallback unavailable"),
            "runtime_s": 0.0,
            "sample_shape": None,
            "mean_energy": None,
            "energy_histogram": None,
            "scheduled_spin_updates": _scheduled_spin_updates(case, int(sample_count), schedule),
            "acceptance_count_available": False,
            "acceptance_count": None,
        }
    )
    thrml_row = (
        _sampler_row(
            label="thrml_software",
            case=case,
            sampler=thrml_sampler,
            seed=DEFAULT_THRML_SEED,
            sample_count=int(sample_count),
            schedule=schedule,
        )
        if bool(pre.get("thrml_import_available"))
        else {
            "label": "thrml_software",
            "ran": False,
            "error": str(pre.get("thrml_import_error") or "THRML import unavailable"),
            "runtime_s": 0.0,
            "sample_shape": None,
            "mean_energy": None,
            "energy_histogram": None,
            "scheduled_spin_updates": _scheduled_spin_updates(case, int(sample_count), schedule),
            "acceptance_count_available": False,
            "acceptance_count": None,
        }
    )
    comparison = _compare_rows(local_row, thrml_row)
    blocked_reason = _blocked_reason(
        preconditions=pre,
        local_row=local_row,
        thrml_row=thrml_row,
        comparison=comparison,
    )
    ready = blocked_reason == "none"
    if ready:
        honest_verdict = (
            "complete: thrml_sampler_portability_smoke_passed_simulator_only_no_hardware_claim"
        )
    elif blocked_reason == "blocked_thrml_unavailable" and bool(local_row.get("ran")):
        honest_verdict = "complete: blocked_thrml_unavailable_local_fallback_ran_no_hardware_claim"
    elif blocked_reason == "local_fallback_failed":
        honest_verdict = (
            "complete: sampler_portability_smoke_blocked_local_fallback_failed_no_hardware_claim"
        )
    else:
        honest_verdict = "complete: thrml_sampler_portability_smoke_blocked_no_hardware_claim"

    artifact = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "project_root": str(PROJECT_ROOT),
            "python_executable": pre.get("python_executable"),
            "python_version": pre.get("python_version"),
            "platform": pre.get("platform"),
            "jax_available": bool(pre.get("jax_available")),
            "jax_version": pre.get("jax_version"),
            "jax_default_backend": pre.get("jax_default_backend"),
            "thrml_version": pre.get("thrml_version"),
            "thrml_import_path": pre.get("thrml_import_path"),
            "local_seed": DEFAULT_LOCAL_SEED,
            "thrml_seed": DEFAULT_THRML_SEED,
            "schedule": schedule,
        },
        "honest_verdict": honest_verdict,
        "thrml_portability_ready": bool(ready),
        "blocked_reason": blocked_reason,
        "preconditions_checked": checked,
        "thrml_import_available": bool(pre.get("thrml_import_available")),
        "jax_devices": list(pre.get("jax_devices") or []),
        "local_fallback_ran": bool(local_row.get("ran")),
        "problem_spec": _problem_spec(case),
        "sample_count": int(sample_count),
        "parity_metrics": comparison,
        "hardware_claim_made": False,
        "tests_run": [str(command) for command in tests_run],
        "field_principles": {
            "software_bridge_to_future_thermodynamic_hardware": True,
            "block_gibbs_sparse_pgm_path_preserved": True,
            "no_tsu_access_claim": True,
            "no_hardware_acceleration_claim": True,
            "no_pip_install_attempted": True,
            "fixed_seeds": True,
            "tiny_problem_only": True,
        },
        "run_date": RUN_DATE,
        "duration_s": _round_metric(time.perf_counter() - start),
    }
    validate_artifact(artifact)
    if output_path is None:
        return artifact
    return write_artifact(output_path, artifact)


def main() -> None:  # pragma: no cover - exercised by operator command.
    run_sampler_portability_smoke()


if __name__ == "__main__":  # pragma: no cover
    main()
