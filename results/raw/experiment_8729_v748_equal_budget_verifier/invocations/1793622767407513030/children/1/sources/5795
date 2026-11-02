"""Exp 1530 sampled n=128 THRML/Carnot simulator parity.

This module extends the Exp 1528/1529 sampled THRML parity pattern to the
production-scale n=128 spin count. It is deliberately a software simulator
comparison: the Carnot lane and THRML lane use identical Ising parameters,
fixed seeds, and comparable schedules, then report sampled energy,
magnetization, autocorrelation, empirical KL, runtime, and process-memory
diagnostics. It does not execute or claim Extropic TSU hardware.

Spec traces: REQ-SAMPLE-051, SCENARIO-SAMPLE-079.
"""

from __future__ import annotations

import importlib
import json
import resource
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from carnot.samplers import thrml_carnot_parity_n64_sample as parity_n64
from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

PROJECT_ROOT = parity_n64.PROJECT_ROOT
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1530_thrml_carnot_parity_n128_production_scale.json"
)
PARITY_MANIFEST_PATH = PROJECT_ROOT / "results" / "thrml_carnot_parity_n128_1530.jsonl"
EXP1529_PATH = PROJECT_ROOT / "results" / "experiment_1529_thrml_carnot_parity_n64_sample.json"

EXPERIMENT_ID = 1530
RUN_DATE = "20260508"
SCHEMA = "thrml_carnot_parity_n128_production_scale_v1"
DEFAULT_SEEDS = parity_n64.DEFAULT_SEEDS
DEFAULT_SAMPLE_COUNT_PER_SEED = parity_n64.DEFAULT_SAMPLE_COUNT_PER_SEED
DEFAULT_N_WARMUP = 512
DEFAULT_STEPS_PER_SAMPLE = parity_n64.DEFAULT_STEPS_PER_SAMPLE
DEFAULT_ENERGY_BIN_COUNT = parity_n64.DEFAULT_ENERGY_BIN_COUNT

THRESHOLDS = {
    "mean_energy_delta_abs_max": 0.60,
    "mean_energy_delta_percent_max": 0.10,
    "magnetization_delta_abs_max": 0.035,
    "kl_divergence_max": 0.15,
    "kl_min_samples_per_backend": 10_000,
    "autocorrelation_lag1_delta_abs_max": 0.15,
}
TERMINAL_VERDICT_PREFIXES = parity_n64.TERMINAL_VERDICT_PREFIXES
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "thrml_parity_n128_passed",
    "simulator_only",
    "no_tsu_hardware_claim",
    "n_spins",
    "topology",
    "seeds",
    "n_samples_per_backend",
    "mean_energy_delta",
    "magnetization_delta",
    "autocorrelation_summary",
    "kl_divergence",
    "runtime_seconds_by_backend",
    "memory_summary",
    "parity_manifest_path",
    "blockers",
    "honest_verdict",
}

ImportModule = parity_n64.ImportModule
BackendFactory = parity_n64.BackendFactory
ParityIsingCase = parity_n64.ParityIsingCase


def n128_signed_ring_chord_case() -> ParityIsingCase:
    """Return the deterministic 128-spin signed ring-chord parity case.

    The topology keeps the scale sweep lineage stable: every spin has one
    signed nearest-neighbor ring edge and one signed distance-two chord. The
    explicit periodic weights avoid hidden randomness while still exercising
    mixed ferro/antiferromagnetic structure at the production spin count.

    Spec traces: REQ-SAMPLE-051.
    """

    n_spins = 128
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    ring_weights = np.tile(
        np.array(
            [
                0.16,
                -0.10,
                0.13,
                -0.06,
                0.11,
                -0.08,
                0.14,
                -0.04,
                0.12,
                -0.09,
                0.15,
                -0.05,
                0.10,
                -0.07,
                0.17,
                -0.03,
            ],
            dtype=np.float64,
        ),
        8,
    )
    chord_weights = np.tile(
        np.array(
            [
                -0.060,
                0.050,
                -0.040,
                0.025,
                -0.055,
                0.035,
                -0.030,
                0.015,
                -0.050,
                0.045,
                -0.035,
                0.020,
                -0.045,
                0.030,
                -0.025,
                0.010,
            ],
            dtype=np.float64,
        ),
        8,
    )
    for idx, weight in enumerate(ring_weights):
        left, right = idx, (idx + 1) % n_spins
        j_matrix[left, right] = j_matrix[right, left] = float(weight)
    for idx, weight in enumerate(chord_weights):
        left, right = idx, (idx + 2) % n_spins
        j_matrix[left, right] = j_matrix[right, left] = float(weight)
    bias = np.tile(
        np.array(
            [
                0.010,
                -0.018,
                0.014,
                -0.022,
                0.018,
                -0.014,
                0.006,
                -0.009,
                0.012,
                -0.016,
                0.020,
                -0.012,
                0.008,
                -0.010,
                0.016,
                -0.006,
            ],
            dtype=np.float64,
        ),
        8,
    )
    return ParityIsingCase(
        name="n128_signed_ring_chord",
        topology="signed_ring_chord",
        j_matrix=j_matrix,
        bias=bias,
        beta=1.0,
    )


def _display_path(path: str | Path) -> str:
    return parity_n64._display_path(path)


def _round_metric(value: float) -> float:
    return parity_n64._round_metric(value)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return parity_n64._write_json(path, payload)


def _write_manifest(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    parity_n64._write_manifest(path, [dict(row) for row in rows])


def _load_exp1529_ready(path: str | Path) -> tuple[bool, dict[str, Any], dict[str, str] | None]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return False, {}, {"blocker": "exp1529_evidence_missing", "detail": str(exc)}
    except json.JSONDecodeError as exc:
        return False, {}, {"blocker": "exp1529_evidence_malformed", "detail": str(exc)}
    ready = (
        payload.get("status") == "complete"
        and payload.get("thrml_parity_n64_passed") is True
        and payload.get("simulator_only") is True
        and payload.get("no_tsu_hardware_claim") is True
    )
    if not ready:
        return (
            False,
            payload,
            {
                "blocker": "exp1529_parity_not_passed",
                "detail": "Exp1529 must be complete, n=64-passed, simulator-only, and no-TSU-claim",
            },
        )
    return True, payload, None


def write_in_progress_artifact(
    path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
) -> dict[str, Any]:
    """Write the bootstrap artifact before import probing or parity execution.

    Spec traces: REQ-SAMPLE-051.
    """

    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "board_execution": False,
            "synthesis_run": False,
            "bitstream_generated": False,
            "hardware_acceleration_claimed": False,
        },
        "status": "in_progress",
        "thrml_parity_n128_passed": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "n_spins": 128,
        "topology": "signed_ring_chord",
        "seeds": list(DEFAULT_SEEDS),
        "n_samples_per_backend": 0,
        "sample_count_per_seed": DEFAULT_SAMPLE_COUNT_PER_SEED,
        "warmup": DEFAULT_N_WARMUP,
        "thinning": DEFAULT_STEPS_PER_SAMPLE,
        "mean_energy_delta": None,
        "mean_energy_delta_percent": None,
        "magnetization_delta": None,
        "autocorrelation_summary": {},
        "kl_divergence": None,
        "kl_estimate_stable": False,
        "stability_diagnostics_present": False,
        "thresholds": dict(THRESHOLDS),
        "energy_bin_count": DEFAULT_ENERGY_BIN_COUNT,
        "runtime_seconds_by_backend": {},
        "memory_summary": {},
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": [{"blocker": "parity_run_not_completed", "detail": "bootstrap artifact only"}],
        "honest_verdict": "success_in_progress_thrml_carnot_parity_n128_simulator_only",
    }
    validate_artifact(artifact)
    return _write_json(path, artifact)


def _blocked_artifact(
    *,
    manifest_path: str | Path,
    seeds: Sequence[int],
    blockers: list[dict[str, str]],
    verdict: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "board_execution": False,
            "synthesis_run": False,
            "bitstream_generated": False,
            "hardware_acceleration_claimed": False,
            **dict(metadata or {}),
        },
        "status": "blocked",
        "thrml_parity_n128_passed": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "n_spins": 128,
        "topology": "signed_ring_chord",
        "seeds": [int(seed) for seed in seeds],
        "n_samples_per_backend": 0,
        "sample_count_per_seed": DEFAULT_SAMPLE_COUNT_PER_SEED,
        "warmup": DEFAULT_N_WARMUP,
        "thinning": DEFAULT_STEPS_PER_SAMPLE,
        "mean_energy_delta": None,
        "mean_energy_delta_percent": None,
        "magnetization_delta": None,
        "autocorrelation_summary": {},
        "kl_divergence": None,
        "kl_estimate_stable": False,
        "stability_diagnostics_present": False,
        "thresholds": dict(THRESHOLDS),
        "energy_bin_count": DEFAULT_ENERGY_BIN_COUNT,
        "runtime_seconds_by_backend": {},
        "memory_summary": {},
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": blockers,
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def sampled_backend_row(
    case: ParityIsingCase,
    *,
    seed: int,
    backend_label: str,
    backend_name: str,
    samples: np.ndarray,
    schedule: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize one Exp 1530 seed/backend sample group.

    Spec traces: REQ-SAMPLE-051.
    """

    row = parity_n64.sampled_backend_row(
        case,
        seed=seed,
        backend_label=backend_label,
        backend_name=backend_name,
        samples=samples,
        schedule=schedule,
    )
    row["case_id"] = f"exp1530:{case.name}:seed_{int(seed)}:{backend_label}"
    return row


def summarize_sampled_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    thresholds: Mapping[str, float],
    energy_bin_count: int,
) -> dict[str, Any]:
    """Build the cross-seed n=128 sampled parity summary row.

    Spec traces: REQ-SAMPLE-051, SCENARIO-SAMPLE-079.
    """

    summary = parity_n64.summarize_sampled_rows(
        rows,
        seeds=seeds,
        thresholds=thresholds,
        energy_bin_count=energy_bin_count,
    )
    summary.update(
        {
            "case_id": "exp1530:n128_signed_ring_chord:sampled_summary",
            "thresholds": dict(thresholds),
        }
    )
    return summary


def _memory_snapshot() -> dict[str, int]:
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    return {
        "tracemalloc_current_bytes": int(current),
        "tracemalloc_peak_bytes": int(peak),
        "rss_max_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _runtime_diagnostics_present(runtime: Mapping[str, Any]) -> bool:
    return all(key in runtime for key in ("carnot", "thrml")) and all(
        float(runtime[key]) >= 0.0 for key in ("carnot", "thrml")
    )


def _memory_diagnostics_present(memory: Mapping[str, Any]) -> bool:
    return all(
        key in memory
        for key in (
            "method",
            "rss_max_kib_start",
            "rss_max_kib_end",
            "tracemalloc_peak_bytes",
        )
    )


def run_parity_n128(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
    exp1529_path: str | Path = EXP1529_PATH,
    importer: ImportModule = importlib.import_module,
    carnot_backend_factory: BackendFactory = CpuBackend,
    thrml_backend_factory: BackendFactory = ThrmlSamplerBackend,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    sample_count_per_seed: int = DEFAULT_SAMPLE_COUNT_PER_SEED,
    n_warmup: int = DEFAULT_N_WARMUP,
    steps_per_sample: int = DEFAULT_STEPS_PER_SAMPLE,
    thresholds: Mapping[str, float] = THRESHOLDS,
    energy_bin_count: int = DEFAULT_ENERGY_BIN_COUNT,
) -> dict[str, Any]:
    """Run sampled n=128 THRML/Carnot parity and write JSON/JSONL evidence."""

    write_in_progress_artifact(output_path, manifest_path)
    exp1529_ready, exp1529_payload, exp1529_blocker = _load_exp1529_ready(exp1529_path)
    if not exp1529_ready:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                seeds=seeds,
                blockers=[
                    exp1529_blocker
                    or {
                        "blocker": "exp1529_parity_not_passed",
                        "detail": "unknown Exp1529 blocker",
                    }
                ],
                verdict="complete_thrml_carnot_parity_n128_blocked_exp1529_no_tsu_hardware_claim",
                metadata={"exp1529_status": exp1529_payload.get("status")},
            ),
        )
    _thrml_modules, thrml_details, import_blocker = (
        parity_n64.parity_n32.parity_n16.parity_n8._import_thrml(importer)
    )
    if import_blocker is not None:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                seeds=seeds,
                blockers=[import_blocker],
                verdict=(
                    "complete_thrml_carnot_parity_n128_blocked_simulator_dependency_"
                    "no_tsu_hardware_claim"
                ),
                metadata={"exp1529_status": exp1529_payload.get("status")},
            ),
        )

    case = n128_signed_ring_chord_case()
    schedule = {
        "beta": float(case.beta),
        "n_warmup": int(n_warmup),
        "steps_per_sample": int(steps_per_sample),
        "use_checkerboard": True,
    }
    memory_start = _memory_snapshot()
    runtime_seconds = {"carnot": 0.0, "thrml": 0.0}
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        carnot_backend = carnot_backend_factory(int(seed))
        thrml_backend = thrml_backend_factory(int(seed))

        carnot_start = time.perf_counter()
        carnot_samples = np.asarray(
            carnot_backend.sample(case.bias, case.j_matrix, int(sample_count_per_seed), schedule)
        )
        runtime_seconds["carnot"] += time.perf_counter() - carnot_start

        thrml_start = time.perf_counter()
        thrml_samples = np.asarray(
            thrml_backend.sample(case.bias, case.j_matrix, int(sample_count_per_seed), schedule)
        )
        runtime_seconds["thrml"] += time.perf_counter() - thrml_start

        rows.append(
            sampled_backend_row(
                case,
                seed=int(seed),
                backend_label="carnot",
                backend_name=str(getattr(carnot_backend, "backend_name", "<unknown>")),
                samples=carnot_samples,
                schedule=schedule,
            )
        )
        rows.append(
            sampled_backend_row(
                case,
                seed=int(seed),
                backend_label="thrml",
                backend_name=str(getattr(thrml_backend, "backend_name", "<unknown>")),
                samples=thrml_samples,
                schedule=schedule,
            )
        )

    summary_row = summarize_sampled_rows(
        rows,
        seeds=seeds,
        thresholds=thresholds,
        energy_bin_count=energy_bin_count,
    )
    _write_manifest(manifest_path, [*rows, summary_row])
    passed = bool(summary_row["passed_thresholds"])
    memory_end = _memory_snapshot()
    runtime_seconds_by_backend = {
        "carnot": _round_metric(runtime_seconds["carnot"]),
        "thrml": _round_metric(runtime_seconds["thrml"]),
    }
    memory_summary = {
        "method": "python_tracemalloc_resource",
        "rss_max_kib_start": memory_start["rss_max_kib"],
        "rss_max_kib_end": memory_end["rss_max_kib"],
        "tracemalloc_current_bytes_start": memory_start["tracemalloc_current_bytes"],
        "tracemalloc_peak_bytes_start": memory_start["tracemalloc_peak_bytes"],
        "tracemalloc_current_bytes": memory_end["tracemalloc_current_bytes"],
        "tracemalloc_peak_bytes": memory_end["tracemalloc_peak_bytes"],
    }
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "board_execution": False,
            "synthesis_run": False,
            "bitstream_generated": False,
            "hardware_acceleration_claimed": False,
            "exp1529_status": exp1529_payload.get("status"),
            **thrml_details,
        },
        "status": "complete",
        "thrml_parity_n128_passed": passed,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "n_spins": case.n_spins,
        "topology": case.topology,
        "seeds": [int(seed) for seed in seeds],
        "n_samples_per_backend": int(summary_row["n_samples_per_backend"]),
        "sample_count_per_seed": int(sample_count_per_seed),
        "warmup": int(n_warmup),
        "thinning": int(steps_per_sample),
        "mean_energy_delta": summary_row["mean_energy_delta"],
        "mean_energy_delta_percent": summary_row["mean_energy_delta_percent"],
        "magnetization_delta": summary_row["magnetization_delta"],
        "autocorrelation_summary": summary_row["autocorrelation_summary"],
        "kl_divergence": summary_row["kl_divergence"],
        "kl_estimate_stable": summary_row["kl_estimate_stable"],
        "stability_diagnostics_present": summary_row["stability_diagnostics_present"],
        "distribution_summary": summary_row["distribution_summary"],
        "thresholds": dict(thresholds),
        "energy_bin_count": int(energy_bin_count),
        "runtime_seconds_by_backend": runtime_seconds_by_backend,
        "memory_summary": memory_summary,
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": []
        if passed
        else [{"blocker": "sampled_parity_threshold_failed", "detail": str(summary_row)}],
        "honest_verdict": (
            "complete_thrml_carnot_parity_n128_passed_no_tsu_hardware_claim"
            if passed
            else "complete_thrml_carnot_parity_n128_failed_sampled_thresholds_no_tsu_hardware_claim"
        ),
    }
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, sampled pass gates, and no-TSU boundaries."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact.get("status") not in {"in_progress", "blocked", "complete"}:
        raise ValueError(f"invalid status: {artifact.get('status')!r}")
    if artifact.get("simulator_only") is not True:
        raise ValueError("simulator_only must remain true for Exp 1530")
    if artifact.get("no_tsu_hardware_claim") is not True:
        raise ValueError("no_tsu_hardware_claim must remain true for Exp 1530")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_VERDICT_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if artifact.get("thrml_parity_n128_passed") is True:
        thresholds = dict(artifact.get("thresholds") or THRESHOLDS)
        autocorr = artifact.get("autocorrelation_summary")
        autocorr_present = isinstance(autocorr, Mapping) and all(
            key in autocorr
            for key in ("carnot_energy_lag1_mean", "thrml_energy_lag1_mean", "lag1_delta")
        )
        mean_energy_gate = float(artifact.get("mean_energy_delta") or 0.0) <= float(
            thresholds["mean_energy_delta_abs_max"]
        ) or float(artifact.get("mean_energy_delta_percent") or 0.0) <= float(
            thresholds["mean_energy_delta_percent_max"]
        )
        runtime_present = _runtime_diagnostics_present(
            dict(artifact.get("runtime_seconds_by_backend") or {})
        )
        memory_present = _memory_diagnostics_present(dict(artifact.get("memory_summary") or {}))
        if not runtime_present:
            raise ValueError("thrml_parity_n128_passed requires runtime diagnostics")
        if not memory_present:
            raise ValueError("thrml_parity_n128_passed requires memory diagnostics")
        pass_metrics = (
            int(artifact.get("n_spins") or 0) == 128
            and len(artifact.get("seeds") or []) > 0
            and int(artifact.get("n_samples_per_backend") or 0)
            >= int(thresholds["kl_min_samples_per_backend"])
            and artifact.get("kl_estimate_stable") is True
            and artifact.get("stability_diagnostics_present") is True
            and autocorr_present
            and mean_energy_gate
            and float(artifact.get("magnetization_delta") or 0.0)
            <= float(thresholds["magnetization_delta_abs_max"])
            and float(artifact.get("kl_divergence") or 0.0)
            <= float(thresholds["kl_divergence_max"])
            and float(autocorr["lag1_delta"])
            <= float(thresholds["autocorrelation_lag1_delta_abs_max"])
        )
        if not pass_metrics:
            raise ValueError("thrml_parity_n128_passed requires sampled pass metrics")
