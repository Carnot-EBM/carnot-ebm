"""Exp 1528 sampled n=32 THRML/Carnot simulator parity.

This module is the first THRML/Carnot parity scaling step where exact
enumeration is deliberately not used. The n=32 state space has 4,294,967,296
states, so the helper records repeated fixed-seed sampler comparisons instead:
mean energy, magnetization, lag-one energy autocorrelation, and an empirical
energy-histogram KL estimate. The run remains simulator-only and does not claim
Extropic TSU hardware execution.

Spec traces: REQ-SAMPLE-049, SCENARIO-SAMPLE-077.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from carnot.samplers import thrml_carnot_parity_n16 as parity_n16
from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

PROJECT_ROOT = parity_n16.PROJECT_ROOT
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1528_thrml_carnot_parity_n32_sample.json"
)
PARITY_MANIFEST_PATH = PROJECT_ROOT / "results" / "thrml_carnot_parity_n32_1528.jsonl"
EXP1527_PATH = PROJECT_ROOT / "results" / "experiment_1527_thrml_carnot_parity_n16.json"

EXPERIMENT_ID = 1528
RUN_DATE = "20260508"
SCHEMA = "thrml_carnot_parity_n32_sample_v1"
DEFAULT_SEEDS = (20260508, 20260509, 20260510, 20260511, 20260512)
DEFAULT_SAMPLE_COUNT_PER_SEED = 2048
DEFAULT_N_WARMUP = 256
DEFAULT_STEPS_PER_SAMPLE = 4
DEFAULT_ENERGY_BIN_COUNT = 32

THRESHOLDS = {
    "mean_energy_delta_abs_max": 0.15,
    "magnetization_delta_abs_max": 0.025,
    "kl_divergence_max": 0.05,
    "kl_min_samples_per_backend": 10_000,
}
TERMINAL_VERDICT_PREFIXES = parity_n16.TERMINAL_VERDICT_PREFIXES
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "thrml_parity_n32_passed",
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
    "parity_manifest_path",
    "blockers",
    "honest_verdict",
}

ImportModule = parity_n16.ImportModule
BackendFactory = parity_n16.BackendFactory
ParityIsingCase = parity_n16.ParityIsingCase
ising_energy = parity_n16.ising_energy


def n32_signed_ring_chord_case() -> ParityIsingCase:
    """Return the deterministic 32-spin signed ring-chord sampled parity case.

    The topology keeps the Exp 1526/1527 pattern: every spin has one signed
    nearest-neighbor ring edge and one signed distance-two chord. The weights
    are periodic and explicit so the case is reproducible without enumerating
    the full n=32 Boltzmann distribution.

    Spec traces: REQ-SAMPLE-049.
    """

    n_spins = 32
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    ring_weights = np.tile(np.array([0.20, -0.10, 0.15, -0.05], dtype=np.float64), 8)
    chord_weights = np.tile(np.array([-0.08, 0.06, -0.04, 0.02], dtype=np.float64), 8)
    for idx, weight in enumerate(ring_weights):
        left, right = idx, (idx + 1) % n_spins
        j_matrix[left, right] = j_matrix[right, left] = float(weight)
    for idx, weight in enumerate(chord_weights):
        left, right = idx, (idx + 2) % n_spins
        j_matrix[left, right] = j_matrix[right, left] = float(weight)
    bias = np.tile(np.array([0.01, -0.02, 0.03, -0.04], dtype=np.float64), 8)
    return ParityIsingCase(
        name="n32_signed_ring_chord",
        topology="signed_ring_chord",
        j_matrix=j_matrix,
        bias=bias,
        beta=1.10,
    )


def _display_path(path: str | Path) -> str:
    return parity_n16._display_path(path)


def _round_metric(value: float) -> float:
    return parity_n16.parity_n8._round_metric(value)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return parity_n16._write_json(path, payload)


def _write_manifest(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    parity_n16._write_manifest(path, [dict(row) for row in rows])


def _load_exp1527_ready(path: str | Path) -> tuple[bool, dict[str, Any], dict[str, str] | None]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return False, {}, {"blocker": "exp1527_evidence_missing", "detail": str(exc)}
    except json.JSONDecodeError as exc:
        return False, {}, {"blocker": "exp1527_evidence_malformed", "detail": str(exc)}
    ready = (
        payload.get("status") == "complete"
        and payload.get("thrml_parity_n16_passed") is True
        and payload.get("simulator_only") is True
        and payload.get("no_tsu_hardware_claim") is True
    )
    if not ready:
        return (
            False,
            payload,
            {
                "blocker": "exp1527_parity_not_passed",
                "detail": "Exp1527 must be complete, n=16-passed, simulator-only, and no-TSU-claim",
            },
        )
    return True, payload, None


def write_in_progress_artifact(
    path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
) -> dict[str, Any]:
    """Write the bootstrap artifact before import probing or parity execution.

    Spec traces: REQ-SAMPLE-049.
    """

    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
        },
        "status": "in_progress",
        "thrml_parity_n32_passed": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "n_spins": 32,
        "topology": "signed_ring_chord",
        "seeds": list(DEFAULT_SEEDS),
        "n_samples_per_backend": 0,
        "sample_count_per_seed": DEFAULT_SAMPLE_COUNT_PER_SEED,
        "warmup": DEFAULT_N_WARMUP,
        "thinning": DEFAULT_STEPS_PER_SAMPLE,
        "mean_energy_delta": None,
        "magnetization_delta": None,
        "autocorrelation_summary": {},
        "kl_divergence": None,
        "kl_estimate_stable": False,
        "thresholds": dict(THRESHOLDS),
        "energy_bin_count": DEFAULT_ENERGY_BIN_COUNT,
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": [{"blocker": "parity_run_not_completed", "detail": "bootstrap artifact only"}],
        "honest_verdict": "success_in_progress_thrml_carnot_parity_n32_simulator_only",
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
            **dict(metadata or {}),
        },
        "status": "blocked",
        "thrml_parity_n32_passed": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "n_spins": 32,
        "topology": "signed_ring_chord",
        "seeds": [int(seed) for seed in seeds],
        "n_samples_per_backend": 0,
        "sample_count_per_seed": DEFAULT_SAMPLE_COUNT_PER_SEED,
        "warmup": DEFAULT_N_WARMUP,
        "thinning": DEFAULT_STEPS_PER_SAMPLE,
        "mean_energy_delta": None,
        "magnetization_delta": None,
        "autocorrelation_summary": {},
        "kl_divergence": None,
        "kl_estimate_stable": False,
        "thresholds": dict(THRESHOLDS),
        "energy_bin_count": DEFAULT_ENERGY_BIN_COUNT,
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": blockers,
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def _sample_energies(case: ParityIsingCase, samples: np.ndarray) -> np.ndarray:
    spin_samples = np.where(np.asarray(samples, dtype=bool), 1, -1).astype(np.int8)
    return np.asarray([ising_energy(case, state) for state in spin_samples], dtype=np.float64)


def _sample_magnetization(samples: np.ndarray) -> float:
    spin_samples = np.where(np.asarray(samples, dtype=bool), 1.0, -1.0)
    return float(np.mean(spin_samples))


def _lag_one_autocorrelation(values: np.ndarray) -> float:
    series = np.asarray(values, dtype=np.float64)
    if series.size < 2:
        return 0.0
    centered = series - float(np.mean(series))
    denom = float(centered @ centered)
    if denom <= 1.0e-15:
        return 0.0
    return float((centered[:-1] @ centered[1:]) / denom)


def _energy_quantiles(energies: np.ndarray) -> dict[str, float]:
    q0, q25, q50, q75, q100 = np.quantile(np.asarray(energies, dtype=np.float64), [0, 0.25, 0.5, 0.75, 1.0])
    return {
        "q0": _round_metric(float(q0)),
        "q25": _round_metric(float(q25)),
        "q50": _round_metric(float(q50)),
        "q75": _round_metric(float(q75)),
        "q100": _round_metric(float(q100)),
    }


def sampled_backend_row(
    case: ParityIsingCase,
    *,
    seed: int,
    backend_label: str,
    backend_name: str,
    samples: np.ndarray,
    schedule: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize one sampled chain group for one seed/backend pair.

    Spec traces: REQ-SAMPLE-049.
    """

    energies = _sample_energies(case, samples)
    mean_energy = float(np.mean(energies))
    row = {
        "case_id": f"exp1528:{case.name}:seed_{int(seed)}:{backend_label}",
        "case_type": "sampled_seed_backend",
        "seed": int(seed),
        "backend": str(backend_label),
        "backend_name": str(backend_name),
        "n_spins": case.n_spins,
        "sample_count": int(np.asarray(samples).shape[0]),
        "schedule": dict(schedule),
        "mean_energy": _round_metric(mean_energy),
        "energy_std": _round_metric(float(np.std(energies))),
        "best_energy": _round_metric(float(np.min(energies))),
        "worst_energy": _round_metric(float(np.max(energies))),
        "energy_quantiles": _energy_quantiles(energies),
        "magnetization": _round_metric(_sample_magnetization(samples)),
        "energy_autocorrelation_lag1": _round_metric(_lag_one_autocorrelation(energies)),
        "energy_trace": [_round_metric(float(value)) for value in energies],
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
    }
    return row


def _energy_histogram_kl(
    carnot_energies: np.ndarray,
    thrml_energies: np.ndarray,
    *,
    energy_bin_count: int,
) -> tuple[float, dict[str, Any]]:
    all_energies = np.concatenate([carnot_energies, thrml_energies])
    min_energy = float(np.min(all_energies))
    max_energy = float(np.max(all_energies))
    lower = min_energy if max_energy > min_energy else min_energy - 0.5
    upper = max_energy if max_energy > min_energy else max_energy + 0.5
    edges = np.linspace(lower, upper, int(energy_bin_count) + 1)
    carnot_counts, _ = np.histogram(carnot_energies, bins=edges)
    thrml_counts, _ = np.histogram(thrml_energies, bins=edges)
    carnot_probs = (carnot_counts.astype(np.float64) + 0.5) / (
        float(np.sum(carnot_counts)) + 0.5 * len(carnot_counts)
    )
    thrml_probs = (thrml_counts.astype(np.float64) + 0.5) / (
        float(np.sum(thrml_counts)) + 0.5 * len(thrml_counts)
    )
    kl_divergence = float(np.sum(carnot_probs * np.log(carnot_probs / thrml_probs)))
    histogram = {
        "energy_bin_count": int(len(carnot_counts)),
        "bin_edges": [_round_metric(float(edge)) for edge in edges],
        "carnot_counts": [int(value) for value in carnot_counts],
        "thrml_counts": [int(value) for value in thrml_counts],
    }
    return _round_metric(kl_divergence), histogram


def summarize_sampled_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    thresholds: Mapping[str, float],
    energy_bin_count: int,
) -> dict[str, Any]:
    """Build the cross-seed sampled parity summary row.

    Spec traces: REQ-SAMPLE-049, SCENARIO-SAMPLE-077.
    """

    carnot_rows = [row for row in rows if row["backend"] == "carnot"]
    thrml_rows = [row for row in rows if row["backend"] == "thrml"]
    carnot_energies = np.asarray(
        [value for row in carnot_rows for value in row["energy_trace"]], dtype=np.float64
    )
    thrml_energies = np.asarray(
        [value for row in thrml_rows for value in row["energy_trace"]], dtype=np.float64
    )
    kl_divergence, histogram = _energy_histogram_kl(
        carnot_energies,
        thrml_energies,
        energy_bin_count=energy_bin_count,
    )
    carnot_mean_energy = float(np.mean(carnot_energies))
    thrml_mean_energy = float(np.mean(thrml_energies))
    carnot_magnetization = float(np.mean([float(row["magnetization"]) for row in carnot_rows]))
    thrml_magnetization = float(np.mean([float(row["magnetization"]) for row in thrml_rows]))
    carnot_lag1 = float(
        np.mean([float(row["energy_autocorrelation_lag1"]) for row in carnot_rows])
    )
    thrml_lag1 = float(np.mean([float(row["energy_autocorrelation_lag1"]) for row in thrml_rows]))
    n_samples_per_backend = int(min(carnot_energies.size, thrml_energies.size))
    mean_energy_delta = abs(carnot_mean_energy - thrml_mean_energy)
    magnetization_delta = abs(carnot_magnetization - thrml_magnetization)
    kl_estimate_stable = n_samples_per_backend >= int(thresholds["kl_min_samples_per_backend"])
    passed_thresholds = (
        mean_energy_delta <= float(thresholds["mean_energy_delta_abs_max"])
        and magnetization_delta <= float(thresholds["magnetization_delta_abs_max"])
        and kl_divergence <= float(thresholds["kl_divergence_max"])
        and kl_estimate_stable
    )
    return {
        "case_id": "exp1528:n32_signed_ring_chord:sampled_summary",
        "case_type": "sampled_distribution_summary",
        "seeds": [int(seed) for seed in seeds],
        "n_samples_per_backend": n_samples_per_backend,
        "carnot_mean_energy": _round_metric(carnot_mean_energy),
        "thrml_mean_energy": _round_metric(thrml_mean_energy),
        "mean_energy_delta": _round_metric(mean_energy_delta),
        "carnot_magnetization": _round_metric(carnot_magnetization),
        "thrml_magnetization": _round_metric(thrml_magnetization),
        "magnetization_delta": _round_metric(magnetization_delta),
        "autocorrelation_summary": {
            "carnot_energy_lag1_mean": _round_metric(carnot_lag1),
            "thrml_energy_lag1_mean": _round_metric(thrml_lag1),
            "lag1_delta": _round_metric(abs(carnot_lag1 - thrml_lag1)),
        },
        "kl_divergence": kl_divergence,
        "kl_estimate_stable": bool(kl_estimate_stable),
        "distribution_summary": histogram,
        "thresholds": dict(thresholds),
        "passed_thresholds": bool(passed_thresholds),
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
    }


def run_parity_n32(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
    exp1527_path: str | Path = EXP1527_PATH,
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
    """Run sampled n=32 THRML/Carnot parity and write JSON/JSONL evidence."""

    write_in_progress_artifact(output_path, manifest_path)
    exp1527_ready, exp1527_payload, exp1527_blocker = _load_exp1527_ready(exp1527_path)
    if not exp1527_ready:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                seeds=seeds,
                blockers=[
                    exp1527_blocker
                    or {
                        "blocker": "exp1527_parity_not_passed",
                        "detail": "unknown Exp1527 blocker",
                    }
                ],
                verdict="complete_thrml_carnot_parity_n32_blocked_exp1527_no_tsu_hardware_claim",
                metadata={"exp1527_status": exp1527_payload.get("status")},
            ),
        )
    _thrml_modules, thrml_details, import_blocker = parity_n16.parity_n8._import_thrml(importer)
    if import_blocker is not None:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                seeds=seeds,
                blockers=[import_blocker],
                verdict="complete_thrml_carnot_parity_n32_blocked_simulator_dependency_no_tsu_hardware_claim",
                metadata={"exp1527_status": exp1527_payload.get("status")},
            ),
        )

    case = n32_signed_ring_chord_case()
    schedule = {
        "beta": float(case.beta),
        "n_warmup": int(n_warmup),
        "steps_per_sample": int(steps_per_sample),
        "use_checkerboard": True,
    }
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        carnot_backend = carnot_backend_factory(int(seed))
        thrml_backend = thrml_backend_factory(int(seed))
        carnot_samples = np.asarray(
            carnot_backend.sample(case.bias, case.j_matrix, int(sample_count_per_seed), schedule)
        )
        thrml_samples = np.asarray(
            thrml_backend.sample(case.bias, case.j_matrix, int(sample_count_per_seed), schedule)
        )
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
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "exp1527_status": exp1527_payload.get("status"),
            **thrml_details,
        },
        "status": "complete",
        "thrml_parity_n32_passed": passed,
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
        "magnetization_delta": summary_row["magnetization_delta"],
        "autocorrelation_summary": summary_row["autocorrelation_summary"],
        "kl_divergence": summary_row["kl_divergence"],
        "kl_estimate_stable": summary_row["kl_estimate_stable"],
        "distribution_summary": summary_row["distribution_summary"],
        "thresholds": dict(thresholds),
        "energy_bin_count": int(energy_bin_count),
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": [] if passed else [{"blocker": "sampled_parity_threshold_failed", "detail": str(summary_row)}],
        "honest_verdict": (
            "complete_thrml_carnot_parity_n32_passed_no_tsu_hardware_claim"
            if passed
            else "complete_thrml_carnot_parity_n32_failed_sampled_thresholds_no_tsu_hardware_claim"
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
        raise ValueError("simulator_only must remain true for Exp 1528")
    if artifact.get("no_tsu_hardware_claim") is not True:
        raise ValueError("no_tsu_hardware_claim must remain true for Exp 1528")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_VERDICT_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if artifact.get("thrml_parity_n32_passed") is True:
        thresholds = dict(artifact.get("thresholds") or THRESHOLDS)
        pass_metrics = (
            int(artifact.get("n_spins") or 0) == 32
            and len(artifact.get("seeds") or []) > 0
            and int(artifact.get("n_samples_per_backend") or 0)
            >= int(thresholds["kl_min_samples_per_backend"])
            and artifact.get("kl_estimate_stable") is True
            and float(artifact.get("mean_energy_delta") or 0.0)
            <= float(thresholds["mean_energy_delta_abs_max"])
            and float(artifact.get("magnetization_delta") or 0.0)
            <= float(thresholds["magnetization_delta_abs_max"])
            and float(artifact.get("kl_divergence") or 0.0)
            <= float(thresholds["kl_divergence_max"])
        )
        if not pass_metrics:
            raise ValueError("thrml_parity_n32_passed requires sampled pass metrics")
