"""Exp 1543 n=256 THRML/Carnot simulator parity under schedule stress.

This module extends the Exp 1530 sampled parity lane to a 256-spin Ising case
and changes the stressor from graph family to sampler schedule. The comparison
is deliberately software-only: Carnot and the THRML adapter receive the same
Ising model and comparable fixed-temperature schedules, while the artifact
keeps Extropic TSU/Z1/XTR-0 hardware claims disabled.

Spec traces: REQ-SAMPLE-053, SCENARIO-SAMPLE-081.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from carnot.samplers import thrml_carnot_parity_n128_production_scale as parity_n128
from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

PROJECT_ROOT = parity_n128.PROJECT_ROOT
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1543_thrml_carnot_parity_n256_schedule_stress.json"
)
PARITY_MANIFEST_PATH = PROJECT_ROOT / "results" / "thrml_carnot_parity_n256_schedule_stress_1543.jsonl"
EXP1530_PATH = (
    PROJECT_ROOT / "results" / "experiment_1530_thrml_carnot_parity_n128_production_scale.json"
)
EXP1531_PATH = PROJECT_ROOT / "results" / "experiment_1531_thrml_diverse_topology_parity_n32.json"

EXPERIMENT_ID = 1543
RUN_DATE = "20260508"
MILESTONE = "20260508"
SCHEMA = "thrml_carnot_parity_n256_schedule_stress_v1"
DEFAULT_BASE_SEED = 20260508
DEFAULT_THRML_SEED_OFFSET = 100_000
DEFAULT_SAMPLES_PER_SCHEDULE = 4096
DEFAULT_ENERGY_BIN_COUNT = parity_n128.DEFAULT_ENERGY_BIN_COUNT

THRESHOLDS = {
    "mean_energy_delta_abs_max": parity_n128.THRESHOLDS["mean_energy_delta_abs_max"],
    "max_energy_delta_abs_max": parity_n128.THRESHOLDS["mean_energy_delta_abs_max"],
    "magnetization_delta_abs_max": parity_n128.THRESHOLDS["magnetization_delta_abs_max"],
    "kl_divergence_max": parity_n128.THRESHOLDS["kl_divergence_max"],
    "kl_min_samples_per_backend": parity_n128.THRESHOLDS["kl_min_samples_per_backend"],
    "autocorrelation_lag1_delta_abs_max": parity_n128.THRESHOLDS[
        "autocorrelation_lag1_delta_abs_max"
    ],
}
TERMINAL_VERDICT_PREFIXES = parity_n128.TERMINAL_VERDICT_PREFIXES
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "thrml_parity_n256_schedule_ready",
    "n_spins",
    "schedules_tested",
    "samples_per_schedule",
    "mean_energy_delta",
    "max_energy_delta",
    "kl_divergence",
    "autocorrelation_delta",
    "parity_passed",
    "simulator_only",
    "no_tsu_hardware_claim",
    "parity_report_path",
    "focused_tests_passed",
    "honest_verdict",
}

ImportModule = parity_n128.ImportModule
BackendFactory = parity_n128.BackendFactory
ParityIsingCase = parity_n128.ParityIsingCase


def _display_path(path: str | Path) -> str:
    return parity_n128._display_path(path)


def _round_metric(value: float) -> float:
    return parity_n128._round_metric(value)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return parity_n128._write_json(path, payload)


def _write_manifest(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    parity_n128._write_manifest(path, [dict(row) for row in rows])


def n256_signed_ring_chord_case() -> ParityIsingCase:
    """Return the deterministic 256-spin schedule-stress Ising case.

    The case uses three explicit periodic edge families so the larger spin
    count exercises more coupling reads without introducing hidden graph
    randomness. Keeping the weights modest makes sampled distribution metrics
    useful under bounded conductor runtime.

    Spec traces: REQ-SAMPLE-053.
    """

    n_spins = 256
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    ring_weights = np.tile(
        np.array(
            [
                0.120,
                -0.080,
                0.105,
                -0.055,
                0.095,
                -0.070,
                0.115,
                -0.045,
                0.090,
                -0.060,
                0.110,
                -0.050,
                0.100,
                -0.065,
                0.125,
                -0.040,
            ],
            dtype=np.float64,
        ),
        16,
    )
    chord2_weights = np.tile(
        np.array(
            [
                -0.035,
                0.030,
                -0.025,
                0.018,
                -0.032,
                0.024,
                -0.020,
                0.014,
                -0.030,
                0.026,
                -0.022,
                0.016,
                -0.028,
                0.020,
                -0.018,
                0.012,
            ],
            dtype=np.float64,
        ),
        16,
    )
    chord17_weights = np.tile(
        np.array(
            [
                0.018,
                -0.016,
                0.014,
                -0.012,
                0.017,
                -0.015,
                0.013,
                -0.011,
                0.016,
                -0.014,
                0.012,
                -0.010,
                0.015,
                -0.013,
                0.011,
                -0.009,
            ],
            dtype=np.float64,
        ),
        16,
    )
    for idx, weight in enumerate(ring_weights):
        left, right = idx, (idx + 1) % n_spins
        j_matrix[left, right] = j_matrix[right, left] = float(weight)
    for idx, weight in enumerate(chord2_weights):
        left, right = idx, (idx + 2) % n_spins
        j_matrix[left, right] = j_matrix[right, left] = float(weight)
    for idx, weight in enumerate(chord17_weights):
        left, right = idx, (idx + 17) % n_spins
        j_matrix[left, right] = j_matrix[right, left] = float(weight)
    bias = np.tile(
        np.array(
            [
                0.008,
                -0.014,
                0.011,
                -0.017,
                0.014,
                -0.011,
                0.005,
                -0.007,
                0.010,
                -0.013,
                0.016,
                -0.009,
                0.006,
                -0.008,
                0.012,
                -0.004,
            ],
            dtype=np.float64,
        ),
        16,
    )
    return ParityIsingCase(
        name="n256_signed_ring_chord",
        topology="signed_ring_chord",
        j_matrix=j_matrix,
        bias=bias,
        beta=1.0,
    )


def default_schedule_variants() -> tuple[dict[str, Any], ...]:
    """Return the fixed schedule variants used by Exp 1543.

    Spec traces: REQ-SAMPLE-053.
    """

    return (
        {
            "schedule_id": "low_beta_short_warmup",
            "beta": 0.90,
            "n_warmup": 384,
            "steps_per_sample": 3,
            "use_checkerboard": True,
            "seed_offset": 0,
        },
        {
            "schedule_id": "baseline_n128_style",
            "beta": 1.00,
            "n_warmup": 512,
            "steps_per_sample": 4,
            "use_checkerboard": True,
            "seed_offset": 17,
        },
        {
            "schedule_id": "high_beta_longer_thinning",
            "beta": 1.10,
            "n_warmup": 640,
            "steps_per_sample": 6,
            "use_checkerboard": True,
            "seed_offset": 31,
        },
    )


def validate_schedule_manifest(schedules: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Validate and normalize schedule variants before sampling.

    Spec traces: REQ-SAMPLE-053.
    """

    normalized = tuple(dict(schedule) for schedule in schedules)
    if len(normalized) < 3:
        raise ValueError("schedule manifest must include at least three variants")
    schedule_ids = [str(schedule.get("schedule_id", "")) for schedule in normalized]
    if any(not schedule_id for schedule_id in schedule_ids):
        raise ValueError("each schedule requires a non-empty schedule_id")
    if len(set(schedule_ids)) != len(schedule_ids):
        raise ValueError("schedule manifest requires unique schedule_id values")
    for schedule in normalized:
        if float(schedule.get("beta", 0.0)) <= 0.0:
            raise ValueError("schedule manifest requires positive beta")
        if int(schedule.get("n_warmup", 0)) < 0:
            raise ValueError("schedule manifest requires non-negative n_warmup")
        if int(schedule.get("steps_per_sample", 0)) <= 0:
            raise ValueError("schedule manifest requires positive steps_per_sample")
        if "use_checkerboard" not in schedule:
            raise ValueError("schedule manifest requires use_checkerboard")
        schedule["beta"] = float(schedule["beta"])
        schedule["n_warmup"] = int(schedule["n_warmup"])
        schedule["steps_per_sample"] = int(schedule["steps_per_sample"])
        schedule["use_checkerboard"] = bool(schedule["use_checkerboard"])
        schedule["seed_offset"] = int(schedule.get("seed_offset", 0))
    return normalized


def _load_exp1530_ready(path: str | Path) -> tuple[bool, dict[str, Any], dict[str, str] | None]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return False, {}, {"blocker": "exp1530_evidence_missing", "detail": str(exc)}
    except json.JSONDecodeError as exc:
        return False, {}, {"blocker": "exp1530_evidence_malformed", "detail": str(exc)}
    ready = (
        payload.get("status") == "complete"
        and payload.get("thrml_parity_n128_passed") is True
        and payload.get("simulator_only") is True
        and payload.get("no_tsu_hardware_claim") is True
    )
    if not ready:
        return (
            False,
            payload,
            {
                "blocker": "exp1530_parity_not_passed",
                "detail": "Exp1530 must be complete, n=128-passed, simulator-only, and no-TSU-claim",
            },
        )
    return True, payload, None


def _load_exp1531_ready(path: str | Path) -> tuple[bool, dict[str, Any], dict[str, str] | None]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return False, {}, {"blocker": "exp1531_evidence_missing", "detail": str(exc)}
    except json.JSONDecodeError as exc:
        return False, {}, {"blocker": "exp1531_evidence_malformed", "detail": str(exc)}
    ready = (
        payload.get("status") == "complete"
        and payload.get("diverse_topology_parity_ready") is True
        and payload.get("simulator_only") is True
        and payload.get("no_tsu_hardware_claim") is True
    )
    if not ready:
        return (
            False,
            payload,
            {
                "blocker": "exp1531_parity_not_ready",
                "detail": "Exp1531 must be complete, diverse-ready, simulator-only, and no-TSU-claim",
            },
        )
    return True, payload, None


def _import_thrml(importer: ImportModule) -> tuple[dict[str, Any], dict[str, str] | None]:
    try:
        thrml = importer("thrml")
        importer("thrml.models")
    except ModuleNotFoundError as exc:
        return (
            {},
            {
                "blocker": "thrml_local_import_unavailable",
                "detail": f"local THRML import failed: {exc}",
            },
        )
    return (
        {
            "thrml_version": str(getattr(thrml, "__version__", "<unknown>")),
            "thrml_import_path": str(getattr(thrml, "__file__", "<unknown>")),
        },
        None,
    )


def _tolerance_sources(exp1530: Mapping[str, Any], exp1531: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "exp1530_path": _display_path(EXP1530_PATH),
        "exp1531_path": _display_path(EXP1531_PATH),
        "exp1530_thresholds": dict(exp1530.get("thresholds") or {}),
        "exp1531_thresholds": dict(exp1531.get("thresholds") or {}),
        "local_reason_for_max_energy_delta_abs_max": (
            "Exp1530 had no separate max-energy threshold; Exp1543 uses the "
            "n=128 mean-energy absolute threshold as the per-schedule worst-case gate."
        ),
    }


def write_in_progress_artifact(
    path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
) -> dict[str, Any]:
    """Write the bootstrap artifact before schedule stress execution.

    Spec traces: REQ-SAMPLE-053.
    """

    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "z1_hardware_execution": False,
            "xtr0_hardware_execution": False,
        },
        "status": "in_progress",
        "milestone": MILESTONE,
        "thrml_parity_n256_schedule_ready": False,
        "n_spins": 256,
        "schedules_tested": 0,
        "samples_per_schedule": 0,
        "mean_energy_delta": None,
        "max_energy_delta": None,
        "kl_divergence": None,
        "autocorrelation_delta": None,
        "parity_passed": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "parity_report_path": _display_path(manifest_path),
        "focused_tests_passed": False,
        "thresholds": dict(THRESHOLDS),
        "schedule_results": {},
        "blockers": [{"blocker": "parity_run_not_completed", "detail": "bootstrap artifact only"}],
        "honest_verdict": "success_in_progress_thrml_parity_n256_schedule_simulator_only",
    }
    validate_artifact(artifact)
    return _write_json(path, artifact)


def _blocked_artifact(
    *,
    manifest_path: str | Path,
    blockers: list[dict[str, str]],
    verdict: str,
    metadata: Mapping[str, Any] | None = None,
    schedules_tested: int = 0,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "z1_hardware_execution": False,
            "xtr0_hardware_execution": False,
            **dict(metadata or {}),
        },
        "status": "blocked",
        "milestone": MILESTONE,
        "thrml_parity_n256_schedule_ready": False,
        "n_spins": 256,
        "schedules_tested": int(schedules_tested),
        "samples_per_schedule": 0,
        "mean_energy_delta": None,
        "max_energy_delta": None,
        "kl_divergence": None,
        "autocorrelation_delta": None,
        "parity_passed": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "parity_report_path": _display_path(manifest_path),
        "focused_tests_passed": False,
        "thresholds": dict(THRESHOLDS),
        "schedule_results": {},
        "blockers": blockers,
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def sampled_schedule_backend_row(
    case: ParityIsingCase,
    *,
    schedule: Mapping[str, Any],
    backend_label: str,
    backend_name: str,
    seed: int,
    samples: np.ndarray,
) -> dict[str, Any]:
    """Summarize sampled states for one schedule/backend pair.

    Spec traces: REQ-SAMPLE-053.
    """

    schedule_id = str(schedule["schedule_id"])
    row = parity_n128.sampled_backend_row(
        case,
        seed=int(seed),
        backend_label=backend_label,
        backend_name=backend_name,
        samples=samples,
        schedule=schedule,
    )
    row.update(
        {
            "case_id": f"exp1543:{schedule_id}:{backend_label}",
            "case_type": "schedule_backend_sampled",
            "schedule_id": schedule_id,
            "n_spins": 256,
        }
    )
    return row


def _schedule_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    schedule_id: str,
    thresholds: Mapping[str, float],
    energy_bin_count: int,
) -> dict[str, Any]:
    schedule_rows = [dict(row) for row in rows if row.get("schedule_id") == schedule_id]
    summary = parity_n128.summarize_sampled_rows(
        schedule_rows,
        seeds=[int(row["seed"]) for row in schedule_rows],
        thresholds={**dict(thresholds), "kl_min_samples_per_backend": 1},
        energy_bin_count=energy_bin_count,
    )
    autocorr = dict(summary["autocorrelation_summary"])
    passed = (
        float(summary["mean_energy_delta"]) <= float(thresholds["max_energy_delta_abs_max"])
        and float(summary["magnetization_delta"]) <= float(thresholds["magnetization_delta_abs_max"])
        and float(summary["kl_divergence"]) <= float(thresholds["kl_divergence_max"])
        and float(autocorr["lag1_delta"])
        <= float(thresholds["autocorrelation_lag1_delta_abs_max"])
    )
    summary.update(
        {
            "case_id": f"exp1543:{schedule_id}:summary",
            "schedule_id": schedule_id,
            "autocorrelation_delta": autocorr["lag1_delta"],
            "passed_thresholds": bool(passed),
        }
    )
    return summary


def summarize_schedule_stress_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    schedules: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
    energy_bin_count: int,
) -> dict[str, Any]:
    """Aggregate Exp 1543 schedule/backend rows into terminal parity metrics.

    Spec traces: REQ-SAMPLE-053, SCENARIO-SAMPLE-081.
    """

    normalized_schedules = validate_schedule_manifest(schedules)
    schedule_ids = [str(schedule["schedule_id"]) for schedule in normalized_schedules]
    schedule_results = {
        schedule_id: _schedule_summary(
            rows,
            schedule_id=schedule_id,
            thresholds=thresholds,
            energy_bin_count=energy_bin_count,
        )
        for schedule_id in schedule_ids
    }
    aggregate = parity_n128.summarize_sampled_rows(
        rows,
        seeds=[int(row["seed"]) for row in rows],
        thresholds=thresholds,
        energy_bin_count=energy_bin_count,
    )
    aggregate_autocorr = dict(aggregate["autocorrelation_summary"])
    samples_per_schedule = min(
        int(result["n_samples_per_backend"]) for result in schedule_results.values()
    )
    max_energy_delta = max(float(result["mean_energy_delta"]) for result in schedule_results.values())
    max_autocorrelation_delta = max(
        float(result["autocorrelation_delta"]) for result in schedule_results.values()
    )
    all_schedules_passed = all(
        bool(result["passed_thresholds"]) for result in schedule_results.values()
    )
    aggregate_passed = (
        float(aggregate["mean_energy_delta"]) <= float(thresholds["mean_energy_delta_abs_max"])
        and max_energy_delta <= float(thresholds["max_energy_delta_abs_max"])
        and float(aggregate["kl_divergence"]) <= float(thresholds["kl_divergence_max"])
        and int(aggregate["n_samples_per_backend"]) >= int(thresholds["kl_min_samples_per_backend"])
        and aggregate["kl_estimate_stable"] is True
        and max_autocorrelation_delta <= float(thresholds["autocorrelation_lag1_delta_abs_max"])
    )
    parity_passed = bool(all_schedules_passed and aggregate_passed)
    return {
        "case_id": "exp1543:n256_schedule_stress:summary",
        "case_type": "schedule_stress_summary",
        "n_spins": 256,
        "schedule_ids": schedule_ids,
        "schedules_tested": len(schedule_ids),
        "samples_per_schedule": samples_per_schedule,
        "n_samples_per_backend": int(aggregate["n_samples_per_backend"]),
        "mean_energy_delta": aggregate["mean_energy_delta"],
        "max_energy_delta": _round_metric(max_energy_delta),
        "kl_divergence": aggregate["kl_divergence"],
        "autocorrelation_delta": _round_metric(max_autocorrelation_delta),
        "aggregate_autocorrelation_delta": aggregate_autocorr["lag1_delta"],
        "schedule_results": schedule_results,
        "thresholds": dict(thresholds),
        "parity_passed": parity_passed,
        "thrml_parity_n256_schedule_ready": parity_passed,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
    }


def run_schedule_stress_n256(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
    exp1530_path: str | Path = EXP1530_PATH,
    exp1531_path: str | Path = EXP1531_PATH,
    importer: ImportModule = importlib.import_module,
    carnot_backend_factory: BackendFactory = CpuBackend,
    thrml_backend_factory: BackendFactory = ThrmlSamplerBackend,
    schedules: Sequence[Mapping[str, Any]] = default_schedule_variants(),
    samples_per_schedule: int = DEFAULT_SAMPLES_PER_SCHEDULE,
    thresholds: Mapping[str, float] = THRESHOLDS,
    energy_bin_count: int = DEFAULT_ENERGY_BIN_COUNT,
    base_seed: int = DEFAULT_BASE_SEED,
    thrml_seed_offset: int = DEFAULT_THRML_SEED_OFFSET,
    focused_tests_passed: bool = False,
) -> dict[str, Any]:
    """Run the n=256 schedule-stress parity probe and write JSON/JSONL evidence."""

    write_in_progress_artifact(output_path, manifest_path)
    try:
        normalized_schedules = validate_schedule_manifest(schedules)
    except ValueError as exc:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                blockers=[{"blocker": "invalid_schedule_manifest", "detail": str(exc)}],
                verdict="complete_thrml_parity_n256_schedule_blocked_invalid_manifest_no_tsu_hardware_claim",
            ),
        )
    exp1530_ready, exp1530_payload, exp1530_blocker = _load_exp1530_ready(exp1530_path)
    if not exp1530_ready:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                blockers=[exp1530_blocker or {"blocker": "exp1530_parity_not_passed", "detail": ""}],
                verdict="complete_thrml_parity_n256_schedule_blocked_exp1530_no_tsu_hardware_claim",
                metadata={"exp1530_status": exp1530_payload.get("status")},
                schedules_tested=len(normalized_schedules),
            ),
        )
    exp1531_ready, exp1531_payload, exp1531_blocker = _load_exp1531_ready(exp1531_path)
    if not exp1531_ready:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                blockers=[exp1531_blocker or {"blocker": "exp1531_parity_not_ready", "detail": ""}],
                verdict="complete_thrml_parity_n256_schedule_blocked_exp1531_no_tsu_hardware_claim",
                metadata={
                    "exp1530_status": exp1530_payload.get("status"),
                    "exp1531_status": exp1531_payload.get("status"),
                },
                schedules_tested=len(normalized_schedules),
            ),
        )
    thrml_details, import_blocker = _import_thrml(importer)
    if import_blocker is not None:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                blockers=[import_blocker],
                verdict=(
                    "complete_thrml_parity_n256_schedule_blocked_simulator_dependency_"
                    "no_tsu_hardware_claim"
                ),
                metadata={
                    "exp1530_status": exp1530_payload.get("status"),
                    "exp1531_status": exp1531_payload.get("status"),
                },
                schedules_tested=len(normalized_schedules),
            ),
        )

    case = n256_signed_ring_chord_case()
    rows: list[dict[str, Any]] = []
    for schedule in normalized_schedules:
        schedule_seed = int(base_seed) + int(schedule["seed_offset"])
        carnot_backend = carnot_backend_factory(schedule_seed)
        thrml_backend = thrml_backend_factory(schedule_seed + int(thrml_seed_offset))
        carnot_samples = np.asarray(
            carnot_backend.sample(case.bias, case.j_matrix, int(samples_per_schedule), dict(schedule))
        )
        thrml_samples = np.asarray(
            thrml_backend.sample(case.bias, case.j_matrix, int(samples_per_schedule), dict(schedule))
        )
        rows.append(
            sampled_schedule_backend_row(
                case,
                schedule=schedule,
                backend_label="carnot",
                backend_name=str(getattr(carnot_backend, "backend_name", "<unknown>")),
                seed=schedule_seed,
                samples=carnot_samples,
            )
        )
        rows.append(
            sampled_schedule_backend_row(
                case,
                schedule=schedule,
                backend_label="thrml",
                backend_name=str(getattr(thrml_backend, "backend_name", "<unknown>")),
                seed=schedule_seed + int(thrml_seed_offset),
                samples=thrml_samples,
            )
        )

    summary_row = summarize_schedule_stress_rows(
        rows,
        schedules=normalized_schedules,
        thresholds=thresholds,
        energy_bin_count=energy_bin_count,
    )
    _write_manifest(manifest_path, [*rows, summary_row])
    passed = bool(summary_row["parity_passed"])
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "z1_hardware_execution": False,
            "xtr0_hardware_execution": False,
            "exp1530_status": exp1530_payload.get("status"),
            "exp1531_status": exp1531_payload.get("status"),
            "thrml_execution_path": "local_software_simulator_or_cpu_fallback",
            "independent_rng_streams": int(thrml_seed_offset) != 0,
            **thrml_details,
        },
        "status": "complete",
        "milestone": MILESTONE,
        "thrml_parity_n256_schedule_ready": passed,
        "n_spins": 256,
        "schedules_tested": summary_row["schedules_tested"],
        "samples_per_schedule": int(samples_per_schedule),
        "mean_energy_delta": summary_row["mean_energy_delta"],
        "max_energy_delta": summary_row["max_energy_delta"],
        "kl_divergence": summary_row["kl_divergence"],
        "autocorrelation_delta": summary_row["autocorrelation_delta"],
        "parity_passed": passed,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "parity_report_path": _display_path(manifest_path),
        "focused_tests_passed": bool(focused_tests_passed),
        "schedule_manifest": list(normalized_schedules),
        "schedule_results": summary_row["schedule_results"],
        "thresholds": dict(thresholds),
        "tolerance_sources": _tolerance_sources(exp1530_payload, exp1531_payload),
        "energy_bin_count": int(energy_bin_count),
        "n_samples_per_backend": summary_row["n_samples_per_backend"],
        "blockers": []
        if passed
        else [{"blocker": "schedule_parity_threshold_failed", "detail": str(summary_row)}],
        "honest_verdict": (
            "complete_thrml_parity_n256_schedule_passed_simulator_only_no_tsu_hardware_claim"
            if passed
            else "complete_thrml_parity_n256_schedule_failed_thresholds_simulator_only_no_tsu_hardware_claim"
        ),
    }
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 1543 schema, metric gates, and no-hardware boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact.get("status") not in {"in_progress", "blocked", "complete"}:
        raise ValueError(f"invalid status: {artifact.get('status')!r}")
    if artifact.get("simulator_only") is not True:
        raise ValueError("simulator_only must remain true for Exp 1543")
    if artifact.get("no_tsu_hardware_claim") is not True:
        raise ValueError("no_tsu_hardware_claim must remain true for Exp 1543")
    if int(artifact.get("n_spins") or 0) != 256:
        raise ValueError("Exp 1543 artifacts must remain at n_spins=256")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_VERDICT_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if artifact.get("thrml_parity_n256_schedule_ready") is True or artifact.get("parity_passed") is True:
        thresholds = dict(artifact.get("thresholds") or THRESHOLDS)
        schedule_results = dict(artifact.get("schedule_results") or {})
        gates_ok = (
            artifact.get("thrml_parity_n256_schedule_ready") is True
            and artifact.get("parity_passed") is True
            and int(artifact.get("schedules_tested") or 0) >= 3
            and int(artifact.get("samples_per_schedule") or 0) > 0
            and len(schedule_results) == int(artifact.get("schedules_tested") or 0)
            and float(artifact.get("mean_energy_delta") or 0.0)
            <= float(thresholds["mean_energy_delta_abs_max"])
            and float(artifact.get("max_energy_delta") or 0.0)
            <= float(thresholds["max_energy_delta_abs_max"])
            and float(artifact.get("kl_divergence") or 0.0)
            <= float(thresholds["kl_divergence_max"])
            and float(artifact.get("autocorrelation_delta") or 0.0)
            <= float(thresholds["autocorrelation_lag1_delta_abs_max"])
        )
        for result in schedule_results.values():
            result_map = dict(result)
            gates_ok = gates_ok and bool(result_map.get("passed_thresholds"))
            gates_ok = gates_ok and int(result_map.get("n_samples_per_backend") or 0) > 0
        if not gates_ok:
            raise ValueError("schedule readiness requires passing n=256 parity metrics")
