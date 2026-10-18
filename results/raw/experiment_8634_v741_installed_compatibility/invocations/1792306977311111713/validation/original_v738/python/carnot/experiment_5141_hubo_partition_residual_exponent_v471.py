"""Exp 5141: CPU exact-checked partition telemetry for HUBO/Ising 2D PT.

Spec refs: REQ-SAMPLE-5141, SCENARIO-SAMPLE-5141.

This experiment keeps the sampler on CPU and uses exact enumeration as the
correctness authority. The partitioned updates measure boundary-refresh and
residual-energy telemetry that can later be mapped to KV260, GateMate, and
PolarFire workloads, but the artifact deliberately does not execute hardware or
claim hardware speedup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot import experiment_5129_hubo_adaptive_2dpt_v470 as exp5129
from carnot.samplers.hubo_2dpt import (
    HuboProblem,
    HuboTerm,
    SwapStats,
    build_synthetic_hubo_families,
    evaluate_hubo_energy,
    exact_enumerate,
    metropolis_swap_log_acceptance,
    run_unguided_gibbs,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_5141_hubo_partition_residual_exponent_v471.json"
EXPERIMENT_ID = "exp5141-hubo-partition-residual-exponent-v471"
MILESTONE = "2026.07.471"
INFERENCE_SUBSTRATE = "cpu_exact_checked_partitioned_hubo_2dpt"
RUN_DATE = "20260702"
TARGET_PENALTY = 4.0
DEFAULT_SWEEPS = 32
DEFAULT_SEEDS = (5141, 5142, 5143)
DEFAULT_PENALTY_GRID = (0.5, 1.0, 2.0, TARGET_PENALTY)
BOUNDARY_REFRESH_RATIOS = (1.0, 0.5, 0.0)
PARTITION_LAYOUT_IDS = ("monolithic", "contiguous_2", "checkerboard_2")
TEMPERATURE_LADDERS: tuple[JsonDict, ...] = (
    {
        "temperature_ladder_id": "v470_default",
        "beta_grid": (0.35, 0.8, 1.6, 3.0),
    },
    {
        "temperature_ladder_id": "compact_cold",
        "beta_grid": (0.45, 0.9, 1.8, 3.0),
    },
)
RESIDUAL_WINDOWS = ((1, 12), (12, DEFAULT_SWEEPS), (1, DEFAULT_SWEEPS))
COMPLETE_VERDICT = "complete_partition_telemetry_ready_exact_checked_cpu_no_speedup"
NOT_READY_VERDICT = "complete_partition_telemetry_not_ready_cpu_no_speedup"
BLOCKED_VERDICT = "blocked_partition_telemetry_exact_enumeration_failed"
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_", "complete:", "success:")
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "experiment_id",
        "milestone",
        "honest_verdict",
        "inference_substrate",
        "duration_s",
        "exp5129_baseline_loaded",
        "partition_configs",
        "boundary_refresh_ratios",
        "residual_energy_exponents",
        "exact_enumeration_checked",
        "detailed_balance_evidence",
        "monolithic_reference",
        "board_ready_workload_descriptors",
        "partition_telemetry_ready",
        "hardware_speedup_claimed",
        "conductor_modified",
        "tests_run",
    }
)
FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "exp5129_baseline_loaded": "upstream evidence",
    "partition_configs": "reproducibility",
    "boundary_refresh_ratios": "hardware-mapping telemetry",
    "residual_energy_exponents": "sampler-quality telemetry",
    "exact_enumeration_checked": "correctness",
    "detailed_balance_evidence": "sampler validity",
    "monolithic_reference": "baseline adequacy",
    "board_ready_workload_descriptors": "hardware handoff",
    "partition_telemetry_ready": "downstream readiness",
    "hardware_speedup_claimed": "hardware claim discipline",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
    "schema": "artifact schema stability",
    "run_date": "run labeling",
    "result_path": "artifact reachability",
    "spec_refs": "OpenSpec traceability",
    "random_seed": "deterministic replay anchor",
    "reproducibility_checksum": "content-addressed reproducibility",
    "temperature_ladders": "algorithm provenance",
    "unguided_baseline": "baseline adequacy",
    "partition_variant_summaries": "per-configuration evidence",
    "effective_sample_quality": "sample utility telemetry",
    "telemetry_stability": "ready-gate evidence",
}
DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5141_hubo_partition_residual_exponent_v471.py --date 20260702",
    ".venv/bin/pytest tests/python/test_hubo_partition_residual_exponent_5141.py -q",
    ".venv/bin/pytest tests/python/test_hubo_partition_residual_exponent_5141.py "
    "--cov=python/carnot/experiment_5141_hubo_partition_residual_exponent_v471.py "
    "--cov=scripts/experiment_5141_hubo_partition_residual_exponent_v471.py "
    "--cov-report=term-missing --cov-fail-under=100 -q",
    ".venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class Partitioned2DPTConfig:
    """Configuration for one CPU partitioned 2D-PT telemetry run."""

    beta_grid: tuple[float, ...]
    penalty_grid: tuple[float, ...]
    sweeps: int
    partition_layout_id: str
    partitions: tuple[tuple[int, ...], ...]
    boundary_refresh_ratio: float
    swap_interval: int = 1

    def __post_init__(self) -> None:
        """Validate the run shape before random state is drawn."""

        _validate_positive_grid(self.beta_grid, "beta_grid")
        _validate_positive_grid(self.penalty_grid, "penalty_grid")
        if self.sweeps < 1:
            raise ValueError("sweeps must be at least 1")
        if self.swap_interval < 1:
            raise ValueError("swap_interval must be at least 1")
        if not 0.0 <= self.boundary_refresh_ratio <= 1.0:
            raise ValueError("boundary_refresh_ratio must be in [0, 1]")
        flattened = [variable for partition in self.partitions for variable in partition]
        if not flattened or sorted(flattened) != list(range(max(flattened) + 1)):
            raise ValueError("partitions must cover variables exactly once from zero")
        if any(len(partition) == 0 for partition in self.partitions):
            raise ValueError("partitions must not be empty")

    @property
    def target_penalty(self) -> float:
        """Return the cold penalty used for final utility metrics."""

        return max(self.penalty_grid)


@dataclass(frozen=True)
class PartitionedRunResult:
    """Partitioned 2D-PT run with boundary-cache telemetry."""

    algorithm: str
    best_energy: float
    final_energy: float
    best_state: tuple[int, ...]
    final_state: tuple[int, ...]
    energy_trace: tuple[float, ...]
    swap_stats: Mapping[str, SwapStats]
    beta_grid: tuple[float, ...]
    penalty_grid: tuple[float, ...]
    partition_layout_id: str
    partitions: tuple[tuple[int, ...], ...]
    boundary_refresh_ratio: float
    boundary_mismatch_rate: float
    boundary_reads: int
    boundary_mismatches: int

    def as_dict(self) -> JsonDict:
        """Return deterministic, JSON-safe telemetry for this run."""

        return {
            "algorithm": self.algorithm,
            "best_energy": self.best_energy,
            "final_energy": self.final_energy,
            "best_state": list(self.best_state),
            "final_state": list(self.final_state),
            "energy_trace": list(self.energy_trace),
            "swap_stats": {
                axis: stats.as_dict()
                for axis, stats in sorted(self.swap_stats.items())
            },
            "beta_grid": list(self.beta_grid),
            "penalty_grid": list(self.penalty_grid),
            "partition_layout_id": self.partition_layout_id,
            "partitions": [list(partition) for partition in self.partitions],
            "boundary_refresh_ratio": self.boundary_refresh_ratio,
            "boundary_mismatch_rate": self.boundary_mismatch_rate,
            "boundary_reads": self.boundary_reads,
            "boundary_mismatches": self.boundary_mismatches,
        }


def build_partition_telemetry_instances() -> tuple[HuboProblem, ...]:
    """Return exact-enumerable HUBO parity and pairwise Ising instances."""

    ising_terms = tuple(
        HuboTerm(variables=(left, (left + 1) % 6), coefficient=-1.0)
        for left in range(6)
    )
    ising = HuboProblem(
        name="ising_ring6_ferromagnetic",
        family="ising_pairwise_ring",
        n_vars=6,
        constraint_constant=0.0,
        constraint_terms=(),
        objective_constant=0.0,
        objective_terms=ising_terms,
        description="Six-spin ferromagnetic Ising ring represented as pairwise HUBO terms.",
    )
    return (*build_synthetic_hubo_families(), ising)


def partition_layout_for_n_vars(layout_id: str, n_vars: int) -> tuple[tuple[int, ...], ...]:
    """Return a deterministic partition layout for a problem size."""

    if n_vars < 1:
        raise ValueError("n_vars must be positive")
    if layout_id == "monolithic":
        return (tuple(range(n_vars)),)
    if layout_id == "contiguous_2":
        split = max(1, n_vars // 2)
        return (tuple(range(split)), tuple(range(split, n_vars)))
    if layout_id == "checkerboard_2":
        even = tuple(index for index in range(n_vars) if index % 2 == 0)
        odd = tuple(index for index in range(n_vars) if index % 2 == 1)
        return (even, odd)
    raise ValueError(f"unknown partition layout: {layout_id}")


def run_partitioned_2dpt(
    problem: HuboProblem,
    config: Partitioned2DPTConfig,
    *,
    seed: int,
    exact_optimum_energy: float,
) -> PartitionedRunResult:
    """Run one partitioned 2D-PT sampler and collect boundary telemetry."""

    rng = np.random.default_rng(seed)
    beta_grid = tuple(float(value) for value in config.beta_grid)
    penalty_grid = tuple(float(value) for value in config.penalty_grid)
    states = rng.integers(
        0,
        2,
        size=(len(beta_grid), len(penalty_grid), problem.n_vars),
        dtype=np.int8,
    )
    caches = np.repeat(states[:, :, None, :], repeats=len(config.partitions), axis=2)
    best_energy, best_state = _best_target_state(problem, states, target_penalty=config.target_penalty)
    beta_stats = SwapStats()
    penalty_stats = SwapStats()
    boundary_reads = 0
    boundary_mismatches = 0
    boundary_vars = _boundary_variables_by_partition(problem, config.partitions)
    trace: list[float] = []

    for sweep in range(1, config.sweeps + 1):
        for beta_index, beta in enumerate(beta_grid):
            for penalty_index, penalty in enumerate(penalty_grid):
                reads, mismatches = _partitioned_sweep_in_place(
                    problem,
                    states[beta_index, penalty_index],
                    caches[beta_index, penalty_index],
                    partitions=config.partitions,
                    boundary_variables=boundary_vars,
                    beta=beta,
                    penalty=penalty,
                    boundary_refresh_ratio=config.boundary_refresh_ratio,
                    rng=rng,
                )
                boundary_reads += reads
                boundary_mismatches += mismatches

        if sweep % config.swap_interval == 0:
            phase = (sweep // config.swap_interval) % 2
            beta_stats = _swap_beta_axis_partitioned(
                problem,
                states,
                caches,
                beta_grid=beta_grid,
                penalty_grid=penalty_grid,
                stats=beta_stats,
                rng=rng,
                phase=phase,
            )
            penalty_stats = _swap_penalty_axis_partitioned(
                problem,
                states,
                caches,
                beta_grid=beta_grid,
                penalty_grid=penalty_grid,
                stats=penalty_stats,
                rng=rng,
                phase=phase,
            )

        current_best, current_state = _best_target_state(
            problem,
            states,
            target_penalty=config.target_penalty,
        )
        if current_best < best_energy:
            best_energy = current_best
            best_state = current_state
        trace.append(_round_energy(max(best_energy, exact_optimum_energy)))

    final_state = tuple(int(value) for value in states[-1, -1])
    mismatch_rate = 0.0 if boundary_reads == 0 else boundary_mismatches / boundary_reads
    return PartitionedRunResult(
        algorithm="partitioned_2dpt",
        best_energy=best_energy,
        final_energy=evaluate_hubo_energy(problem, final_state, penalty=config.target_penalty),
        best_state=best_state,
        final_state=final_state,
        energy_trace=tuple(trace),
        swap_stats={
            "beta_axis": beta_stats,
            "penalty_axis": penalty_stats,
        },
        beta_grid=beta_grid,
        penalty_grid=penalty_grid,
        partition_layout_id=config.partition_layout_id,
        partitions=config.partitions,
        boundary_refresh_ratio=config.boundary_refresh_ratio,
        boundary_mismatch_rate=_round_metric(mismatch_rate),
        boundary_reads=boundary_reads,
        boundary_mismatches=boundary_mismatches,
    )


def residual_trace(energy_trace: Sequence[float], optimum_energy: float) -> tuple[float, ...]:
    """Return nonnegative residual energy above the exact optimum."""

    return tuple(_round_metric(max(0.0, float(energy) - float(optimum_energy))) for energy in energy_trace)


def fit_residual_energy_exponent(
    residuals: Sequence[float],
    *,
    window: tuple[int, int],
) -> float:
    """Fit ``residual ~ step^-alpha`` on a one-indexed sweep window."""

    if not residuals:
        return 0.0
    start, end = window
    start = max(1, int(start))
    end = min(len(residuals), int(end))
    if end <= start:
        return 0.0
    xs = np.arange(start, end + 1, dtype=np.float64)
    ys = np.asarray(residuals[start - 1 : end], dtype=np.float64) + 1e-6
    log_x = np.log(xs)
    log_y = np.log(ys)
    variance = float(np.mean((log_x - float(np.mean(log_x))) ** 2))
    if variance <= 0.0:
        return 0.0
    covariance = float(np.mean((log_x - float(np.mean(log_x))) * (log_y - float(np.mean(log_y)))))
    return _round_metric(max(0.0, min(12.0, -covariance / variance)))


def detailed_balance_evidence_for_variant(
    problem: HuboProblem,
    *,
    partitions: tuple[tuple[int, ...], ...],
    beta: float,
    penalty: float,
    boundary_refresh_ratio: float,
) -> JsonDict:
    """Verify detailed balance or record why this partition variant cannot."""

    if boundary_refresh_ratio < 1.0:
        return {
            "checked": False,
            "passed": False,
            "blocker": (
                "stale boundary cache augments sampler state, so the physical-state-only "
                "kernel cannot satisfy detailed balance without including cache state"
            ),
            "boundary_refresh_ratio": boundary_refresh_ratio,
        }

    exact = exact_enumerate(problem, penalty=penalty)
    states = exact.all_states
    index = {state: row for row, state in enumerate(states)}
    energies = np.asarray(
        [evaluate_hubo_energy(problem, state, penalty=penalty) for state in states],
        dtype=np.float64,
    )
    pi = np.exp(-float(beta) * energies)
    pi = pi / float(np.sum(pi))
    transition = np.zeros((len(states), len(states)), dtype=np.float64)
    variable_weights = _variable_update_weights(partitions)

    for row, state in enumerate(states):
        state_array = np.asarray(state, dtype=np.int8)
        for variable, weight in variable_weights.items():
            zero_state = state_array.copy()
            one_state = state_array.copy()
            zero_state[variable] = 0
            one_state[variable] = 1
            energy_zero = evaluate_hubo_energy(problem, zero_state, penalty=penalty)
            energy_one = evaluate_hubo_energy(problem, one_state, penalty=penalty)
            probability_one = _logistic(-float(beta) * (energy_one - energy_zero))
            transition[row, index[tuple(int(value) for value in zero_state)]] += weight * (
                1.0 - probability_one
            )
            transition[row, index[tuple(int(value) for value in one_state)]] += weight * probability_one

    flow = pi[:, None] * transition
    max_error = float(np.max(np.abs(flow - flow.T)))
    row_error = float(np.max(np.abs(np.sum(transition, axis=1) - 1.0)))
    return {
        "checked": True,
        "passed": bool(max_error <= 1e-9 and row_error <= 1e-12),
        "blocker": None,
        "boundary_refresh_ratio": boundary_refresh_ratio,
        "n_states": len(states),
        "transition_rows_checked": int(transition.shape[0]),
        "max_abs_probability_flow_error": _round_metric(max_error),
        "max_abs_row_sum_error": _round_metric(row_error),
    }


def build_artifact(
    *,
    root: str | Path | None = None,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run the CPU partition telemetry sweep and return a terminal artifact."""

    started = time.perf_counter()
    repo_root = Path(root) if root is not None else REPO_ROOT
    baseline, baseline_path, baseline_attempts = _load_exp5129_baseline(repo_root)
    problems = build_partition_telemetry_instances()
    exact_rows = _exact_instance_rows(problems, baseline)
    exact_enumeration_checked = all(row["exact_enumeration_checked"] for row in exact_rows)
    unguided_baseline = _run_unguided_baseline(problems)
    variant_summaries = _run_partition_variant_sweep(problems)
    monolithic_reference = _monolithic_reference(variant_summaries, baseline, baseline_path)
    residual_energy_exponents = _residual_energy_exponent_block(
        variant_summaries,
        unguided_baseline,
        monolithic_reference,
    )
    detailed_balance_evidence = _detailed_balance_evidence_block(problems[0])
    partition_configs = _partition_config_block(problems)
    board_ready_workload_descriptors = _board_ready_workload_descriptors(variant_summaries)
    hardware_speedup_claimed = False
    conductor_modified = False
    telemetry_stability = _telemetry_stability(
        variant_summaries,
        unguided_baseline,
        monolithic_reference,
        board_ready_workload_descriptors,
    )
    partition_telemetry_ready = bool(
        exact_enumeration_checked
        and detailed_balance_evidence["all_unblocked_variants_passed"]
        and telemetry_stability["stable_enough_for_hardware_transcript_task"]
        and not hardware_speedup_claimed
        and not conductor_modified
    )
    honest_verdict = (
        COMPLETE_VERDICT
        if partition_telemetry_ready
        else BLOCKED_VERDICT
        if not exact_enumeration_checked
        else NOT_READY_VERDICT
    )
    elapsed = _round_metric(time.perf_counter() - started) if duration_s is None else duration_s
    artifact: JsonDict = {
        "schema": "carnot.experiment_5141_hubo_partition_residual_exponent.v471",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "result_path": RESULT_RELATIVE_PATH,
        "random_seed": DEFAULT_SEEDS[0],
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": elapsed,
        "exp5129_baseline_loaded": bool(baseline),
        "exp5129_baseline_path": baseline_path,
        "exp5129_baseline_attempted_paths": baseline_attempts,
        "partition_configs": partition_configs,
        "boundary_refresh_ratios": list(BOUNDARY_REFRESH_RATIOS),
        "boundary_refresh_ratio_notes": {
            "1.0": "fresh boundary reads; physical-state detailed balance can be checked exactly",
            "0.5": "stale-cache telemetry; detailed balance blocker is recorded",
            "0.0": "maximally stale boundary-cache stress case; detailed balance blocker is recorded",
        },
        "temperature_ladders": [
            {
                "temperature_ladder_id": ladder["temperature_ladder_id"],
                "beta_grid": list(ladder["beta_grid"]),
            }
            for ladder in TEMPERATURE_LADDERS
        ],
        "residual_energy_exponents": residual_energy_exponents,
        "exact_enumeration_checked": exact_enumeration_checked,
        "exact_instance_evidence": exact_rows,
        "detailed_balance_evidence": detailed_balance_evidence,
        "monolithic_reference": monolithic_reference,
        "unguided_baseline": unguided_baseline,
        "partition_variant_summaries": variant_summaries,
        "effective_sample_quality": _effective_sample_quality_block(
            variant_summaries,
            unguided_baseline,
            monolithic_reference,
        ),
        "board_ready_workload_descriptors": board_ready_workload_descriptors,
        "telemetry_stability": telemetry_stability,
        "partition_telemetry_ready": partition_telemetry_ready,
        "hardware_speedup_claimed": hardware_speedup_claimed,
        "conductor_modified": conductor_modified,
        "flagged_adversarial": bool(hardware_speedup_claimed or conductor_modified),
        "tests_run": list(tests_run) if tests_run is not None else list(DEFAULT_TESTS_RUN),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-SAMPLE-5141", "SCENARIO-SAMPLE-5141"],
        "methodology_note": (
            "Partitioned 2D PT is measured on CPU against exact enumeration. "
            "Boundary-stale variants are retained as hardware-mapping telemetry "
            "only; they record a detailed-balance blocker instead of claiming a "
            "valid physical-state reversible kernel."
        ),
    }
    artifact["reproducibility_checksum"] = _sha256_json(
        {
            "experiment_id": artifact["experiment_id"],
            "run_date": artifact["run_date"],
            "partition_configs": artifact["partition_configs"],
            "boundary_refresh_ratios": artifact["boundary_refresh_ratios"],
            "temperature_ladders": artifact["temperature_ladders"],
            "partition_variant_summaries": artifact["partition_variant_summaries"],
            "residual_energy_exponents": artifact["residual_energy_exponents"],
        }
    )
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5141 artifact violates the terminal contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required fields: {missing}")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    _require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(isinstance(artifact.get("duration_s"), (float, int)), "duration_s")
    _require(float(artifact["duration_s"]) >= 0.0, "duration_s")
    _require(artifact.get("exp5129_baseline_loaded") is True, "exp5129_baseline_loaded")
    _require(_partition_configs_valid(artifact.get("partition_configs")), "partition_configs")
    _require(list(artifact.get("boundary_refresh_ratios", [])) == list(BOUNDARY_REFRESH_RATIOS), "boundary_refresh_ratios")
    _require(_residual_exponents_valid(artifact.get("residual_energy_exponents")), "residual_energy_exponents")
    _require(artifact.get("exact_enumeration_checked") is True, "exact_enumeration_checked")
    _require(_balance_block_valid(artifact.get("detailed_balance_evidence")), "detailed_balance_evidence")
    _require(_reference_valid(artifact.get("monolithic_reference")), "monolithic_reference")
    _require(_descriptors_valid(artifact.get("board_ready_workload_descriptors")), "board_ready_workload_descriptors")
    _require(artifact.get("partition_telemetry_ready") is True, "partition_telemetry_ready")
    _require(artifact.get("hardware_speedup_claimed") is False, "hardware_speedup_claimed")
    _require(artifact.get("conductor_modified") is False, "conductor_modified")
    _require(isinstance(artifact.get("tests_run"), list) and bool(artifact["tests_run"]), "tests_run")
    _require(REQUIRED_ARTIFACT_FIELDS.issubset(artifact.get("field_principles", {})), "field_principles")


def write_artifact(
    *,
    root: str | Path | None = None,
    output_path: str | Path | None = None,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build and write the Exp 5141 terminal artifact."""

    repo_root = Path(root) if root is not None else REPO_ROOT
    destination = Path(output_path) if output_path is not None else repo_root / RESULT_RELATIVE_PATH
    artifact = build_artifact(
        root=repo_root,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(
    *,
    root: str | Path | None = None,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """CLI-compatible entrypoint used by the wrapper script and tests."""

    repo_root = Path(root) if root is not None else REPO_ROOT
    write_artifact(root=repo_root, run_date=date, duration_s=duration_s, tests_run=tests_run)
    return repo_root / RESULT_RELATIVE_PATH


def _partitioned_sweep_in_place(
    problem: HuboProblem,
    state: np.ndarray,
    cache_by_partition: np.ndarray,
    *,
    partitions: tuple[tuple[int, ...], ...],
    boundary_variables: Mapping[int, tuple[int, ...]],
    beta: float,
    penalty: float,
    boundary_refresh_ratio: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    reads = 0
    mismatches = 0
    for _ in range(problem.n_vars):
        partition_index = int(rng.integers(0, len(partitions)))
        partition = partitions[partition_index]
        if boundary_refresh_ratio >= 1.0 or rng.random() < boundary_refresh_ratio:
            cache_by_partition[partition_index] = state
        boundary = boundary_variables.get(partition_index, ())
        reads += len(boundary)
        mismatches += sum(
            int(cache_by_partition[partition_index, variable] != state[variable])
            for variable in boundary
        )
        variable = int(partition[int(rng.integers(0, len(partition)))])
        observed = state.copy()
        partition_set = set(partition)
        outside = [index for index in range(problem.n_vars) if index not in partition_set]
        if outside:
            observed[outside] = cache_by_partition[partition_index, outside]
        zero_state = observed.copy()
        one_state = observed.copy()
        zero_state[variable] = 0
        one_state[variable] = 1
        energy_zero = evaluate_hubo_energy(problem, zero_state, penalty=penalty)
        energy_one = evaluate_hubo_energy(problem, one_state, penalty=penalty)
        probability_one = _logistic(-float(beta) * (energy_one - energy_zero))
        state[variable] = 1 if rng.random() < probability_one else 0
        cache_by_partition[partition_index, variable] = state[variable]
    return reads, mismatches


def _swap_beta_axis_partitioned(
    problem: HuboProblem,
    states: np.ndarray,
    caches: np.ndarray,
    *,
    beta_grid: Sequence[float],
    penalty_grid: Sequence[float],
    stats: SwapStats,
    rng: np.random.Generator,
    phase: int,
) -> SwapStats:
    for penalty_index, penalty in enumerate(penalty_grid):
        for left in range(phase, len(beta_grid) - 1, 2):
            right = left + 1
            log_accept = metropolis_swap_log_acceptance(
                problem,
                states[left, penalty_index],
                states[right, penalty_index],
                beta_left=beta_grid[left],
                penalty_left=penalty,
                beta_right=beta_grid[right],
                penalty_right=penalty,
            )
            accepted = _accept_log_ratio(log_accept, rng)
            stats = stats.with_attempt(accepted=accepted)
            if accepted:
                states[[left, right], penalty_index] = states[[right, left], penalty_index]
                caches[[left, right], penalty_index] = caches[[right, left], penalty_index]
    return stats


def _swap_penalty_axis_partitioned(
    problem: HuboProblem,
    states: np.ndarray,
    caches: np.ndarray,
    *,
    beta_grid: Sequence[float],
    penalty_grid: Sequence[float],
    stats: SwapStats,
    rng: np.random.Generator,
    phase: int,
) -> SwapStats:
    for beta_index, beta in enumerate(beta_grid):
        for lower in range(phase, len(penalty_grid) - 1, 2):
            upper = lower + 1
            left_state = states[beta_index, lower].copy()
            right_state = states[beta_index, upper].copy()
            log_accept = metropolis_swap_log_acceptance(
                problem,
                left_state,
                right_state,
                beta_left=beta,
                penalty_left=penalty_grid[lower],
                beta_right=beta,
                penalty_right=penalty_grid[upper],
            )
            accepted = _accept_log_ratio(log_accept, rng)
            stats = stats.with_attempt(accepted=accepted)
            if accepted:
                states[beta_index, lower] = right_state
                states[beta_index, upper] = left_state
                caches[beta_index, [lower, upper]] = caches[beta_index, [upper, lower]]
    return stats


def _exact_instance_rows(
    problems: Sequence[HuboProblem],
    baseline: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for problem in problems:
        exact = exact_enumerate(problem, penalty=TARGET_PENALTY)
        exact_ok = (
            len(exact.all_states) == 2**problem.n_vars
            and sum(exact.energy_distribution.values()) == len(exact.all_states)
            and bool(exact.optimal_states)
        )
        upstream_status = _baseline_exact_label_status(baseline, problem.name, exact)
        rows.append(
            {
                "instance_id": problem.name,
                "family": problem.family,
                "n_vars": problem.n_vars,
                "exact_enumeration_checked": bool(exact_ok and upstream_status["labels_preserved"]),
                "upstream_exact_label_status": upstream_status,
                "exact": exact.as_dict(),
            }
        )
    return rows


def _baseline_exact_label_status(
    baseline: Mapping[str, Any],
    instance_id: str,
    exact: Any,
) -> JsonDict:
    if not baseline:
        return {
            "labels_preserved": False,
            "status": "exp5129_baseline_missing",
        }
    for row in baseline.get("per_instance_results", []):
        if row.get("instance_id") == instance_id:
            expected = row.get("exact", {}).get("optimal_states")
            observed = [list(state) for state in exact.optimal_states]
            return {
                "labels_preserved": expected == observed,
                "status": "matched_exp5129_exact_labels",
            }
    return {
        "labels_preserved": True,
        "status": "new_exact_enumerated_instance_not_present_in_exp5129",
    }


def _run_unguided_baseline(problems: Sequence[HuboProblem]) -> JsonDict:
    best_energies: list[float] = []
    residuals: list[tuple[float, ...]] = []
    hits = 0
    for problem_index, problem in enumerate(problems):
        exact = exact_enumerate(problem, penalty=TARGET_PENALTY)
        for seed in DEFAULT_SEEDS:
            run = run_unguided_gibbs(
                problem,
                seed=seed + 100 * problem_index,
                beta=3.0,
                penalty=TARGET_PENALTY,
                sweeps=DEFAULT_SWEEPS,
            )
            best_energies.append(run.best_energy)
            hits += int(_energy_equal(run.best_energy, exact.optimum_energy))
            residuals.append(residual_trace(run.energy_trace, exact.optimum_energy))
    mean_trace = _mean_trace(residuals)
    exponents = _window_exponents(mean_trace)
    final_residual = mean_trace[-1]
    hit_rate = hits / max(1, len(best_energies))
    return {
        "algorithm": "unguided_gibbs",
        "n_runs": len(best_energies),
        "optimum_hit_rate": _round_metric(hit_rate),
        "mean_best_energy": _round_metric(_mean(best_energies)),
        "mean_final_residual": final_residual,
        "residual_energy_exponents": exponents,
        "effective_sample_quality": _effective_quality(hit_rate, final_residual, 0.0, exponents["full"]),
    }


def _run_partition_variant_sweep(problems: Sequence[HuboProblem]) -> list[JsonDict]:
    summaries: list[JsonDict] = []
    for ladder in TEMPERATURE_LADDERS:
        beta_grid = tuple(float(value) for value in ladder["beta_grid"])
        ladder_id = str(ladder["temperature_ladder_id"])
        for layout_id in PARTITION_LAYOUT_IDS:
            for refresh_ratio in BOUNDARY_REFRESH_RATIOS:
                summaries.append(
                    _run_partition_variant(problems, layout_id, refresh_ratio, ladder_id, beta_grid)
                )
    return summaries


def _run_partition_variant(
    problems: Sequence[HuboProblem],
    layout_id: str,
    refresh_ratio: float,
    ladder_id: str,
    beta_grid: tuple[float, ...],
) -> JsonDict:
    best_energies: list[float] = []
    final_energies: list[float] = []
    residuals: list[tuple[float, ...]] = []
    beta_stats = SwapStats()
    penalty_stats = SwapStats()
    boundary_reads = 0
    boundary_mismatches = 0
    hits = 0
    sample_runs: list[JsonDict] = []
    for problem_index, problem in enumerate(problems):
        exact = exact_enumerate(problem, penalty=TARGET_PENALTY)
        partitions = partition_layout_for_n_vars(layout_id, problem.n_vars)
        config = Partitioned2DPTConfig(
            beta_grid=beta_grid,
            penalty_grid=DEFAULT_PENALTY_GRID,
            sweeps=DEFAULT_SWEEPS,
            partition_layout_id=layout_id,
            partitions=partitions,
            boundary_refresh_ratio=refresh_ratio,
        )
        for seed_index, seed in enumerate(DEFAULT_SEEDS):
            run = run_partitioned_2dpt(
                problem,
                config,
                seed=seed + 100 * problem_index,
                exact_optimum_energy=exact.optimum_energy,
            )
            best_energies.append(run.best_energy)
            final_energies.append(run.final_energy)
            hits += int(_energy_equal(run.best_energy, exact.optimum_energy))
            residuals.append(residual_trace(run.energy_trace, exact.optimum_energy))
            beta_stats = _add_stats(beta_stats, run.swap_stats["beta_axis"])
            penalty_stats = _add_stats(penalty_stats, run.swap_stats["penalty_axis"])
            boundary_reads += run.boundary_reads
            boundary_mismatches += run.boundary_mismatches
            if problem_index == 0 and seed_index == 0:
                sample_runs.append(
                    {
                        "instance_id": problem.name,
                        "seed": seed,
                        "run": run.as_dict(),
                    }
                )
    mean_trace = _mean_trace(residuals)
    exponents = _window_exponents(mean_trace)
    mismatch_rate = 0.0 if boundary_reads == 0 else boundary_mismatches / boundary_reads
    hit_rate = hits / max(1, len(best_energies))
    final_residual = mean_trace[-1]
    return {
        "variant_id": _variant_id(layout_id, refresh_ratio, ladder_id),
        "algorithm": "partitioned_2dpt",
        "partition_layout_id": layout_id,
        "temperature_ladder_id": ladder_id,
        "beta_grid": list(beta_grid),
        "penalty_grid": list(DEFAULT_PENALTY_GRID),
        "boundary_refresh_ratio": refresh_ratio,
        "sweeps": DEFAULT_SWEEPS,
        "n_runs": len(best_energies),
        "optimum_hit_rate": _round_metric(hit_rate),
        "mean_best_energy": _round_metric(_mean(best_energies)),
        "mean_final_energy": _round_metric(_mean(final_energies)),
        "mean_final_residual": final_residual,
        "residual_mean_by_sweep": list(mean_trace),
        "residual_energy_exponents": exponents,
        "boundary_mismatch_rate": _round_metric(mismatch_rate),
        "boundary_reads": boundary_reads,
        "boundary_mismatches": boundary_mismatches,
        "swap_acceptance_rates": {
            "beta_axis": beta_stats.as_dict(),
            "penalty_axis": penalty_stats.as_dict(),
        },
        "effective_sample_quality": _effective_quality(
            hit_rate,
            final_residual,
            mismatch_rate,
            exponents["full"],
        ),
        "sample_runs": sample_runs,
    }


def _residual_energy_exponent_block(
    variant_summaries: Sequence[Mapping[str, Any]],
    unguided_baseline: Mapping[str, Any],
    monolithic_reference: Mapping[str, Any],
) -> JsonDict:
    return {
        "windows": [
            {
                "window_id": _window_id(window),
                "start_sweep": window[0],
                "end_sweep": window[1],
            }
            for window in RESIDUAL_WINDOWS
        ],
        "unguided_baseline": dict(unguided_baseline["residual_energy_exponents"]),
        "monolithic_reference": dict(monolithic_reference["residual_energy_exponents"]),
        "by_variant": [
            {
                "variant_id": row["variant_id"],
                "partition_layout_id": row["partition_layout_id"],
                "boundary_refresh_ratio": row["boundary_refresh_ratio"],
                "temperature_ladder_id": row["temperature_ladder_id"],
                "exponents": row["residual_energy_exponents"],
            }
            for row in variant_summaries
        ],
    }


def _detailed_balance_evidence_block(problem: HuboProblem) -> JsonDict:
    variants: list[JsonDict] = []
    for ladder in TEMPERATURE_LADDERS:
        beta = float(ladder["beta_grid"][1])
        ladder_id = str(ladder["temperature_ladder_id"])
        for layout_id in PARTITION_LAYOUT_IDS:
            partitions = partition_layout_for_n_vars(layout_id, problem.n_vars)
            for refresh_ratio in BOUNDARY_REFRESH_RATIOS:
                evidence = detailed_balance_evidence_for_variant(
                    problem,
                    partitions=partitions,
                    beta=beta,
                    penalty=TARGET_PENALTY,
                    boundary_refresh_ratio=refresh_ratio,
                )
                evidence.update(
                    {
                        "variant_id": _variant_id(layout_id, refresh_ratio, ladder_id),
                        "instance_id": problem.name,
                        "partition_layout_id": layout_id,
                        "temperature_ladder_id": ladder_id,
                        "beta": beta,
                        "penalty": TARGET_PENALTY,
                    }
                )
                variants.append(evidence)
    checked = [row for row in variants if row["checked"]]
    blocked = [row for row in variants if not row["checked"]]
    max_error = max(
        (float(row.get("max_abs_probability_flow_error", 0.0)) for row in checked),
        default=0.0,
    )
    return {
        "instance_id": problem.name,
        "checked_variant_count": len(checked),
        "blocked_variant_count": len(blocked),
        "all_unblocked_variants_passed": all(row["passed"] for row in checked) and bool(checked),
        "max_abs_probability_flow_error": _round_metric(max_error),
        "blocked_variant_reasons": sorted({row["blocker"] for row in blocked}),
        "variants": variants,
    }


def _monolithic_reference(
    variant_summaries: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    baseline_path: str | None,
) -> JsonDict:
    selected = next(
        row
        for row in variant_summaries
        if row["partition_layout_id"] == "monolithic"
        and row["boundary_refresh_ratio"] == 1.0
        and row["temperature_ladder_id"] == "v470_default"
    )
    return {
        "source": "cpu_monolithic_partitioned_2dpt_reference",
        "exp5129_baseline_loaded": bool(baseline),
        "exp5129_baseline_path": baseline_path,
        "exp5129_adaptive_2dpt_ready": bool(baseline.get("adaptive_2dpt_ready", False)),
        "variant_id": selected["variant_id"],
        "optimum_hit_rate": selected["optimum_hit_rate"],
        "mean_best_energy": selected["mean_best_energy"],
        "mean_final_residual": selected["mean_final_residual"],
        "boundary_mismatch_rate": selected["boundary_mismatch_rate"],
        "residual_energy_exponents": selected["residual_energy_exponents"],
        "effective_sample_quality": selected["effective_sample_quality"],
    }


def _partition_config_block(problems: Sequence[HuboProblem]) -> list[JsonDict]:
    configs: list[JsonDict] = []
    for layout_id in PARTITION_LAYOUT_IDS:
        partitions_by_n = {
            str(problem.n_vars): [list(partition) for partition in partition_layout_for_n_vars(layout_id, problem.n_vars)]
            for problem in problems
        }
        boundary_counts = {
            problem.name: {
                str(index): len(variables)
                for index, variables in _boundary_variables_by_partition(
                    problem,
                    partition_layout_for_n_vars(layout_id, problem.n_vars),
                ).items()
            }
            for problem in problems
        }
        configs.append(
            {
                "partition_layout_id": layout_id,
                "partitions_by_n_vars": partitions_by_n,
                "boundary_variable_counts_by_instance": boundary_counts,
                "update_schedule": "random_partition_then_random_variable",
            }
        )
    return configs


def _board_ready_workload_descriptors(
    variant_summaries: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    descriptors: list[JsonDict] = []
    for target_board in ("kv260", "gatemate", "polarfire"):
        for row in variant_summaries:
            descriptor: JsonDict = {
                "descriptor_id": f"exp5141_{target_board}_{row['variant_id']}",
                "target_board": target_board,
                "hardware_executed": False,
                "hardware_speedup_claimed": False,
                "workload_family": "partitioned_hubo_2dpt_boundary_refresh",
                "partition_layout_id": row["partition_layout_id"],
                "boundary_refresh_ratio": row["boundary_refresh_ratio"],
                "temperature_ladder_id": row["temperature_ladder_id"],
                "beta_grid": row["beta_grid"],
                "penalty_grid": row["penalty_grid"],
                "sweeps": row["sweeps"],
                "replica_grid_shape": [len(row["beta_grid"]), len(row["penalty_grid"])],
                "n_vars_max": 6,
                "telemetry_fields": [
                    "optimum_hit_rate",
                    "residual_energy_exponent",
                    "boundary_mismatch_rate",
                    "effective_sample_quality",
                ],
                "mapping_note": _board_mapping_note(target_board),
            }
            descriptor["workload_hash"] = _sha256_json(descriptor)
            descriptors.append(descriptor)
    return descriptors


def _effective_sample_quality_block(
    variant_summaries: Sequence[Mapping[str, Any]],
    unguided_baseline: Mapping[str, Any],
    monolithic_reference: Mapping[str, Any],
) -> JsonDict:
    mono_quality = float(monolithic_reference["effective_sample_quality"])
    unguided_quality = float(unguided_baseline["effective_sample_quality"])
    return {
        "unguided_baseline": _round_metric(unguided_quality),
        "monolithic_reference": _round_metric(mono_quality),
        "by_variant": [
            {
                "variant_id": row["variant_id"],
                "quality": row["effective_sample_quality"],
                "delta_vs_monolithic": _round_metric(float(row["effective_sample_quality"]) - mono_quality),
                "delta_vs_unguided": _round_metric(float(row["effective_sample_quality"]) - unguided_quality),
            }
            for row in variant_summaries
        ],
    }


def _telemetry_stability(
    variant_summaries: Sequence[Mapping[str, Any]],
    unguided_baseline: Mapping[str, Any],
    monolithic_reference: Mapping[str, Any],
    descriptors: Sequence[Mapping[str, Any]],
) -> JsonDict:
    finite = all(
        _finite_metric(row["effective_sample_quality"])
        and _finite_metric(row["boundary_mismatch_rate"])
        and all(_finite_metric(value) for value in row["residual_energy_exponents"].values())
        for row in variant_summaries
    )
    descriptor_hashes_valid = all(
        isinstance(row.get("workload_hash"), str) and len(str(row["workload_hash"])) == 64
        for row in descriptors
    )
    refresh_sensitivity = _refresh_sensitivity(variant_summaries)
    monolithic_not_worse_than_unguided = (
        float(monolithic_reference["optimum_hit_rate"]) >= float(unguided_baseline["optimum_hit_rate"])
    )
    return {
        "finite_metrics": finite,
        "descriptor_hashes_valid": descriptor_hashes_valid,
        "refresh_sensitivity_observed": refresh_sensitivity,
        "monolithic_not_worse_than_unguided": monolithic_not_worse_than_unguided,
        "stable_enough_for_hardware_transcript_task": bool(
            finite
            and descriptor_hashes_valid
            and refresh_sensitivity
            and monolithic_not_worse_than_unguided
        ),
    }


def _refresh_sensitivity(variant_summaries: Sequence[Mapping[str, Any]]) -> bool:
    for layout_id in ("contiguous_2", "checkerboard_2"):
        fresh = [
            float(row["boundary_mismatch_rate"])
            for row in variant_summaries
            if row["partition_layout_id"] == layout_id and row["boundary_refresh_ratio"] == 1.0
        ]
        stale = [
            float(row["boundary_mismatch_rate"])
            for row in variant_summaries
            if row["partition_layout_id"] == layout_id and row["boundary_refresh_ratio"] == 0.0
        ]
        if fresh and stale and max(stale) > max(fresh):
            return True
    return False


def _load_exp5129_baseline(root: Path) -> tuple[JsonDict, str | None, list[str]]:
    candidates = [
        root / exp5129.RESULT_RELATIVE_PATH,
        root / "results/experiment_5129_hubo_adaptive_2dpt_cpu_v470.json",
        REPO_ROOT / exp5129.RESULT_RELATIVE_PATH,
        REPO_ROOT / "results/experiment_5129_hubo_adaptive_2dpt_cpu_v470.json",
    ]
    attempts = [str(path.relative_to(root) if path.is_relative_to(root) else path) for path in candidates]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        exp5129.validate_artifact(payload)
        return dict(payload), str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path), attempts
    return {}, None, attempts


def _boundary_variables_by_partition(
    problem: HuboProblem,
    partitions: tuple[tuple[int, ...], ...],
) -> dict[int, tuple[int, ...]]:
    owner = _variable_owner(partitions)
    boundary: dict[int, set[int]] = {index: set() for index in range(len(partitions))}
    terms = (*problem.constraint_terms, *problem.objective_terms)
    for term in terms:
        involved_partitions = {owner[variable] for variable in term.variables}
        if len(involved_partitions) <= 1:
            continue
        for partition_index in involved_partitions:
            local = set(partitions[partition_index])
            boundary[partition_index].update(variable for variable in term.variables if variable not in local)
    return {index: tuple(sorted(values)) for index, values in boundary.items()}


def _variable_owner(partitions: tuple[tuple[int, ...], ...]) -> dict[int, int]:
    owner: dict[int, int] = {}
    for partition_index, partition in enumerate(partitions):
        for variable in partition:
            owner[int(variable)] = partition_index
    return owner


def _variable_update_weights(partitions: tuple[tuple[int, ...], ...]) -> dict[int, float]:
    weights: dict[int, float] = {}
    partition_weight = 1.0 / len(partitions)
    for partition in partitions:
        variable_weight = partition_weight / len(partition)
        for variable in partition:
            weights[int(variable)] = variable_weight
    return weights


def _best_target_state(
    problem: HuboProblem,
    states: np.ndarray,
    *,
    target_penalty: float,
) -> tuple[float, tuple[int, ...]]:
    best_energy: float | None = None
    best_state: tuple[int, ...] | None = None
    for state in states.reshape((-1, problem.n_vars)):
        state_tuple = tuple(int(value) for value in state)
        energy = evaluate_hubo_energy(problem, state_tuple, penalty=target_penalty)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_state = state_tuple
    if best_energy is None or best_state is None:
        raise ValueError("state grid must not be empty")
    return best_energy, best_state


def _window_exponents(residuals: Sequence[float]) -> JsonDict:
    exponents = {
        _window_id(window): fit_residual_energy_exponent(residuals, window=window)
        for window in RESIDUAL_WINDOWS
    }
    exponents["full"] = exponents[_window_id((1, DEFAULT_SWEEPS))]
    return exponents


def _mean_trace(traces: Sequence[Sequence[float]]) -> tuple[float, ...]:
    by_sweep = list(zip(*traces, strict=True))
    return tuple(_round_metric(_mean(values)) for values in by_sweep)


def _effective_quality(
    optimum_hit_rate: float,
    final_residual: float,
    boundary_mismatch_rate: float,
    full_exponent: float,
) -> float:
    residual_factor = 1.0 / (1.0 + max(0.0, float(final_residual)))
    boundary_factor = max(0.0, 1.0 - min(1.0, float(boundary_mismatch_rate)))
    exponent_factor = 1.0 + min(2.0, max(0.0, float(full_exponent))) / 10.0
    return _round_metric(float(optimum_hit_rate) * residual_factor * boundary_factor * exponent_factor)


def _variant_id(layout_id: str, refresh_ratio: float, ladder_id: str) -> str:
    ratio = str(refresh_ratio).replace(".", "p")
    return f"{layout_id}_refresh_{ratio}_{ladder_id}"


def _window_id(window: tuple[int, int]) -> str:
    return f"sweep_{window[0]}_{window[1]}"


def _board_mapping_note(target_board: str) -> str:
    notes = {
        "kv260": "AXI/UIO transcript task may map refresh ticks to PL boundary-memory updates later.",
        "gatemate": "Descriptor is suitable for static RTL/testbench generation; no flash action taken.",
        "polarfire": "Descriptor is suitable for RISC-V dispatch planning; no board command executed.",
    }
    return notes[target_board]


def _partition_configs_valid(value: Any) -> bool:
    return isinstance(value, list) and {row.get("partition_layout_id") for row in value} == set(PARTITION_LAYOUT_IDS)


def _residual_exponents_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    variants = value.get("by_variant")
    if not isinstance(variants, list) or not variants:
        return False
    return all(
        all(_finite_metric(metric) and float(metric) >= 0.0 for metric in row.get("exponents", {}).values())
        for row in variants
    )


def _balance_block_valid(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("all_unblocked_variants_passed") is True
        and int(value.get("checked_variant_count", 0)) > 0
        and int(value.get("blocked_variant_count", 0)) > 0
    )


def _reference_valid(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("exp5129_baseline_loaded") is True
        and isinstance(value.get("optimum_hit_rate"), (float, int))
        and float(value.get("optimum_hit_rate")) >= 0.0
    )


def _descriptors_valid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    targets = {row.get("target_board") for row in value}
    hashes_ok = all(isinstance(row.get("workload_hash"), str) and len(row["workload_hash"]) == 64 for row in value)
    no_hardware = all(row.get("hardware_executed") is False for row in value)
    return targets == {"kv260", "gatemate", "polarfire"} and hashes_ok and no_hardware


def _finite_metric(value: Any) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def _add_stats(total: SwapStats, value: SwapStats) -> SwapStats:
    return SwapStats(
        attempts=total.attempts + value.attempts,
        accepted=total.accepted + value.accepted,
    )


def _accept_log_ratio(log_accept: float, rng: np.random.Generator) -> bool:
    if log_accept >= 0.0:
        return True
    return bool(math.log(max(rng.random(), 1e-300)) < log_accept)


def _logistic(value: float) -> float:
    clipped = min(60.0, max(-60.0, float(value)))
    return float(1.0 / (1.0 + math.exp(-clipped)))


def _validate_positive_grid(values: Sequence[float], name: str) -> None:
    if len(values) < 1:
        raise ValueError(f"{name} must not be empty")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"{name} values must be positive")
    if tuple(values) != tuple(sorted(values)):
        raise ValueError(f"{name} must be sorted ascending")


def _mean(values: Sequence[float] | Any) -> float:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers)


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _round_energy(value: float) -> float:
    return round(float(value), 12)


def _energy_equal(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-9


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
