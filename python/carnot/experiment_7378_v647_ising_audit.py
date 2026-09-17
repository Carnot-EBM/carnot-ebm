"""Audit the shipped CPU Ising sampler against the frozen Exp7377 law.

The exact energy fixture says what distribution is correct. This experiment
adds the missing empirical question: do fixed Markov chains reproduce bounded
observables from that distribution within a fixed sample budget?

Spec refs: REQ-SAMPLER-7378 and SCENARIO-SAMPLER-7378-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

# This experiment is explicitly CPU-only. Set the JAX platform before its first
# import so a host with visible GPUs cannot silently change the execution venue.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7377_v647_ising_law as exact_law
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.samplers.parallel_ising import ParallelIsingSampler


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260917"
MILESTONE = "2026.09.647"
PHASE = 3
EXPERIMENT_ID = "exp7378-v647-ising-audit"
SCHEMA = "carnot.exp7378.v647.ising_audit.v1"
RESULT_PATH = Path("results/experiment_7378_v647_ising_audit.json")
RAW_DIR = Path("results/raw/experiment_7378_v647_ising_audit")
TRACE_PATH = RAW_DIR / "ordered_chain_traces.jsonl.gz"
UPSTREAM_PATH = Path("results/experiment_7377_v647_ising_law.json")
MODULE_PATH = Path("python/carnot/experiment_7378_v647_ising_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7378_v647_ising_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7378_v647_ising_audit.py")
SPEC_PATH = Path("openspec/capabilities/samplers/spec.md")

BETA_GRID = (0.5, 1.0, 2.0)
CHAIN_SEEDS = (7_378_101, 7_378_211, 7_378_307, 7_378_401)
CHAIN_COUNT = 4
WARMUP_STEPS = 1_000
RECORDED_SAMPLES = 4_000
STEPS_PER_SAMPLE = 1
OBSERVABLE_ERROR_LIMIT = 0.05
ESS_MINIMUM = 1_000.0
ESS_LAG_WINDOW = 512
NEGATIVE_FORMULA_INDICES = (0, 8, 16)
FAITHFUL_CONDITIONS = ("source_only", "proof_assisted_source_only")
NEGATIVE_CONDITION = "appended_implied_clause_negative_control"

REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
E2E_CHECK_NAMES = ("e2e_002_python_jax", "e2e_python_rust_energy")
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_CURRENT_INVOCATIONS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
PRIOR_ATTEMPT_FINDINGS = (
    {
        "attempt": 1,
        "artifact_sha256": "sha256:feed952f9783bdcfe7805a195580eba12fba6fbcb4194709367b8d615cdeabed",
        "finding": "EXECUTION_VENUE_INVALID",
        "observed": "host_cpu",
        "expected": ["gatemate", "host", "kv260", "polarfire"],
        "disposition": "corrected_execution_venue_to_closed_value_host",
        "sampling_protocol_changed": False,
        "sample_budget_increased": False,
    },
)
REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7377_v647_ising_law.py"),
    Path("openspec/capabilities/ising-backend/spec.md"),
    SPEC_PATH,
    Path("python/carnot/samplers/parallel_ising.py"),
    Path("python/carnot/phase3/k_sat_ising.py"),
    Path("tests/python/test_e2e_training_sampling.py"),
    Path("tests/python/test_e2e_serialization.py"),
    UPSTREAM_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)
V647_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for traces, settings, and reduced evidence."""

    return validation_contract.canonical_hash(value)


def _sha256_file(path: Path) -> str:
    """Hash exact bytes without loading a potentially large trace at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def independent_source_energy(bits: Sequence[int], clauses: Sequence[Sequence[int]]) -> float:
    """Count unsatisfied clauses without calling the Exp7377 evaluator."""

    violations = 0
    for clause in clauses:
        clause_satisfied = False
        for raw_literal in clause:
            literal = int(raw_literal)
            index = abs(literal) - 1
            if index < 0 or index >= len(bits):
                raise ValueError("literal_out_of_range")
            value = int(bits[index])
            clause_satisfied = clause_satisfied or (value == (1 if literal > 0 else 0))
        violations += int(not clause_satisfied)
    return float(violations)


def _fixed_bits(assumptions: Sequence[int], n_vars: int) -> tuple[dict[int, int], bool]:
    """Convert unit assumptions to fixed bits and expose contradictions."""

    fixed: dict[int, int] = {}
    contradiction = False
    for raw_literal in assumptions:
        literal = int(raw_literal)
        index = abs(literal) - 1
        if index < 0 or index >= n_vars:
            raise ValueError("assumption_out_of_range")
        value = 1 if literal > 0 else 0
        if index in fixed and fixed[index] != value:
            contradiction = True
        fixed[index] = value
    return fixed, contradiction


def conditioned_sampler_parameters(
    formula: Mapping[str, Any], clauses: Sequence[Sequence[int]]
) -> JsonDict:
    """Algebraically clamp fixed spins before calling the unchanged sampler.

    Rejection after sampling would distort the retained chain and could hide a
    sampler defect. Eliminating fixed variables first samples the exact
    conditioned law and restores each sample without filtering.
    """

    n_vars = int(formula["n_vars"])
    fixed, contradiction = _fixed_bits(formula.get("assumptions") or [], n_vars)
    free = [index for index in range(n_vars) if index not in fixed]
    law = exact_law.compile_2cnf(n_vars, clauses)
    dense = np.zeros((n_vars, n_vars), dtype=np.float64)
    for left, right, value in law.couplings:
        dense[left, right] = float(value)
        dense[right, left] = float(value)

    fixed_spins = {index: 2 * bit - 1 for index, bit in fixed.items()}
    reduced_biases_pm = []
    for index in free:
        field = float(law.biases[index])
        field += math.fsum(dense[index, other] * spin for other, spin in fixed_spins.items())
        reduced_biases_pm.append(field)
    reduced_couplings_pm = dense[np.ix_(free, free)] if free else np.zeros((0, 0))

    # ParallelIsingSampler stores Boolean states but applies a +/-1 conditional.
    # This exact change of variables makes its sigmoid target the source law.
    sampler_biases = np.asarray(reduced_biases_pm, dtype=np.float64) - np.sum(
        reduced_couplings_pm, axis=1
    )
    sampler_couplings = 2.0 * reduced_couplings_pm

    constant = float(law.offset)
    constant -= math.fsum(float(law.biases[index]) * spin for index, spin in fixed_spins.items())
    for left, right, value in law.couplings:
        if left in fixed_spins and right in fixed_spins:
            constant -= float(value) * fixed_spins[left] * fixed_spins[right]
    boolean_offset = constant + math.fsum(reduced_biases_pm)
    boolean_offset -= math.fsum(
        reduced_couplings_pm[left, right]
        for left in range(len(free))
        for right in range(left + 1, len(free))
    )
    return {
        "n_vars": n_vars,
        "free_indices": free,
        "fixed_bits": fixed,
        "empty_support": contradiction,
        "biases": sampler_biases.astype(np.float32),
        "coupling_matrix": sampler_couplings.astype(np.float32),
        "boolean_offset": boolean_offset,
        "source_clauses": [list(map(int, clause)) for clause in clauses],
    }


def restore_bits(free_bits: Sequence[int], parameters: Mapping[str, Any]) -> tuple[int, ...]:
    """Restore one reduced sample to the original variable order."""

    free = list(parameters["free_indices"])
    if len(free_bits) != len(free):
        raise ValueError("free_state_length")
    restored = [0] * int(parameters["n_vars"])
    for index, value in dict(parameters["fixed_bits"]).items():
        restored[int(index)] = int(value)
    for index, value in zip(free, free_bits, strict=True):
        restored[int(index)] = int(value)
    return tuple(restored)


def reduced_boolean_energy(free_bits: Sequence[int], parameters: Mapping[str, Any]) -> float:
    """Evaluate the transformed Boolean polynomial used by the sampler."""

    values = np.asarray(free_bits, dtype=np.float64)
    biases = np.asarray(parameters["biases"], dtype=np.float64)
    coupling = np.asarray(parameters["coupling_matrix"], dtype=np.float64)
    pair = math.fsum(
        2.0 * coupling[left, right] * values[left] * values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return float(parameters["boolean_offset"]) - 2.0 * float(biases @ values) - pair


def _state_index(bits: Sequence[int]) -> int:
    """Encode bits in their retained lexical order."""

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def enumerated_target(
    formula: Mapping[str, Any],
    beta: float,
    *,
    clauses: Sequence[Sequence[int]] | None = None,
) -> JsonDict:
    """Enumerate the exact conditioned law and every bounded target moment."""

    n_vars = int(formula["n_vars"])
    source_clauses = [list(map(int, clause)) for clause in formula["original_clauses"]]
    law_clauses = [list(map(int, clause)) for clause in (clauses or source_clauses)]
    fixed, contradiction = _fixed_bits(formula.get("assumptions") or [], n_vars)
    states = exact_law.enumerate_bits(n_vars)
    support = [
        not contradiction and all(bits[index] == value for index, value in fixed.items())
        for bits in states
    ]
    independent = [independent_source_energy(bits, law_clauses) for bits in states]
    compiled = exact_law.compile_2cnf(n_vars, law_clauses)
    compiled_energy = [compiled.energy(exact_law.bits_to_spins(bits)) for bits in states]
    parity = max(abs(left - right) for left, right in zip(independent, compiled_energy))
    probabilities, normalizer = exact_law.normalized_probabilities(independent, support, beta)
    sparse_probabilities = {
        str(index): probability
        for index, probability in enumerate(probabilities)
        if probability > 0.0
    }
    if normalizer == 0.0:
        observables: list[JsonDict] = []
    else:
        series: dict[str, list[float]] = {
            "energy_per_clause": [
                independent_source_energy(bits, source_clauses) / len(source_clauses)
                for bits in states
            ]
        }
        series.update(
            {f"bit_{index}": [float(bits[index]) for bits in states] for index in range(n_vars)}
        )
        observables = []
        for name, values in series.items():
            mean = math.fsum(
                probability * value for probability, value in zip(probabilities, values)
            )
            variance = math.fsum(
                probability * (value - mean) ** 2
                for probability, value in zip(probabilities, values)
            )
            observables.append({"observable": name, "mean": mean, "variance": max(0.0, variance)})
    return {
        "formula_id": formula.get("formula_id"),
        "beta": float(beta),
        "support_size": sum(support),
        "normalizer": normalizer,
        "state_probabilities": sparse_probabilities,
        "observables": observables,
        "independent_energy_parity_max_error": parity,
        "law_clause_count": len(law_clauses),
        "source_clause_count": len(source_clauses),
        "target_sha256": canonical_hash(
            {
                "formula_id": formula.get("formula_id"),
                "beta": beta,
                "law_clauses": law_clauses,
                "support": support,
                "probabilities": probabilities,
            }
        ),
    }


def autocorrelation_diagnostics(values: Sequence[float], *, target_variance: float) -> JsonDict:
    """Estimate ESS with a fixed initial-positive paired autocorrelation sum."""

    array = np.asarray(values, dtype=np.float64)
    count = int(array.size)
    structurally_degenerate = target_variance <= 1e-15
    constant = count == 0 or bool(np.all(array == array[0]))
    if structurally_degenerate:
        return {
            "draw_count": count,
            "target_variance": float(target_variance),
            "structurally_degenerate": True,
            "constant_observed": constant,
            "lag_correlations": None,
            "integrated_autocorrelation_time": None,
            "effective_sample_size": None,
            "qualified": True,
            "estimator": "fixed_window_geyer_initial_positive_pairs_fft",
        }
    if count < 2 or constant:
        return {
            "draw_count": count,
            "target_variance": float(target_variance),
            "structurally_degenerate": False,
            "constant_observed": constant,
            "lag_correlations": None,
            "integrated_autocorrelation_time": None,
            "effective_sample_size": None,
            "qualified": False,
            "estimator": "fixed_window_geyer_initial_positive_pairs_fft",
        }

    centered = array - float(np.mean(array))
    denominator = float(np.dot(centered, centered))
    used = min(ESS_LAG_WINDOW, count - 1)
    transform_size = 1 << (2 * count - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=transform_size)
    products = np.fft.irfft(spectrum * np.conjugate(spectrum), n=transform_size)[: used + 1]
    correlations = products / denominator
    positive_pair_sum = 0.0
    stop_lag = 0
    for first in range(1, used, 2):
        second = first + 1
        pair = float(correlations[first])
        if second <= used:
            pair += float(correlations[second])
        if pair <= 0.0:
            break
        positive_pair_sum += pair
        stop_lag = second if second <= used else first
    integrated = max(1.0, 1.0 + 2.0 * positive_pair_sum)
    effective = min(float(count), max(1.0, float(count) / integrated))
    return {
        "draw_count": count,
        "target_variance": float(target_variance),
        "structurally_degenerate": False,
        "constant_observed": False,
        "lag_correlations": [float(value) for value in correlations[: min(33, len(correlations))]],
        "fixed_lag_window": ESS_LAG_WINDOW,
        "positive_sequence_stop_lag": stop_lag,
        "integrated_autocorrelation_time": integrated,
        "effective_sample_size": effective,
        "qualified": effective >= ESS_MINIMUM,
        "estimator": "fixed_window_geyer_initial_positive_pairs_fft",
    }


def _observable_series(
    samples: np.ndarray, source_clauses: Sequence[Sequence[int]]
) -> dict[str, list[float]]:
    energy = [
        independent_source_energy(bits, source_clauses) / len(source_clauses) for bits in samples
    ]
    result = {"energy_per_clause": energy}
    result.update(
        {
            f"bit_{index}": [float(value) for value in samples[:, index]]
            for index in range(samples.shape[1])
        }
    )
    return result


def _analyze_samples(
    samples: np.ndarray,
    target: Mapping[str, Any],
    source_clauses: Sequence[Sequence[int]],
) -> tuple[dict[str, list[float]], list[JsonDict]]:
    series = _observable_series(samples, source_clauses)
    target_by_name = {row["observable"]: row for row in target["observables"]}
    rows: list[JsonDict] = []
    for name, values in series.items():
        exact = target_by_name[name]
        observed = float(np.mean(values))
        diagnostics = autocorrelation_diagnostics(values, target_variance=float(exact["variance"]))
        rows.append(
            {
                "observable": name,
                "exact_mean": float(exact["mean"]),
                "observed_mean": observed,
                "absolute_error": abs(observed - float(exact["mean"])),
                "autocorrelation": diagnostics,
            }
        )
    return series, rows


def _support_violation_count(samples: np.ndarray, assumptions: Sequence[int]) -> int:
    fixed, contradiction = _fixed_bits(assumptions, samples.shape[1])
    if contradiction:
        return len(samples)
    return sum(any(int(bits[index]) != value for index, value in fixed.items()) for bits in samples)


def run_sampler_chain(
    formula: Mapping[str, Any],
    *,
    beta: float,
    condition: str,
    chain_order: int,
    seed: int,
    clauses: Sequence[Sequence[int]] | None = None,
    n_warmup: int = WARMUP_STEPS,
    n_samples: int = RECORDED_SAMPLES,
    steps_per_sample: int = STEPS_PER_SAMPLE,
    target: Mapping[str, Any] | None = None,
) -> tuple[JsonDict, JsonDict | None]:
    """Run one fixed chain and keep exact timing and ordered-trace evidence."""

    started = time.monotonic()
    source_clauses = [list(map(int, clause)) for clause in formula["original_clauses"]]
    law_clauses = [list(map(int, clause)) for clause in (clauses or source_clauses)]
    build_started = time.monotonic()
    exact_target = dict(target or enumerated_target(formula, beta, clauses=law_clauses))
    build_time = time.monotonic() - build_started
    condition_started = time.monotonic()
    parameters = conditioned_sampler_parameters(formula, law_clauses)
    conditioning_time = time.monotonic() - condition_started
    base = {
        "row_type": "chain",
        "cell_id": f"{formula.get('formula_id')}|{beta}|{condition}",
        "formula_id": formula.get("formula_id"),
        "source_hash": formula.get("source_hash"),
        "beta": float(beta),
        "condition": condition,
        "chain_order": int(chain_order),
        "seed": int(seed),
        "warmup_steps": int(n_warmup),
        "planned_recorded_samples": int(n_samples),
        "steps_per_sample": int(steps_per_sample),
        "target_support_size": exact_target["support_size"],
        "target_sha256": exact_target["target_sha256"],
    }
    if parameters["empty_support"]:
        row = {
            **base,
            "outcome": "structural_empty_support",
            "recorded_samples": 0,
            "state_counts": {},
            "trace_sha256": None,
            "support_violation_count": 0,
            "observable_results": [],
            "observable_series": {},
            "autocorrelation_diagnostics": [],
            "timing_s": {
                "complete_elapsed": time.monotonic() - started,
                "build": build_time,
                "conditioning": conditioning_time,
                "updates": 0.0,
                "analysis": 0.0,
            },
            "costs": {
                "warmup_updates": 0,
                "recorded_updates": 0,
                "current_model_calls": 0,
            },
            "failures": ["empty_conditioned_support_has_no_probability_law"],
            "censored": True,
        }
        return row, None

    update_started = time.monotonic()
    free_count = len(parameters["free_indices"])
    if free_count == 0:
        reduced_samples = np.zeros((n_samples, 0), dtype=bool)
    else:
        sampler = ParallelIsingSampler(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=steps_per_sample,
            schedule=None,
            use_checkerboard=True,
        )
        result = sampler.sample(
            jax.random.PRNGKey(seed),
            jnp.asarray(parameters["biases"]),
            jnp.asarray(parameters["coupling_matrix"]),
            beta=float(beta),
        )
        jax.block_until_ready(result)
        reduced_samples = np.asarray(result, dtype=bool)
    update_time = time.monotonic() - update_started

    analysis_started = time.monotonic()
    samples = np.asarray(
        [restore_bits(tuple(map(int, row)), parameters) for row in reduced_samples],
        dtype=np.int8,
    )
    state_indices = [_state_index(row) for row in samples]
    state_counts = {str(index): count for index, count in sorted(Counter(state_indices).items())}
    series, observable_rows = _analyze_samples(samples, exact_target, source_clauses)
    trace_hash = canonical_hash(state_indices)
    support_violations = _support_violation_count(samples, formula.get("assumptions") or [])
    analysis_time = time.monotonic() - analysis_started
    failures = [] if support_violations == 0 else ["conditioned_support_violation"]
    row = {
        **base,
        "outcome": "complete" if not failures else "failed",
        "recorded_samples": len(samples),
        "state_counts": state_counts,
        "trace_sha256": trace_hash,
        "support_violation_count": support_violations,
        "observable_results": observable_rows,
        "observable_series": series,
        "autocorrelation_diagnostics": [
            {
                "observable": item["observable"],
                **deepcopy(item["autocorrelation"]),
            }
            for item in observable_rows
        ],
        "timing_s": {
            "complete_elapsed": time.monotonic() - started,
            "build": build_time,
            "conditioning": conditioning_time,
            "updates": update_time,
            "analysis": analysis_time,
        },
        "costs": {
            "warmup_updates": n_warmup,
            "recorded_updates": n_samples * steps_per_sample,
            "current_model_calls": 0,
        },
        "failures": failures,
        "censored": False,
    }
    trace = {
        "cell_id": base["cell_id"],
        "formula_id": formula.get("formula_id"),
        "condition": condition,
        "beta": float(beta),
        "chain_order": int(chain_order),
        "seed": int(seed),
        "source_hash": formula.get("source_hash"),
        "state_indices": state_indices,
        "trace_sha256": trace_hash,
    }
    return row, trace


def _total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * math.fsum(
        abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys
    )


def aggregate_cell(
    formula_id: str,
    beta: float,
    condition: str,
    chains: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> JsonDict:
    """Combine four chains without pretending correlated draws are independent."""

    cell_id = f"{formula_id}|{beta}|{condition}"
    if target.get("support_size") == 0 and not target.get("state_probabilities"):
        return {
            "row_type": "cell",
            "cell_kind": "source_faithful",
            "cell_id": cell_id,
            "formula_id": formula_id,
            "beta": float(beta),
            "condition": condition,
            "outcome": "complete_structural_empty_support",
            "support_status": "empty_conditioned_support",
            "chain_count": len(chains),
            "recorded_samples": 0,
            "observable_results": [],
            "max_observable_error": None,
            "minimum_effective_samples": None,
            "observable_error_gate_passed": False,
            "effective_sample_gate_passed": False,
            "qualified": False,
            "interval_coverage": {"covered": 0, "total": 0, "rate": None},
            "descriptive_sparse_histogram_tv": None,
            "histogram_is_gate": False,
            "target_energy_parity_max_error": float(
                target.get("independent_energy_parity_max_error", math.inf)
            ),
            "target_sha256": target.get("target_sha256"),
            "costs": {"recorded_samples": 0, "current_model_calls": 0},
            "failures": ["empty_conditioned_support_has_no_probability_law"],
            "censored": False,
        }

    target_by_name = {row["observable"]: row for row in target["observables"]}
    observable_rows: list[JsonDict] = []
    for name, exact in target_by_name.items():
        chain_values = [list(map(float, row["observable_series"][name])) for row in chains]
        values = [value for chain in chain_values for value in chain]
        observed = float(np.mean(values))
        diagnostics = [
            autocorrelation_diagnostics(chain, target_variance=float(exact["variance"]))
            for chain in chain_values
        ]
        structurally_degenerate = float(exact["variance"]) <= 1e-15
        effective_values = [row["effective_sample_size"] for row in diagnostics]
        aggregate_effective = (
            None
            if structurally_degenerate
            else (
                math.fsum(float(value) for value in effective_values)
                if all(value is not None for value in effective_values)
                else None
            )
        )
        sample_variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
        if structurally_degenerate:
            low = high = observed
            ess_passed = True
        elif aggregate_effective is None:
            low = high = None
            ess_passed = False
        else:
            half_width = 1.96 * math.sqrt(sample_variance / aggregate_effective)
            low = max(0.0, observed - half_width)
            high = min(1.0, observed + half_width)
            ess_passed = aggregate_effective >= ESS_MINIMUM
        covered = low is not None and low <= float(exact["mean"]) <= high
        error = abs(observed - float(exact["mean"]))
        observable_rows.append(
            {
                "observable": name,
                "exact_mean": float(exact["mean"]),
                "exact_variance": float(exact["variance"]),
                "observed_mean": observed,
                "absolute_error": error,
                "error_limit": OBSERVABLE_ERROR_LIMIT,
                "error_gate_passed": error <= OBSERVABLE_ERROR_LIMIT,
                "aggregate_effective_samples": aggregate_effective,
                "effective_sample_minimum": ESS_MINIMUM,
                "effective_sample_gate_passed": ess_passed,
                "ci95_low": low,
                "ci95_high": high,
                "exact_mean_covered": covered,
                "per_chain_autocorrelation": diagnostics,
            }
        )

    counts: Counter[str] = Counter()
    total = 0
    for chain in chains:
        counts.update({str(key): int(value) for key, value in chain["state_counts"].items()})
        total += int(chain["recorded_samples"])
    empirical = {key: count / total for key, count in counts.items()} if total else {}
    histogram_tv = _total_variation(empirical, target["state_probabilities"])
    nondegenerate_ess = [
        float(row["aggregate_effective_samples"])
        for row in observable_rows
        if row["aggregate_effective_samples"] is not None
    ]
    minimum_ess = min(nondegenerate_ess) if nondegenerate_ess else float(total)
    error_passed = all(row["error_gate_passed"] for row in observable_rows)
    ess_passed = all(row["effective_sample_gate_passed"] for row in observable_rows)
    complete = (
        len(chains) == CHAIN_COUNT
        and {int(row["chain_order"]) for row in chains} == set(range(CHAIN_COUNT))
        and all(row.get("outcome") == "complete" for row in chains)
    )
    covered_count = sum(bool(row["exact_mean_covered"]) for row in observable_rows)
    return {
        "row_type": "cell",
        "cell_kind": (
            "wrong_law_negative_control" if condition == NEGATIVE_CONDITION else "source_faithful"
        ),
        "cell_id": cell_id,
        "formula_id": formula_id,
        "beta": float(beta),
        "condition": condition,
        "outcome": "complete" if complete else "incomplete",
        "support_status": "nonempty_conditioned_support",
        "chain_count": len(chains),
        "recorded_samples": total,
        "observable_results": observable_rows,
        "max_observable_error": max(row["absolute_error"] for row in observable_rows),
        "minimum_effective_samples": minimum_ess,
        "observable_error_gate_passed": error_passed,
        "effective_sample_gate_passed": ess_passed,
        "qualified": complete and error_passed and ess_passed,
        "interval_coverage": {
            "covered": covered_count,
            "total": len(observable_rows),
            "rate": covered_count / len(observable_rows),
        },
        "descriptive_sparse_histogram_tv": histogram_tv,
        "histogram_source": "sparse state-index histogram from four fixed 4000-draw chains",
        "histogram_is_gate": False,
        "target_energy_parity_max_error": float(
            target.get("independent_energy_parity_max_error", math.inf)
        ),
        "target_sha256": target.get("target_sha256"),
        "costs": {"recorded_samples": total, "current_model_calls": 0},
        "failures": [] if complete else ["chain_roster_incomplete"],
        "censored": False,
    }


def distribution_shift_control(
    formula: Mapping[str, Any],
    beta: float,
    source_target: Mapping[str, Any],
    appended_target: Mapping[str, Any],
) -> JsonDict:
    """Measure the deliberate wrong-law change independently of chain quality."""

    clauses = [list(map(int, clause)) for clause in formula["original_clauses"]]
    appended = [*clauses, list(map(int, formula["implied_clause"]))]
    states = exact_law.enumerate_bits(int(formula["n_vars"]))
    source_energy = [independent_source_energy(bits, clauses) for bits in states]
    appended_energy = [independent_source_energy(bits, appended) for bits in states]
    source_minimum = min(source_energy)
    appended_minimum = min(appended_energy)
    source_states = {index for index, value in enumerate(source_energy) if value == source_minimum}
    appended_states = {
        index for index, value in enumerate(appended_energy) if value == appended_minimum
    }
    distance = _total_variation(
        source_target["state_probabilities"], appended_target["state_probabilities"]
    )
    return {
        "row_type": "distribution_shift_control",
        "control_id": f"{formula.get('formula_id')}|{beta}|appended_clause",
        "formula_id": formula.get("formula_id"),
        "source_hash": formula.get("source_hash"),
        "beta": float(beta),
        "appended_clause": list(formula["implied_clause"]),
        "exact_total_variation": distance,
        "source_satisfying_minima_preserved": source_states == appended_states,
        "wrong_law_detected": distance > 1e-10 and source_states == appended_states,
        "source_target_sha256": source_target["target_sha256"],
        "appended_target_sha256": appended_target["target_sha256"],
        "outcome": "complete",
        "costs": {"states_enumerated": len(states), "current_model_calls": 0},
        "failures": [],
        "censored": False,
    }


def _public_chain_row(row: Mapping[str, Any]) -> JsonDict:
    """Remove series duplicated by the lossless raw trace archive."""

    result = deepcopy(dict(row))
    result.pop("observable_series", None)
    return result


def run_sampling_panel(
    formulas: Sequence[Mapping[str, Any]], *, started: float
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Run the frozen faithful panel and hostile appended-clause subset."""

    chain_rows: list[JsonDict] = []
    cell_rows: list[JsonDict] = []
    controls: list[JsonDict] = []
    traces: list[JsonDict] = []
    total_cells = len(formulas) * len(BETA_GRID) * len(FAITHFUL_CONDITIONS)
    total_cells += len(NEGATIVE_FORMULA_INDICES) * len(BETA_GRID)
    completed = 0
    for formula in formulas:
        for beta in BETA_GRID:
            target = enumerated_target(formula, beta)
            for condition in FAITHFUL_CONDITIONS:
                private_rows: list[JsonDict] = []
                for order, seed in enumerate(CHAIN_SEEDS):
                    row, trace = run_sampler_chain(
                        formula,
                        beta=beta,
                        condition=condition,
                        chain_order=order,
                        seed=seed,
                        target=target,
                    )
                    private_rows.append(row)
                    chain_rows.append(_public_chain_row(row))
                    if trace is not None:
                        traces.append(trace)
                cell_rows.append(
                    aggregate_cell(
                        str(formula["formula_id"]), beta, condition, private_rows, target
                    )
                )
                completed += 1
                progress(
                    started,
                    "evaluate",
                    "cell_complete",
                    completed=f"{completed}/{total_cells}",
                    cell=cell_rows[-1]["cell_id"],
                )

    for formula_index in NEGATIVE_FORMULA_INDICES:
        formula = formulas[formula_index]
        clauses = [*formula["original_clauses"], formula["implied_clause"]]
        for beta in BETA_GRID:
            source_target = enumerated_target(formula, beta)
            appended_target = enumerated_target(formula, beta, clauses=clauses)
            private_rows = []
            for order, seed in enumerate(CHAIN_SEEDS):
                row, trace = run_sampler_chain(
                    formula,
                    beta=beta,
                    condition=NEGATIVE_CONDITION,
                    chain_order=order,
                    seed=seed,
                    clauses=clauses,
                    target=appended_target,
                )
                private_rows.append(row)
                chain_rows.append(_public_chain_row(row))
                if trace is not None:
                    traces.append(trace)
            cell = aggregate_cell(
                str(formula["formula_id"]), beta, NEGATIVE_CONDITION, private_rows, appended_target
            )
            source_by_name = {
                row["observable"]: float(row["mean"]) for row in source_target["observables"]
            }
            cell["maximum_observed_error_from_original_source_target"] = max(
                abs(float(row["observed_mean"]) - source_by_name[row["observable"]])
                for row in cell["observable_results"]
            )
            cell_rows.append(cell)
            control = distribution_shift_control(formula, beta, source_target, appended_target)
            control["sampled_appended_target_qualified"] = cell["qualified"]
            control["sampled_appended_target_max_error"] = cell["max_observable_error"]
            control["sampled_error_from_original_source_target"] = cell[
                "maximum_observed_error_from_original_source_target"
            ]
            controls.append(control)
            completed += 1
            progress(
                started,
                "evaluate",
                "negative_cell_complete",
                completed=f"{completed}/{total_cells}",
                cell=cell["cell_id"],
            )
    return chain_rows, cell_rows, controls, traces


def atomic_gzip_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Write ordered chain traces through a local fsync and atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for record in records:
                line = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                compressed.write(line + b"\n")
        raw.flush()
        os.fsync(raw.fileno())
    temporary.replace(path)
    return {
        "path": path.as_posix(),
        "compression": "gzip_json_lines_mtime_zero",
        "record_count": len(records),
        "byte_count": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def load_trace_archive(path: Path, manifest: Mapping[str, Any]) -> list[JsonDict]:
    """Reload exact trace bytes and reject count, length, or hash drift."""

    if not path.is_file():
        raise ValueError("trace_archive_missing")
    if path.stat().st_size != int(manifest["byte_count"]):
        raise ValueError("trace_archive_byte_count")
    if _sha256_file(path) != manifest["sha256"]:
        raise ValueError("trace_archive_sha256")
    records: list[JsonDict] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("trace_archive_row_not_object")
            records.append(value)
    if len(records) != int(manifest["record_count"]):
        raise ValueError("trace_archive_record_count")
    return records


def validate_raw_evidence(artifact: Mapping[str, Any], repo_root: Path = REPO_ROOT) -> JsonDict:
    """Cold-reduce ordered traces against every nonempty public chain row."""

    manifest = dict(artifact.get("raw_trace_archive") or {})
    try:
        path = Path(str(manifest["path"]))
        if not path.is_absolute():
            path = repo_root / path
        records = load_trace_archive(path, manifest)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        return {"passed": False, "error": str(error), "record_count": 0}
    expected = {
        (str(row["cell_id"]), int(row["chain_order"])): row
        for row in artifact.get("per_chain_results") or []
        if int(row.get("recorded_samples", 0)) > 0
    }
    observed: dict[tuple[str, int], Mapping[str, Any]] = {}
    errors: list[str] = []
    for record in records:
        key = (str(record.get("cell_id")), int(record.get("chain_order", -1)))
        if key in observed:
            errors.append(f"duplicate_trace:{key}")
        observed[key] = record
        indices = [int(value) for value in record.get("state_indices") or []]
        if canonical_hash(indices) != record.get("trace_sha256"):
            errors.append(f"trace_hash:{key}")
    if set(expected) != set(observed):
        errors.append("trace_roster")
    for key in set(expected) & set(observed):
        row = expected[key]
        record = observed[key]
        indices = [int(value) for value in record["state_indices"]]
        counts = {str(index): count for index, count in sorted(Counter(indices).items())}
        if len(indices) != int(row["recorded_samples"]):
            errors.append(f"trace_length:{key}")
        if counts != row["state_counts"]:
            errors.append(f"trace_counts:{key}")
        if record.get("trace_sha256") != row.get("trace_sha256"):
            errors.append(f"public_trace_hash:{key}")
    return {
        "passed": not errors,
        "record_count": len(records),
        "expected_record_count": len(expected),
        "errors": errors,
        "archive_sha256": manifest.get("sha256"),
    }


def _precondition(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool | None = None,
    terminal_blocking: bool = True,
) -> JsonDict:
    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else passed,
        "terminal_blocking": terminal_blocking,
    }


def authenticate_upstream(path: Path) -> tuple[list[JsonDict], dict[str, str], JsonDict]:
    """Authenticate the exact Exp7377 producer before using its formulas."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    if not path.is_file() or path.stat().st_size == 0:
        checks.append(
            _precondition(
                "upstream_artifact_bytes",
                path.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "missing",
            )
        )
        return checks, hashes, {}
    hashes[UPSTREAM_PATH.as_posix()] = _sha256_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        checks.append(
            _precondition(
                "upstream_artifact_bytes",
                path.as_posix(),
                "bytes",
                "valid_json_object",
                "malformed_json",
            )
        )
        return checks, hashes, {}
    if not isinstance(value, dict):
        value = {}
    checks.append(
        _precondition(
            "upstream_artifact_bytes",
            path.as_posix(),
            "bytes",
            "readable_nonempty_bytes",
            "readable_nonempty_bytes",
        )
    )
    expected_equal = {
        "experiment_id": "exp7377-v647-ising-law",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "law_fixture_ready_score": 1,
        "flagged_adversarial": False,
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
    }
    for field, expected in expected_equal.items():
        checks.append(
            _precondition(f"upstream_{field}", path.as_posix(), field, expected, value.get(field))
        )
    verdict = value.get("verdict_class")
    allowed = ["positive", "circular_positive", "null"]
    checks.append(
        _precondition(
            "upstream_verdict_class",
            path.as_posix(),
            "verdict_class",
            allowed,
            verdict,
            passed=verdict in allowed,
        )
    )
    formulas = value.get("frozen_formulas") or []
    protocol = value.get("frozen_sampling_protocol") or {}
    fixture_hash = exact_law.fixture_hash(formulas) if isinstance(formulas, list) else None
    protocol_expected = {
        "formula_count": 24,
        "beta_grid": list(BETA_GRID),
        "chain_count": CHAIN_COUNT,
        "chain_seeds": list(CHAIN_SEEDS),
        "warmup_steps_per_chain": WARMUP_STEPS,
        "recorded_samples_per_chain": RECORDED_SAMPLES,
        "steps_per_recorded_sample": STEPS_PER_SAMPLE,
        "tuning_after_observation": False,
    }
    protocol_observed = {
        "formula_count": len(formulas),
        "beta_grid": protocol.get("beta_grid"),
        "chain_count": protocol.get("chain_count"),
        "chain_seeds": protocol.get("chain_seeds"),
        "warmup_steps_per_chain": protocol.get("warmup_steps_per_chain"),
        "recorded_samples_per_chain": protocol.get("recorded_samples_per_chain"),
        "steps_per_recorded_sample": protocol.get("steps_per_recorded_sample"),
        "tuning_after_observation": protocol.get("tuning_after_observation"),
    }
    checks.append(
        _precondition(
            "upstream_frozen_protocol",
            path.as_posix(),
            "frozen_sampling_protocol",
            protocol_expected,
            protocol_observed,
        )
    )
    checks.append(
        _precondition(
            "upstream_fixture_hash",
            path.as_posix(),
            "frozen_sampling_protocol.fixture_sha256",
            fixture_hash,
            protocol.get("fixture_sha256"),
        )
    )
    producer_errors = exact_law.validate_artifact(value)
    checks.append(
        _precondition(
            "upstream_producer_validation",
            path.as_posix(),
            "producer_validator_errors",
            [],
            producer_errors,
        )
    )
    return checks, hashes, value


def collect_preconditions(
    repo_root: Path,
) -> tuple[list[JsonDict], dict[str, str], JsonDict, list[JsonDict]]:
    """Check exact local sources, upstream gates, CPU resources, and quarantine."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_PATHS:
        if relative == UPSTREAM_PATH:
            continue
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if present else "missing",
            )
        )
        if present:
            hashes[relative.as_posix()] = _sha256_file(path)
    upstream_checks, upstream_hashes, upstream = authenticate_upstream(root / UPSTREAM_PATH)
    checks.extend(upstream_checks)
    hashes.update(upstream_hashes)
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-SAMPLER-7378",
            "REQ-SAMPLER-7378" if "REQ-SAMPLER-7378" in spec_text else None,
        )
    )
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7378" in exclusion_text or "exp7378-ising-audit" in exclusion_text
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            excluded,
        )
    )
    checks.append(
        _precondition(
            "cpu_only_jax_backend",
            "jax.default_backend",
            "backend",
            "cpu",
            jax.default_backend(),
        )
    )
    try:
        from carnot import _rust as rust_extension  # noqa: PLC0415

        rust_path = Path(rust_extension.__file__).resolve()
        rust_available = rust_path.is_file()
        rust_observed: Any = str(rust_path) if rust_available else "missing"
    except ImportError:
        rust_available = False
        rust_observed = "missing"
    checks.append(
        _precondition(
            "existing_rust_extension_available",
            "python/carnot/_rust.*.so",
            "compiled_extension",
            "available_without_rebuild",
            "available_without_rebuild" if rust_available else rust_observed,
        )
    )
    sidecars = [
        {
            "label": "exp7377_historical_inference_provenance",
            "source_path": UPSTREAM_PATH.as_posix(),
            "source_sha256": hashes.get(UPSTREAM_PATH.as_posix()),
            "counted_as_current": False,
            "receipts": deepcopy(upstream.get("historical_inference_sidecars") or []),
        }
    ]
    return checks, hashes, upstream, sidecars


def build_validation_plan(
    repo_root: Path, private_root: Path
) -> list[validation_scope.CommandSpec]:
    """Extend the exact Exp7358 affected plan with two bounded E2E commands."""

    commands = validation_contract.build_command_plan(repo_root, V647_MANIFEST, private_root)
    basetemp = private_root / "basetemp"
    coverage = private_root / "coverage"
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage.mkdir(parents=True, exist_ok=True)
    pytest = ".venv/bin/pytest"
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    commands.extend(
        [
            validation_scope.CommandSpec(
                "e2e_002_python_jax",
                (
                    pytest,
                    *common,
                    f"--basetemp={basetemp / 'e2e-python-jax'}",
                    "tests/python/test_e2e_training_sampling.py",
                    "-q",
                ),
                "applicable_e2e_002_python_jax",
            ),
            validation_scope.CommandSpec(
                "e2e_python_rust_energy",
                (
                    pytest,
                    *common,
                    f"--basetemp={basetemp / 'e2e-python-rust'}",
                    (
                        "tests/python/test_e2e_serialization.py::"
                        "TestE2ESerializationPyO3CrossLanguage::"
                        "test_ising_rust_python_energy_agreement"
                    ),
                    "-q",
                ),
                "existing_compiled_python_rust_energy_harness",
            ),
        ]
    )
    return commands


def validate_validation_plan(
    repo_root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject command expansion, missing private parents, and E2E drift."""

    errors = validation_contract.validate_command_plan(
        repo_root, V647_MANIFEST, commands[: len(REQUIRED_CHECK_NAMES)]
    )
    names = [command.name for command in commands]
    expected = [*REQUIRED_CHECK_NAMES, *E2E_CHECK_NAMES]
    if names != expected:
        errors.append("validation_command_roster")
    if any(
        argument.rstrip("/") in {"tests", "tests/python"}
        for row in commands
        for argument in row.argv
    ):
        errors.append("broad_pytest_target")
    for command in commands:
        for argument in command.argv:
            if (
                argument.startswith("--basetemp=")
                and not Path(argument.split("=", 1)[1]).parent.is_dir()
            ):
                errors.append(f"missing_basetemp_parent:{command.name}")
    return list(dict.fromkeys(errors))


def _receipt_set_passes(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in names
    )


def classify_terminal(
    *,
    prerequisites_passed: bool,
    required_validation_passed: bool,
    capture_complete: bool,
    science_qualified: bool,
    flagged_adversarial: bool,
) -> JsonDict:
    """Keep external absence, invalid evidence, completion, and value distinct."""

    if not prerequisites_passed:
        verdict = "blocked"
        honest = "blocked_external_ising_law_prerequisite"
        capture = 0
        value = 0
    elif not required_validation_passed or not capture_complete or flagged_adversarial:
        verdict = "disqualified"
        honest = "complete_disqualified_ising_sampler_audit_evidence"
        capture = 0
        value = 0
    elif science_qualified:
        verdict = "circular_positive"
        honest = "complete_circular_positive_source_law_sampling_qualified"
        capture = 1
        value = 1
    else:
        verdict = "null"
        honest = "complete_null_source_law_sampling_not_fully_qualified"
        capture = 1
        value = 0
    return {
        "verdict_class": verdict,
        "honest_verdict": honest,
        "ising_sample_capture_complete_score": capture,
        "ising_law_value_score": value,
        "promotion_score": 0,
    }


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute evidence coverage, validations, hostile controls, and scores."""

    preconditions = [
        row
        for row in artifact.get("preconditions_checked") or []
        if isinstance(row, Mapping) and row.get("terminal_blocking") is True
    ]
    prerequisites = bool(preconditions) and all(row.get("passed") is True for row in preconditions)
    chains = list(artifact.get("per_chain_results") or [])
    cells = list(artifact.get("cell_results") or [])
    controls = list(artifact.get("distribution_shift_controls") or [])
    receipts = list(artifact.get("validation_receipts") or [])
    source_cells = [row for row in cells if row.get("cell_kind") == "source_faithful"]
    negative_cells = [row for row in cells if row.get("cell_kind") == "wrong_law_negative_control"]
    source_chains = [row for row in chains if row.get("condition") in FAITHFUL_CONDITIONS]
    negative_chains = [row for row in chains if row.get("condition") == NEGATIVE_CONDITION]
    expected_source_cell_ids = {
        f"{formula_id}|{beta}|{condition}"
        for formula_id in artifact.get("formula_ids") or []
        for beta in BETA_GRID
        for condition in FAITHFUL_CONDITIONS
    }
    expected_negative_cell_ids = {
        f"{formula_id}|{beta}|{NEGATIVE_CONDITION}"
        for formula_id in artifact.get("negative_control_formula_ids") or []
        for beta in BETA_GRID
    }

    def chains_complete(rows: Sequence[Mapping[str, Any]], cell_ids: set[str]) -> bool:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("cell_id")), []).append(row)
        return set(grouped) == cell_ids and all(
            len(group) == CHAIN_COUNT
            and {int(row.get("chain_order", -1)) for row in group} == set(range(CHAIN_COUNT))
            and all(row.get("outcome") in {"complete", "structural_empty_support"} for row in group)
            for group in grouped.values()
        )

    source_cells_complete = (
        len(source_cells) == 24 * len(BETA_GRID) * len(FAITHFUL_CONDITIONS)
        and {str(row.get("cell_id")) for row in source_cells} == expected_source_cell_ids
        and all(
            row.get("outcome") in {"complete", "complete_structural_empty_support"}
            for row in source_cells
        )
    )
    negative_cells_complete = (
        len(negative_cells) == len(NEGATIVE_FORMULA_INDICES) * len(BETA_GRID)
        and {str(row.get("cell_id")) for row in negative_cells} == expected_negative_cell_ids
        and all(row.get("outcome") == "complete" for row in negative_cells)
    )
    chain_evidence = chains_complete(source_chains, expected_source_cell_ids) and chains_complete(
        negative_chains, expected_negative_cell_ids
    )
    raw_passed = (artifact.get("raw_trace_reduction") or {}).get("passed") is True
    energy_controls = bool(source_cells) and all(
        float(row.get("target_energy_parity_max_error", math.inf)) <= 1e-12 for row in source_cells
    )
    hostile_controls = (
        len(controls) == len(NEGATIVE_FORMULA_INDICES) * len(BETA_GRID)
        and all(row.get("wrong_law_detected") is True for row in controls)
        and all(row.get("sampled_appended_target_qualified") is True for row in controls)
    )
    affected = _receipt_set_passes(receipts, REQUIRED_CHECK_NAMES)
    e2e = _receipt_set_passes(receipts, E2E_CHECK_NAMES)
    terminal = _receipt_set_passes(receipts, TERMINAL_CHECK_NAMES)
    validations = affected and e2e and terminal
    complete = (
        source_cells_complete
        and negative_cells_complete
        and chain_evidence
        and raw_passed
        and energy_controls
    )
    all_source_qualified = bool(source_cells) and all(
        row.get("qualified") is True for row in source_cells
    )
    science = complete and all_source_qualified and hostile_controls
    classification = classify_terminal(
        prerequisites_passed=prerequisites,
        required_validation_passed=validations,
        capture_complete=complete,
        science_qualified=science,
        flagged_adversarial=artifact.get("flagged_adversarial") is True,
    )
    return {
        "preconditions_passed": prerequisites,
        "source_cells_complete": source_cells_complete,
        "negative_control_cells_complete": negative_cells_complete,
        "chain_evidence_complete": chain_evidence,
        "raw_trace_evidence_passed": raw_passed,
        "independent_source_energy_controls_passed": energy_controls,
        "wrong_law_negative_controls_passed": hostile_controls,
        "affected_validation_passed": affected,
        "e2e_validation_passed": e2e,
        "terminal_validation_passed": terminal,
        "required_validation_passed": validations,
        "empty_support_cell_count": sum(
            row.get("support_status") == "empty_conditioned_support" for row in source_cells
        ),
        "qualified_source_cell_count": sum(row.get("qualified") is True for row in source_cells),
        "source_cell_count": len(source_cells),
        "all_source_cells_qualified": all_source_qualified,
        "capture_complete": complete,
        "science_qualified": science,
        **classification,
    }


def _gate(
    check: str, category: str, expected: Any, observed: Any, *, terminal_blocking: bool
) -> JsonDict:
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        "terminal_blocking": terminal_blocking,
    }


def build_acceptance_gates(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Separate completion, safety, scientific value, validation, and promotion."""

    reduced = independent_reduce(artifact)
    definitions = (
        ("preconditions_passed", "prerequisite", True, True),
        ("source_cells_complete", "completion", True, True),
        ("negative_control_cells_complete", "completion", True, True),
        ("chain_evidence_complete", "completion", True, True),
        ("raw_trace_evidence_passed", "completion", True, True),
        ("independent_source_energy_controls_passed", "safety", True, True),
        ("wrong_law_negative_controls_passed", "safety", True, True),
        ("all_source_cells_qualified", "scientific_efficacy", True, False),
        ("affected_validation_passed", "required_validation", True, True),
        ("e2e_validation_passed", "required_validation", True, True),
        ("terminal_validation_passed", "required_validation", True, True),
    )
    gates = [
        _gate(name, category, expected, reduced[name], terminal_blocking=blocking)
        for name, category, expected, blocking in definitions
    ]
    gates.append(_gate("automatic_promotion", "promotion", 0, 0, terminal_blocking=False))
    return gates


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every failure and identify the first exact blocking field."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    blocking = [row for row in failures if row.get("terminal_blocking") is True]
    return {
        "passed": not blocking,
        "failed_count": len(failures),
        "blocking_failed_count": len(blocking),
        "first_failure": blocking[0] if blocking else (failures[0] if failures else None),
        "failures": failures,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind code, upstream bytes, settings, traces, rows, and command receipts."""

    bound = {
        key: artifact.get(key)
        for key in (
            "schema",
            "experiment_id",
            "milestone",
            "run_date",
            "random_seed",
            "source_artifact_hashes",
            "sampler_protocol",
            "formula_ids",
            "negative_control_formula_ids",
            "per_chain_results",
            "cell_results",
            "distribution_shift_controls",
            "raw_trace_archive",
            "raw_trace_reduction",
            "sample_size_budget",
            "validation_receipts",
            "verdict_class",
        )
    }
    return canonical_hash(bound)


def field_principles(artifact: Mapping[str, Any]) -> dict[str, str]:
    """Explain required fields while preserving their ordinary JSON types."""

    specific = {
        "schema": "Use a versioned schema with ordinary top-level experiment_id and milestone.",
        "status": "Use a terminal status only after actual work and required validation.",
        "run_date": "Use 20260917 with actual start and end UTC timestamps.",
        "preconditions_checked": "Record exact paths, producer identity, hashes, class, and resources before dependent work.",
        "MODEL_SPECS": "List intended current models; this host-only sampler audit has none.",
        "model_invoked": "Set true for any attempted current model load or generation.",
        "invocation_counts": "Record attempted, completed, failed, cancelled, and in-flight current operations as zero.",
        "inference_substrate": "Describe actual CPU exact enumeration and JAX sampling with historical provenance in sidecars.",
        "inference_substrate_class": "Use the closed CPU exact solver or simulator class.",
        "execution_venue": "Record measured host CPU work and no board execution.",
        "duration_s": "Use measured monotonic duration without invented time or delay padding.",
        "phase_spans": "Record measured read, build, load, generate, evaluate, validate, and write spans.",
        "random_seed": "Freeze formula, chain, and protocol seeds.",
        "reproducibility_checksum": "Bind exact code, settings, source formulas, protocol, raw evidence, and rows.",
        "source_artifact_hashes": "Hash every exact producer and source path used by the audit.",
        "rows": "Retain every chain, cell, control, metric, cost, failure, and censoring disposition.",
        "sample_size_budget": "Predeclare planned, attempted, completed, censored, and remaining units without outcome-driven extension.",
        "acceptance_gate_results": "Separate expected, observed, and passed values for completion, safety, value, and validation.",
        "gate_check_summary": "Name each failed check with its expected and observed value.",
        "verifier_is_oracle": "True because the formal source energy defines the exact target probabilities.",
        "honest_verdict": "Use complete_ for finished work and blocked_ for an unavailable required input.",
        "verdict_class": "Use exactly positive, circular_positive, null, blocked, disqualified, or partial.",
        "flagged_adversarial": "Set true only for a critical independent finding and exclude the producer from readiness.",
        "validation_receipts": "Retain argv, environment, scope, exit code, duration, and log hash for every executed check.",
        "repository_health": "Keep dated unrelated failures separate from required affected checks.",
        "field_principles": "Explain each field without wrapping dictionaries or numeric gates.",
        "promotion_score": "Remain zero because this milestone cannot roll out or publish automatically.",
        "ising_sample_capture_complete_score": "One requires complete chain and cell evidence plus valid required checks, independent of efficacy.",
        "ising_law_value_score": "One requires exact energy controls, hostile controls, and every fixed observable and ESS gate.",
        "per_chain_results": "Retain raw chain identity, budgets, errors, autocorrelation, effective counts, and cost.",
        "distribution_shift_controls": "Retain independent original-versus-appended implied-clause law results.",
        "raw_trace_archive": "Bind ordered chain samples losslessly outside the summary JSON.",
        "source_law_preservation_recommendation": "Recommend no enablement unless every preregistered source-law cell qualifies.",
    }
    return {
        key: specific.get(key, "Retain this supporting evidence in its ordinary JSON type.")
        for key in artifact
        if key != "field_principles"
    } | {"field_principles": specific["field_principles"]}


def _base_artifact(
    *,
    upstream: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    sidecars: Sequence[Mapping[str, Any]],
    chains: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    trace_manifest: Mapping[str, Any],
    raw_reduction: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    repository_health: Mapping[str, Any],
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build and independently classify one ordinary terminal-shaped record."""

    formulas = list(upstream.get("frozen_formulas") or [])
    formula_ids = [str(row["formula_id"]) for row in formulas]
    negative_ids = [formula_ids[index] for index in NEGATIVE_FORMULA_INDICES] if formula_ids else []
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "building_terminal_record",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "current": deepcopy(ZERO_CURRENT_INVOCATIONS),
            "historical": {"counted_as_current": False, "sidecar_count": len(sidecars)},
        },
        "inference_substrate": "host_cpu_jax_parallel_ising_sampling_with_exact_enumeration_no_model",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "host_computation": {
            "node": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor() or "host_cpu",
            "python": platform.python_version(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "current_model_operations": 0,
            "current_board_operations": 0,
        },
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "formula_seeds": deepcopy(
                upstream.get("frozen_sampling_protocol", {}).get("formula_seeds")
            ),
            "chain_seeds": list(CHAIN_SEEDS),
        },
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": [deepcopy(dict(row)) for row in sidecars],
        "upstream_producer": {
            "path": UPSTREAM_PATH.as_posix(),
            "experiment_id": upstream.get("experiment_id"),
            "schema": upstream.get("schema"),
            "verdict_class": upstream.get("verdict_class"),
            "flagged_adversarial": upstream.get("flagged_adversarial"),
            "law_fixture_ready_score": upstream.get("law_fixture_ready_score"),
            "byte_sha256": source_hashes.get(UPSTREAM_PATH.as_posix()),
        },
        "formula_ids": formula_ids,
        "negative_control_formula_ids": negative_ids,
        "sampler_protocol": {
            "sampler": "carnot.samplers.parallel_ising.ParallelIsingSampler",
            "implementation_changed": False,
            "use_checkerboard": True,
            "schedule": None,
            "beta_grid": list(BETA_GRID),
            "conditions": list(FAITHFUL_CONDITIONS),
            "chain_count": CHAIN_COUNT,
            "chain_seeds": list(CHAIN_SEEDS),
            "warmup_steps_per_chain": WARMUP_STEPS,
            "recorded_samples_per_chain": RECORDED_SAMPLES,
            "steps_per_recorded_sample": STEPS_PER_SAMPLE,
            "observable_error_limit": OBSERVABLE_ERROR_LIMIT,
            "effective_sample_minimum_per_cell": ESS_MINIMUM,
            "ess_lag_window": ESS_LAG_WINDOW,
            "support_filtering": False,
            "tuning_after_observation": False,
        },
        "per_chain_results": [deepcopy(dict(row)) for row in chains],
        "cell_results": [deepcopy(dict(row)) for row in cells],
        "distribution_shift_controls": [deepcopy(dict(row)) for row in controls],
        "raw_trace_archive": deepcopy(dict(trace_manifest)),
        "raw_trace_reduction": deepcopy(dict(raw_reduction)),
        "rows": [
            *[deepcopy(dict(row)) for row in chains],
            *[deepcopy(dict(row)) for row in cells],
            *[deepcopy(dict(row)) for row in controls],
        ],
        "sample_size_budget": {
            "planned_formulas": 24,
            "attempted_formulas": len({row.get("formula_id") for row in cells}),
            "planned_source_cells": 24 * len(BETA_GRID) * len(FAITHFUL_CONDITIONS),
            "attempted_source_cells": sum(
                row.get("cell_kind") == "source_faithful" for row in cells
            ),
            "completed_source_cells": sum(
                row.get("cell_kind") == "source_faithful"
                and row.get("outcome") in {"complete", "complete_structural_empty_support"}
                for row in cells
            ),
            "planned_negative_cells": len(NEGATIVE_FORMULA_INDICES) * len(BETA_GRID),
            "attempted_negative_cells": sum(
                row.get("cell_kind") == "wrong_law_negative_control" for row in cells
            ),
            "planned_chains": (
                24 * len(BETA_GRID) * len(FAITHFUL_CONDITIONS)
                + len(NEGATIVE_FORMULA_INDICES) * len(BETA_GRID)
            )
            * CHAIN_COUNT,
            "attempted_chains": len(chains),
            "completed_nonempty_chains": sum(row.get("outcome") == "complete" for row in chains),
            "structural_empty_support_chains": sum(
                row.get("outcome") == "structural_empty_support" for row in chains
            ),
            "planned_recorded_samples_per_nonempty_chain": RECORDED_SAMPLES,
            "censored_samples_due_to_empty_support": sum(
                int(row.get("planned_recorded_samples", 0))
                for row in chains
                if row.get("outcome") == "structural_empty_support"
            ),
            "stopping_rule": "Run the frozen budget once; never extend or tune after observed metrics.",
            "remaining_work": max(
                0,
                (
                    24 * len(BETA_GRID) * len(FAITHFUL_CONDITIONS)
                    + len(NEGATIVE_FORMULA_INDICES) * len(BETA_GRID)
                )
                * CHAIN_COUNT
                - len(chains),
            ),
        },
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "repository_health": deepcopy(dict(repository_health)),
        "prior_attempt_findings": deepcopy(PRIOR_ATTEMPT_FINDINGS),
        "verifier_is_oracle": True,
        "flagged_adversarial": flagged_adversarial,
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
        "rust_changed": False,
        "hardware_speed_claimed": False,
        "bit_equivalent_energy_is_speedup": False,
        "source_law_preservation_recommendation": {
            "recommend_enable_new_sampler": False,
            "recommendation": "pending_independent_reduction",
            "claim_boundary": "CPU empirical source-law preservation only; no hardware or speed claim.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "field_principles": {},
        "independent_reduction": {},
        "ising_sample_capture_complete_score": 0,
        "ising_law_value_score": 0,
        "promotion_score": 0,
        "honest_verdict": "complete_disqualified_building_terminal_record",
        "verdict_class": "disqualified",
    }
    reduction = independent_reduce(artifact)
    artifact["independent_reduction"] = reduction
    for key in (
        "ising_sample_capture_complete_score",
        "ising_law_value_score",
        "promotion_score",
        "honest_verdict",
        "verdict_class",
    ):
        artifact[key] = reduction[key]
    artifact["status"] = {
        "blocked": "blocked_ising_audit_prerequisite",
        "disqualified": "complete_ising_audit_disqualified",
        "null": "complete_ising_audit_null",
        "circular_positive": "complete_ising_audit_source_law_qualified",
    }.get(str(artifact["verdict_class"]), "complete_ising_audit_terminal")
    artifact["source_law_preservation_recommendation"] = {
        "recommend_enable_new_sampler": False,
        "recommendation": (
            "source_law_preservation_supported_within_fixed_observable_and_ess_gates"
            if artifact["ising_law_value_score"] == 1
            else "do_not_enable_new_sampler_not_all_fixed_source_law_cells_qualified"
        ),
        "claim_boundary": "CPU empirical source-law preservation only; no hardware or speed claim.",
    }
    artifact["acceptance_gate_results"] = build_acceptance_gates(artifact)
    artifact["gate_check_summary"] = gate_summary(artifact["acceptance_gate_results"])
    artifact["field_principles"] = field_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _synthetic_rows(
    upstream: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Create bounded complete evidence used only by cold reducer unit tests."""

    chains: list[JsonDict] = []
    cells: list[JsonDict] = []
    controls: list[JsonDict] = []
    formulas = list(upstream["frozen_formulas"])
    for formula in formulas:
        empty = conditioned_sampler_parameters(formula, formula["original_clauses"])[
            "empty_support"
        ]
        for beta in BETA_GRID:
            for condition in FAITHFUL_CONDITIONS:
                cell_id = f"{formula['formula_id']}|{beta}|{condition}"
                for order, seed in enumerate(CHAIN_SEEDS):
                    chains.append(
                        {
                            "row_type": "chain",
                            "cell_id": cell_id,
                            "formula_id": formula["formula_id"],
                            "condition": condition,
                            "beta": beta,
                            "chain_order": order,
                            "seed": seed,
                            "outcome": "structural_empty_support" if empty else "complete",
                            "recorded_samples": 0 if empty else RECORDED_SAMPLES,
                            "planned_recorded_samples": RECORDED_SAMPLES,
                            "state_counts": {} if empty else {"0": RECORDED_SAMPLES},
                            "trace_sha256": None
                            if empty
                            else canonical_hash([0] * RECORDED_SAMPLES),
                            "costs": {"current_model_calls": 0},
                            "failures": (
                                ["empty_conditioned_support_has_no_probability_law"]
                                if empty
                                else []
                            ),
                            "censored": empty,
                        }
                    )
                cells.append(
                    {
                        "row_type": "cell",
                        "cell_kind": "source_faithful",
                        "cell_id": cell_id,
                        "formula_id": formula["formula_id"],
                        "condition": condition,
                        "beta": beta,
                        "outcome": "complete_structural_empty_support" if empty else "complete",
                        "support_status": (
                            "empty_conditioned_support" if empty else "nonempty_conditioned_support"
                        ),
                        "target_energy_parity_max_error": 0.0,
                        "qualified": not empty,
                        "costs": {"current_model_calls": 0},
                        "failures": (
                            ["empty_conditioned_support_has_no_probability_law"] if empty else []
                        ),
                        "censored": False,
                    }
                )
    for index in NEGATIVE_FORMULA_INDICES:
        formula = formulas[index]
        for beta in BETA_GRID:
            cell_id = f"{formula['formula_id']}|{beta}|{NEGATIVE_CONDITION}"
            for order, seed in enumerate(CHAIN_SEEDS):
                chains.append(
                    {
                        "row_type": "chain",
                        "cell_id": cell_id,
                        "formula_id": formula["formula_id"],
                        "condition": NEGATIVE_CONDITION,
                        "beta": beta,
                        "chain_order": order,
                        "seed": seed,
                        "outcome": "complete",
                        "recorded_samples": RECORDED_SAMPLES,
                        "planned_recorded_samples": RECORDED_SAMPLES,
                        "state_counts": {"0": RECORDED_SAMPLES},
                        "trace_sha256": canonical_hash([0] * RECORDED_SAMPLES),
                        "costs": {"current_model_calls": 0},
                        "failures": [],
                        "censored": False,
                    }
                )
            cells.append(
                {
                    "row_type": "cell",
                    "cell_kind": "wrong_law_negative_control",
                    "cell_id": cell_id,
                    "formula_id": formula["formula_id"],
                    "condition": NEGATIVE_CONDITION,
                    "beta": beta,
                    "outcome": "complete",
                    "support_status": "nonempty_conditioned_support",
                    "target_energy_parity_max_error": 0.0,
                    "qualified": True,
                    "costs": {"current_model_calls": 0},
                    "failures": [],
                    "censored": False,
                }
            )
            controls.append(
                {
                    "row_type": "distribution_shift_control",
                    "formula_id": formula["formula_id"],
                    "beta": beta,
                    "wrong_law_detected": True,
                    "sampled_appended_target_qualified": True,
                    "outcome": "complete",
                    "costs": {"current_model_calls": 0},
                    "failures": [],
                    "censored": False,
                }
            )
    return chains, cells, controls


def build_artifact_for_test(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build a complete-null fixture without running production chains."""

    upstream = json.loads((REPO_ROOT / UPSTREAM_PATH).read_text(encoding="utf-8"))
    chains, cells, controls = _synthetic_rows(upstream)
    preconditions = [_precondition("unit_precondition", "unit", "ready", True, True)]
    nonempty = sum(int(row["recorded_samples"]) > 0 for row in chains)
    return _base_artifact(
        upstream=upstream,
        preconditions=preconditions,
        source_hashes={UPSTREAM_PATH.as_posix(): _sha256_file(REPO_ROOT / UPSTREAM_PATH)},
        sidecars=[],
        chains=chains,
        cells=cells,
        controls=controls,
        trace_manifest={
            "path": "/tmp/unit-traces.jsonl.gz",
            "compression": "gzip_json_lines_mtime_zero",
            "record_count": nonempty,
            "byte_count": 1,
            "sha256": "sha256:" + "b" * 64,
        },
        raw_reduction={"passed": True, "record_count": nonempty, "errors": []},
        receipts=receipts,
        phase_spans=[],
        started_at_utc="2026-09-17T00:00:00+00:00",
        completed_at_utc="2026-09-17T00:00:01+00:00",
        duration_s=1.0,
        repository_health={
            "as_of": RUN_DATE,
            "status": "not_assessed_by_unit_fixture",
            "affects_required_checks": False,
            "historical_failures": [],
        },
    )


def build_blocked_artifact_for_test() -> JsonDict:
    """Build the terminal external-absence shape used by validator tests."""

    precondition = _precondition(
        "upstream_artifact_bytes",
        UPSTREAM_PATH.as_posix(),
        "bytes",
        "readable_nonempty_bytes",
        "missing",
    )
    return build_blocked_artifact(
        preconditions=[precondition],
        source_hashes={},
        sidecars=[],
        started_at_utc="2026-09-17T00:00:00+00:00",
        completed_at_utc="2026-09-17T00:00:01+00:00",
        duration_s=1.0,
        phase_spans=[],
    )


def build_blocked_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    sidecars: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Publish external absence with exact failed fields and no dependent work."""

    failures = [deepcopy(dict(row)) for row in preconditions if row.get("passed") is not True]
    first = failures[0] if failures else None
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "blocked_ising_audit_prerequisite",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "current": deepcopy(ZERO_CURRENT_INVOCATIONS),
            "historical": {"counted_as_current": False, "sidecar_count": len(sidecars)},
        },
        "inference_substrate": "host_cpu_precondition_checks_only",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {"chain_seeds": list(CHAIN_SEEDS)},
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": [deepcopy(dict(row)) for row in sidecars],
        "upstream_producer": {"path": UPSTREAM_PATH.as_posix()},
        "formula_ids": [],
        "negative_control_formula_ids": [],
        "sampler_protocol": {
            "beta_grid": list(BETA_GRID),
            "chain_seeds": list(CHAIN_SEEDS),
            "warmup_steps_per_chain": WARMUP_STEPS,
            "recorded_samples_per_chain": RECORDED_SAMPLES,
            "tuning_after_observation": False,
        },
        "per_chain_results": [],
        "cell_results": [],
        "distribution_shift_controls": [],
        "raw_trace_archive": {},
        "raw_trace_reduction": {"passed": False, "error": "blocked_before_sampling"},
        "rows": [],
        "sample_size_budget": {
            "planned_chains": 612,
            "attempted_chains": 0,
            "completed_nonempty_chains": 0,
            "censored_chains": 612,
            "remaining_work": 612,
            "stopping_rule": "Stop before dependent work when a required external input fails.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "passed": False,
            "failed_count": len(failures),
            "blocking_failed_count": len(failures),
            "first_failure": first,
            "failures": failures,
        },
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_ising_law_prerequisite",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {
            "as_of": RUN_DATE,
            "status": "not_assessed_blocked_before_validation",
            "affects_required_checks": False,
            "historical_failures": [],
        },
        "prior_attempt_findings": deepcopy(PRIOR_ATTEMPT_FINDINGS),
        "field_principles": {},
        "independent_reduction": {},
        "promotion_score": 0,
        "ising_sample_capture_complete_score": 0,
        "ising_law_value_score": 0,
        "source_law_preservation_recommendation": {
            "recommend_enable_new_sampler": False,
            "recommendation": "blocked_external_prerequisite",
            "claim_boundary": "No sampling claim was measured.",
        },
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
        "rust_changed": False,
        "hardware_speed_claimed": False,
        "bit_equivalent_energy_is_speedup": False,
    }
    artifact["field_principles"] = field_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, declarations, reduction, scores, and checksum."""

    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("current_model_declaration_invalid")
    if (artifact.get("invocation_counts") or {}).get("current") != ZERO_CURRENT_INVOCATIONS:
        errors.append("current_invocation_counts_invalid")
    if artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("substrate_class_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("verdict_class") not in CLOSED_VERDICTS:
        errors.append("verdict_class_invalid")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_nonzero")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("circularity_declaration_invalid")
    if set(artifact.get("field_principles") or {}) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("verdict_class") == "blocked":
        if (
            artifact.get("rows")
            or artifact.get("per_chain_results")
            or artifact.get("validation_receipts")
        ):
            errors.append("blocked_artifact_has_dependent_work")
        if (artifact.get("gate_check_summary") or {}).get("first_failure") is None:
            errors.append("blocked_gate_summary_missing")
        if any(
            artifact.get(field) != 0
            for field in (
                "ising_sample_capture_complete_score",
                "ising_law_value_score",
                "promotion_score",
            )
        ):
            errors.append("blocked_scores_nonzero")
    else:
        reduced = independent_reduce(artifact)
        if artifact.get("independent_reduction") != reduced:
            errors.append("stored_reduction_mismatch")
        expected = {
            key: reduced[key]
            for key in (
                "verdict_class",
                "honest_verdict",
                "ising_sample_capture_complete_score",
                "ising_law_value_score",
                "promotion_score",
            )
        }
        if any(artifact.get(key) != value for key, value in expected.items()):
            errors.append("score_mismatch")
    if artifact.get("flagged_adversarial") is True and artifact.get("ising_law_value_score") != 0:
        errors.append("adversarial_value_nonzero")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def utc_now() -> str:  # pragma: no cover - runtime timestamp.
    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Emit a flushed monotonic boundary or loop heartbeat."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7378] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, *, operation: str = "executed"
) -> JsonDict:  # pragma: no cover - runtime evidence.
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "operation": operation,
        "checkpoint_at_utc": utc_now(),
    }


def _terminal_commands(
    candidate: Path,
) -> list[validation_contract.PlannedCommand]:  # pragma: no cover
    """Build cold replay, raw reduction, hostile lint, and row checks."""

    python = ".venv/bin/python"
    reducer_code = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7378_v647_ising_audit import validate_artifact,validate_raw_evidence;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_artifact(v);r=validate_raw_evidence(v);"
        "print({'artifact_errors':e,'raw':r},flush=True);"
        "raise SystemExit(bool(e) or not r.get('passed',False))"
    )
    return [
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "cold_artifact_replay",
                (
                    python,
                    "-u",
                    WRAPPER_PATH.as_posix(),
                    "--date",
                    RUN_DATE,
                    "--replay-artifact",
                    str(candidate),
                ),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "independent_raw_reduction",
                (python, "-u", "-c", reducer_code, str(candidate)),
                "measured_candidate_and_raw_trace",
            ),
            "completion",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "measured_candidate",
            ),
            "safety",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
    ]


def run_experiment(  # pragma: no cover - executed through the declared entrypoint.
    repo_root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:
    """Run fixed chains, scoped validation, hostile checks, and atomic write."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    preconditions, hashes, upstream, sidecars = collect_preconditions(root)
    spans.append(_span("read", phase_started, started))
    preconditions_passed = all(
        row["passed"] for row in preconditions if row["terminal_blocking"] is True
    )
    progress(started, "preconditions", "end", passed=preconditions_passed)
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    validation_contract.atomic_json(raw_dir / "historical_model_receipts.json", {"rows": sidecars})
    if not preconditions_passed:
        blocked = build_blocked_artifact(
            preconditions=preconditions,
            source_hashes=hashes,
            sidecars=sidecars,
            started_at_utc=started_at,
            completed_at_utc=utc_now(),
            duration_s=time.monotonic() - started,
            phase_spans=spans,
        )
        progress(started, "write", "before_atomic_blocked", path=output_path)
        validation_contract.atomic_json(root / output_path, blocked)
        progress(started, "write", "after_atomic_blocked", status=blocked["status"])
        return blocked

    phase_started = time.monotonic()
    progress(started, "build", "start")
    formulas = list(upstream["frozen_formulas"])
    spans.append(_span("build", phase_started, started))
    progress(started, "build", "end", formulas=len(formulas))

    phase_started = time.monotonic()
    progress(started, "load", "before", operation="no_model_load_required")
    spans.append(_span("load", phase_started, started, operation="not_applicable_no_model"))
    progress(started, "load", "after", model_invoked=False)
    phase_started = time.monotonic()
    progress(started, "generate", "before", operation="no_generation_required")
    spans.append(_span("generate", phase_started, started, operation="not_applicable_no_model"))
    progress(started, "generate", "after", generation_calls=0)

    phase_started = time.monotonic()
    progress(started, "evaluate", "before_fixed_sampling", planned_chains=612)
    chains, cells, controls, traces = run_sampling_panel(formulas, started=started)
    trace_manifest = atomic_gzip_jsonl(root / TRACE_PATH, traces)
    raw_shell = {"raw_trace_archive": trace_manifest, "per_chain_results": chains}
    raw_reduction = validate_raw_evidence(raw_shell, root)
    spans.append(_span("evaluate", phase_started, started))
    progress(
        started,
        "evaluate",
        "after_fixed_sampling",
        chains=len(chains),
        cells=len(cells),
        traces=len(traces),
        raw_passed=raw_reduction["passed"],
    )

    private = Path(tempfile.mkdtemp(prefix="exp7378-validation-", dir="/tmp"))
    commands = build_validation_plan(root, private)
    plan_errors = validate_validation_plan(root, commands)
    phase_started = time.monotonic()
    progress(started, "validate", "before_required_subprocesses", plan_errors=len(plan_errors))
    validation_rows: list[JsonDict] = []
    if not plan_errors:
        validation_rows = validation_contract.run_categorized_commands(
            root,
            [
                validation_contract.PlannedCommand(command, "required_validation", True)
                for command in commands
            ],
            log_dir=raw_dir / "validation/required",
        )
    progress(
        started,
        "validate",
        "after_required_subprocesses",
        passed=(
            _receipt_set_passes(validation_rows, REQUIRED_CHECK_NAMES)
            and _receipt_set_passes(validation_rows, E2E_CHECK_NAMES)
        ),
    )
    repository_health = {
        "as_of": RUN_DATE,
        "status": "not_assessed_by_scoped_experiment",
        "scope": "affected and applicable E2E checks only; no broad fallback command",
        "affects_required_checks": False,
        "historical_failures": [],
    }
    candidate = _base_artifact(
        upstream=upstream,
        preconditions=preconditions,
        source_hashes=hashes,
        sidecars=sidecars,
        chains=chains,
        cells=cells,
        controls=controls,
        trace_manifest=trace_manifest,
        raw_reduction=raw_reduction,
        receipts=validation_rows,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        repository_health=repository_health,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    validation_contract.atomic_json(candidate_path, candidate)

    progress(started, "validate", "before_terminal_subprocesses")
    terminal_rows = validation_contract.run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_span("validate", phase_started, started))
    terminal_passed = _receipt_set_passes(terminal_rows, TERMINAL_CHECK_NAMES)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal_rows)
    progress(
        started,
        "validate",
        "after_terminal_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )

    phase_started = time.monotonic()
    final = _base_artifact(
        upstream=upstream,
        preconditions=preconditions,
        source_hashes=hashes,
        sidecars=sidecars,
        chains=chains,
        cells=cells,
        controls=controls,
        trace_manifest=trace_manifest,
        raw_reduction=raw_reduction,
        receipts=[*validation_rows, *terminal_rows],
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        repository_health=repository_health,
        flagged_adversarial=critical,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    spans.append(_span("write", phase_started, started, operation="atomic_terminal_publication"))
    final["phase_spans"] = spans
    final["duration_s"] = time.monotonic() - started
    final["completed_at_utc"] = utc_now()
    final["field_principles"] = field_principles(final)
    final["reproducibility_checksum"] = reproducibility_checksum(final)
    progress(started, "write", "before_atomic_terminal", path=output_path)
    validation_contract.atomic_json(candidate_path, final)
    validation_contract.atomic_json(root / output_path, final)
    progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the thin public experiment entrypoint arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--replay-artifact", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the audit or cold-validate one measured candidate."""

    args = parse_args(argv)
    if args.replay_artifact is not None:
        artifact = json.loads(args.replay_artifact.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        raw = validate_raw_evidence(artifact, REPO_ROOT)
        print(json.dumps({"artifact_errors": errors, "raw_trace_reduction": raw}), flush=True)
        return int(bool(errors) or not raw.get("passed", False))
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
