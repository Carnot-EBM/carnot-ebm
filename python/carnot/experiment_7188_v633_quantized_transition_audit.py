"""Audit fixed-width energy feedback on exact fixed-cardinality Ising slices.

The experiment separates two effects that are easy to mix together. A naive
quantized Metropolis rule samples a quantized target. Two-stage delayed
acceptance uses the cheap target only as a screen, then restores the original
full-precision target with a derived residual acceptance law.

Spec refs: REQ-SAMPLER-7188 and SCENARIO-SAMPLER-7188-*.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from carnot import experiment_7187_v633_slice_sampler as slices


JsonDict = dict[str, Any]
State = tuple[int, ...]

RUN_DATE = "20260910"
TASK_ID = "exp7188-quantized-transition-audit"
MILESTONE = "2026.09.633"
RESULT_PATH = Path("results/experiment_7188_v633_quantized_transition_audit.json")
CHECKPOINT_PATH = Path(
    "results/checkpoints/experiment_7188_v633_quantized_transition_audit_running.json"
)
SPEC_PATH = Path("openspec/capabilities/samplers/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
UPSTREAM_PATH = Path("results/experiment_7187_v633_slice_sampler.json")
TOLERANCE = 1.0e-10

QUANTIZER_BITS = (4, 8, 16)
FULL_ARM = "full_precision_pair_swap_mh"
NAIVE_ARM = "naive_quantized_energy_mh"
CORRECTED_ARM = "two_stage_delayed_acceptance"
PERTURBED_ARM = "matched_random_perturbation_mh"
ARMS = (FULL_ARM, NAIVE_ARM, CORRECTED_ARM, PERTURBED_ARM)

SMALL_GRAPH_SEEDS = slices.SMALL_GRAPH_SEEDS
TRAJECTORY_GRAPH_SEED = 718701
TRAJECTORY_K = 2
TRAJECTORY_BETA = 2.0
TRAJECTORY_SEEDS = tuple(range(718800, 718810))
TRAJECTORY_PROPOSALS = 4_000
TRAJECTORY_BURN_IN = 500

EXPECTED_TASK_CONTRACT = {
    "id": TASK_ID,
    "milestone": MILESTONE,
    "deliverable": str(RESULT_PATH),
    "gated_on": [
        {
            "upstream": "exp7187-slice-sampler",
            "artifact_field": "slice_sampler_ready_score",
            "op": "==",
            "value": 1,
            "principle": "Require complete valid inputs, not a positive scientific outcome.",
        }
    ],
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    ROADMAP_PATH,
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    UPSTREAM_PATH,
    Path("python/carnot/analysis/pbit_sampler_portability.py"),
    Path("python/carnot/experiment_7133_v626_multiscale_sampler_prototype.py"),
    Path("python/carnot/experiment_7187_v633_slice_sampler.py"),
    Path("python/carnot/experiment_7188_v633_quantized_transition_audit.py"),
    SPEC_PATH,
    Path("openspec/capabilities/ising-backend/spec.md"),
    Path("scripts/experiments/experiment_7188_v633_quantized_transition_audit.py"),
    Path("tests/python/test_experiment_7188_v633_quantized_transition_audit.py"),
)
HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")

FIELD_PRINCIPLES = {
    "field_principles": "Echo each field reason so the artifact explains its evidence contract.",
    "status": "Use a terminal state only after the task work is complete or externally blocked.",
    "preconditions_checked": (
        "Name each resource and record its actual availability before measurement."
    ),
    "run_date": "Use 20260910; never copy a historical run date.",
    "inference_substrate": "Describe the computation actually executed, not merely planned.",
    "execution_venue": "Host or device identity limits where the evidence applies.",
    "duration_s": "Measure elapsed work with a monotonic clock; never pad or invent runtime.",
    "source_artifact_hashes": "Hashes bind inputs, code, and frozen contracts to the result.",
    "rows": "Emit one row per unit and arm or condition, including errors and abstentions.",
    "random_seed": "Freeze randomness so another process can reconstruct the study.",
    "reproducibility_checksum": (
        "Hash input contracts, code, seeds, and raw rows to expose drift."
    ),
    "gate_check_summary": (
        "Every blocked verdict names the exact failed check, upstream, field, expected value, "
        "and observed value."
    ),
    "verifier_is_oracle": (
        "Declare whether the scored verifier uses the same authority that labels the outcome."
    ),
    "verdict_class": (
        "Use positive | circular_positive | null | blocked | disqualified | partial. "
        "Only unfinished own work is partial."
    ),
    "honest_verdict": (
        "A terminal description distinguishes useful evidence, null findings, "
        "disqualification, and external blocks."
    ),
    "inference_substrate_class": (
        "Use cpu_exact_solver_or_simulator when the declared work runs; use blocked_no_run "
        "only before any qualifying work."
    ),
    "quantized_audit_complete_score": (
        "One means complete comparison, including biased and null arms."
    ),
    "corrected_kernel_ready_score": ("Only the full-target law licenses future implementation."),
    "quantizer_contract": "Rounding and saturation determine the actual distribution.",
    "law_comparison_rows": "Exact full-target errors separate bias from sampling variance.",
    "cost_rows": "Correction overhead may erase any computational benefit.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_SCHEMA_FIELDS = {
    *REQUIRED_ARTIFACT_FIELDS,
    "quantizer_rows",
    "trajectory_rows",
    "hardware_execution_claimed",
    "soft_spin_execution_claimed",
    "tsu_execution_claimed",
    "paper_replication_claimed",
    "fixed_width_accumulator_claimed",
    "useful_acceleration_claimed",
    "delayed_acceptance_derivation",
    "arithmetic_assumptions",
}

DELAYED_ACCEPTANCE_DERIVATION = (
    "Let r_q=exp(-beta*DeltaE_q) and r_c=exp(-beta*(DeltaE-DeltaE_q)). "
    "The forward acceptance is min(1,r_q)*min(1,r_c). For positive a, "
    "min(1,a)/min(1,1/a)=a. Dividing the forward product by its reverse "
    "therefore gives r_q*r_c=exp(-beta*DeltaE). The pair-swap proposal is "
    "symmetric, so detailed balance holds for the full-precision target."
)


@dataclass(frozen=True)
class QuantizedValues:
    """Keep integer codes and decoded values under one explicit scale."""

    bits: int
    qmin: int
    qmax: int
    scale: float
    codes: tuple[int, ...]
    dequantized: tuple[float, ...]
    saturation_count: int


@dataclass(frozen=True)
class QuantizedInstance:
    """Store one fixed-width graph without implying a hardware accumulator."""

    n: int
    bits: int
    qmin: int
    qmax: int
    scale: float
    edge_codes: tuple[int, ...]
    field_codes: tuple[int, ...]
    edge_sites: tuple[tuple[int, int], ...]
    saturation_count: int


def _progress(phase: int, boundary: str, operation: str) -> None:
    """Flush phase boundaries so the conductor can distinguish work from a stall."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def canonical_json(value: Any) -> str:
    """Encode stable JSON and reject nonfinite evidence before publication."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except ValueError as exc:
        raise ValueError("nonfinite value in canonical JSON") from exc


def sha256_json(value: Any) -> str:
    """Return a tagged digest of one canonical JSON value."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash required bytes without treating a missing file as an empty file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind all artifact fields except the digest that stores this calculation."""

    material = dict(payload)
    material["reproducibility_checksum"] = ""
    return sha256_json(material)


def quantize_values(values: Sequence[float], *, bits: int, scale: float) -> QuantizedValues:
    """Round to nearest-even, then saturate to a signed two's-complement range."""

    if bits not in QUANTIZER_BITS:
        raise ValueError(f"bits must be one of {QUANTIZER_BITS}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    rounded = np.rint(array / scale)
    saturation_count = int(np.count_nonzero((rounded < qmin) | (rounded > qmax)))
    codes_array = np.clip(rounded, qmin, qmax).astype(np.int64)
    codes = tuple(int(value) for value in codes_array)
    decoded = tuple(float(value * scale) for value in codes)
    return QuantizedValues(bits, qmin, qmax, scale, codes, decoded, saturation_count)


def quantize_instance(instance: slices.SliceInstance, bits: int) -> QuantizedInstance:
    """Quantize every coupling and field with one per-instance precision scale."""

    slices.validate_instance(instance)
    coefficients = tuple(edge[2] for edge in instance.edges) + tuple(instance.fields)
    qmax = (2 ** (bits - 1)) - 1 if bits in QUANTIZER_BITS else 0
    scale = max(abs(value) for value in coefficients) / qmax if qmax else 0.0
    encoded = quantize_values(coefficients, bits=bits, scale=scale)
    edge_count = len(instance.edges)
    return QuantizedInstance(
        n=instance.n,
        bits=bits,
        qmin=encoded.qmin,
        qmax=encoded.qmax,
        scale=encoded.scale,
        edge_codes=encoded.codes[:edge_count],
        field_codes=encoded.codes[edge_count:],
        edge_sites=tuple((left, right) for left, right, _ in instance.edges),
        saturation_count=encoded.saturation_count,
    )


def _validated_state(n: int, state: Sequence[int]) -> State:
    """Reject a malformed state before either energy authority reads it."""

    if len(state) != n or any(spin not in (-1, 1) for spin in state):
        raise ValueError("state must contain exactly n signed spins")
    return tuple(state)


def full_precision_authority_energy(instance: slices.SliceInstance, state: Sequence[int]) -> float:
    """Evaluate full energy in a scalar path that never calls quantized energy."""

    spins = _validated_state(instance.n, state)
    favorable = 0.0
    for left, right, coupling in instance.edges:
        favorable += coupling * spins[left] * spins[right]
    for index, field in enumerate(instance.fields):
        favorable += field * spins[index]
    return float(-favorable)


def quantized_energy(instance: QuantizedInstance, state: Sequence[int]) -> float:
    """Accumulate integer terms exactly, then apply the shared scale once."""

    spins = _validated_state(instance.n, state)
    accumulator = 0
    for (left, right), code in zip(instance.edge_sites, instance.edge_codes, strict=True):
        accumulator += code * spins[left] * spins[right]
    for index, code in enumerate(instance.field_codes):
        accumulator += code * spins[index]
    return float(-accumulator * instance.scale)


def distribution_from_energies(energies: Sequence[float], beta: float) -> tuple[float, ...]:
    """Normalize one finite Boltzmann law with a stable log-weight shift."""

    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    values = np.asarray(energies, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("energies must be a nonempty finite vector")
    log_weights = -beta * values
    weights = np.exp(log_weights - float(np.max(log_weights)))
    probabilities = weights / float(np.sum(weights))
    return tuple(float(value) for value in probabilities)


def delayed_acceptance_log_terms(
    beta: float, delta_full: float, delta_quantized: float
) -> tuple[float, float, float]:
    """Return both derived acceptance stages and their log-product."""

    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    if not math.isfinite(delta_full) or not math.isfinite(delta_quantized):
        raise ValueError("energy differences must be finite")
    stage_one = min(0.0, -beta * delta_quantized)
    stage_two = min(0.0, -beta * (delta_full - delta_quantized))
    return stage_one, stage_two, stage_one + stage_two


def _neighbor_indices(states: Sequence[State]) -> tuple[tuple[int, ...], ...]:
    """Enumerate every positive-negative swap in the supplied stable state order."""

    state_index = {state: index for index, state in enumerate(states)}
    neighbors: list[tuple[int, ...]] = []
    for state in states:
        positive = [index for index, spin in enumerate(state) if spin == 1]
        negative = [index for index, spin in enumerate(state) if spin == -1]
        adjacent: list[int] = []
        for left in positive:
            for right in negative:
                candidate = list(state)
                candidate[left], candidate[right] = candidate[right], candidate[left]
                adjacent.append(state_index[tuple(candidate)])
        neighbors.append(tuple(adjacent))
    return tuple(neighbors)


def build_transition_matrix(
    states: Sequence[State],
    beta: float,
    full_energies: Sequence[float],
    approximate_energies: Sequence[float],
    arm: str,
) -> np.ndarray:
    """Assemble an explicit pair-swap matrix from the selected acceptance law."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if len(states) != len(full_energies) or len(states) != len(approximate_energies):
        raise ValueError("state and energy vectors must have the same length")
    distribution_from_energies(full_energies, beta)
    distribution_from_energies(approximate_energies, beta)
    full = np.asarray(full_energies, dtype=np.float64)
    approximate = np.asarray(approximate_energies, dtype=np.float64)
    matrix = np.zeros((len(states), len(states)), dtype=np.float64)
    for source, adjacent in enumerate(_neighbor_indices(states)):
        if not adjacent:
            matrix[source, source] = 1.0
            continue
        proposal = 1.0 / len(adjacent)
        for target in adjacent:
            delta_full = float(full[target] - full[source])
            delta_approximate = float(approximate[target] - approximate[source])
            if arm == FULL_ARM:
                log_acceptance = min(0.0, -beta * delta_full)
            elif arm in {NAIVE_ARM, PERTURBED_ARM}:
                log_acceptance = min(0.0, -beta * delta_approximate)
            else:
                log_acceptance = delayed_acceptance_log_terms(beta, delta_full, delta_approximate)[
                    2
                ]
            matrix[source, target] += proposal * math.exp(log_acceptance)
        matrix[source, source] = 1.0 - float(np.sum(matrix[source]))
    return matrix


def transition_diagnostics(matrix: np.ndarray, target_probabilities: Sequence[float]) -> JsonDict:
    """Measure matrix support, reversibility, and stationarity for one target."""

    target = np.asarray(target_probabilities, dtype=np.float64)
    if matrix.shape != (len(target), len(target)):
        raise ValueError("transition shape must match target probabilities")
    flow = target[:, np.newaxis] * matrix
    return {
        "transition_normalization_error_max": float(np.max(np.abs(np.sum(matrix, axis=1) - 1.0))),
        "transition_support_min": float(np.min(matrix)),
        "detailed_balance_error_max": float(np.max(np.abs(flow - flow.T))),
        "stationary_residual_max": float(np.max(np.abs(target @ matrix - target))),
    }


def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    """Return half the L1 distance between two finite probability laws."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("probability vectors must have the same shape")
    return float(0.5 * np.sum(np.abs(left_array - right_array)))


def state_moments(
    states: Sequence[State], probabilities: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute all spin means and unique pair moments for a finite law."""

    state_array = np.asarray(states, dtype=np.float64)
    weights = np.asarray(probabilities, dtype=np.float64)
    if state_array.ndim != 2 or weights.shape != (len(state_array),):
        raise ValueError("probabilities must match the state rows")
    first = weights @ state_array
    second = tuple(
        float(np.sum(weights * state_array[:, left] * state_array[:, right]))
        for left in range(state_array.shape[1])
        for right in range(left + 1, state_array.shape[1])
    )
    return tuple(float(value) for value in first), second


def moment_bias(
    states: Sequence[State], reference: Sequence[float], observed: Sequence[float]
) -> JsonDict:
    """Report maximum first- and second-order errors without pooling their meaning."""

    reference_first, reference_second = state_moments(states, reference)
    observed_first, observed_second = state_moments(states, observed)
    first_error = np.abs(np.asarray(reference_first) - np.asarray(observed_first))
    second_error = np.abs(np.asarray(reference_second) - np.asarray(observed_second))
    return {
        "first_moment_bias_max": float(np.max(first_error, initial=0.0)),
        "second_moment_bias_max": float(np.max(second_error, initial=0.0)),
    }


def energy_order_metrics(
    full_energies: Sequence[float], approximate_energies: Sequence[float]
) -> JsonDict:
    """Count strict order reversals and approximation-created energy ties."""

    if len(full_energies) != len(approximate_energies):
        raise ValueError("energy vectors must have the same length")
    inversions = 0
    comparable = 0
    ties = 0
    for left in range(len(full_energies)):
        for right in range(left + 1, len(full_energies)):
            full_delta = full_energies[right] - full_energies[left]
            if full_delta == 0.0:
                continue
            comparable += 1
            approximate_delta = approximate_energies[right] - approximate_energies[left]
            if approximate_delta == 0.0:
                ties += 1
            elif full_delta * approximate_delta < 0.0:
                inversions += 1
    return {
        "energy_order_inversion_count": inversions,
        "energy_order_comparable_pair_count": comparable,
        "energy_order_inversion_rate": inversions / comparable if comparable else 0.0,
        "quantized_tie_count": ties,
    }


def _matched_error_vector(errors: np.ndarray, *, seed: int) -> np.ndarray:
    """Draw a random direction whose L2 norm exactly matches measured error."""

    target_norm = float(np.linalg.norm(errors))
    if target_norm == 0.0:
        return np.zeros_like(errors, dtype=np.float64)
    rng = random.Random(seed)
    direction = np.asarray([rng.gauss(0.0, 1.0) for _ in errors], dtype=np.float64)
    return direction * (target_norm / float(np.linalg.norm(direction)))


def _derived_seed(*parts: object) -> int:
    """Domain-separate replay streams without reading a current chain state."""

    material = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def matched_perturbation(
    instance: slices.SliceInstance, quantized: QuantizedInstance, *, seed: int
) -> slices.SliceInstance:
    """Match coupling and field quantization norms with random coefficient errors."""

    edge_values = np.asarray([edge[2] for edge in instance.edges], dtype=np.float64)
    field_values = np.asarray(instance.fields, dtype=np.float64)
    edge_errors = np.asarray(quantized.edge_codes, dtype=np.float64) * quantized.scale - edge_values
    field_errors = (
        np.asarray(quantized.field_codes, dtype=np.float64) * quantized.scale - field_values
    )
    edge_noise = _matched_error_vector(
        edge_errors, seed=_derived_seed(seed, instance.instance_hash, quantized.bits, "edges")
    )
    field_noise = _matched_error_vector(
        field_errors, seed=_derived_seed(seed, instance.instance_hash, quantized.bits, "fields")
    )
    changed = slices.SliceInstance(
        n=instance.n,
        seed=instance.seed,
        edges=tuple(
            (left, right, float(coupling + edge_noise[index]))
            for index, (left, right, coupling) in enumerate(instance.edges)
        ),
        fields=tuple(float(value) for value in field_values + field_noise),
    )
    slices.validate_instance(changed)
    return changed


def _accept(log_threshold: float, rng: random.Random) -> bool:
    """Compare in log space so a large unfavorable energy cannot overflow."""

    return math.log(max(rng.random(), sys.float_info.min)) < log_threshold


def run_trajectory(
    *,
    states: Sequence[State],
    beta: float,
    full_energies: Sequence[float],
    approximate_energies: Sequence[float],
    full_target: Sequence[float],
    own_target: Sequence[float],
    arm: str,
    seed: int,
    proposals: int,
    burn_in: int,
) -> JsonDict:
    """Run one bounded chain and keep Monte Carlo error separate from law bias."""

    if proposals <= 0 or burn_in < 0 or burn_in >= proposals:
        raise ValueError("burn-in must be nonnegative and shorter than proposals")
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if not (
        len(states)
        == len(full_energies)
        == len(approximate_energies)
        == len(full_target)
        == len(own_target)
    ):
        raise ValueError("trajectory vectors must have matching lengths")
    rng = random.Random(seed)
    adjacent = _neighbor_indices(states)
    current = rng.randrange(len(states))
    counts = np.zeros(len(states), dtype=np.int64)
    trace: list[int] = []
    accepted = 0
    stage_one_accepted = 0
    full_energy_calls = 1 if arm in {FULL_ARM, CORRECTED_ARM} else 0
    started = time.monotonic()
    for step in range(proposals):
        candidate = rng.choice(adjacent[current])
        delta_full = full_energies[candidate] - full_energies[current]
        delta_approximate = approximate_energies[candidate] - approximate_energies[current]
        move = False
        if arm == FULL_ARM:
            full_energy_calls += 1
            move = _accept(min(0.0, -beta * delta_full), rng)
        elif arm in {NAIVE_ARM, PERTURBED_ARM}:
            move = _accept(min(0.0, -beta * delta_approximate), rng)
        else:
            stage_one, stage_two, _ = delayed_acceptance_log_terms(
                beta, delta_full, delta_approximate
            )
            if _accept(stage_one, rng):
                stage_one_accepted += 1
                full_energy_calls += 1
                move = _accept(stage_two, rng)
        if move:
            current = candidate
            accepted += 1
        if step >= burn_in:
            counts[current] += 1
            trace.append(current)
    elapsed = time.monotonic() - started
    empirical = counts.astype(np.float64) / float(np.sum(counts))
    monte_carlo_moments = moment_bias(states, own_target, empirical)
    total_moments = moment_bias(states, full_target, empirical)
    baseline_calls = proposals + 1
    return {
        "status": "complete",
        "failure": None,
        "proposals": proposals,
        "burn_in": burn_in,
        "sample_count": len(trace),
        "accepted_moves": accepted,
        "empirical_acceptance": accepted / proposals,
        "stage_one_acceptance": (stage_one_accepted / proposals if arm == CORRECTED_ARM else None),
        "full_energy_calls": full_energy_calls,
        "full_energy_calls_saved": baseline_calls - full_energy_calls,
        "elapsed_wall_time_s": elapsed,
        "exact_distortion_tv_from_full": total_variation(full_target, own_target),
        "monte_carlo_tv_to_own_target": total_variation(empirical, own_target),
        "empirical_tv_to_full_target": total_variation(empirical, full_target),
        "monte_carlo_first_moment_error_max": monte_carlo_moments["first_moment_bias_max"],
        "monte_carlo_second_moment_error_max": monte_carlo_moments["second_moment_bias_max"],
        "empirical_first_moment_bias_from_full_max": total_moments["first_moment_bias_max"],
        "empirical_second_moment_bias_from_full_max": total_moments["second_moment_bias_max"],
        "trace_sha256": sha256_json(trace),
    }


def _task_contract(root: Path) -> JsonDict | None:
    """Read only the exact roadmap fields that gate this same-milestone task."""

    path = root / ROADMAP_PATH
    if not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or not isinstance(document.get("tasks"), list):
        return None
    task = next(
        (row for row in document["tasks"] if isinstance(row, Mapping) and row.get("id") == TASK_ID),
        None,
    )
    if task is None:
        return None
    return {
        "id": task.get("id"),
        "milestone": task.get("milestone"),
        "deliverable": task.get("deliverable"),
        "gated_on": task.get("gated_on"),
    }


def collect_preconditions(
    root: Path,
    *,
    result_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str]]:
    """Record source bytes, contracts, tools, output paths, hashes, and upstream gate."""

    result = result_path or root / RESULT_PATH
    checkpoint = checkpoint_path or root / CHECKPOINT_PATH
    result.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source_sizes = {
        str(path): (root / path).stat().st_size if (root / path).is_file() else None
        for path in REQUIRED_SOURCE_PATHS
    }
    source_hashes = {
        str(path): sha256_file(root / path)
        for path in REQUIRED_SOURCE_PATHS
        if source_sizes[str(path)] is not None and source_sizes[str(path)] > 0
    }
    spec_file = root / SPEC_PATH
    spec_text = spec_file.read_text(encoding="utf-8") if spec_file.is_file() else ""
    observed_task = _task_contract(root)
    upstream_file = root / UPSTREAM_PATH
    try:
        upstream = json.loads(upstream_file.read_text(encoding="utf-8"))
        upstream_score = upstream.get("slice_sampler_ready_score")
    except (OSError, json.JSONDecodeError, AttributeError):
        upstream_score = None
    checks = [
        {
            "check": "driving_capability_spec",
            "upstream": str(SPEC_PATH),
            "field": "REQ-SAMPLER-7188",
            "expected_value": {"exists": True, "req_present": True},
            "observed_value": {
                "exists": spec_file.is_file(),
                "req_present": "REQ-SAMPLER-7188" in spec_text,
            },
            "passed": spec_file.is_file() and "REQ-SAMPLER-7188" in spec_text,
        },
        {
            "check": "required_source_bytes",
            "upstream": "repository",
            "field": "byte_count",
            "expected_value": "every required source is nonempty",
            "observed_value": source_sizes,
            "passed": all(size is not None and size > 0 for size in source_sizes.values()),
        },
        {
            "check": "same_milestone_gate_fields",
            "upstream": str(ROADMAP_PATH),
            "field": "task_contract",
            "expected_value": EXPECTED_TASK_CONTRACT,
            "observed_value": observed_task,
            "passed": observed_task == EXPECTED_TASK_CONTRACT,
        },
        {
            "check": "required_tools",
            "upstream": "host_python_environment",
            "field": "python_numpy_pyyaml",
            "expected_value": {"python": True, "numpy": True, "pyyaml": True},
            "observed_value": {
                "python": bool(sys.executable),
                "numpy": bool(np.__version__),
                "pyyaml": bool(yaml.__version__),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "pyyaml_version": yaml.__version__,
            },
            "passed": bool(sys.executable and np.__version__ and yaml.__version__),
        },
        {
            "check": "output_directories",
            "upstream": "filesystem",
            "field": "result_and_checkpoint_parent",
            "expected_value": {"result_parent": True, "checkpoint_parent": True},
            "observed_value": {
                "result_parent": result.parent.is_dir(),
                "checkpoint_parent": checkpoint.parent.is_dir(),
            },
            "passed": result.parent.is_dir() and checkpoint.parent.is_dir(),
        },
        {
            "check": "source_artifact_hashes",
            "upstream": "required_source_bytes",
            "field": "sha256",
            "expected_value": len(REQUIRED_SOURCE_PATHS),
            "observed_value": len(source_hashes),
            "passed": len(source_hashes) == len(REQUIRED_SOURCE_PATHS)
            and all(HASH_PATTERN.fullmatch(value) for value in source_hashes.values()),
        },
        {
            "check": "upstream_slice_sampler_ready",
            "upstream": str(UPSTREAM_PATH),
            "field": "slice_sampler_ready_score",
            "expected_value": 1,
            "observed_value": upstream_score,
            "passed": upstream_score == 1,
        },
    ]
    return checks, source_hashes


def _quantizer_contract() -> JsonDict:
    """Return the frozen arithmetic rules in machine-readable form."""

    return {
        "precisions_bits": list(QUANTIZER_BITS),
        "signed_code_range": "[-2^(bits-1), 2^(bits-1)-1]",
        "shared_scale_scope": "all stored couplings and fields in one instance and precision",
        "scale_rule": "max_abs_coefficient / (2^(bits-1)-1)",
        "rounding_rule": "IEEE-754 round-to-nearest, ties-to-even via numpy.rint",
        "saturation_rule": "clip rounded code to the signed code range",
        "dequantization_rule": "decoded_coefficient = integer_code * shared_scale",
        "energy_rule": "exact Python integer sum, then one multiplication by shared_scale",
    }


def _base_artifact(
    *, root: Path, run_date: str, checks: list[JsonDict], hashes: dict[str, str]
) -> JsonDict:
    """Create every field before either blocked or measured evidence is added."""

    return {
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "running",
        "preconditions_checked": checks,
        "run_date": run_date,
        "inference_substrate": "not_started",
        "execution_venue": "host",
        "host_identity": platform.node() or "unknown-host",
        "duration_s": 0.0,
        "source_artifact_hashes": hashes,
        "rows": [],
        "random_seed": {
            "small_graph_seeds": list(SMALL_GRAPH_SEEDS),
            "trajectory_seeds": list(TRAJECTORY_SEEDS),
            "stream_derivation": "sha256(seed, instance_hash, precision_bits, arm)",
        },
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "running",
        "inference_substrate_class": "blocked_no_run",
        "quantized_audit_complete_score": 0,
        "corrected_kernel_ready_score": 0,
        "quantizer_contract": _quantizer_contract(),
        "law_comparison_rows": [],
        "cost_rows": [],
        "quantizer_rows": [],
        "trajectory_rows": [],
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "root": str(root.resolve()),
        "spec_refs": ["REQ-SAMPLER-7188", "SCENARIO-SAMPLER-7188-*"],
        "delayed_acceptance_derivation": DELAYED_ACCEPTANCE_DERIVATION,
        "arithmetic_assumptions": [
            "Python integers provide an exact accumulator for the bounded small fixtures.",
            "Decoded energy uses binary64 only after the integer sum is complete.",
            "No fixed accumulator width, overflow rule, or device timing is assumed.",
        ],
        "full_energy_call_contract": (
            "Charge one initial full-table access and one candidate access per full evaluation; "
            "the corrected arm accesses a candidate only after stage one accepts."
        ),
        "trajectory_contract": {
            "n": 8,
            "k": TRAJECTORY_K,
            "beta": TRAJECTORY_BETA,
            "graph_seed": TRAJECTORY_GRAPH_SEED,
            "proposals": TRAJECTORY_PROPOSALS,
            "burn_in": TRAJECTORY_BURN_IN,
            "seeds": list(TRAJECTORY_SEEDS),
        },
        "hardware_execution_claimed": False,
        "soft_spin_execution_claimed": False,
        "tsu_execution_claimed": False,
        "paper_replication_claimed": False,
        "fixed_width_accumulator_claimed": False,
        "useful_acceleration_claimed": False,
        "quantization_defines_different_target": False,
    }


def _coefficient_metrics(instance: slices.SliceInstance, quantized: QuantizedInstance) -> JsonDict:
    """Measure coupling and field errors without using state probabilities."""

    original_edges = np.asarray([edge[2] for edge in instance.edges], dtype=np.float64)
    original_fields = np.asarray(instance.fields, dtype=np.float64)
    edge_errors = np.asarray(quantized.edge_codes) * quantized.scale - original_edges
    field_errors = np.asarray(quantized.field_codes) * quantized.scale - original_fields
    combined = np.concatenate((edge_errors, field_errors))
    return {
        "coupling_error_l2": float(np.linalg.norm(edge_errors)),
        "field_error_l2": float(np.linalg.norm(field_errors)),
        "coefficient_error_max": float(np.max(np.abs(combined))),
        "coefficient_error_rms": float(np.sqrt(np.mean(combined**2))),
    }


def _expected_stage_one_acceptance(
    states: Sequence[State],
    beta: float,
    full_target: Sequence[float],
    approximate_energies: Sequence[float],
) -> float:
    """Average the quantized screen acceptance under the full target."""

    total = 0.0
    for source, adjacent in enumerate(_neighbor_indices(states)):
        proposal = 1.0 / len(adjacent)
        for target in adjacent:
            delta = approximate_energies[target] - approximate_energies[source]
            total += full_target[source] * proposal * math.exp(min(0.0, -beta * delta))
    return float(total)


def _condition_id(n: int, k: int, beta: float, graph_seed: int) -> str:
    return f"n{n}:k{k}:beta{beta:g}:graph{graph_seed}"


def _enumerate_exact_rows() -> tuple[list[JsonDict], list[JsonDict]]:
    """Build every quantizer row and every independently assembled finite law row."""

    quantizer_rows: list[JsonDict] = []
    law_rows: list[JsonDict] = []
    instances = {
        (n, graph_seed): slices.make_frustrated_instance(n, graph_seed)
        for n in (8, 12)
        for graph_seed in SMALL_GRAPH_SEEDS
    }
    quantized_instances: dict[tuple[int, int, int], QuantizedInstance] = {}
    perturbed_instances: dict[tuple[int, int, int], slices.SliceInstance] = {}
    for (n, graph_seed), instance in instances.items():
        for bits in QUANTIZER_BITS:
            quantized = quantize_instance(instance, bits)
            perturbed = matched_perturbation(instance, quantized, seed=7188)
            quantized_instances[(n, graph_seed, bits)] = quantized
            perturbed_instances[(n, graph_seed, bits)] = perturbed
            quantizer_rows.append(
                {
                    "row_type": "quantizer",
                    "unit_id": f"n{n}:graph{graph_seed}:bits{bits}",
                    "n": n,
                    "graph_seed": graph_seed,
                    "instance_hash": instance.instance_hash,
                    "precision_bits": bits,
                    "qmin": quantized.qmin,
                    "qmax": quantized.qmax,
                    "shared_scale": quantized.scale,
                    "coupling_count": len(instance.edges),
                    "field_count": len(instance.fields),
                    "saturation_count": quantized.saturation_count,
                    **_coefficient_metrics(instance, quantized),
                    "passed": True,
                }
            )

    total = 54 * len(QUANTIZER_BITS) * len(ARMS)
    completed = 0
    started = time.monotonic()
    heartbeat = started
    for n in (8, 12):
        for k in (1, 2, n // 2):
            states = slices.enumerate_slice(n, k)
            for beta in (0.5, 2.0, 5.0):
                for graph_seed in SMALL_GRAPH_SEEDS:
                    instance = instances[(n, graph_seed)]
                    full_energies = tuple(
                        full_precision_authority_energy(instance, state) for state in states
                    )
                    full_target = distribution_from_energies(full_energies, beta)
                    condition_id = _condition_id(n, k, beta, graph_seed)
                    for bits in QUANTIZER_BITS:
                        quantized = quantized_instances[(n, graph_seed, bits)]
                        perturbed = perturbed_instances[(n, graph_seed, bits)]
                        quantized_energies = tuple(
                            quantized_energy(quantized, state) for state in states
                        )
                        perturbed_energies = tuple(
                            full_precision_authority_energy(perturbed, state) for state in states
                        )
                        quantized_target = distribution_from_energies(quantized_energies, beta)
                        perturbed_target = distribution_from_energies(perturbed_energies, beta)
                        quantized_tv = total_variation(full_target, quantized_target)
                        for arm in ARMS:
                            if arm == PERTURBED_ARM:
                                approximate = perturbed_energies
                                own_target = perturbed_target
                            elif arm == NAIVE_ARM:
                                approximate = quantized_energies
                                own_target = quantized_target
                            else:
                                approximate = quantized_energies
                                own_target = full_target
                            matrix = build_transition_matrix(
                                states, beta, full_energies, approximate, arm
                            )
                            own_diagnostics = transition_diagnostics(matrix, own_target)
                            full_diagnostics = transition_diagnostics(matrix, full_target)
                            moments = moment_bias(states, full_target, own_target)
                            order = energy_order_metrics(full_energies, approximate)
                            stage_one = (
                                _expected_stage_one_acceptance(
                                    states, beta, full_target, approximate
                                )
                                if arm == CORRECTED_ARM
                                else None
                            )
                            exact_acceptance = float(
                                np.sum(
                                    np.asarray(own_target, dtype=np.float64)
                                    * (1.0 - np.diag(matrix))
                                )
                            )
                            passed = (
                                own_diagnostics["transition_normalization_error_max"] <= TOLERANCE
                                and own_diagnostics["transition_support_min"] >= -TOLERANCE
                                and own_diagnostics["detailed_balance_error_max"] <= TOLERANCE
                                and own_diagnostics["stationary_residual_max"] <= TOLERANCE
                                and (
                                    arm not in {FULL_ARM, CORRECTED_ARM}
                                    or (
                                        full_diagnostics["detailed_balance_error_max"] <= TOLERANCE
                                        and full_diagnostics["stationary_residual_max"] <= TOLERANCE
                                    )
                                )
                            )
                            law_rows.append(
                                {
                                    "row_type": "law_comparison",
                                    "unit_id": f"{condition_id}:bits{bits}:{arm}",
                                    "condition_id": condition_id,
                                    "n": n,
                                    "k": k,
                                    "beta": beta,
                                    "graph_seed": graph_seed,
                                    "instance_hash": instance.instance_hash,
                                    "precision_bits": bits,
                                    "arm": arm,
                                    "state_count": len(states),
                                    "saturation_count": quantized.saturation_count,
                                    "exact_acceptance": exact_acceptance,
                                    "stage_one_acceptance_under_full_target": stage_one,
                                    "expected_full_energy_calls_saved_fraction": (
                                        1.0 - stage_one
                                        if stage_one is not None
                                        else (1.0 if arm in {NAIVE_ARM, PERTURBED_ARM} else 0.0)
                                    ),
                                    "quantized_target_tv_from_full": quantized_tv,
                                    "exact_target_tv_from_full": total_variation(
                                        full_target, own_target
                                    ),
                                    **moments,
                                    **order,
                                    **{
                                        f"own_target_{key}": value
                                        for key, value in own_diagnostics.items()
                                    },
                                    **{
                                        f"full_target_{key}": value
                                        for key, value in full_diagnostics.items()
                                    },
                                    "quantization_error_is_exact": True,
                                    "monte_carlo_error_included": False,
                                    "passed": passed,
                                }
                            )
                            completed += 1
                            now = time.monotonic()
                            if now - heartbeat >= 50.0:  # pragma: no cover - slow host only.
                                print(
                                    f"[heartbeat] elapsed_s={now - started:.3f} "
                                    f"completed={completed}/{total} "
                                    "operation=exact_transition_enumeration",
                                    flush=True,
                                )
                                heartbeat = now
    return quantizer_rows, law_rows


def _trajectory_rows() -> list[JsonDict]:
    """Run the fixed ten-stream anchor for each precision and law."""

    instance = slices.make_frustrated_instance(8, TRAJECTORY_GRAPH_SEED)
    states = slices.enumerate_slice(8, TRAJECTORY_K)
    full_energies = tuple(full_precision_authority_energy(instance, state) for state in states)
    full_target = distribution_from_energies(full_energies, TRAJECTORY_BETA)
    rows: list[JsonDict] = []
    total = len(QUANTIZER_BITS) * len(ARMS) * len(TRAJECTORY_SEEDS)
    completed = 0
    started = time.monotonic()
    heartbeat = started
    for bits in QUANTIZER_BITS:
        quantized = quantize_instance(instance, bits)
        perturbed = matched_perturbation(instance, quantized, seed=7188)
        quantized_energies = tuple(quantized_energy(quantized, state) for state in states)
        perturbed_energies = tuple(
            full_precision_authority_energy(perturbed, state) for state in states
        )
        quantized_target = distribution_from_energies(quantized_energies, TRAJECTORY_BETA)
        perturbed_target = distribution_from_energies(perturbed_energies, TRAJECTORY_BETA)
        for arm in ARMS:
            approximate = perturbed_energies if arm == PERTURBED_ARM else quantized_energies
            if arm == PERTURBED_ARM:
                own_target = perturbed_target
            elif arm == NAIVE_ARM:
                own_target = quantized_target
            else:
                own_target = full_target
            for base_seed in TRAJECTORY_SEEDS:
                stream_seed = _derived_seed(base_seed, instance.instance_hash, bits, arm)
                measured = run_trajectory(
                    states=states,
                    beta=TRAJECTORY_BETA,
                    full_energies=full_energies,
                    approximate_energies=approximate,
                    full_target=full_target,
                    own_target=own_target,
                    arm=arm,
                    seed=stream_seed,
                    proposals=TRAJECTORY_PROPOSALS,
                    burn_in=TRAJECTORY_BURN_IN,
                )
                rows.append(
                    {
                        "row_type": "trajectory",
                        "unit_id": f"bits{bits}:{arm}:seed{base_seed}",
                        "condition_id": _condition_id(
                            8, TRAJECTORY_K, TRAJECTORY_BETA, TRAJECTORY_GRAPH_SEED
                        ),
                        "precision_bits": bits,
                        "arm": arm,
                        "random_seed": base_seed,
                        "stream_seed": stream_seed,
                        "quantized_target_tv_from_full": total_variation(
                            full_target, quantized_target
                        ),
                        "quantization_error_is_exact": True,
                        "monte_carlo_error_is_separate": True,
                        **measured,
                    }
                )
                completed += 1
                now = time.monotonic()
                if now - heartbeat >= 50.0:  # pragma: no cover - slow host only.
                    print(
                        f"[heartbeat] elapsed_s={now - started:.3f} "
                        f"completed={completed}/{total} operation=seeded_trajectories",
                        flush=True,
                    )
                    heartbeat = now
    return rows


def _cost_rows(trajectory_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Aggregate calls and measured wall time without turning bias into a speed win."""

    rows: list[JsonDict] = []
    for bits in QUANTIZER_BITS:
        by_arm = {
            arm: [
                row
                for row in trajectory_rows
                if row["precision_bits"] == bits and row["arm"] == arm
            ]
            for arm in ARMS
        }
        full_calls = sum(row["full_energy_calls"] for row in by_arm[FULL_ARM])
        full_time = sum(row["elapsed_wall_time_s"] for row in by_arm[FULL_ARM])
        for arm in ARMS:
            selected = by_arm[arm]
            calls = sum(row["full_energy_calls"] for row in selected)
            elapsed = sum(row["elapsed_wall_time_s"] for row in selected)
            rows.append(
                {
                    "row_type": "cost",
                    "unit_id": f"bits{bits}:{arm}",
                    "precision_bits": bits,
                    "arm": arm,
                    "trajectory_count": len(selected),
                    "total_proposals": sum(row["proposals"] for row in selected),
                    "full_energy_calls": calls,
                    "full_energy_calls_saved_vs_full": full_calls - calls,
                    "fewer_full_energy_calls_than_full": calls < full_calls,
                    "total_elapsed_wall_time_s": elapsed,
                    "mean_elapsed_wall_time_s": elapsed / len(selected),
                    "lower_wall_time_than_full": elapsed < full_time,
                    "full_target_fidelity": arm in {FULL_ARM, CORRECTED_ARM},
                }
            )
    return rows


def build_artifact(
    *,
    root: Path,
    run_date: str = RUN_DATE,
    preconditions: list[JsonDict] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> JsonDict:
    """Build complete evidence, or stop before measurement on an external gate failure."""

    started = time.monotonic()
    _progress(0, "start", "precondition checks")
    if preconditions is None or source_hashes is None:
        measured_checks, measured_hashes = collect_preconditions(root)
        checks = measured_checks if preconditions is None else preconditions
        hashes = measured_hashes if source_hashes is None else source_hashes
    else:
        checks, hashes = preconditions, source_hashes
    artifact = _base_artifact(root=root, run_date=run_date, checks=checks, hashes=hashes)
    failed = next((row for row in checks if row.get("passed") is not True), None)
    _progress(0, "end", "precondition checks")
    if failed is not None:
        artifact.update(
            {
                "status": "blocked_external_precondition",
                "inference_substrate": "no qualifying computation executed",
                "duration_s": time.monotonic() - started,
                "gate_check_summary": {
                    "passed": False,
                    "failed_check": failed.get("check"),
                    "upstream": failed.get("upstream"),
                    "field": failed.get("field"),
                    "expected_value": failed.get("expected_value"),
                    "observed_value": failed.get("observed_value"),
                },
                "verdict_class": "blocked",
                "honest_verdict": (
                    f"blocked_external_precondition: {failed.get('check')} failed before measurement"
                ),
                "inference_substrate_class": "blocked_no_run",
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact

    artifact["gate_check_summary"] = {
        "passed": True,
        "failed_check": None,
        "upstream": "repository_preflight",
        "field": "all_required_preconditions",
        "expected_value": "all pass",
        "observed_value": "all pass",
    }
    _progress(1, "start", "signed quantizer enumeration")
    quantizer_rows, law_rows = _enumerate_exact_rows()
    artifact["quantizer_rows"] = quantizer_rows
    _progress(1, "end", f"signed quantizer enumeration rows={len(quantizer_rows)}")
    _progress(2, "start", "exact transition law comparison")
    artifact["law_comparison_rows"] = law_rows
    _progress(2, "end", f"exact transition law comparison rows={len(law_rows)}")
    _progress(3, "start", "ten-seed trajectory benchmark")
    trajectory_rows = _trajectory_rows()
    artifact["trajectory_rows"] = trajectory_rows
    _progress(3, "end", f"ten-seed trajectory benchmark rows={len(trajectory_rows)}")
    _progress(4, "start", "full-energy and wall-time cost aggregation")
    cost_rows = _cost_rows(trajectory_rows)
    artifact["cost_rows"] = cost_rows
    _progress(4, "end", f"full-energy and wall-time cost aggregation rows={len(cost_rows)}")
    _progress(5, "start", "terminal evidence assembly")
    artifact["rows"] = quantizer_rows + law_rows + trajectory_rows + cost_rows
    corrected_rows = [row for row in law_rows if row["arm"] == CORRECTED_ARM]
    corrected_costs = [row for row in cost_rows if row["arm"] == CORRECTED_ARM]
    corrected_ready = all(
        row["full_target_stationary_residual_max"] <= TOLERANCE
        and row["full_target_detailed_balance_error_max"] <= TOLERANCE
        and row["passed"] is True
        for row in corrected_rows
    )
    acceleration_eligible = all(
        row["fewer_full_energy_calls_than_full"] and row["lower_wall_time_than_full"]
        for row in corrected_costs
    )
    changed_targets = sum(
        row["exact_target_tv_from_full"] > TOLERANCE for row in law_rows if row["arm"] == NAIVE_ARM
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": (
                "cpu_exact_solver_or_simulator: CPU NumPy exact enumeration of Exp7187 "
                "fixed-cardinality slices plus bounded Python pair-swap trajectories with "
                "integer-coded energies"
            ),
            "inference_substrate_class": "cpu_exact_solver_or_simulator",
            "duration_s": time.monotonic() - started,
            "verdict_class": "positive" if changed_targets else "null",
            "honest_verdict": (
                f"positive: naive quantized energy defines a different target in "
                f"{changed_targets}/162 precision-conditioned exact laws; the derived "
                "delayed-acceptance kernel preserved the full target. This is CPU fidelity "
                "evidence, not a soft-spin, FPGA, TSU, or paper-reproduction result."
                if changed_targets
                else "null: no exact target change was resolved on the frozen precision roster; "
                "the corrected law still preserved the full target."
            ),
            "quantized_audit_complete_score": 1,
            "corrected_kernel_ready_score": int(corrected_ready),
            "useful_acceleration_claimed": bool(acceleration_eligible),
            "quantization_defines_different_target": bool(changed_targets),
            "naive_changed_target_law_count": changed_targets,
            "naive_planned_law_count": 162,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _progress(5, "end", "terminal evidence assembly")
    return artifact


def _expected_quantizer_units() -> set[str]:
    return {
        f"n{n}:graph{seed}:bits{bits}"
        for n in (8, 12)
        for seed in SMALL_GRAPH_SEEDS
        for bits in QUANTIZER_BITS
    }


def _expected_law_units() -> set[str]:
    return {
        f"{_condition_id(n, k, beta, seed)}:bits{bits}:{arm}"
        for n in (8, 12)
        for k in (1, 2, n // 2)
        for beta in (0.5, 2.0, 5.0)
        for seed in SMALL_GRAPH_SEEDS
        for bits in QUANTIZER_BITS
        for arm in ARMS
    }


def _expected_trajectory_units() -> set[str]:
    return {
        f"bits{bits}:{arm}:seed{seed}"
        for bits in QUANTIZER_BITS
        for arm in ARMS
        for seed in TRAJECTORY_SEEDS
    }


def _expected_cost_units() -> set[str]:
    return {f"bits{bits}:{arm}" for bits in QUANTIZER_BITS for arm in ARMS}


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Recompute row completeness, correction fidelity, claims, and checksum."""

    if not REQUIRED_SCHEMA_FIELDS.issubset(payload):
        return ["missing_required_fields"]
    errors: list[str] = []
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_invalid")
    if payload.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_authority_invalid")
    if any(
        payload.get(field) is not False
        for field in (
            "hardware_execution_claimed",
            "soft_spin_execution_claimed",
            "tsu_execution_claimed",
            "paper_replication_claimed",
            "fixed_width_accumulator_claimed",
        )
    ):
        errors.append("claim_boundary_invalid")

    if payload.get("verdict_class") == "blocked":
        gate = payload.get("gate_check_summary", {})
        if (
            payload.get("status") != "blocked_external_precondition"
            or payload.get("inference_substrate_class") != "blocked_no_run"
            or payload.get("quantized_audit_complete_score") != 0
            or payload.get("corrected_kernel_ready_score") != 0
            or any(
                payload.get(field)
                for field in (
                    "rows",
                    "quantizer_rows",
                    "law_comparison_rows",
                    "trajectory_rows",
                    "cost_rows",
                )
            )
            or not isinstance(gate, Mapping)
            or gate.get("passed") is not False
            or not gate.get("failed_check")
            or not gate.get("upstream")
            or not gate.get("field")
        ):
            errors.append("blocked_state_invalid")
        return list(dict.fromkeys(errors))

    quantizer_rows = payload.get("quantizer_rows", [])
    law_rows = payload.get("law_comparison_rows", [])
    trajectory_rows = payload.get("trajectory_rows", [])
    cost_rows = payload.get("cost_rows", [])
    if (
        len(quantizer_rows) != 18
        or {row.get("unit_id") for row in quantizer_rows} != _expected_quantizer_units()
        or any(row.get("passed") is not True for row in quantizer_rows)
    ):
        errors.append("quantizer_rows_incomplete")
    if len(law_rows) != 648 or {row.get("unit_id") for row in law_rows} != _expected_law_units():
        errors.append("law_rows_incomplete")
    elif any(row.get("passed") is not True for row in law_rows):
        errors.append("law_rows_invalid")
    if (
        len(trajectory_rows) != 120
        or {row.get("unit_id") for row in trajectory_rows} != _expected_trajectory_units()
        or any(row.get("status") != "complete" for row in trajectory_rows)
    ):
        errors.append("trajectory_rows_incomplete")
    if (
        len(cost_rows) != 12
        or {row.get("unit_id") for row in cost_rows} != _expected_cost_units()
        or any(row.get("trajectory_count") != 10 for row in cost_rows)
    ):
        errors.append("cost_rows_incomplete")
    if len(payload.get("rows", [])) != 798:
        errors.append("rows_incomplete")

    corrected = [row for row in law_rows if row.get("arm") == CORRECTED_ARM]
    corrected_valid = len(corrected) == 162 and all(
        row.get("full_target_stationary_residual_max", math.inf) <= TOLERANCE
        and row.get("full_target_detailed_balance_error_max", math.inf) <= TOLERANCE
        and row.get("passed") is True
        for row in corrected
    )
    if not corrected_valid:
        errors.append("corrected_law_invalid")
    roster_errors = {
        "quantizer_rows_incomplete",
        "law_rows_incomplete",
        "law_rows_invalid",
        "trajectory_rows_incomplete",
        "cost_rows_incomplete",
        "rows_incomplete",
    }
    complete = not any(error in roster_errors for error in errors)
    if payload.get("quantized_audit_complete_score") != int(complete):
        errors.append("readiness_invalid")
    expected_corrected = complete and corrected_valid
    if payload.get("corrected_kernel_ready_score") != int(expected_corrected):
        errors.append("readiness_invalid")

    corrected_costs = [row for row in cost_rows if row.get("arm") == CORRECTED_ARM]
    acceleration_eligible = len(corrected_costs) == 3 and all(
        row.get("fewer_full_energy_calls_than_full") is True
        and row.get("lower_wall_time_than_full") is True
        and row.get("full_target_fidelity") is True
        for row in corrected_costs
    )
    if payload.get("useful_acceleration_claimed") is True and not acceleration_eligible:
        errors.append("acceleration_claim_invalid")
    if payload.get("status") != "complete" or payload.get("verdict_class") not in {
        "positive",
        "null",
    }:
        errors.append("terminal_verdict_invalid")
    if payload.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("substrate_class_invalid")
    summary = payload.get("gate_check_summary", {})
    if not isinstance(summary, Mapping) or summary.get("passed") is not True:
        errors.append("gate_summary_invalid")
    if payload.get("delayed_acceptance_derivation") != DELAYED_ACCEPTANCE_DERIVATION:
        errors.append("correction_derivation_invalid")
    return list(dict.fromkeys(errors))


def atomic_write(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Publish one complete JSON file through same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - interrupted replacement only.
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "atomic_replace": True,
    }


def run_experiment(*, root: Path, output: Path, run_date: str) -> JsonDict:
    """Build, validate, and publish the requested terminal artifact."""

    artifact = build_artifact(root=root, run_date=run_date)
    _progress(6, "start", "artifact validation")
    errors = validate_artifact(artifact)
    _progress(6, "end", f"artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7188 artifact: {errors}")
    _progress(7, "start", "final atomic write")
    receipt = atomic_write(output, artifact)
    _progress(7, "end", f"final atomic write bytes={receipt['bytes']}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit or validate one caller-selected artifact without rewriting it."""

    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.validate is not None:
        _progress(6, "start", f"validation path={args.validate}")
        try:
            payload = json.loads(args.validate.read_text(encoding="utf-8"))
            errors = validate_artifact(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"validation_error: {exc}", flush=True)
            _progress(6, "end", "validation errors=1")
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        _progress(6, "end", f"validation errors={len(errors)}")
        return 0 if not errors else 2
    output = args.output or root / RESULT_PATH
    try:
        artifact = run_experiment(root=root, output=output, run_date=args.date)
    except (OSError, TypeError, ValueError) as exc:
        print(f"experiment_error: {exc}", flush=True)
        return 2
    print(
        canonical_json(
            {
                "corrected_kernel_ready_score": artifact["corrected_kernel_ready_score"],
                "output": str(output),
                "quantized_audit_complete_score": artifact["quantized_audit_complete_score"],
                "verdict_class": artifact["verdict_class"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the executable entrypoint.
    raise SystemExit(main())
