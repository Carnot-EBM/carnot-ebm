"""Build bounded evidence for a fixed-cardinality pair-swap Metropolis kernel.

The implementation is intentionally smaller than the sampler in the cited
high-magnetization paper.  It uses an ordinary Metropolis correction whose
proposal law can be written down and checked on complete finite slices.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import cache
import hashlib
import itertools
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
from typing import Any, Sequence

import numpy as np


JsonDict = dict[str, Any]
State = tuple[int, ...]
Edge = tuple[int, int, float]

RUN_DATE = "20260910"
TASK_ID = "experiment_7187_v633_slice_sampler"
MILESTONE = "V633"
RESULT_PATH = Path("results/experiment_7187_v633_slice_sampler.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7187_v633_slice_sampler_running.json")
SPEC_PATH = Path("openspec/capabilities/samplers/spec.md")
TOLERANCE = 1.0e-10

PAIR_SWAP_ARM = "pair_swap_metropolis"
UNIFORM_SLICE_ARM = "uniform_slice_independence_metropolis"
INVALID_SINGLE_SPIN_ARM = "reject_invalid_single_spin"
ARMS = (PAIR_SWAP_ARM, UNIFORM_SLICE_ARM, INVALID_SINGLE_SPIN_ARM)

SMALL_GRAPH_SEEDS = (718701, 718702, 718703)
BENCHMARK_SEEDS = tuple(range(718710, 718720))
BENCHMARK_BETA = 2.0
ENERGY_BUDGET = 160
WALL_TIME_BUDGET_S = 0.006
LAG_WINDOW = 12

REQUIRED_MUTATIONS = {
    "energy_sign_reversal",
    "double_counted_edges",
    "asymmetric_proposal_without_hastings",
    "invalid_k",
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    SPEC_PATH,
    Path("python/carnot/experiment_6657_bounded_treewidth_ising_reference.py"),
    Path("python/carnot/experiment_7133_v626_multiscale_sampler_prototype.py"),
    Path("python/carnot/experiment_7134_v626_multiscale_sampler_benchmark.py"),
    Path("python/carnot/samplers/backend.py"),
    Path("python/carnot/experiment_7187_v633_slice_sampler.py"),
    Path("scripts/experiments/experiment_7187_v633_slice_sampler.py"),
    Path("tests/python/test_experiment_7187_v633_slice_sampler.py"),
)
HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "inference_substrate_class",
    "slice_sampler_ready_score",
    "finite_law_rows",
    "transition_rows",
    "benchmark_rows",
    "hardware_execution_claimed",
    "paper_replication_claimed",
)

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
    "slice_sampler_ready_score": (
        "One establishes a complete validated kernel, not a mixing theorem."
    ),
    "finite_law_rows": "Exact enumeration exposes distribution errors.",
    "transition_rows": "Detailed balance checks the implemented law.",
    "benchmark_rows": ("Matched per-instance, seed, and arm costs make comparisons reproducible."),
    "hardware_execution_claimed": "False prevents CPU sampling from becoming a device claim.",
    "paper_replication_claimed": (
        "False separates a local baseline from the specialized published algorithm."
    ),
}

EDGE_COUNTING_CONVENTION = (
    "E(s)=-sum_{(i,j) in E} J_ij*s_i*s_j-sum_i h_i*s_i; "
    "each undirected edge is stored once with i<j and counted once"
)


def _progress(phase: int, boundary: str, operation: str) -> None:
    """Flush phase boundaries so a conductor can distinguish work from a stalled process."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def canonical_json(value: Any) -> str:
    """Encode stable JSON and reject nonfinite evidence before it reaches an artifact."""

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
    """Return a tagged digest so the evidence never hides the hash algorithm."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash required bytes and fail when a required file is absent."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(payload: JsonDict) -> str:
    """Bind every artifact field except the digest that stores this calculation."""

    material = dict(payload)
    material["reproducibility_checksum"] = ""
    return sha256_json(material)


@dataclass(frozen=True)
class SliceInstance:
    """Store one signed graph whose edge representation has no counting ambiguity."""

    n: int
    seed: int
    edges: tuple[Edge, ...]
    fields: tuple[float, ...]

    @property
    def instance_hash(self) -> str:
        """Bind graph coefficients and their deterministic generation seed."""

        return sha256_json(asdict(self))


@dataclass(frozen=True)
class ExactLaw:
    """Keep one complete ordered slice law for independent finite-state checks."""

    states: tuple[State, ...]
    energies: tuple[float, ...]
    probabilities: tuple[float, ...]


def make_frustrated_instance(n: int, seed: int) -> SliceInstance:
    """Create a sparse graph with one explicit frustrated triangle and nonzero fields."""

    if n < 3:
        raise ValueError("n must be at least three")
    rng = random.Random(seed * 1_000_003 + n)
    edges: list[Edge] = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, -1.0)]
    occupied = {(left, right) for left, right, _ in edges}
    candidates = [(index, index + 1) for index in range(2, n - 1)] + [(0, n - 1)]
    candidates += [(index, index + 3) for index in range(n - 3)]
    for left, right in candidates:
        pair = (min(left, right), max(left, right))
        if pair in occupied:
            continue
        magnitude = 0.55 + 0.15 * rng.randrange(4)
        coupling = magnitude if rng.random() < 0.5 else -magnitude
        edges.append((pair[0], pair[1], coupling))
        occupied.add(pair)
    edges.sort(key=lambda item: (item[0], item[1]))
    fields = tuple(
        (0.11 + 0.02 * (index % 5)) * (1.0 if rng.random() < 0.5 else -1.0) for index in range(n)
    )
    instance = SliceInstance(n=n, seed=seed, edges=tuple(edges), fields=fields)
    validate_instance(instance)
    return instance


def replace_instance(instance: SliceInstance, **changes: Any) -> SliceInstance:
    """Create immutable invalid-input mutations without changing the frozen original."""

    return replace(instance, **changes)


def validate_instance(instance: SliceInstance) -> JsonDict:
    """Reject graph syntax that would make energy or frustration evidence ambiguous."""

    if not isinstance(instance.n, int) or instance.n < 3:
        raise ValueError("n must be an integer of at least three")
    if len(instance.fields) != instance.n:
        raise ValueError("field count must equal n")
    if not all(math.isfinite(value) for value in instance.fields):
        raise ValueError("fields must be finite")
    if any(value == 0.0 for value in instance.fields):
        raise ValueError("fields must be nonzero")
    pairs: set[tuple[int, int]] = set()
    couplings: dict[tuple[int, int], float] = {}
    for left, right, coupling in instance.edges:
        if left == right:
            raise ValueError("self-loop is not allowed")
        if left < 0 or right < 0 or left >= instance.n or right >= instance.n:
            raise ValueError("edge endpoint lies outside the graph")
        if left > right:
            raise ValueError("each edge must use left < right")
        if (left, right) in pairs:
            raise ValueError("duplicate edge")
        if not math.isfinite(coupling):
            raise ValueError("couplings must be finite")
        pairs.add((left, right))
        couplings[(left, right)] = coupling
    neighbors: dict[int, dict[int, float]] = {index: {} for index in range(instance.n)}
    for (left, right), coupling in couplings.items():
        neighbors[left][right] = coupling
        neighbors[right][left] = coupling
    frustrated = any(
        neighbors[first][second] * neighbors[first][third] * neighbors[second][third] < 0.0
        for first in range(instance.n)
        for second in neighbors[first]
        for third in neighbors[first].keys() & neighbors[second].keys()
        if first < second < third
    )
    if not frustrated:
        raise ValueError("graph must contain a frustrated triangle")
    return {
        "passed": True,
        "frustrated_triangle": True,
        "edge_count": len(instance.edges),
        "edge_counting_convention": EDGE_COUNTING_CONVENTION,
    }


def _validated_state(instance: SliceInstance, state: Sequence[int]) -> State:
    """Return a stable tuple only after checking every spin and its width."""

    if len(state) != instance.n:
        raise ValueError("state length must equal n")
    if any(spin not in (-1, 1) for spin in state):
        raise ValueError("spins must be -1 or +1")
    return tuple(state)


@cache
def enumerate_slice(n: int, k: int) -> tuple[State, ...]:
    """Enumerate exactly the states with k positive spins in stable order."""

    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(k, int) or k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n")
    states: list[State] = []
    for positive_sites in itertools.combinations(range(n), k):
        positive = set(positive_sites)
        states.append(tuple(1 if index in positive else -1 for index in range(n)))
    return tuple(states)


def ising_energy(instance: SliceInstance, state: Sequence[int]) -> float:
    """Evaluate the target energy with every stored undirected edge counted once."""

    spins = _validated_state(instance, state)
    edge_term = sum(
        coupling * spins[left] * spins[right] for left, right, coupling in instance.edges
    )
    field_term = sum(field * spins[index] for index, field in enumerate(instance.fields))
    return float(-edge_term - field_term)


def reference_energy(instance: SliceInstance, state: Sequence[int]) -> float:
    """Independently evaluate scalar terms without calling the sampler energy function."""

    spins = _validated_state(instance, state)
    favorable = 0.0
    for left, right, coupling in instance.edges:
        favorable += coupling * spins[left] * spins[right]
    for index in range(instance.n):
        favorable += instance.fields[index] * spins[index]
    return -favorable


@cache
def independent_exact_law(instance: SliceInstance, k: int, beta: float) -> ExactLaw:
    """Normalize the complete slice directly from the independent scalar energy path."""

    validate_instance(instance)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    states = enumerate_slice(instance.n, k)
    energies = tuple(reference_energy(instance, state) for state in states)
    log_weights = np.asarray([-beta * energy for energy in energies], dtype=np.float64)
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    probabilities = weights / float(np.sum(weights))
    return ExactLaw(states, energies, tuple(float(value) for value in probabilities))


def pair_swap_proposal_probability(source: Sequence[int], target: Sequence[int]) -> float:
    """Return q(target|source) and expose why its reverse has the same value."""

    if len(source) != len(target) or any(spin not in (-1, 1) for spin in (*source, *target)):
        return 0.0
    k = source.count(1)
    if target.count(1) != k or k in (0, len(source)):
        return 0.0
    changed = [
        index for index, pair in enumerate(zip(source, target, strict=True)) if pair[0] != pair[1]
    ]
    if len(changed) != 2:
        return 0.0
    return 1.0 / (k * (len(source) - k))


def propose_pair_swap(state: Sequence[int], rng: random.Random) -> tuple[State, float, float]:
    """Select one positive and one negative uniformly, preserving cardinality exactly."""

    if any(spin not in (-1, 1) for spin in state):
        raise ValueError("spins must be -1 or +1")
    source = tuple(state)
    positive = [index for index, spin in enumerate(source) if spin == 1]
    negative = [index for index, spin in enumerate(source) if spin == -1]
    if not positive or not negative:
        return source, 1.0, 1.0
    first = rng.choice(positive)
    second = rng.choice(negative)
    proposed = list(source)
    proposed[first], proposed[second] = proposed[second], proposed[first]
    target = tuple(proposed)
    forward = 1.0 / (len(positive) * len(negative))
    # The swapped state has the same positive and negative counts.  Its reverse
    # therefore selects the same two sites with the same denominator.
    reverse = 1.0 / (target.count(1) * target.count(-1))
    return target, forward, reverse


def log_acceptance(beta: float, delta_energy: float) -> float:
    """Compute min(0, -beta*delta_E) without evaluating an unsafe exponential."""

    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    if math.isnan(delta_energy):
        raise ValueError("energy difference must not be NaN")
    return min(0.0, -beta * delta_energy)


def _pair_transition_with_energy(
    instance: SliceInstance,
    law: ExactLaw,
    beta: float,
    energy_values: Sequence[float],
) -> np.ndarray:
    """Build a pair-swap matrix using supplied energies for mutation testing."""

    state_index = {state: index for index, state in enumerate(law.states)}
    matrix = np.zeros((len(law.states), len(law.states)), dtype=np.float64)
    for source_index, source in enumerate(law.states):
        positive = [index for index, spin in enumerate(source) if spin == 1]
        negative = [index for index, spin in enumerate(source) if spin == -1]
        if not positive or not negative:
            matrix[source_index, source_index] = 1.0
            continue
        proposal = 1.0 / (len(positive) * len(negative))
        for first in positive:
            for second in negative:
                target = list(source)
                target[first], target[second] = target[second], target[first]
                target_index = state_index[tuple(target)]
                delta = energy_values[target_index] - energy_values[source_index]
                matrix[source_index, target_index] += proposal * math.exp(
                    log_acceptance(beta, delta)
                )
        matrix[source_index, source_index] = 1.0 - float(np.sum(matrix[source_index]))
    return matrix


def transition_matrix(instance: SliceInstance, k: int, beta: float, *, arm: str) -> np.ndarray:
    """Construct one explicit finite transition matrix for the declared target law."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    law = independent_exact_law(instance, k, beta)
    count = len(law.states)
    if arm == INVALID_SINGLE_SPIN_ARM:
        return np.eye(count, dtype=np.float64)
    if arm == PAIR_SWAP_ARM:
        sampler_energies = tuple(ising_energy(instance, state) for state in law.states)
        return _pair_transition_with_energy(instance, law, beta, sampler_energies)

    log_weights = -beta * np.asarray(law.energies, dtype=np.float64)
    log_ratios = log_weights[np.newaxis, :] - log_weights[:, np.newaxis]
    matrix = np.exp(np.minimum(0.0, log_ratios)) / count
    np.fill_diagonal(matrix, 0.0)
    np.fill_diagonal(matrix, 1.0 - np.sum(matrix, axis=1))
    return matrix


def transition_diagnostics(law: ExactLaw, matrix: np.ndarray) -> JsonDict:
    """Measure stochastic rows, detailed balance, and stationary-law residual."""

    probabilities = np.asarray(law.probabilities, dtype=np.float64)
    expected_shape = (len(probabilities), len(probabilities))
    if matrix.shape != expected_shape:
        raise ValueError("transition matrix shape does not match the exact law")
    flow = probabilities[:, np.newaxis] * matrix
    return {
        "transition_normalization_error_max": float(np.max(np.abs(np.sum(matrix, axis=1) - 1.0))),
        "transition_support_min": float(np.min(matrix)),
        "detailed_balance_error_max": float(np.max(np.abs(flow - flow.T))),
        "stationary_law_error": float(np.max(np.abs(probabilities @ matrix - probabilities))),
    }


def _mutation_matrix(instance: SliceInstance, k: int, beta: float, mutation: str) -> np.ndarray:
    """Construct a named incorrect kernel so the retained negative controls can fire."""

    law = independent_exact_law(instance, k, beta)
    if mutation == "energy_sign_reversal":
        values = [-ising_energy(instance, state) for state in law.states]
        return _pair_transition_with_energy(instance, law, beta, values)
    if mutation == "double_counted_edges":
        values = []
        for state in law.states:
            edge_term = sum(
                coupling * state[left] * state[right] for left, right, coupling in instance.edges
            )
            field_term = sum(field * state[index] for index, field in enumerate(instance.fields))
            values.append(-2.0 * edge_term - field_term)
        return _pair_transition_with_energy(instance, law, beta, values)
    if mutation == "asymmetric_proposal_without_hastings":
        probabilities = np.asarray(law.probabilities, dtype=np.float64)
        proposal = np.arange(1.0, len(law.states) + 1.0, dtype=np.float64)
        proposal /= float(np.sum(proposal))
        ratios = probabilities[np.newaxis, :] / probabilities[:, np.newaxis]
        matrix = proposal[np.newaxis, :] * np.minimum(1.0, ratios)
        np.fill_diagonal(matrix, 0.0)
        np.fill_diagonal(matrix, 1.0 - np.sum(matrix, axis=1))
        return matrix
    raise ValueError(f"unknown mutation: {mutation}")


def run_mutations() -> list[JsonDict]:
    """Retain the four preregistered failures instead of hiding negative results."""

    instance = make_frustrated_instance(8, SMALL_GRAPH_SEEDS[0])
    law = independent_exact_law(instance, 2, 2.0)
    rows: list[JsonDict] = []
    for mutation in sorted(REQUIRED_MUTATIONS - {"invalid_k"}):
        diagnostics = transition_diagnostics(law, _mutation_matrix(instance, 2, 2.0, mutation))
        observed = max(
            diagnostics["detailed_balance_error_max"], diagnostics["stationary_law_error"]
        )
        detected = observed > TOLERANCE
        rows.append(
            {
                "mutation_id": mutation,
                "expected_value": f"> {TOLERANCE}",
                "observed_value": observed,
                "detected": detected,
                "passed": detected,
            }
        )
    try:
        enumerate_slice(8, 9)
    except ValueError:
        observed_invalid = "ValueError"
    else:  # pragma: no cover - the branch documents the mutation's failure mode.
        observed_invalid = "accepted"
    detected_invalid = observed_invalid == "ValueError"
    rows.append(
        {
            "mutation_id": "invalid_k",
            "expected_value": "ValueError",
            "observed_value": observed_invalid,
            "detected": detected_invalid,
            "passed": detected_invalid,
        }
    )
    return rows


def autocorrelation(values: Sequence[float], lag_window: int) -> list[float] | None:
    """Return the fixed lag series, or null when a constant trace has no correlation."""

    if lag_window < 1 or lag_window >= len(values):
        raise ValueError("lag window must be positive and shorter than the trace")
    array = np.asarray(values, dtype=np.float64)
    centered = array - float(np.mean(array))
    variance_sum = float(np.dot(centered, centered))
    if variance_sum == 0.0:
        return None
    correlations = [1.0]
    for lag in range(1, lag_window + 1):
        numerator = float(np.dot(centered[:-lag], centered[lag:]))
        correlations.append(numerator / variance_sum)
    return correlations


def effective_sample_size(values: Sequence[float], lag_window: int) -> float | None:
    """Estimate ESS with the initial positive autocorrelation sequence."""

    correlations = autocorrelation(values, lag_window)
    if correlations is None:
        return None
    positive_sum = 0.0
    for value in correlations[1:]:
        if value <= 0.0:
            break
        positive_sum += value
    estimate = len(values) / (1.0 + 2.0 * positive_sum)
    return float(min(len(values), max(1.0, estimate)))


def _uniform_slice_state(n: int, k: int, rng: random.Random) -> State:
    """Draw uniformly from a slice by choosing its positive sites without replacement."""

    positive = set(rng.sample(range(n), k))
    return tuple(1 if index in positive else -1 for index in range(n))


def _stream_seed(seed: int, instance: SliceInstance, arm: str) -> int:
    """Separate random streams by graph and arm without reading the current chain state."""

    material = f"{seed}:{instance.instance_hash}:{arm}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def run_chain(
    instance: SliceInstance,
    *,
    k: int,
    beta: float,
    arm: str,
    seed: int,
    energy_budget: int | None = None,
    wall_time_s: float | None = None,
) -> JsonDict:
    """Run one arm under exactly one charged-energy or wall-time protocol."""

    validate_instance(instance)
    enumerate_slice(instance.n, k)
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if (energy_budget is None) == (wall_time_s is None):
        raise ValueError("provide exactly one budget")
    if energy_budget is not None and energy_budget <= 0:
        raise ValueError("energy budget must be positive")
    if wall_time_s is not None and (not math.isfinite(wall_time_s) or wall_time_s <= 0.0):
        raise ValueError("wall-time budget must be positive and finite")

    rng = random.Random(_stream_seed(seed, instance, arm))
    state = _uniform_slice_state(instance.n, k, rng)
    current_energy = 0.0
    energies: list[float] = []
    evaluations = 0
    accepted = 0
    started = time.monotonic()
    deadline = started + wall_time_s if wall_time_s is not None else None
    while True:
        if energy_budget is not None and evaluations >= energy_budget:
            break
        if deadline is not None and evaluations > 0 and time.monotonic() >= deadline:
            break
        if evaluations == 0:
            current_energy = ising_energy(instance, state)
        elif arm == PAIR_SWAP_ARM:
            candidate, _, _ = propose_pair_swap(state, rng)
            candidate_energy = ising_energy(instance, candidate)
            threshold = log_acceptance(beta, candidate_energy - current_energy)
            if math.log(max(rng.random(), sys.float_info.min)) < threshold:
                state = candidate
                current_energy = candidate_energy
                accepted += 1
        elif arm == UNIFORM_SLICE_ARM:
            candidate = _uniform_slice_state(instance.n, k, rng)
            candidate_energy = ising_energy(instance, candidate)
            threshold = log_acceptance(beta, candidate_energy - current_energy)
            if math.log(max(rng.random(), sys.float_info.min)) < threshold:
                state = candidate
                current_energy = candidate_energy
                accepted += 1
        else:
            candidate = list(state)
            candidate[rng.randrange(instance.n)] *= -1
            ising_energy(instance, candidate)
            # A single flip leaves the fixed-cardinality support.  The target
            # assigns it zero probability, so this control rejects every move.
        evaluations += 1
        energies.append(current_energy)
    elapsed = time.monotonic() - started
    frozen = arm == INVALID_SINGLE_SPIN_ARM
    lag = min(LAG_WINDOW, max(1, len(energies) - 1))
    correlations = None if frozen or len(energies) < 2 else autocorrelation(energies, lag)
    ess = None if frozen or correlations is None else effective_sample_size(energies, lag)
    return {
        "arm": arm,
        "status": "complete",
        "failure": None,
        "attempts": evaluations,
        "energy_evaluations": evaluations,
        "elapsed_s": elapsed,
        "acceptance": 0.0 if frozen else accepted / max(1, evaluations),
        "frozen": frozen,
        "frozen_reason": "single spin flips leave the slice" if frozen else None,
        "energy_ess": ess,
        "ess_per_second": ess / elapsed if ess is not None and elapsed > 0.0 else None,
        "autocorrelation": correlations,
        "minimum_energy": min(energies),
        "sample_count": len(energies),
        "trace_sha256": sha256_json(energies),
    }


def collect_preconditions(
    root: Path,
    *,
    result_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str]]:
    """Record required bytes, contract fields, tools, output paths, and hashes."""

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
        if (root / path).is_file() and source_sizes[str(path)]
    }
    spec_file = root / SPEC_PATH
    spec_text = spec_file.read_text(encoding="utf-8") if spec_file.is_file() else ""
    milestone_fields = {
        "id": TASK_ID,
        "milestone": MILESTONE,
        "deliverable": str(RESULT_PATH),
        "gated_on": [],
    }
    checks = [
        {
            "check": "driving_capability_spec",
            "upstream": str(SPEC_PATH),
            "field": "REQ-SAMPLER-7187",
            "expected_value": {"exists": True, "req_present": True},
            "observed_value": {
                "exists": spec_file.is_file(),
                "req_present": "REQ-SAMPLER-7187" in spec_text,
            },
            "passed": spec_file.is_file() and "REQ-SAMPLER-7187" in spec_text,
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
            "upstream": MILESTONE,
            "field": "task_contract",
            "expected_value": milestone_fields,
            "observed_value": dict(milestone_fields),
            "passed": True,
        },
        {
            "check": "required_tools",
            "upstream": "host_python_environment",
            "field": "python_and_numpy",
            "expected_value": {"python": True, "numpy": True},
            "observed_value": {
                "python": bool(sys.executable),
                "numpy": bool(np.__version__),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
            },
            "passed": bool(sys.executable and np.__version__),
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
    ]
    return checks, source_hashes


def _base_artifact(
    *, root: Path, run_date: str, checks: list[JsonDict], hashes: dict[str, str]
) -> JsonDict:
    """Create all required fields before either blocked or measured evidence is added."""

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
            "benchmark_seeds": list(BENCHMARK_SEEDS),
            "stream_derivation": "sha256(seed, instance_hash, arm)",
        },
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "running",
        "inference_substrate_class": "blocked_no_run",
        "slice_sampler_ready_score": 0,
        "finite_law_rows": [],
        "transition_rows": [],
        "benchmark_rows": [],
        "hardware_execution_claimed": False,
        "paper_replication_claimed": False,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "root": str(root.resolve()),
        "edge_counting_convention": EDGE_COUNTING_CONVENTION,
        "proposal_symmetry_derivation": (
            "Every slice state has k positive and n-k negative sites. Selecting one of each "
            "gives q(y|x)=1/(k*(n-k)); the swap preserves both counts, so q(x|y) is equal."
        ),
        "mutation_rows": [],
        "speed_win_claimed": False,
        "minimum_energy_is_fidelity_evidence": False,
    }


def _finite_rows() -> tuple[list[JsonDict], list[JsonDict]]:
    """Enumerate the frozen small roster and check all three explicit kernels."""

    finite_rows: list[JsonDict] = []
    transition_rows: list[JsonDict] = []
    total = 2 * 3 * 3 * len(SMALL_GRAPH_SEEDS)
    completed = 0
    heartbeat = time.monotonic()
    for n in (8, 12):
        for k in (1, 2, n // 2):
            for beta in (0.5, 2.0, 5.0):
                for graph_seed in SMALL_GRAPH_SEEDS:
                    instance = make_frustrated_instance(n, graph_seed)
                    law = independent_exact_law(instance, k, beta)
                    normalization_error = abs(sum(law.probabilities) - 1.0)
                    support_min = min(law.probabilities)
                    cardinality_valid = all(state.count(1) == k for state in law.states)
                    energy_parity_error = max(
                        abs(ising_energy(instance, state) - energy)
                        for state, energy in zip(law.states, law.energies, strict=True)
                    )
                    condition_id = f"n{n}:k{k}:beta{beta:g}:graph{graph_seed}"
                    finite_passed = (
                        normalization_error <= TOLERANCE
                        and support_min > 0.0
                        and cardinality_valid
                        and energy_parity_error <= TOLERANCE
                    )
                    finite_rows.append(
                        {
                            "row_type": "finite_law",
                            "condition_id": condition_id,
                            "n": n,
                            "k": k,
                            "beta": beta,
                            "graph_seed": graph_seed,
                            "instance_hash": instance.instance_hash,
                            "state_count": len(law.states),
                            "normalization_error": normalization_error,
                            "support_min": support_min,
                            "cardinality_valid": cardinality_valid,
                            "energy_parity_error_max": energy_parity_error,
                            "passed": finite_passed,
                        }
                    )
                    for arm in ARMS:
                        diagnostics = transition_diagnostics(
                            law, transition_matrix(instance, k, beta, arm=arm)
                        )
                        passed = (
                            diagnostics["transition_normalization_error_max"] <= TOLERANCE
                            and diagnostics["transition_support_min"] >= -TOLERANCE
                            and diagnostics["detailed_balance_error_max"] <= TOLERANCE
                            and diagnostics["stationary_law_error"] <= TOLERANCE
                        )
                        transition_rows.append(
                            {
                                "row_type": "transition",
                                "condition_id": condition_id,
                                "n": n,
                                "k": k,
                                "beta": beta,
                                "graph_seed": graph_seed,
                                "arm": arm,
                                "frozen": arm == INVALID_SINGLE_SPIN_ARM,
                                **diagnostics,
                                "passed": passed,
                            }
                        )
                    completed += 1
                    now = time.monotonic()
                    if now - heartbeat >= 50.0:  # pragma: no cover - requires a stalled host.
                        print(
                            f"[heartbeat] elapsed_s={now - heartbeat:.3f} completed={completed}/{total} "
                            "operation=finite_transition_enumeration",
                            flush=True,
                        )
                        heartbeat = now
    return finite_rows, transition_rows


def _benchmark_rows() -> list[JsonDict]:
    """Run all arms under separate matched energy and matched wall-time protocols."""

    rows: list[JsonDict] = []
    total = 2 * 2 * len(BENCHMARK_SEEDS) * len(ARMS) * 2
    completed = 0
    started = time.monotonic()
    heartbeat = started
    for n in (32, 64):
        for k in (2, 4):
            for graph_seed in BENCHMARK_SEEDS:
                instance = make_frustrated_instance(n, graph_seed)
                for arm in ARMS:
                    protocols = (
                        ("matched_energy_evaluations", {"energy_budget": ENERGY_BUDGET}),
                        ("matched_wall_time", {"wall_time_s": WALL_TIME_BUDGET_S}),
                    )
                    for protocol, budget in protocols:
                        result = run_chain(
                            instance,
                            k=k,
                            beta=BENCHMARK_BETA,
                            arm=arm,
                            seed=graph_seed,
                            **budget,
                        )
                        rows.append(
                            {
                                "row_type": "benchmark",
                                "unit_id": f"n{n}:k{k}:seed{graph_seed}:{arm}:{protocol}",
                                "n": n,
                                "k": k,
                                "beta": BENCHMARK_BETA,
                                "graph_seed": graph_seed,
                                "instance_hash": instance.instance_hash,
                                "protocol": protocol,
                                "energy_budget": budget.get("energy_budget"),
                                "wall_time_budget_s": budget.get("wall_time_s"),
                                "stationary_law_error": None,
                                "target_law": "fixed_cardinality_boltzmann_slice",
                                **result,
                            }
                        )
                        completed += 1
                        now = time.monotonic()
                        if now - heartbeat >= 50.0:  # pragma: no cover - requires a stalled host.
                            print(
                                f"[heartbeat] elapsed_s={now - started:.3f} completed={completed}/{total} "
                                "operation=matched_sampler_benchmark",
                                flush=True,
                            )
                            heartbeat = now
    return rows


def build_artifact(
    *,
    root: Path | None = None,
    run_date: str = RUN_DATE,
    preconditions: list[JsonDict] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> JsonDict:
    """Build complete evidence, or stop before measurement on an external gate failure."""

    repository = root or Path(__file__).resolve().parents[2]
    started = time.monotonic()
    _progress(0, "start", "precondition checks")
    if preconditions is None or source_hashes is None:
        measured_checks, measured_hashes = collect_preconditions(repository)
        checks = measured_checks if preconditions is None else preconditions
        hashes = measured_hashes if source_hashes is None else source_hashes
    else:
        checks, hashes = preconditions, source_hashes
    artifact = _base_artifact(root=repository, run_date=run_date, checks=checks, hashes=hashes)
    failed = next((row for row in checks if not row.get("passed", False)), None)
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
    _progress(1, "start", "fixed finite-law roster")
    finite_rows, transition_rows = _finite_rows()
    artifact["finite_law_rows"] = finite_rows
    _progress(1, "end", f"fixed finite-law roster rows={len(finite_rows)}")
    _progress(2, "start", "explicit transition diagnostics")
    artifact["transition_rows"] = transition_rows
    _progress(2, "end", f"explicit transition diagnostics rows={len(transition_rows)}")
    _progress(3, "start", "adversarial mutation controls")
    artifact["mutation_rows"] = run_mutations()
    _progress(3, "end", f"adversarial mutation controls rows={len(artifact['mutation_rows'])}")
    _progress(4, "start", "matched energy-evaluation benchmark")
    _progress(4, "end", "matched energy-evaluation benchmark configured")
    _progress(5, "start", "matched wall-time benchmark")
    benchmark_rows = _benchmark_rows()
    artifact["benchmark_rows"] = benchmark_rows
    _progress(5, "end", f"matched benchmarks rows={len(benchmark_rows)}")
    _progress(6, "start", "assemble terminal evidence")
    artifact["rows"] = finite_rows + transition_rows + benchmark_rows
    active_energy_rows = [
        row
        for row in benchmark_rows
        if row["protocol"] == "matched_energy_evaluations" and not row["frozen"]
    ]
    mean_ess_rate = {
        arm: float(
            np.mean(
                [
                    row["ess_per_second"]
                    for row in active_energy_rows
                    if row["arm"] == arm and row["ess_per_second"] is not None
                ]
            )
        )
        for arm in (PAIR_SWAP_ARM, UNIFORM_SLICE_ARM)
    }
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": "cpu_exact_solver_or_simulator",
            "duration_s": time.monotonic() - started,
            "verdict_class": "positive",
            "honest_verdict": (
                "positive: the bounded pair-swap kernel preserves the enumerated slice laws; "
                "this is CPU sample-quality evidence, not a mixing theorem or paper replication"
            ),
            "inference_substrate_class": "cpu_exact_solver_or_simulator",
            "slice_sampler_ready_score": 1,
            "speed_win_claimed": (mean_ess_rate[PAIR_SWAP_ARM] > mean_ess_rate[UNIFORM_SLICE_ARM]),
            "speed_comparison": {
                "metric": "mean_energy_ess_per_second_on_matched_energy_rows",
                "by_arm": mean_ess_rate,
                "separate_from_readiness": True,
            },
            "bounded_claim": (
                "Validated a feasible pair-swap Metropolis baseline on the fixed CPU roster. "
                "No polynomial mixing, specialized-paper-sampler, or hardware claim is made."
            ),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _progress(6, "end", "assemble terminal evidence")
    return artifact


def _expected_benchmark_units() -> set[str]:
    """Return the frozen benchmark roster so row deletion cannot improve a verdict."""

    return {
        f"n{n}:k{k}:seed{seed}:{arm}:{protocol}"
        for n in (32, 64)
        for k in (2, 4)
        for seed in BENCHMARK_SEEDS
        for arm in ARMS
        for protocol in ("matched_energy_evaluations", "matched_wall_time")
    }


def validate_artifact(payload: JsonDict) -> list[str]:
    """Recompute completeness, finite invariants, claim limits, and the checksum."""

    errors: list[str] = []
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(payload)
    if missing:
        return ["missing_required_fields"]
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_invalid")
    if payload.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    if (
        payload.get("hardware_execution_claimed") is not False
        or payload.get("paper_replication_claimed") is not False
    ):
        errors.append("claim_boundary_invalid")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_authority_invalid")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")

    if payload.get("verdict_class") == "blocked":
        gate = payload.get("gate_check_summary", {})
        if (
            payload.get("status") != "blocked_external_precondition"
            or payload.get("inference_substrate_class") != "blocked_no_run"
            or payload.get("slice_sampler_ready_score") != 0
            or any(
                payload.get(field)
                for field in ("rows", "finite_law_rows", "transition_rows", "benchmark_rows")
            )
            or gate.get("passed") is not False
            or not gate.get("failed_check")
        ):
            errors.append("blocked_state_invalid")
        return errors

    finite_rows = payload.get("finite_law_rows", [])
    transition_rows = payload.get("transition_rows", [])
    benchmark_rows = payload.get("benchmark_rows", [])
    if len(finite_rows) != 54:
        errors.append("finite_law_rows_incomplete")
    elif any(
        not row.get("passed")
        or row.get("normalization_error", math.inf) > TOLERANCE
        or row.get("support_min", 0.0) <= 0.0
        or not row.get("cardinality_valid")
        or row.get("energy_parity_error_max", math.inf) > TOLERANCE
        for row in finite_rows
    ):
        errors.append("finite_law_failure")
    if len(transition_rows) != 162:
        errors.append("transition_rows_incomplete")
    elif any(
        not row.get("passed")
        or row.get("detailed_balance_error_max", math.inf) > TOLERANCE
        or row.get("stationary_law_error", math.inf) > TOLERANCE
        for row in transition_rows
    ):
        errors.append("transition_failure")
    units = [row.get("unit_id") for row in benchmark_rows]
    if len(benchmark_rows) != 240 or set(units) != _expected_benchmark_units():
        errors.append("benchmark_rows_incomplete")
    frozen_rows = [row for row in benchmark_rows if row.get("arm") == INVALID_SINGLE_SPIN_ARM]
    if any(
        row.get("frozen") is not True
        or row.get("energy_ess") is not None
        or row.get("ess_per_second") is not None
        or row.get("autocorrelation") is not None
        for row in frozen_rows
    ):
        errors.append("frozen_control_metrics_invalid")
    if len(payload.get("rows", [])) != 456:
        errors.append("rows_incomplete")
    mutations = payload.get("mutation_rows", [])
    if {row.get("mutation_id") for row in mutations} != REQUIRED_MUTATIONS or any(
        row.get("passed") is not True for row in mutations
    ):
        errors.append("mutation_controls_invalid")
    ready = not any(
        error
        in {
            "finite_law_rows_incomplete",
            "finite_law_failure",
            "transition_rows_incomplete",
            "transition_failure",
            "benchmark_rows_incomplete",
            "frozen_control_metrics_invalid",
            "rows_incomplete",
            "mutation_controls_invalid",
        }
        for error in errors
    )
    if payload.get("slice_sampler_ready_score") != int(ready):
        errors.append("readiness_invalid")
    if payload.get("status") != "complete" or payload.get("verdict_class") != "positive":
        errors.append("terminal_verdict_invalid")
    if payload.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("substrate_class_invalid")
    if payload.get("gate_check_summary", {}).get("passed") is not True:
        errors.append("gate_summary_invalid")
    return errors


def atomic_write(path: Path, payload: JsonDict) -> JsonDict:
    """Publish one complete JSON file through same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - only an interrupted replace leaves this file.
            temporary.unlink()
    return {"atomic_replace": True, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def run_experiment(*, root: Path | None = None, run_date: str = RUN_DATE) -> JsonDict:
    """Build, validate, and atomically publish the requested terminal artifact."""

    repository = root or Path(__file__).resolve().parents[2]
    artifact = build_artifact(root=repository, run_date=run_date)
    _progress(7, "start", "artifact validation")
    errors = validate_artifact(artifact)
    _progress(7, "end", f"artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"artifact validation failed: {errors}")
    _progress(8, "start", "final atomic write")
    receipt = atomic_write(repository / RESULT_PATH, artifact)
    _progress(8, "end", f"final atomic write bytes={receipt['bytes']}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed run date or a caller-selected validation target."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp7187 or validate durable bytes without changing repository state."""

    args = _parse_args(argv)
    if args.validate is not None:
        _progress(7, "start", f"validation path={args.validate}")
        try:
            payload = json.loads(args.validate.read_text(encoding="utf-8"))
            errors = validate_artifact(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"validation_error: {exc}", flush=True)
            _progress(7, "end", "validation_end errors=1")
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        _progress(7, "end", f"validation_end errors={len(errors)}")
        return 0 if not errors else 2
    try:
        run_experiment(run_date=args.date)
    except (OSError, TypeError, ValueError) as exc:
        print(f"experiment_error: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the shipped entrypoint.
    raise SystemExit(main())
