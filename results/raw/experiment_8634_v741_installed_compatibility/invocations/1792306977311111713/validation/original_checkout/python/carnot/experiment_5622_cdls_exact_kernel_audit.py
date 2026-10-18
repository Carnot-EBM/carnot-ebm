"""Exp5622 exact cDLS transition-kernel audit.

Spec refs: REQ-SAMPLE-5622, SCENARIO-SAMPLE-5622.

This module answers one narrow scientific question before another timing run:
does the continuous-intermediate projection target the intended discrete Ising
Boltzmann distribution? The verifier builds exact small transition matrices.
The uncorrected projection is kept as a named positive control because it is the
failure mode we need the audit to catch. The corrected kernel uses the standard
Metropolis-Hastings ratio with the exact discrete energy and the exact
probability that the Gaussian intermediate projects to a requested Ising state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import erfc, exp, log, sqrt
from pathlib import Path
from typing import Any

import numpy as np


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5622_cdls_exact_kernel_audit.json")

EXPERIMENT = 5622
EXPERIMENT_ID = "exp5622-cdls-exact-kernel-audit"
MILESTONE = "2026.07.507"
RUN_DATE = "2026-07-14"
SCHEMA = "carnot.experiment_5622.cdls_exact_kernel_audit.v1"
SPEC_REFS = ("REQ-SAMPLE-5622", "SCENARIO-SAMPLE-5622")
INFERENCE_SUBSTRATE = "deterministic_verifier"

DEFAULT_RANDOM_SEEDS = (5622, 5623, 5624, 5625, 5626)
DEFAULT_RETAINED_SAMPLES = 4096
DEFAULT_BURN_IN_STEPS = 512
EXACT_ENUMERATION_LIMIT_STATES = 4096
EXACT_ROW_SUM_TOLERANCE = 1e-12
EXACT_BALANCE_TOLERANCE = 1e-10
EXACT_TV_TOLERANCE = 1e-10
BIASED_CONTROL_TV_FLOOR = 1e-3
IRREDUCIBLE_TOLERANCE = 1e-15
TERMINAL_PREFIXES = ("complete:", "blocked:")

CDLS_PROPOSAL_STD = 0.72
CDLS_DRIFT_SCALE = 0.17
CDLS_CONTINUOUS_BOUND = 3.0

MODEL_IDS = (
    "discrete_dls_heat_bath",
    "uncorrected_cdls_projection",
    "corrected_cdls_projection_mh",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Explains why each audit field exists before Exp5623 can consume it.",
    "target_descriptors": "Makes every exact Ising system explicit enough to recompute the target distribution.",
    "models_tested": "Keeps the discrete baseline, biased cDLS control, and corrected cDLS kernel separately named.",
    "state_space_sizes": "Shows the exact enumeration limit and the number of states audited per topology.",
    "transition_row_sum_error_max": "Confirms every accepted transition matrix is normalized.",
    "detailed_balance_residual_max": "Measures reversibility or invariant-condition residual instead of assuming it.",
    "exact_distribution_tv_max": "Quantifies stationary-target parity for the final corrected kernel.",
    "empirical_distribution_intervals": "Bounds finite-sample exact-versus-empirical total variation across independent seeds.",
    "broken_kernel_controls_rejected": "Proves the audit rejects biased and broken positive controls.",
    "correction_applied": "Declares whether projection bias was corrected before any timing run.",
    "correction_spec": "Documents the exact acceptance rule and proposal probability used by the corrected kernel.",
    "quality_gate_specification": "Predeclares the large-n Exp5623 quality gates before timing evidence exists.",
    "quality_gate_specified_count": "Makes downstream gate coverage numeric rather than prose-only.",
    "kernel_audit_ready_score": "Equals 1.0 only when exactness and control-rejection gates pass.",
    "inference_substrate": "Declares that the artifact came from deterministic enumeration and simulation, not LLM judgment.",
    "random_seeds": "Records replay seeds for empirical checks.",
    "reproducibility_checksum": "Content-addresses the audit artifact for deterministic replay.",
    "honest_verdict": "States whether a biased kernel blocks timing.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class IsingSystem:
    """Small exact Ising target used to audit sampler transition kernels."""

    system_id: str
    topology: str
    n_spins: int
    temperature: float
    couplings: np.ndarray
    fields: np.ndarray
    constraint_indices: tuple[int, ...]
    target_spins: tuple[int, ...]


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for reproducible audit hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content using Carnot's SHA-256 convention."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _coupling_matrix(n_spins: int, edges: Sequence[tuple[int, int, float]]) -> np.ndarray:
    matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    for i, j, value in edges:
        matrix[i, j] = float(value)
        matrix[j, i] = float(value)
    return matrix


def exact_ising_systems() -> list[IsingSystem]:
    """Return finite-temperature, nontrivial Ising systems small enough to enumerate."""

    return [
        IsingSystem(
            system_id="ising_n3_mixed_line",
            topology="line_mixed_ferro_antiferro",
            n_spins=3,
            temperature=1.35,
            couplings=_coupling_matrix(3, [(0, 1, 0.62), (1, 2, -0.48)]),
            fields=np.array([0.18, -0.27, 0.11], dtype=np.float64),
            constraint_indices=(0, 1, 2),
            target_spins=(1, -1, 1),
        ),
        IsingSystem(
            system_id="ising_n4_frustrated_cycle",
            topology="cycle_with_frustrating_chord",
            n_spins=4,
            temperature=1.10,
            couplings=_coupling_matrix(
                4,
                [(0, 1, 0.51), (1, 2, -0.44), (2, 3, 0.37), (3, 0, -0.29), (0, 2, 0.16)],
            ),
            fields=np.array([0.07, -0.21, 0.19, -0.09], dtype=np.float64),
            constraint_indices=(0, 2, 3),
            target_spins=(1, -1, 1, -1),
        ),
        IsingSystem(
            system_id="ising_n5_weighted_star",
            topology="weighted_star_with_leaf_coupling",
            n_spins=5,
            temperature=1.55,
            couplings=_coupling_matrix(
                5,
                [(0, 1, 0.42), (0, 2, -0.35), (0, 3, 0.28), (0, 4, -0.31), (2, 4, 0.22)],
            ),
            fields=np.array([0.13, -0.08, 0.17, -0.24, 0.06], dtype=np.float64),
            constraint_indices=(0, 1, 4),
            target_spins=(1, 1, -1, 1, -1),
        ),
        IsingSystem(
            system_id="ising_n6_ladder_chord",
            topology="two_by_three_ladder_with_chord",
            n_spins=6,
            temperature=1.25,
            couplings=_coupling_matrix(
                6,
                [
                    (0, 1, 0.33),
                    (1, 2, -0.27),
                    (3, 4, 0.41),
                    (4, 5, -0.36),
                    (0, 3, 0.25),
                    (1, 4, -0.30),
                    (2, 5, 0.21),
                    (0, 5, -0.18),
                ],
            ),
            fields=np.array([-0.12, 0.22, -0.16, 0.09, -0.05, 0.14], dtype=np.float64),
            constraint_indices=(1, 2, 4, 5),
            target_spins=(-1, 1, -1, 1, -1, 1),
        ),
    ]


def enumerate_states(n_spins: int) -> np.ndarray:
    """Enumerate all Ising states as rows with values in {-1, +1}."""

    state_count = 2**int(n_spins)
    if state_count > EXACT_ENUMERATION_LIMIT_STATES:
        raise ValueError("state_count exceeds exact enumeration limit")  # pragma: no cover
    states = np.empty((state_count, n_spins), dtype=np.int8)
    for row in range(state_count):
        for bit in range(n_spins):
            states[row, bit] = 1 if (row >> bit) & 1 else -1
    return states


def energy_vector(system: IsingSystem, states: np.ndarray) -> np.ndarray:
    """Compute E(x) = -0.5 x^T J x - h^T x for every enumerated state."""

    spin_values = states.astype(np.float64)
    pair_term = -0.5 * np.einsum("bi,ij,bj->b", spin_values, system.couplings, spin_values)
    field_term = -spin_values @ system.fields
    return pair_term + field_term


def target_distribution(system: IsingSystem, states: np.ndarray) -> np.ndarray:
    """Return the normalized finite-temperature Boltzmann target."""

    beta = 1.0 / float(system.temperature)
    energies = energy_vector(system, states)
    shifted = -beta * (energies - float(np.min(energies)))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def _state_index(states: np.ndarray) -> dict[tuple[int, ...], int]:
    return {tuple(int(value) for value in row): index for index, row in enumerate(states)}


def heat_bath_transition_matrix(system: IsingSystem, states: np.ndarray) -> np.ndarray:
    """Build an exact single-site heat-bath transition matrix."""

    beta = 1.0 / float(system.temperature)
    index = _state_index(states)
    n_states = len(states)
    matrix = np.zeros((n_states, n_states), dtype=np.float64)
    for row_index, state in enumerate(states.astype(np.float64)):
        for spin_index in range(system.n_spins):
            field = float(system.couplings[spin_index] @ state + system.fields[spin_index])
            prob_plus = 1.0 / (1.0 + exp(-2.0 * beta * field))
            for spin_value, probability in ((1, prob_plus), (-1, 1.0 - prob_plus)):
                proposed = state.astype(np.int8).copy()
                proposed[spin_index] = spin_value
                matrix[row_index, index[tuple(int(value) for value in proposed)]] += probability / system.n_spins
    return matrix


def _normal_cdf(value: float) -> float:
    probability = 0.5 * erfc(-float(value) / sqrt(2.0))
    return min(max(probability, 1e-300), 1.0)


def cdls_projected_proposal_matrix(system: IsingSystem, states: np.ndarray) -> np.ndarray:
    """Return q(y|x), the exact probability that the Gaussian intermediate projects to y."""

    beta = 1.0 / float(system.temperature)
    spin_values = states.astype(np.float64)
    n_states = len(states)
    matrix = np.zeros((n_states, n_states), dtype=np.float64)
    for row_index, source in enumerate(spin_values):
        field = system.couplings @ source + system.fields
        mean = source + CDLS_DRIFT_SCALE * beta * field
        for col_index, projected in enumerate(spin_values):
            log_probability = 0.0
            for sign, coordinate_mean in zip(projected, mean, strict=True):
                probability = _normal_cdf(float(sign) * float(coordinate_mean) / CDLS_PROPOSAL_STD)
                log_probability += log(probability)
            matrix[row_index, col_index] = exp(log_probability)
        matrix[row_index, :] /= float(np.sum(matrix[row_index, :]))
    return matrix


def corrected_cdls_transition_matrix(system: IsingSystem, states: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Apply an exact Metropolis-Hastings correction to the projected cDLS proposal."""

    proposal = cdls_projected_proposal_matrix(system, states)
    log_target = np.log(target)
    n_states = len(states)
    matrix = np.zeros((n_states, n_states), dtype=np.float64)
    log_proposal = np.log(proposal)
    for source in range(n_states):
        off_diagonal_mass = 0.0
        for proposed in range(n_states):
            if proposed == source:
                continue
            log_accept = min(
                0.0,
                float(log_target[proposed] + log_proposal[proposed, source] - log_target[source] - log_proposal[source, proposed]),
            )
            transition_probability = float(proposal[source, proposed] * exp(log_accept))
            matrix[source, proposed] = transition_probability
            off_diagonal_mass += transition_probability
        matrix[source, source] = max(0.0, 1.0 - off_diagonal_mass)
    return matrix


def biased_temperature_control_matrix(system: IsingSystem, states: np.ndarray) -> np.ndarray:
    """Build a normalized but deliberately wrong-temperature heat-bath control."""

    biased = IsingSystem(
        system_id=f"{system.system_id}_biased_temperature",
        topology=system.topology,
        n_spins=system.n_spins,
        temperature=system.temperature * 1.8,
        couplings=system.couplings,
        fields=system.fields,
        constraint_indices=system.constraint_indices,
        target_spins=system.target_spins,
    )
    return heat_bath_transition_matrix(biased, states)


def broken_proposal_control_matrix(state_count: int) -> np.ndarray:
    """Return a row-normalized identity kernel that is intentionally reducible."""

    return np.eye(int(state_count), dtype=np.float64)


def transition_matrices(system: IsingSystem, states: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Construct the three named kernels required by the audit."""

    return {
        "discrete_dls_heat_bath": heat_bath_transition_matrix(system, states),
        "uncorrected_cdls_projection": cdls_projected_proposal_matrix(system, states),
        "corrected_cdls_projection_mh": corrected_cdls_transition_matrix(system, states, target),
    }


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    """Return total variation distance between two categorical distributions."""

    return float(0.5 * np.sum(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    """Compute the invariant row distribution by deterministic power iteration."""

    distribution = np.full(matrix.shape[0], 1.0 / matrix.shape[0], dtype=np.float64)
    for _ in range(20_000):
        updated = distribution @ matrix
        if float(np.max(np.abs(updated - distribution))) < 1e-15:
            distribution = updated
            break
        distribution = updated
    distribution = np.maximum(distribution, 0.0)
    return distribution / float(np.sum(distribution))


def _reachable(adjacency: np.ndarray, start: int) -> set[int]:
    seen = {int(start)}
    stack = [int(start)]
    while stack:
        node = stack.pop()
        for neighbor in np.flatnonzero(adjacency[node]):
            value = int(neighbor)
            if value not in seen:
                seen.add(value)
                stack.append(value)
    return seen


def is_irreducible(matrix: np.ndarray) -> bool:
    """Return true when the positive-probability transition graph is strongly connected."""

    adjacency = matrix > IRREDUCIBLE_TOLERANCE
    all_nodes = set(range(matrix.shape[0]))
    return _reachable(adjacency, 0) == all_nodes and _reachable(adjacency.T, 0) == all_nodes


def detailed_balance_residual(target: np.ndarray, matrix: np.ndarray) -> float:
    """Measure max |pi(x)P(x,y) - pi(y)P(y,x)| over the transition matrix."""

    flow = target[:, None] * matrix
    return float(np.max(np.abs(flow - flow.T)))


def energy_histogram_tv(system: IsingSystem, states: np.ndarray, target: np.ndarray, observed: np.ndarray) -> float:
    """Compare target and observed distributions after grouping states by energy."""

    energies = energy_vector(system, states)
    target_bins: dict[float, float] = {}
    observed_bins: dict[float, float] = {}
    for energy, target_probability, observed_probability in zip(energies, target, observed, strict=True):
        key = round(float(energy), 12)
        target_bins[key] = target_bins.get(key, 0.0) + float(target_probability)
        observed_bins[key] = observed_bins.get(key, 0.0) + float(observed_probability)
    keys = sorted(set(target_bins) | set(observed_bins))
    target_values = np.array([target_bins.get(key, 0.0) for key in keys], dtype=np.float64)
    observed_values = np.array([observed_bins.get(key, 0.0) for key in keys], dtype=np.float64)
    return total_variation(target_values, observed_values)


def audit_transition_matrix(
    *,
    system: IsingSystem,
    states: np.ndarray,
    target: np.ndarray,
    matrix: np.ndarray,
    model_id: str,
) -> JsonDict:
    """Audit one transition matrix against exact stochastic and target-invariance gates."""

    row_sums = np.sum(matrix, axis=1)
    row_sum_error = float(np.max(np.abs(row_sums - 1.0)))
    probability_min = float(np.min(matrix))
    stochastic = row_sum_error <= EXACT_ROW_SUM_TOLERANCE and probability_min >= -EXACT_ROW_SUM_TOLERANCE
    irreducible = is_irreducible(matrix) if stochastic else False
    stationary = stationary_distribution(matrix) if stochastic else np.full(len(target), 1.0 / len(target))
    stationary_tv = total_variation(stationary, target)
    balance_residual = detailed_balance_residual(target, matrix) if stochastic else float("inf")
    histogram_tv = energy_histogram_tv(system, states, target, stationary)
    passes = bool(
        stochastic
        and irreducible
        and balance_residual <= EXACT_BALANCE_TOLERANCE
        and stationary_tv <= EXACT_TV_TOLERANCE
        and histogram_tv <= EXACT_TV_TOLERANCE
    )
    return {
        "system_id": system.system_id,
        "model_id": model_id,
        "state_count": int(len(states)),
        "row_sum_error_max": row_sum_error,
        "probability_min": probability_min,
        "irreducible": irreducible,
        "detailed_balance_residual_max": balance_residual,
        "stationary_distribution_tv": stationary_tv,
        "energy_histogram_tv": histogram_tv,
        "passes_exact_target_gate": passes,
    }


def empirical_distribution(
    matrix: np.ndarray,
    *,
    seed: int,
    retained_samples: int,
    burn_in_steps: int,
) -> np.ndarray:
    """Replay a Markov chain and return the retained empirical distribution."""

    rng = np.random.default_rng(int(seed))
    cumulative = np.cumsum(matrix, axis=1)
    cumulative[:, -1] = 1.0
    state = int(rng.integers(0, matrix.shape[0]))
    for _ in range(int(burn_in_steps)):
        state = int(np.searchsorted(cumulative[state], rng.random(), side="right"))
    counts = np.zeros(matrix.shape[0], dtype=np.int64)
    for _ in range(int(retained_samples)):
        state = int(np.searchsorted(cumulative[state], rng.random(), side="right"))
        counts[state] += 1
    return counts.astype(np.float64) / float(retained_samples)


def empirical_tv_interval(
    *,
    system_id: str,
    model_id: str,
    matrix: np.ndarray,
    target: np.ndarray,
    seeds: Sequence[int],
    retained_samples: int,
    burn_in_steps: int,
) -> JsonDict:
    """Summarize exact-versus-empirical TV across independent replay seeds."""

    tv_values: list[float] = []
    distributions: list[np.ndarray] = []
    for seed in seeds:
        distribution = empirical_distribution(
            matrix,
            seed=int(seed),
            retained_samples=int(retained_samples),
            burn_in_steps=int(burn_in_steps),
        )
        distributions.append(distribution)
        tv_values.append(total_variation(distribution, target))
    replay = empirical_distribution(
        matrix,
        seed=int(seeds[0]),
        retained_samples=int(retained_samples),
        burn_in_steps=int(burn_in_steps),
    )
    values = np.asarray(tv_values, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    half_width = 1.96 * std / sqrt(float(len(values))) if len(values) > 1 else 0.0
    return {
        "system_id": system_id,
        "model_id": model_id,
        "seed_count": len(seeds),
        "seeds": [int(seed) for seed in seeds],
        "retained_samples_per_seed": int(retained_samples),
        "burn_in_steps": int(burn_in_steps),
        "tv_by_seed": [round(float(value), 12) for value in tv_values],
        "tv_mean": round(mean, 12),
        "tv_std": round(std, 12),
        "tv_interval_95": [round(max(0.0, mean - half_width), 12), round(min(1.0, mean + half_width), 12)],
        "tv_max": round(float(np.max(values)), 12),
        "uncertainty_method": "five_seed_t_interval_on_seed_tv_values",
        "seed_replay_match": bool(np.array_equal(distributions[0], replay)),
    }


def target_descriptor(system: IsingSystem, states: np.ndarray, target: np.ndarray) -> JsonDict:
    """Describe one exact target with enough detail for recomputation."""

    energies = energy_vector(system, states)
    return {
        "system_id": system.system_id,
        "topology": system.topology,
        "n_spins": system.n_spins,
        "state_count": int(len(states)),
        "temperature": float(system.temperature),
        "couplings": np.round(system.couplings, 12).tolist(),
        "fields": np.round(system.fields, 12).tolist(),
        "constraint_indices": list(system.constraint_indices),
        "target_spins": list(system.target_spins),
        "energy_min": round(float(np.min(energies)), 12),
        "energy_max": round(float(np.max(energies)), 12),
        "target_probability_checksum": sha256_json(np.round(target, 15).tolist()),
    }


def quality_gate_specification() -> list[JsonDict]:
    """Predeclare the large-n gates that Exp5623 must satisfy before timing claims."""

    return [
        {
            "gate_id": "QG-5623-TARGET-TV",
            "category": "target_behavior",
            "requirement": "Corrected cDLS empirical spin/energy marginals must remain noninferior to discrete DLS under the preregistered TV tolerance.",
            "threshold": {"empirical_distribution_tv_delta_max": 0.03},
        },
        {
            "gate_id": "QG-5623-ENERGY-HIST",
            "category": "energy_behavior",
            "requirement": "Corrected cDLS energy histogram distance from the discrete baseline must not exceed the preregistered tolerance before speed is inspected.",
            "threshold": {"energy_histogram_tv_delta_max": 0.03, "mean_energy_worse_abs_max": 0.5},
        },
        {
            "gate_id": "QG-5623-MIXING",
            "category": "mixing",
            "requirement": "Corrected cDLS must meet a minimum effective-sample-size floor and must not hide poor mixing behind wall-clock speed.",
            "threshold": {"min_effective_sample_size": 100.0, "max_integrated_autocorrelation_ratio": 2.0},
        },
        {
            "gate_id": "QG-5623-CONSTRAINT",
            "category": "constraint_validity",
            "requirement": "Corrected cDLS exact constraint-satisfaction rate must be noninferior to discrete DLS within the preregistered tolerance.",
            "threshold": {"constraint_satisfaction_rate_drop_max": 0.03},
        },
    ]


def models_tested() -> list[JsonDict]:
    """Return the named kernels preserved by the audit."""

    return [
        {
            "model_id": "discrete_dls_heat_bath",
            "role": "discrete_baseline_control",
            "description": "Single-site heat-bath Gibbs kernel using the exact Ising conditional.",
            "projection_correction": "not_applicable",
        },
        {
            "model_id": "uncorrected_cdls_projection",
            "role": "biased_projection_positive_control",
            "description": "Continuous Gaussian intermediate projected to signs without MH correction.",
            "projection_correction": "none",
        },
        {
            "model_id": "corrected_cdls_projection_mh",
            "role": "final_kernel",
            "description": "Projected cDLS proposal corrected by exact discrete Metropolis-Hastings.",
            "projection_correction": "metropolis_hastings_exact_projected_proposal",
        },
    ]


def _summarize_model_audits(audit_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    summaries: list[JsonDict] = []
    for model_id in MODEL_IDS:
        rows = [row for row in audit_rows if row["model_id"] == model_id]
        summaries.append(
            {
                "model_id": model_id,
                "systems_tested": len(rows),
                "systems_passing_exact_target_gate": sum(1 for row in rows if row["passes_exact_target_gate"] is True),
                "row_sum_error_max": max(float(row["row_sum_error_max"]) for row in rows),
                "detailed_balance_residual_max": max(float(row["detailed_balance_residual_max"]) for row in rows),
                "stationary_distribution_tv_max": max(float(row["stationary_distribution_tv"]) for row in rows),
                "energy_histogram_tv_max": max(float(row["energy_histogram_tv"]) for row in rows),
            }
        )
    return summaries


def ready_score(payload: Mapping[str, Any]) -> float:
    """Return 1.0 only when exactness and control-rejection gates all pass."""

    model_ids = {row.get("model_id") for row in payload.get("models_tested", []) if isinstance(row, Mapping)}
    intervals = payload.get("empirical_distribution_intervals", [])
    empirical_replay_ok = bool(intervals) and all(
        isinstance(row, Mapping) and row.get("seed_replay_match") is True for row in intervals
    )
    gates = (
        model_ids == set(MODEL_IDS),
        payload.get("inference_substrate") == INFERENCE_SUBSTRATE,
        payload.get("correction_applied") is True,
        isinstance(payload.get("correction_spec"), Mapping)
        and payload["correction_spec"].get("large_n_timing_tuned") is False,
        int(payload.get("quality_gate_specified_count", 0)) >= 3,
        payload.get("broken_kernel_controls_rejected") is True,
        float(payload.get("transition_row_sum_error_max", 1.0)) <= EXACT_ROW_SUM_TOLERANCE,
        float(payload.get("detailed_balance_residual_max", 1.0)) <= EXACT_BALANCE_TOLERANCE,
        float(payload.get("exact_distribution_tv_max", 1.0)) <= EXACT_TV_TOLERANCE,
        float(payload.get("energy_histogram_tv_max", 1.0)) <= EXACT_TV_TOLERANCE,
        empirical_replay_ok,
    )
    return 1.0 if all(gates) else 0.0


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return a terminal verdict that blocks timing when the final kernel is biased."""

    if ready_score(payload) == 1.0:
        return (
            "complete: corrected cDLS exact kernel audit ready; "
            "biased uncorrected kernel blocks timing unless the corrected kernel is used"
        )
    return "blocked: biased kernel blocks timing until exact cDLS kernel gates pass"


def build_artifact(
    *,
    retained_samples: int = DEFAULT_RETAINED_SAMPLES,
    burn_in_steps: int = DEFAULT_BURN_IN_STEPS,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp5622 terminal audit artifact."""

    systems = exact_ising_systems()
    target_descriptors: list[JsonDict] = []
    state_space_rows: list[JsonDict] = []
    audit_rows: list[JsonDict] = []
    control_rows: list[JsonDict] = []
    empirical_rows: list[JsonDict] = []

    for system in systems:
        states = enumerate_states(system.n_spins)
        target = target_distribution(system, states)
        matrices = transition_matrices(system, states, target)
        target_descriptors.append(target_descriptor(system, states, target))
        state_space_rows.append(
            {
                "system_id": system.system_id,
                "n_spins": system.n_spins,
                "state_count": int(len(states)),
                "exact_transition_matrix_feasible": True,
            }
        )
        for model_id, matrix in matrices.items():
            audit_rows.append(
                audit_transition_matrix(
                    system=system,
                    states=states,
                    target=target,
                    matrix=matrix,
                    model_id=model_id,
                )
            )
        control_rows.append(
            audit_transition_matrix(
                system=system,
                states=states,
                target=target,
                matrix=biased_temperature_control_matrix(system, states),
                model_id="biased_temperature_positive_control",
            )
        )
        control_rows.append(
            audit_transition_matrix(
                system=system,
                states=states,
                target=target,
                matrix=broken_proposal_control_matrix(len(states)),
                model_id="broken_zero_support_control",
            )
        )
        for model_id in ("discrete_dls_heat_bath", "corrected_cdls_projection_mh"):
            empirical_rows.append(
                empirical_tv_interval(
                    system_id=system.system_id,
                    model_id=model_id,
                    matrix=matrices[model_id],
                    target=target,
                    seeds=random_seeds,
                    retained_samples=retained_samples,
                    burn_in_steps=burn_in_steps,
                )
            )

    corrected_rows = [row for row in audit_rows if row["model_id"] == "corrected_cdls_projection_mh"]
    uncorrected_rows = [row for row in audit_rows if row["model_id"] == "uncorrected_cdls_projection"]
    biased_controls_rejected = all(row["passes_exact_target_gate"] is False for row in control_rows)
    uncorrected_rejected = all(
        row["passes_exact_target_gate"] is False and float(row["stationary_distribution_tv"]) > BIASED_CONTROL_TV_FLOOR
        for row in uncorrected_rows
    )
    gates = quality_gate_specification()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "target_descriptors": target_descriptors,
        "models_tested": models_tested(),
        "state_space_sizes": {
            "enumeration_limit_states": EXACT_ENUMERATION_LIMIT_STATES,
            "max_exact_enumerated_states": max(row["state_count"] for row in state_space_rows),
            "systems": state_space_rows,
        },
        "exact_audit_rows": audit_rows,
        "model_summaries": _summarize_model_audits(audit_rows),
        "positive_control_audit_rows": control_rows,
        "transition_row_sum_error_max": max(float(row["row_sum_error_max"]) for row in corrected_rows),
        "detailed_balance_residual_max": max(float(row["detailed_balance_residual_max"]) for row in corrected_rows),
        "exact_distribution_tv_max": max(float(row["stationary_distribution_tv"]) for row in corrected_rows),
        "energy_histogram_tv_max": max(float(row["energy_histogram_tv"]) for row in corrected_rows),
        "empirical_distribution_intervals": empirical_rows,
        "broken_kernel_controls_rejected": bool(biased_controls_rejected and uncorrected_rejected),
        "correction_applied": bool(uncorrected_rejected),
        "correction_spec": {
            "final_kernel": "corrected_cdls_projection_mh",
            "proposal": "z_i ~ Normal(x_i + drift_scale * beta * (Jx + h)_i, proposal_std); y_i = sign(z_i)",
            "proposal_probability": "q(y|x) = product_i Phi(y_i * mean_i / proposal_std)",
            "acceptance_rule": "alpha(x,y)=min(1, pi(y) q(x|y) / (pi(x) q(y|x))) with pi(x) proportional to exp(-beta E(x))",
            "continuous_bound": CDLS_CONTINUOUS_BOUND,
            "bound_note": "The symmetric clamp is recorded for Exp5611 parity; for positive bounds it does not change sign-projection probabilities.",
            "proposal_std": CDLS_PROPOSAL_STD,
            "drift_scale": CDLS_DRIFT_SCALE,
            "large_n_timing_tuned": False,
        },
        "quality_gate_specification": gates,
        "quality_gate_specified_count": len(gates),
        "kernel_audit_ready_score": 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": [int(seed) for seed in random_seeds],
        "tests_added_or_reused": list(tests_added_or_reused or [str(RESULT_RELATIVE_PATH), "tests/python/test_experiment_5622_cdls_exact_kernel_audit.py"]),
        "honest_verdict": "",
        "reproducibility_checksum": "",
    }
    artifact["kernel_audit_ready_score"] = ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate Exp5622 fields and fail closed on manually-set readiness."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload.get("correction_applied") is not True:
        raise ValueError("correction_applied must be true")
    correction_spec = payload.get("correction_spec")
    if not isinstance(correction_spec, Mapping):
        raise ValueError("correction_spec must be a mapping")  # pragma: no cover
    if correction_spec.get("large_n_timing_tuned") is not False:
        raise ValueError("large_n_timing_tuned must be false")
    if int(payload.get("quality_gate_specified_count", 0)) < 3:
        raise ValueError("quality_gate_specified_count must be at least 3")
    if int(payload.get("quality_gate_specified_count", 0)) != len(payload.get("quality_gate_specification", [])):
        raise ValueError("quality_gate_specified_count mismatch")  # pragma: no cover
    if payload.get("broken_kernel_controls_rejected") is not True:
        raise ValueError("broken_kernel_controls_rejected must be true")
    model_ids = {row.get("model_id") for row in payload.get("models_tested", []) if isinstance(row, Mapping)}
    if model_ids != set(MODEL_IDS):
        raise ValueError("models_tested mismatch")  # pragma: no cover
    expected_ready = ready_score(payload)
    if float(payload.get("kernel_audit_ready_score", -1.0)) != expected_ready:
        raise ValueError("kernel_audit_ready_score mismatch")
    if expected_ready == 1.0:
        if float(payload.get("transition_row_sum_error_max", 1.0)) > EXACT_ROW_SUM_TOLERANCE:
            raise ValueError("transition_row_sum_error_max exceeds tolerance")  # pragma: no cover
        if float(payload.get("detailed_balance_residual_max", 1.0)) > EXACT_BALANCE_TOLERANCE:
            raise ValueError("detailed_balance_residual_max exceeds tolerance")  # pragma: no cover
        if float(payload.get("exact_distribution_tv_max", 1.0)) > EXACT_TV_TOLERANCE:
            raise ValueError("exact_distribution_tv_max exceeds tolerance")  # pragma: no cover
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal artifact with stable formatting."""

    output_path = Path(root) / RESULT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def run_experiment(repo_root: str | Path = REPO_ROOT) -> Path:  # pragma: no cover - thin live runner.
    """Build, validate, and write the Exp5622 artifact."""

    artifact = build_artifact()
    return write_output(repo_root, artifact)


if __name__ == "__main__":  # pragma: no cover
    print(run_experiment())
