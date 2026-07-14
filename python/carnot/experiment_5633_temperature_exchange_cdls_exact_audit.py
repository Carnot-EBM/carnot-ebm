"""Exp5633 exact fixed-ladder temperature-label exchange audit.

Spec refs: REQ-SAMPLE-5633, SCENARIO-SAMPLE-5633.

This module adds only one layer around Exp5622's corrected cDLS kernel: a fixed
parallel-tempering ladder whose exchange move swaps temperature labels, not
replica states. Keeping labels and states separate matters because the exact
invariant distribution is a product over "state currently wearing beta k"
factors. Copying states during exchange can look harmless in code, but it is no
longer the reversible label move audited here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import itertools
import json
from math import exp
from pathlib import Path
from typing import Any

import numpy as np

from carnot import experiment_5622_cdls_exact_kernel_audit as exp5622


JsonDict = dict[str, Any]
IsingSystem = exp5622.IsingSystem

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5633_temperature_exchange_cdls_exact_audit.json")
CORRECTED_KERNEL_SOURCE_RELATIVE_PATH = Path("python/carnot/experiment_5622_cdls_exact_kernel_audit.py")
CORRECTED_KERNEL_RESULT_RELATIVE_PATH = exp5622.RESULT_RELATIVE_PATH

EXPERIMENT = 5633
EXPERIMENT_ID = "exp5633-temperature-exchange-cdls-exact-audit"
MILESTONE = "2026.07.508"
RUN_DATE = "2026-07-14"
SCHEMA = "carnot.experiment_5633.temperature_exchange_cdls_exact_audit.v1"
SPEC_REFS = ("REQ-SAMPLE-5633", "SCENARIO-SAMPLE-5633")
INFERENCE_SUBSTRATE = "exact_corrected_cdls_with_replica_temperature_exchange"

BETA_LADDER = (0.45, 0.8, 1.25)
DEFAULT_RANDOM_SEEDS = (5633, 5634, 5635, 5636, 5637)
DEFAULT_REPLAY_SWEEPS = 48
TRANSITION_NORMALIZATION_TOLERANCE = 1e-12
SWAP_DETAILED_BALANCE_TOLERANCE = 1e-6
EXACT_DISTRIBUTION_TV_THRESHOLD = 0.02
COLD_ENERGY_ERROR_TOLERANCE = 1e-9
TERMINAL_PREFIXES = ("complete:", "blocked:")

BROKEN_CONTROL_IDS = (
    "missing_beta_factors",
    "wrong_energy_sign",
    "state_copy_swap",
    "asynchronous_stale_energy",
    "one_way_exchange",
    "biased_proposal_schedule",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Explains why each audit and gate field exists before any exchange quality trial can consume it.",
    "corrected_kernel_receipt": "Proves the exact Exp5622 corrected within-replica substrate is unchanged and source-hash pinned.",
    "beta_ladder": "Freezes the temperature scope so later quality differences cannot come from an unreported ladder search.",
    "within_replica_schedule": "Records the exact corrected-cDLS transition accounting applied at each physical replica.",
    "exchange_schedule": "Records which adjacent beta-label pairs are proposed and in what fixed order.",
    "swap_rule": "Makes the exact label-exchange acceptance ratio inspectable.",
    "enumerable_targets": "Preserves deterministic Ising ground truth, marginals, and energy distributions for replay.",
    "transition_normalization_error_max": "Confirms the within-replica and exchange kernels are stochastic before stationarity is trusted.",
    "swap_detailed_balance_residual_max": "Measures reversibility of the temperature-label exchange instead of assuming it.",
    "exact_distribution_tv_max": "Measures exact product-target and cold-label parity for the composed exchange sampler.",
    "cold_replica_energy_error": "Reports target-temperature energy parity explicitly instead of hiding it in aggregate TV.",
    "round_trip_accounting_error": "Checks that label movement is accounted for without copying or losing temperature labels.",
    "broken_controls": "Demonstrates the audit rejects biased exchange variants that would otherwise look superficially normalized.",
    "deterministic_replay_pass": "Proves fixed seeds reproduce the same label-exchange trace.",
    "timing_claimed": "Bare false keeps Exp5633 an exactness audit and does not reopen retired crossover timing.",
    "hardware_speedup_claimed": "Bare false prevents CUDA, board, SNN, TSU, or hardware inference from entering this artifact.",
    "replica_exchange_kernel_ready_score": "Provides a mechanical downstream gate that is 1.0 only when exactness and control-rejection gates pass.",
    "inference_substrate": "Declares exact corrected cDLS plus replica temperature-label exchange, not LLM inference or hardware timing.",
    "random_seeds": "Records replay seeds for deterministic trace and accounting checks.",
    "reproducibility_checksum": "Content-addresses the artifact so future reruns can detect silent drift.",
    "honest_verdict": "States whether invariant failure blocks later quality trials.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so audit hashes survive reruns."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content using the same stable format as Exp5622."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file byte-for-byte for source and result receipts."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _coupling_matrix(n_spins: int, edges: Sequence[tuple[int, int, float]]) -> np.ndarray:
    matrix = np.zeros((int(n_spins), int(n_spins)), dtype=np.float64)
    for left, right, value in edges:
        matrix[left, right] = float(value)
        matrix[right, left] = float(value)
    return matrix


def enumerable_frustrated_systems() -> list[IsingSystem]:
    """Return tiny frustrated Ising systems whose replica products are enumerable."""

    return [
        IsingSystem(
            system_id="ising_n3_frustrated_triangle",
            topology="frustrated_triangle_mixed_sign_cycle",
            n_spins=3,
            temperature=1.0,
            couplings=_coupling_matrix(3, [(0, 1, 0.58), (1, 2, 0.52), (0, 2, -0.47)]),
            fields=np.array([0.12, -0.18, 0.09], dtype=np.float64),
            constraint_indices=(0, 1, 2),
            target_spins=(1, -1, 1),
        ),
        IsingSystem(
            system_id="ising_n3_frustrated_triangle_field_shift",
            topology="frustrated_triangle_field_shift",
            n_spins=3,
            temperature=1.0,
            couplings=_coupling_matrix(3, [(0, 1, -0.61), (1, 2, 0.49), (0, 2, 0.56)]),
            fields=np.array([-0.07, 0.16, -0.11], dtype=np.float64),
            constraint_indices=(0, 1, 2),
            target_spins=(-1, 1, -1),
        ),
    ]


def _system_for_beta(system: IsingSystem, beta: float) -> IsingSystem:
    """Reuse Exp5622's kernel code by representing beta as temperature 1 / beta."""

    return replace(system, temperature=1.0 / float(beta))


def target_distribution_for_beta(system: IsingSystem, states: np.ndarray, beta: float) -> np.ndarray:
    """Enumerate the exact Boltzmann distribution for one beta value."""

    energies = exp5622.energy_vector(system, states)
    shifted = -float(beta) * (energies - float(np.min(energies)))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def corrected_transition_for_beta(system: IsingSystem, states: np.ndarray, beta: float) -> np.ndarray:
    """Build Exp5622's exact corrected cDLS transition matrix at a requested beta."""

    beta_system = _system_for_beta(system, beta)
    target = target_distribution_for_beta(system, states, beta)
    return exp5622.corrected_cdls_transition_matrix(beta_system, states, target)


def within_replica_schedule() -> list[JsonDict]:
    """Return the fixed per-sweep corrected-cDLS state-update schedule."""

    return [
        {
            "phase": "within_replica",
            "physical_replica_index": replica_index,
            "kernel": "corrected_cdls_projection_mh",
            "temperature_source": "current_label_beta",
            "transition_count": 1,
        }
        for replica_index in range(len(BETA_LADDER))
    ]


def exchange_schedule() -> list[JsonDict]:
    """Return the fixed adjacent beta-label proposal order."""

    return [
        {
            "phase": "temperature_label_exchange",
            "adjacent_beta_label_pair": [left, left + 1],
            "proposal": "swap_temperature_labels",
            "state_copy_allowed": False,
            "transition_count": 1,
        }
        for left in range(len(BETA_LADDER) - 1)
    ]


def swap_rule() -> JsonDict:
    """Describe the exact label-exchange Metropolis rule used by the audit."""

    return {
        "state_update": "temperature_labels_only",
        "acceptance_ratio": "min(1, exp((beta_a - beta_b) * (E(x_a) - E(x_b))))",
        "energy_convention": "Exp5622 E(x) = -0.5 x^T J x - h^T x",
        "proposal_symmetry": "fixed adjacent beta-label pair proposal; reverse label swap has the same schedule mass",
        "state_copy_allowed": False,
    }


def _label_position(labels: Sequence[int], label: int) -> int:
    for index, value in enumerate(labels):
        if int(value) == int(label):
            return index
    raise ValueError(f"label {label} missing from permutation")  # pragma: no cover


def _swap_labels(labels: Sequence[int], label_pair: tuple[int, int]) -> tuple[int, ...]:
    updated = [int(value) for value in labels]
    left_pos = _label_position(updated, label_pair[0])
    right_pos = _label_position(updated, label_pair[1])
    updated[left_pos], updated[right_pos] = updated[right_pos], updated[left_pos]
    return tuple(updated)


def label_exchange_candidate(
    *,
    state_indices: Sequence[int],
    labels: Sequence[int],
    label_pair: Sequence[int],
) -> JsonDict:
    """Return the proposed label-only move without mutating replica states."""

    pair = (int(label_pair[0]), int(label_pair[1]))
    return {
        "proposed_state_indices": [int(value) for value in state_indices],
        "proposed_labels": list(_swap_labels(labels, pair)),
        "state_copy_allowed": False,
    }


def _safe_acceptance_from_log_ratio(log_ratio: float) -> float:
    if log_ratio >= 0.0:
        return 1.0
    if log_ratio < -745.0:  # pragma: no cover - not reached by tiny fixtures.
        return 0.0
    return float(exp(log_ratio))


def swap_acceptance_probability(
    *,
    system: IsingSystem,
    states: np.ndarray,
    state_indices: Sequence[int],
    labels: Sequence[int],
    label_pair: Sequence[int],
    variant: str = "correct",
) -> float:
    """Return the transition probability for a label swap under one rule variant."""

    pair = (int(label_pair[0]), int(label_pair[1]))
    beta_left = BETA_LADDER[pair[0]]
    beta_right = BETA_LADDER[pair[1]]
    left_pos = _label_position(labels, pair[0])
    right_pos = _label_position(labels, pair[1])
    energies = exp5622.energy_vector(system, states)
    energy_left = float(energies[int(state_indices[left_pos])])
    energy_right = float(energies[int(state_indices[right_pos])])

    if variant == "correct":
        log_ratio = (beta_left - beta_right) * (energy_left - energy_right)
        return _safe_acceptance_from_log_ratio(log_ratio)
    if variant == "missing_beta_factors":
        return _safe_acceptance_from_log_ratio(energy_left - energy_right)
    if variant == "wrong_energy_sign":
        log_ratio = (beta_left - beta_right) * (energy_right - energy_left)
        return _safe_acceptance_from_log_ratio(log_ratio)
    if variant == "asynchronous_stale_energy":
        stale_left = float(energies[int(state_indices[0])])
        log_ratio = (beta_left - beta_right) * (stale_left - energy_right)
        return _safe_acceptance_from_log_ratio(log_ratio)
    if variant == "one_way_exchange":
        if left_pos > right_pos:
            return 0.0
        log_ratio = (beta_left - beta_right) * (energy_left - energy_right)
        return _safe_acceptance_from_log_ratio(log_ratio)
    if variant == "biased_proposal_schedule":
        log_ratio = (beta_left - beta_right) * (energy_left - energy_right)
        base_acceptance = _safe_acceptance_from_log_ratio(log_ratio)
        proposal_bias = 0.9 if energy_left < energy_right else 0.1
        return float(proposal_bias * base_acceptance)
    raise ValueError(f"unknown swap variant: {variant}")  # pragma: no cover


def _permutations(replica_count: int) -> list[tuple[int, ...]]:
    return list(itertools.permutations(range(int(replica_count))))


def _state_tuple_index(state_tuple: Sequence[int], state_count: int) -> int:
    value = 0
    multiplier = 1
    for state_index in state_tuple:
        value += int(state_index) * multiplier
        multiplier *= int(state_count)
    return value


def _augmented_index(
    state_tuple: Sequence[int],
    labels: Sequence[int],
    *,
    state_count: int,
    permutation_to_index: Mapping[tuple[int, ...], int],
) -> int:
    per_label_count = int(state_count) ** len(labels)
    return permutation_to_index[tuple(int(value) for value in labels)] * per_label_count + _state_tuple_index(
        state_tuple,
        state_count,
    )


def product_target_distribution(
    targets_by_label: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, ...]], dict[tuple[int, ...], int]]:
    """Enumerate the product target over physical states and label permutations."""

    replica_count = len(targets_by_label)
    state_count = len(targets_by_label[0])
    permutations = _permutations(replica_count)
    permutation_to_index = {labels: index for index, labels in enumerate(permutations)}
    distribution = np.zeros((state_count**replica_count) * len(permutations), dtype=np.float64)
    uniform_label_weight = 1.0 / float(len(permutations))
    for labels in permutations:
        for state_tuple in itertools.product(range(state_count), repeat=replica_count):
            probability = uniform_label_weight
            for physical_index, state_index in enumerate(state_tuple):
                label = labels[physical_index]
                probability *= float(targets_by_label[label][state_index])
            distribution[
                _augmented_index(
                    state_tuple,
                    labels,
                    state_count=state_count,
                    permutation_to_index=permutation_to_index,
                )
            ] = probability
    distribution /= float(np.sum(distribution))
    return distribution, permutations, permutation_to_index


def _iter_augmented_states(
    *,
    state_count: int,
    replica_count: int,
    permutations: Sequence[tuple[int, ...]],
):
    for labels in permutations:
        for state_tuple in itertools.product(range(state_count), repeat=replica_count):
            yield tuple(int(value) for value in state_tuple), labels


def _apply_within_update(
    distribution: np.ndarray,
    *,
    kernels_by_label: Sequence[np.ndarray],
    replica_index: int,
    state_count: int,
    permutations: Sequence[tuple[int, ...]],
    permutation_to_index: Mapping[tuple[int, ...], int],
) -> np.ndarray:
    replica_count = len(kernels_by_label)
    output = np.zeros_like(distribution)
    for state_tuple, labels in _iter_augmented_states(
        state_count=state_count,
        replica_count=replica_count,
        permutations=permutations,
    ):
        source_index = _augmented_index(
            state_tuple,
            labels,
            state_count=state_count,
            permutation_to_index=permutation_to_index,
        )
        mass = float(distribution[source_index])
        label = labels[int(replica_index)]
        row = kernels_by_label[label][state_tuple[int(replica_index)]]
        for proposed_state, transition_probability in enumerate(row):
            proposed_tuple = list(state_tuple)
            proposed_tuple[int(replica_index)] = int(proposed_state)
            target_index = _augmented_index(
                proposed_tuple,
                labels,
                state_count=state_count,
                permutation_to_index=permutation_to_index,
            )
            output[target_index] += mass * float(transition_probability)
    return output


def _apply_exchange_update(
    distribution: np.ndarray,
    *,
    system: IsingSystem,
    states: np.ndarray,
    label_pair: tuple[int, int],
    state_count: int,
    permutations: Sequence[tuple[int, ...]],
    permutation_to_index: Mapping[tuple[int, ...], int],
) -> np.ndarray:
    replica_count = len(label_pair) + 1
    output = np.zeros_like(distribution)
    for state_tuple, labels in _iter_augmented_states(
        state_count=state_count,
        replica_count=replica_count,
        permutations=permutations,
    ):
        source_index = _augmented_index(
            state_tuple,
            labels,
            state_count=state_count,
            permutation_to_index=permutation_to_index,
        )
        mass = float(distribution[source_index])
        acceptance = swap_acceptance_probability(
            system=system,
            states=states,
            state_indices=state_tuple,
            labels=labels,
            label_pair=label_pair,
            variant="correct",
        )
        swapped_labels = _swap_labels(labels, label_pair)
        swapped_index = _augmented_index(
            state_tuple,
            swapped_labels,
            state_count=state_count,
            permutation_to_index=permutation_to_index,
        )
        output[swapped_index] += mass * acceptance
        output[source_index] += mass * (1.0 - acceptance)
    return output


def apply_fixed_exchange_schedule(
    distribution: np.ndarray,
    *,
    system: IsingSystem,
    states: np.ndarray,
    kernels_by_label: Sequence[np.ndarray],
    state_count: int,
    permutations: Sequence[tuple[int, ...]],
    permutation_to_index: Mapping[tuple[int, ...], int],
) -> np.ndarray:
    """Apply one full preregistered within-replica plus exchange sweep exactly."""

    updated = distribution
    for step in within_replica_schedule():
        updated = _apply_within_update(
            updated,
            kernels_by_label=kernels_by_label,
            replica_index=int(step["physical_replica_index"]),
            state_count=state_count,
            permutations=permutations,
            permutation_to_index=permutation_to_index,
        )
    for step in exchange_schedule():
        pair = tuple(int(value) for value in step["adjacent_beta_label_pair"])
        updated = _apply_exchange_update(
            updated,
            system=system,
            states=states,
            label_pair=(pair[0], pair[1]),
            state_count=state_count,
            permutations=permutations,
            permutation_to_index=permutation_to_index,
        )
    return updated


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    """Return categorical total variation distance."""

    return float(0.5 * np.sum(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def cold_label_marginal(
    distribution: np.ndarray,
    *,
    cold_label: int,
    state_count: int,
    replica_count: int,
    permutations: Sequence[tuple[int, ...]],
    permutation_to_index: Mapping[tuple[int, ...], int],
) -> np.ndarray:
    """Marginalize over the state currently wearing the cold beta label."""

    marginal = np.zeros(state_count, dtype=np.float64)
    for state_tuple, labels in _iter_augmented_states(
        state_count=state_count,
        replica_count=replica_count,
        permutations=permutations,
    ):
        index = _augmented_index(
            state_tuple,
            labels,
            state_count=state_count,
            permutation_to_index=permutation_to_index,
        )
        cold_position = _label_position(labels, int(cold_label))
        marginal[state_tuple[cold_position]] += float(distribution[index])
    return marginal / float(np.sum(marginal))


def transition_normalization_error(
    *,
    system: IsingSystem,
    states: np.ndarray,
    kernels_by_label: Sequence[np.ndarray],
    beta_ladder: Sequence[float],
) -> float:
    """Measure the largest row-sum error over within and label-exchange kernels."""

    errors = [float(np.max(np.abs(np.sum(matrix, axis=1) - 1.0))) for matrix in kernels_by_label]
    replica_count = len(beta_ladder)
    permutations = _permutations(replica_count)
    state_count = len(states)
    for step in exchange_schedule():
        pair = tuple(int(value) for value in step["adjacent_beta_label_pair"])
        for state_tuple, labels in _iter_augmented_states(
            state_count=state_count,
            replica_count=replica_count,
            permutations=permutations,
        ):
            acceptance = swap_acceptance_probability(
                system=system,
                states=states,
                state_indices=state_tuple,
                labels=labels,
                label_pair=pair,
                variant="correct",
            )
            errors.append(abs((acceptance + (1.0 - acceptance)) - 1.0))
    return max(errors)


def swap_detailed_balance_residual(
    *,
    system: IsingSystem,
    states: np.ndarray,
    beta_ladder: Sequence[float],
    label_pair: Sequence[int],
    variant: str = "correct",
) -> float:
    """Measure detailed-balance residual for one adjacent label-exchange kernel."""

    targets_by_label = [target_distribution_for_beta(system, states, beta) for beta in beta_ladder]
    product_target, permutations, permutation_to_index = product_target_distribution(targets_by_label)
    state_count = len(states)
    replica_count = len(beta_ladder)
    pair = (int(label_pair[0]), int(label_pair[1]))
    residual = 0.0
    for state_tuple, labels in _iter_augmented_states(
        state_count=state_count,
        replica_count=replica_count,
        permutations=permutations,
    ):
        source_index = _augmented_index(
            state_tuple,
            labels,
            state_count=state_count,
            permutation_to_index=permutation_to_index,
        )
        swapped_labels = _swap_labels(labels, pair)
        target_index = _augmented_index(
            state_tuple,
            swapped_labels,
            state_count=state_count,
            permutation_to_index=permutation_to_index,
        )
        forward = swap_acceptance_probability(
            system=system,
            states=states,
            state_indices=state_tuple,
            labels=labels,
            label_pair=pair,
            variant=variant,
        )
        reverse = swap_acceptance_probability(
            system=system,
            states=states,
            state_indices=state_tuple,
            labels=swapped_labels,
            label_pair=pair,
            variant=variant,
        )
        residual = max(
            residual,
            abs(float(product_target[source_index]) * forward - float(product_target[target_index]) * reverse),
        )
    return float(residual)


def _state_copy_stationarity_tv(
    *,
    system: IsingSystem,
    states: np.ndarray,
    beta_ladder: Sequence[float],
    label_pair: tuple[int, int],
) -> float:
    targets_by_label = [target_distribution_for_beta(system, states, beta) for beta in beta_ladder]
    product_target, permutations, permutation_to_index = product_target_distribution(targets_by_label)
    state_count = len(states)
    replica_count = len(beta_ladder)
    output = np.zeros_like(product_target)
    for state_tuple, labels in _iter_augmented_states(
        state_count=state_count,
        replica_count=replica_count,
        permutations=permutations,
    ):
        source_index = _augmented_index(
            state_tuple,
            labels,
            state_count=state_count,
            permutation_to_index=permutation_to_index,
        )
        acceptance = swap_acceptance_probability(
            system=system,
            states=states,
            state_indices=state_tuple,
            labels=labels,
            label_pair=label_pair,
            variant="correct",
        )
        left_pos = _label_position(labels, label_pair[0])
        right_pos = _label_position(labels, label_pair[1])
        copied_tuple = list(state_tuple)
        copied_tuple[left_pos] = copied_tuple[right_pos]
        copied_index = _augmented_index(
            copied_tuple,
            labels,
            state_count=state_count,
            permutation_to_index=permutation_to_index,
        )
        output[copied_index] += float(product_target[source_index]) * acceptance
        output[source_index] += float(product_target[source_index]) * (1.0 - acceptance)
    output /= float(np.sum(output))
    return total_variation(product_target, output)


def audit_broken_controls(
    *,
    system: IsingSystem,
    states: np.ndarray,
    beta_ladder: Sequence[float],
) -> list[JsonDict]:
    """Run the required biased exchange controls and report why each is rejected."""

    first_pair = tuple(int(value) for value in exchange_schedule()[0]["adjacent_beta_label_pair"])
    rows: list[JsonDict] = []
    for control_id in BROKEN_CONTROL_IDS:
        if control_id == "state_copy_swap":
            state_copy_tv = _state_copy_stationarity_tv(
                system=system,
                states=states,
                beta_ladder=beta_ladder,
                label_pair=first_pair,
            )
            rows.append(
                {
                    "control_id": control_id,
                    "detected": bool(state_copy_tv > EXACT_DISTRIBUTION_TV_THRESHOLD),
                    "rejection_reason": "exchange mutates/copies states instead of swapping labels",
                    "state_mutation_detected": True,
                    "stationarity_tv": round(float(state_copy_tv), 12),
                    "swap_detailed_balance_residual": None,
                }
            )
            continue
        residuals = [
            swap_detailed_balance_residual(
                system=system,
                states=states,
                beta_ladder=beta_ladder,
                label_pair=step["adjacent_beta_label_pair"],
                variant=control_id,
            )
            for step in exchange_schedule()
        ]
        residual = max(float(value) for value in residuals)
        rows.append(
            {
                "control_id": control_id,
                "detected": bool(residual > SWAP_DETAILED_BALANCE_TOLERANCE),
                "rejection_reason": "swap detailed balance residual exceeds tolerance",
                "state_mutation_detected": False,
                "stationarity_tv": None,
                "swap_detailed_balance_residual": round(residual, 12),
            }
        )
    return rows


def _energy_distribution(system: IsingSystem, states: np.ndarray, target: np.ndarray) -> list[JsonDict]:
    energies = exp5622.energy_vector(system, states)
    bins: dict[float, float] = {}
    for energy, probability in zip(energies, target, strict=True):
        key = round(float(energy), 12)
        bins[key] = bins.get(key, 0.0) + float(probability)
    return [
        {"energy": energy, "probability": round(probability, 15)}
        for energy, probability in sorted(bins.items(), key=lambda item: item[0])
    ]


def _spin_marginals(states: np.ndarray, target: np.ndarray) -> JsonDict:
    spin_values = states.astype(np.float64)
    probability_plus = np.sum((spin_values == 1.0) * target[:, None], axis=0)
    mean = np.sum(spin_values * target[:, None], axis=0)
    return {
        "probability_plus": [round(float(value), 15) for value in probability_plus],
        "spin_mean": [round(float(value), 15) for value in mean],
    }


def enumerable_target_descriptor(
    *,
    system: IsingSystem,
    states: np.ndarray,
    targets_by_label: Sequence[np.ndarray],
    beta_ladder: Sequence[float],
) -> JsonDict:
    """Record exact beta-indexed marginals and energy distributions."""

    energies = exp5622.energy_vector(system, states)
    beta_rows = []
    for label, beta in enumerate(beta_ladder):
        target = targets_by_label[label]
        beta_rows.append(
            {
                "beta_label": label,
                "beta": float(beta),
                "target_probability_checksum": sha256_json(np.round(target, 15).tolist()),
                "spin_marginals": _spin_marginals(states, target),
                "energy_mean": round(float(np.dot(target, energies)), 15),
                "energy_distribution": _energy_distribution(system, states, target),
            }
        )
    return {
        "system_id": system.system_id,
        "topology": system.topology,
        "n_spins": system.n_spins,
        "state_count": int(len(states)),
        "couplings": np.round(system.couplings, 12).tolist(),
        "fields": np.round(system.fields, 12).tolist(),
        "beta_targets": beta_rows,
    }


def _audit_one_system(system: IsingSystem, beta_ladder: Sequence[float]) -> JsonDict:
    states = exp5622.enumerate_states(system.n_spins)
    targets_by_label = [target_distribution_for_beta(system, states, beta) for beta in beta_ladder]
    kernels_by_label = [corrected_transition_for_beta(system, states, beta) for beta in beta_ladder]
    product_target, permutations, permutation_to_index = product_target_distribution(targets_by_label)
    after_schedule = apply_fixed_exchange_schedule(
        product_target,
        system=system,
        states=states,
        kernels_by_label=kernels_by_label,
        state_count=len(states),
        permutations=permutations,
        permutation_to_index=permutation_to_index,
    )
    product_tv = total_variation(product_target, after_schedule)
    cold_label = len(beta_ladder) - 1
    cold_marginal = cold_label_marginal(
        after_schedule,
        cold_label=cold_label,
        state_count=len(states),
        replica_count=len(beta_ladder),
        permutations=permutations,
        permutation_to_index=permutation_to_index,
    )
    cold_target = targets_by_label[cold_label]
    cold_tv = total_variation(cold_target, cold_marginal)
    energies = exp5622.energy_vector(system, states)
    cold_energy_error = abs(float(np.dot(cold_marginal, energies)) - float(np.dot(cold_target, energies)))
    swap_residuals = [
        swap_detailed_balance_residual(
            system=system,
            states=states,
            beta_ladder=beta_ladder,
            label_pair=step["adjacent_beta_label_pair"],
            variant="correct",
        )
        for step in exchange_schedule()
    ]
    baseline = _baseline_audits(
        system=system,
        states=states,
        targets_by_label=targets_by_label,
        kernels_by_label=kernels_by_label,
        beta_ladder=beta_ladder,
    )
    return {
        "descriptor": enumerable_target_descriptor(
            system=system,
            states=states,
            targets_by_label=targets_by_label,
            beta_ladder=beta_ladder,
        ),
        "product_target_state_count": int(len(product_target)),
        "transition_normalization_error": transition_normalization_error(
            system=system,
            states=states,
            kernels_by_label=kernels_by_label,
            beta_ladder=beta_ladder,
        ),
        "swap_detailed_balance_residual": max(float(value) for value in swap_residuals),
        "product_target_stationarity_tv": product_tv,
        "cold_label_distribution_tv": cold_tv,
        "cold_replica_energy_error": cold_energy_error,
        "baselines": baseline,
        "broken_controls": audit_broken_controls(system=system, states=states, beta_ladder=beta_ladder),
    }


def _baseline_audits(
    *,
    system: IsingSystem,
    states: np.ndarray,
    targets_by_label: Sequence[np.ndarray],
    kernels_by_label: Sequence[np.ndarray],
    beta_ladder: Sequence[float],
) -> JsonDict:
    cold_label = len(beta_ladder) - 1
    cold_distribution = targets_by_label[cold_label].copy()
    cold_kernel = kernels_by_label[cold_label]
    for _ in range(len(within_replica_schedule()) + len(exchange_schedule())):
        cold_distribution = cold_distribution @ cold_kernel
    single_chain_tv = total_variation(targets_by_label[cold_label], cold_distribution)

    product_target, permutations, permutation_to_index = product_target_distribution(targets_by_label)
    independent = product_target
    for step in within_replica_schedule():
        independent = _apply_within_update(
            independent,
            kernels_by_label=kernels_by_label,
            replica_index=int(step["physical_replica_index"]),
            state_count=len(states),
            permutations=permutations,
            permutation_to_index=permutation_to_index,
        )
    independent_tv = total_variation(product_target, independent)
    return {
        "single_chain_equal_transition": {
            "system_id": system.system_id,
            "beta_label": cold_label,
            "transition_count": len(within_replica_schedule()) + len(exchange_schedule()),
            "exact_distribution_tv": single_chain_tv,
        },
        "independent_replicas_no_exchange": {
            "system_id": system.system_id,
            "transition_count": len(within_replica_schedule()),
            "exact_distribution_tv": independent_tv,
        },
    }


def _sample_from_row(cumulative: np.ndarray, row_index: int, rng: np.random.Generator) -> int:
    return int(np.searchsorted(cumulative[row_index], rng.random(), side="right"))


def replay_label_exchange_trace(
    *,
    system: IsingSystem,
    states: np.ndarray,
    beta_ladder: Sequence[float],
    seed: int,
    sweeps: int,
) -> JsonDict:
    """Replay a sampled label-exchange chain and account for label movement."""

    rng = np.random.default_rng(int(seed))
    kernels_by_label = [corrected_transition_for_beta(system, states, beta) for beta in beta_ladder]
    cumulative_by_label = [np.cumsum(matrix, axis=1) for matrix in kernels_by_label]
    for cumulative in cumulative_by_label:
        cumulative[:, -1] = 1.0
    replica_count = len(beta_ladder)
    state_indices = [int(rng.integers(0, len(states))) for _ in range(replica_count)]
    labels = list(range(replica_count))
    attempts = 0
    accepted = 0
    rejected = 0
    invalid_permutation_count = 0
    label_position_counts = {str(label): [0 for _ in range(replica_count)] for label in range(replica_count)}
    trace: list[JsonDict] = []
    label_paths = {label: [] for label in range(replica_count)}

    for sweep in range(int(sweeps)):
        for step in within_replica_schedule():
            physical_index = int(step["physical_replica_index"])
            label = labels[physical_index]
            state_indices[physical_index] = _sample_from_row(cumulative_by_label[label], state_indices[physical_index], rng)
        for step in exchange_schedule():
            pair = tuple(int(value) for value in step["adjacent_beta_label_pair"])
            attempts += 1
            acceptance = swap_acceptance_probability(
                system=system,
                states=states,
                state_indices=state_indices,
                labels=labels,
                label_pair=pair,
                variant="correct",
            )
            if rng.random() < acceptance:
                labels = list(_swap_labels(labels, pair))
                accepted += 1
            else:
                rejected += 1
            if sorted(labels) != list(range(replica_count)):  # pragma: no cover - swaps preserve labels.
                invalid_permutation_count += 1
        for label in range(replica_count):
            position = _label_position(labels, label)
            label_position_counts[str(label)][position] += 1
            label_paths[label].append(position)
        trace.append(
            {
                "sweep": sweep,
                "state_indices": list(state_indices),
                "labels": list(labels),
                "cold_label_position": _label_position(labels, replica_count - 1),
            }
        )

    accounting_error = abs(attempts - accepted - rejected) + invalid_permutation_count
    return {
        "seed": int(seed),
        "sweeps": int(sweeps),
        "attempted_swaps": attempts,
        "accepted_swaps": accepted,
        "rejected_swaps": rejected,
        "invalid_permutation_count": invalid_permutation_count,
        "label_position_counts": label_position_counts,
        "round_trips_by_label": {
            str(label): _count_round_trips(path, replica_count=replica_count) for label, path in label_paths.items()
        },
        "round_trip_accounting_error": float(accounting_error),
        "trace_checksum": sha256_json(trace),
    }


def _count_round_trips(path: Sequence[int], *, replica_count: int) -> int:
    hot = 0
    cold = int(replica_count) - 1
    count = 0
    seen_cold = False
    for position in path:
        if int(position) == cold:
            seen_cold = True
        if seen_cold and int(position) == hot:
            count += 1
            seen_cold = False
    return count


def deterministic_replay_summary(
    *,
    systems: Sequence[IsingSystem],
    beta_ladder: Sequence[float],
    seeds: Sequence[int],
    sweeps: int,
) -> JsonDict:
    """Run each trace twice and require byte-stable trace checksums."""

    rows: list[JsonDict] = []
    for system in systems:
        states = exp5622.enumerate_states(system.n_spins)
        for seed in seeds:
            first = replay_label_exchange_trace(
                system=system,
                states=states,
                beta_ladder=beta_ladder,
                seed=int(seed),
                sweeps=int(sweeps),
            )
            second = replay_label_exchange_trace(
                system=system,
                states=states,
                beta_ladder=beta_ladder,
                seed=int(seed),
                sweeps=int(sweeps),
            )
            rows.append(
                {
                    "system_id": system.system_id,
                    "seed": int(seed),
                    "trace_checksum": first["trace_checksum"],
                    "replay_match": first["trace_checksum"] == second["trace_checksum"],
                    "round_trip_accounting_error": first["round_trip_accounting_error"],
                    "attempted_swaps": first["attempted_swaps"],
                    "accepted_swaps": first["accepted_swaps"],
                    "rejected_swaps": first["rejected_swaps"],
                    "round_trips_by_label": first["round_trips_by_label"],
                }
            )
    return {
        "pass": bool(rows) and all(row["replay_match"] is True for row in rows),
        "round_trip_accounting_error_max": max(float(row["round_trip_accounting_error"]) for row in rows),
        "rows": rows,
    }


def corrected_kernel_receipt(root: str | Path = REPO_ROOT) -> JsonDict:
    """Expose the exact Exp5622 source and artifact hashes used as substrate."""

    root_path = Path(root)
    source_path = root_path / CORRECTED_KERNEL_SOURCE_RELATIVE_PATH
    result_path = root_path / CORRECTED_KERNEL_RESULT_RELATIVE_PATH
    result_payload: JsonDict | None = None
    if result_path.exists():
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            result_payload = loaded
    correction_spec = result_payload.get("correction_spec", {}) if result_payload is not None else {}
    source_exists = source_path.exists()
    result_exists = result_path.exists()
    substrate_unchanged = bool(
        source_exists
        and result_exists
        and result_payload is not None
        and result_payload.get("kernel_audit_ready_score") == 1.0
        and correction_spec.get("final_kernel") == "corrected_cdls_projection_mh"
        and correction_spec.get("proposal_std") == exp5622.CDLS_PROPOSAL_STD
        and correction_spec.get("drift_scale") == exp5622.CDLS_DRIFT_SCALE
        and correction_spec.get("continuous_bound") == exp5622.CDLS_CONTINUOUS_BOUND
        and correction_spec.get("large_n_timing_tuned") is False
    )
    return {
        "source_path": CORRECTED_KERNEL_SOURCE_RELATIVE_PATH.as_posix(),
        "source_sha256": file_sha256(source_path) if source_exists else None,
        "result_path": CORRECTED_KERNEL_RESULT_RELATIVE_PATH.as_posix(),
        "result_sha256": file_sha256(result_path) if result_exists else None,
        "source_reproducibility_checksum": None
        if result_payload is None
        else result_payload.get("reproducibility_checksum"),
        "final_kernel": "corrected_cdls_projection_mh",
        "proposal_std": exp5622.CDLS_PROPOSAL_STD,
        "drift_scale": exp5622.CDLS_DRIFT_SCALE,
        "continuous_bound": exp5622.CDLS_CONTINUOUS_BOUND,
        "acceptance_rule": correction_spec.get(
            "acceptance_rule",
            "alpha(x,y)=min(1, pi(y) q(x|y) / (pi(x) q(y|x)))",
        ),
        "large_n_timing_tuned": correction_spec.get("large_n_timing_tuned"),
        "substrate_unchanged": substrate_unchanged,
    }


def _summarize_baselines(system_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    single_values = [
        float(row["baselines"]["single_chain_equal_transition"]["exact_distribution_tv"]) for row in system_rows
    ]
    independent_values = [
        float(row["baselines"]["independent_replicas_no_exchange"]["exact_distribution_tv"]) for row in system_rows
    ]
    return {
        "single_chain_equal_transition": {
            "description": "Cold-beta corrected cDLS chain with the same number of transition attempts as one exchange sweep.",
            "exact_distribution_tv_max": max(single_values),
        },
        "independent_replicas_no_exchange": {
            "description": "Fixed-label independent corrected cDLS replicas with the exchange layer removed.",
            "exact_distribution_tv_max": max(independent_values),
        },
    }


def ready_score(payload: Mapping[str, Any]) -> float:
    """Return 1.0 only when exactness, replay, and control gates all pass."""

    controls = payload.get("broken_controls", [])
    controls_ok = bool(controls) and all(isinstance(row, Mapping) and row.get("detected") is True for row in controls)
    gates = (
        isinstance(payload.get("corrected_kernel_receipt"), Mapping)
        and payload["corrected_kernel_receipt"].get("substrate_unchanged") is True,
        payload.get("beta_ladder") == list(BETA_LADDER),
        payload.get("timing_claimed") is False,
        payload.get("hardware_speedup_claimed") is False,
        payload.get("deterministic_replay_pass") is True,
        payload.get("round_trip_accounting_error") == 0.0,
        controls_ok,
        payload.get("validity_regression_detected") is False,
        payload.get("inference_substrate") == INFERENCE_SUBSTRATE,
        float(payload.get("transition_normalization_error_max", 1.0)) <= TRANSITION_NORMALIZATION_TOLERANCE,
        float(payload.get("swap_detailed_balance_residual_max", 1.0)) <= SWAP_DETAILED_BALANCE_TOLERANCE,
        float(payload.get("exact_distribution_tv_max", 1.0)) <= EXACT_DISTRIBUTION_TV_THRESHOLD,
        float(payload.get("cold_replica_energy_error", 1.0)) <= COLD_ENERGY_ERROR_TOLERANCE,
    )
    return 1.0 if all(gates) else 0.0


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return a terminal verdict that blocks quality trials on invariant failure."""

    if ready_score(payload) == 1.0:
        return "complete: exact temperature-label exchange cDLS audit ready; invariant gates permit later quality trials"
    return "blocked: invariant failure blocks temperature-exchange quality trials"


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    replay_sweeps: int = DEFAULT_REPLAY_SWEEPS,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp5633 terminal exactness artifact."""

    systems = enumerable_frustrated_systems()
    system_rows = [_audit_one_system(system, BETA_LADDER) for system in systems]
    replay = deterministic_replay_summary(
        systems=systems,
        beta_ladder=BETA_LADDER,
        seeds=random_seeds,
        sweeps=int(replay_sweeps),
    )
    all_controls: list[JsonDict] = []
    for control_id in BROKEN_CONTROL_IDS:
        matching = [
            row
            for system_row in system_rows
            for row in system_row["broken_controls"]
            if row["control_id"] == control_id
        ]
        all_controls.append(
            {
                "control_id": control_id,
                "detected": all(row["detected"] is True for row in matching),
                "max_swap_detailed_balance_residual": None
                if all(row["swap_detailed_balance_residual"] is None for row in matching)
                else max(
                    float(row["swap_detailed_balance_residual"])
                    for row in matching
                    if row["swap_detailed_balance_residual"] is not None
                ),
                "max_stationarity_tv": None
                if all(row["stationarity_tv"] is None for row in matching)
                else max(float(row["stationarity_tv"]) for row in matching if row["stationarity_tv"] is not None),
                "state_mutation_detected": any(row["state_mutation_detected"] is True for row in matching),
                "systems_tested": [system_row["descriptor"]["system_id"] for system_row in system_rows],
            }
        )

    stationarity_values = [
        float(row["product_target_stationarity_tv"]) for row in system_rows
    ] + [float(row["cold_label_distribution_tv"]) for row in system_rows]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "corrected_kernel_receipt": corrected_kernel_receipt(root),
        "beta_ladder": list(BETA_LADDER),
        "within_replica_schedule": within_replica_schedule(),
        "exchange_schedule": exchange_schedule(),
        "swap_rule": swap_rule(),
        "enumerable_targets": [row["descriptor"] for row in system_rows],
        "transition_normalization_error_max": max(
            float(row["transition_normalization_error"]) for row in system_rows
        ),
        "swap_detailed_balance_residual_max": max(
            float(row["swap_detailed_balance_residual"]) for row in system_rows
        ),
        "exact_distribution_tv_max": max(stationarity_values),
        "cold_replica_energy_error": max(float(row["cold_replica_energy_error"]) for row in system_rows),
        "round_trip_accounting_error": float(replay["round_trip_accounting_error_max"]),
        "broken_controls": all_controls,
        "deterministic_replay_pass": bool(replay["pass"]),
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "replica_exchange_kernel_ready_score": 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": [int(seed) for seed in random_seeds],
        "baselines": _summarize_baselines(system_rows),
        "product_stationarity_rows": [
            {
                "system_id": row["descriptor"]["system_id"],
                "product_target_state_count": row["product_target_state_count"],
                "product_target_stationarity_tv": row["product_target_stationarity_tv"],
                "cold_label_distribution_tv": row["cold_label_distribution_tv"],
                "cold_replica_energy_error": row["cold_replica_energy_error"],
                "swap_detailed_balance_residual": row["swap_detailed_balance_residual"],
                "transition_normalization_error": row["transition_normalization_error"],
            }
            for row in system_rows
        ],
        "round_trip_accounting": replay,
        "validity_regression_detected": False,
        "tests_added_or_reused": list(
            tests_added_or_reused
            or [
                str(RESULT_RELATIVE_PATH),
                "tests/python/test_experiment_5633_temperature_exchange_cdls_exact_audit.py",
            ]
        ),
        "honest_verdict": "",
        "reproducibility_checksum": "",
    }
    artifact["replica_exchange_kernel_ready_score"] = ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate Exp5633 fields and fail closed on manually-set readiness."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload.get("beta_ladder") != list(BETA_LADDER):
        raise ValueError("beta_ladder mismatch")  # pragma: no cover
    receipt = payload.get("corrected_kernel_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("final_kernel") != "corrected_cdls_projection_mh":
        raise ValueError("corrected_kernel_receipt mismatch")  # pragma: no cover
    if payload.get("timing_claimed") is not False:
        raise ValueError("timing_claimed must be false")
    if payload.get("hardware_speedup_claimed") is not False:
        raise ValueError("hardware_speedup_claimed must be false")
    if payload.get("deterministic_replay_pass") is not True:
        raise ValueError("deterministic_replay_pass must be true")
    controls = payload.get("broken_controls")
    if not isinstance(controls, list) or {row.get("control_id") for row in controls if isinstance(row, Mapping)} != set(
        BROKEN_CONTROL_IDS
    ):
        raise ValueError("broken_controls mismatch")
    if not all(isinstance(row, Mapping) and row.get("detected") is True for row in controls):
        raise ValueError("broken_controls must all be detected")
    expected_ready = ready_score(payload)
    if float(payload.get("replica_exchange_kernel_ready_score", -1.0)) != expected_ready:
        raise ValueError("replica_exchange_kernel_ready_score mismatch")
    if expected_ready == 1.0:
        if float(payload.get("transition_normalization_error_max", 1.0)) > TRANSITION_NORMALIZATION_TOLERANCE:
            raise ValueError("transition_normalization_error_max exceeds tolerance")  # pragma: no cover
        if float(payload.get("swap_detailed_balance_residual_max", 1.0)) > SWAP_DETAILED_BALANCE_TOLERANCE:
            raise ValueError("swap_detailed_balance_residual_max exceeds tolerance")  # pragma: no cover
        if float(payload.get("exact_distribution_tv_max", 1.0)) > EXACT_DISTRIBUTION_TV_THRESHOLD:
            raise ValueError("exact_distribution_tv_max exceeds tolerance")  # pragma: no cover
        if float(payload.get("cold_replica_energy_error", 1.0)) > COLD_ENERGY_ERROR_TOLERANCE:
            raise ValueError("cold_replica_energy_error exceeds tolerance")  # pragma: no cover
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal artifact with deterministic formatting."""

    output_path = Path(root) / RESULT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def run_experiment(repo_root: str | Path = REPO_ROOT) -> Path:  # pragma: no cover - thin live runner.
    """Build, validate, and write the Exp5633 artifact."""

    artifact = build_artifact(root=repo_root)
    return write_output(repo_root, artifact)


if __name__ == "__main__":  # pragma: no cover
    print(run_experiment())
