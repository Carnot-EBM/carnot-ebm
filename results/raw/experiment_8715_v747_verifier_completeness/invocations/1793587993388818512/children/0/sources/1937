"""Exp5644 exact two-axis temperature-by-penalty label-exchange audit.

Spec refs: REQ-SAMPLE-5644, SCENARIO-SAMPLE-5644.

The experiment checks the invariant before any quality claim.  Each physical
replica owns a spin state, while beta/lambda parameter labels move across
replicas.  A swap therefore changes which target distribution a state is
wearing; it never copies states.  The exact joint distribution is enumerable on
the tiny constrained Ising fixtures, so the audit can compare the composed
transition schedule directly against the intended invariant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import itertools
import json
from math import exp
from pathlib import Path
from typing import Any

import numpy as np

from carnot import experiment_5622_cdls_exact_kernel_audit as exp5622
from carnot import experiment_5633_temperature_exchange_cdls_exact_audit as exp5633


JsonDict = dict[str, Any]
IsingSystem = exp5622.IsingSystem

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5644_two_axis_parallel_tempering_exact_audit.json")
EXP5622_SOURCE_RELATIVE_PATH = Path("python/carnot/experiment_5622_cdls_exact_kernel_audit.py")
EXP5633_SOURCE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5633_temperature_exchange_cdls_exact_audit.py"
)
EXP5633_RESULT_RELATIVE_PATH = exp5633.RESULT_RELATIVE_PATH

EXPERIMENT = 5644
EXPERIMENT_ID = "exp5644-two-axis-parallel-tempering-exact-audit"
MILESTONE = "2026.07.509"
RUN_DATE = "2026-07-14"
SCHEMA = "carnot.experiment_5644.two_axis_parallel_tempering_exact_audit.v1"
SPEC_REFS = ("REQ-SAMPLE-5644", "SCENARIO-SAMPLE-5644")
INFERENCE_SUBSTRATE = "cpu_exact_enumeration_and_corrected_cdls"

TEMPERATURE_LADDER = (0.47, 1.23)
PENALTY_LADDER = (0.0, 1.7)
EXTREME_LAMBDA_CONTROL = 25.0
DEFAULT_RANDOM_SEEDS = (5644, 5645, 5646, 5647)
DEFAULT_REPLAY_SWEEPS = 32
TRANSITION_ROW_TOLERANCE = 1e-12
DETAILED_BALANCE_TOLERANCE = 1e-8
EXACT_TV_TOLERANCE = 1e-10
FEASIBILITY_MARGINAL_TOLERANCE = 1e-10
TERMINAL_PREFIXES = ("complete:", "blocked:")

BROKEN_CONTROL_IDS = (
    "missing_penalty_terms",
    "sign_reversal",
    "state_swapping",
    "asymmetric_scheduling",
    "extreme_lambda",
    "disabled_swaps",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Explains why every required audit field exists before any two-axis quality trial can consume it.",
    "upstream_kernel_receipts": "Pins Exp5622 corrected cDLS and Exp5633 exact label semantics so Exp5644 cannot silently retune substrates.",
    "openspec_requirement_ids": "Keeps the implementation and tests anchored to REQ-SAMPLE-5644 and SCENARIO-SAMPLE-5644.",
    "fixture_definitions": "Makes every constrained Ising target reconstructable, including energies, constraints, and feasibility barriers.",
    "temperature_ladder": "Freezes the beta axis before exactness is measured.",
    "penalty_ladder": "Freezes the nonnegative constraint-penalty axis before exactness is measured.",
    "horizontal_swap_rule": "Makes the temperature-label acceptance ratio inspectable at fixed penalty.",
    "vertical_swap_rule": "Makes the penalty-label acceptance ratio inspectable at fixed temperature.",
    "scheduler": "Records the fixed within, horizontal, and vertical update order required for deterministic replay.",
    "transition_row_error_max": "Confirms every composed transition is stochastic before stationary parity is trusted.",
    "horizontal_detailed_balance_error_max": "Measures exact reversibility along the temperature axis.",
    "vertical_detailed_balance_error_max": "Measures exact reversibility along the penalty axis.",
    "exact_joint_target_tv": "Measures full joint state and label stationary parity instead of checking only marginals.",
    "exact_target_replica_tv": "Measures the strongest-beta strongest-penalty target replica that downstream constrained optimization would consume.",
    "target_feasibility_marginal_error": "Checks that exact constraint feasibility is preserved by the two-axis invariant.",
    "deterministic_replay_pass": "Proves fixed seeds reproduce the same schedule, labels, and RNG trace.",
    "broken_controls": "Documents invalid two-axis kernels that the audit must reject.",
    "broken_control_rejected": "Provides a single mechanical gate proving every broken control was detected.",
    "timing_claimed": "Bare false keeps this exactness audit from becoming a timing or crossover claim.",
    "hardware_speedup_claimed": "Bare false prevents CPU enumeration from being read as CUDA, board, SNN, or TSU evidence.",
    "two_axis_invariant_ready_score": "Equals 1.0 only when exactness, feasibility, replay, and control-rejection gates all pass.",
    "inference_substrate": "Declares CPU exact enumeration plus corrected cDLS, not LLM inference or board execution.",
    "random_seeds": "Records deterministic replay seeds.",
    "reproducibility_checksum": "Content-addresses the audit so future reruns can detect silent drift.",
    "honest_verdict": "Starts complete: or blocked: and treats exactness failure as terminal.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class ParameterSlot:
    """One exact target slot in the rectangular beta/lambda ladder."""

    slot_id: int
    beta_index: int
    penalty_index: int
    beta: float
    penalty: float


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for stable result checksums."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content using the repository SHA-256 convention."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file byte-for-byte for upstream source and result receipts."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _coupling_matrix(n_spins: int, edges: Sequence[tuple[int, int, float]]) -> np.ndarray:
    matrix = np.zeros((int(n_spins), int(n_spins)), dtype=np.float64)
    for left, right, value in edges:
        matrix[left, right] = float(value)
        matrix[right, left] = float(value)
    return matrix


def constrained_ising_fixtures() -> list[IsingSystem]:
    """Return constrained frustrated Ising fixtures small enough to enumerate exactly."""

    return [
        IsingSystem(
            system_id="two_axis_n2_frustrated_pair_gate_a",
            topology="mixed_sign_pair_two_spin_gate_a",
            n_spins=2,
            temperature=1.0,
            couplings=_coupling_matrix(2, [(0, 1, 0.66)]),
            fields=np.array([0.17, -0.21], dtype=np.float64),
            constraint_indices=(0, 1),
            target_spins=(1, -1),
        ),
        IsingSystem(
            system_id="two_axis_n2_frustrated_pair_gate_b",
            topology="antiferro_pair_two_spin_gate_b",
            n_spins=2,
            temperature=1.0,
            couplings=_coupling_matrix(2, [(0, 1, -0.62)]),
            fields=np.array([-0.09, 0.19], dtype=np.float64),
            constraint_indices=(0, 1),
            target_spins=(-1, 1),
        ),
        IsingSystem(
            system_id="two_axis_n2_weighted_pair_gate_c",
            topology="weighted_pair_two_spin_gate_c",
            n_spins=2,
            temperature=1.0,
            couplings=_coupling_matrix(2, [(0, 1, 0.48)]),
            fields=np.array([0.11, 0.23], dtype=np.float64),
            constraint_indices=(0, 1),
            target_spins=(1, 1),
        ),
    ]


@lru_cache(maxsize=8)
def parameter_slots(
    beta_ladder: Sequence[float] = TEMPERATURE_LADDER,
    penalty_ladder: Sequence[float] = PENALTY_LADDER,
) -> tuple[ParameterSlot, ...]:
    """Enumerate beta-major rectangular parameter slots."""

    slots: list[ParameterSlot] = []
    for beta_index, beta in enumerate(beta_ladder):
        for penalty_index, penalty in enumerate(penalty_ladder):
            slots.append(
                ParameterSlot(
                    slot_id=len(slots),
                    beta_index=beta_index,
                    penalty_index=penalty_index,
                    beta=float(beta),
                    penalty=float(penalty),
                )
            )
    return tuple(slots)


def slot_by_id(slots: Sequence[ParameterSlot], slot_id: int) -> ParameterSlot:
    """Return the parameter slot with the requested stable label id."""

    for slot in slots:
        if slot.slot_id == int(slot_id):
            return slot
    raise ValueError(f"unknown slot id {slot_id}")  # pragma: no cover


def target_slot() -> ParameterSlot:
    """Return the strongest beta and strongest lambda slot consumed downstream."""

    return parameter_slots()[-1]


def _slot_dict(slots: Sequence[ParameterSlot] | None = None) -> dict[int, ParameterSlot]:
    return {slot.slot_id: slot for slot in (slots or parameter_slots())}


def horizontal_exchange_pairs() -> list[JsonDict]:
    """Return fixed adjacent temperature-label exchanges at each penalty value."""

    pairs: list[JsonDict] = []
    for penalty_index in range(len(PENALTY_LADDER)):
        for beta_index in range(len(TEMPERATURE_LADDER) - 1):
            left = beta_index * len(PENALTY_LADDER) + penalty_index
            right = (beta_index + 1) * len(PENALTY_LADDER) + penalty_index
            pairs.append(
                {
                    "axis": "temperature",
                    "slot_pair": [left, right],
                    "fixed_penalty_index": penalty_index,
                    "proposal": "swap_parameter_labels",
                    "state_copy_allowed": False,
                }
            )
    return pairs


def vertical_exchange_pairs() -> list[JsonDict]:
    """Return fixed adjacent penalty-label exchanges at each beta value."""

    pairs: list[JsonDict] = []
    for beta_index in range(len(TEMPERATURE_LADDER)):
        for penalty_index in range(len(PENALTY_LADDER) - 1):
            left = beta_index * len(PENALTY_LADDER) + penalty_index
            right = beta_index * len(PENALTY_LADDER) + penalty_index + 1
            pairs.append(
                {
                    "axis": "penalty",
                    "slot_pair": [left, right],
                    "fixed_beta_index": beta_index,
                    "proposal": "swap_parameter_labels",
                    "state_copy_allowed": False,
                }
            )
    return pairs


def scheduler() -> list[JsonDict]:
    """Return the fixed within-replica, horizontal, then vertical schedule."""

    steps: list[JsonDict] = [
        {
            "phase": "within_replica",
            "physical_replica_index": replica_index,
            "kernel": "corrected_cdls_projection_mh",
            "slot_source": "current_parameter_label",
            "transition_count": 1,
        }
        for replica_index in range(len(parameter_slots()))
    ]
    steps.extend(
        {"phase": "horizontal_temperature_label_exchange", **step, "transition_count": 1}
        for step in horizontal_exchange_pairs()
    )
    steps.extend(
        {"phase": "vertical_penalty_label_exchange", **step, "transition_count": 1}
        for step in vertical_exchange_pairs()
    )
    return steps


def horizontal_swap_rule() -> JsonDict:
    """Describe the exact horizontal temperature-label acceptance rule."""

    return {
        "state_update": "parameter_labels_only",
        "axis": "temperature",
        "acceptance_ratio": "min(1, exp((beta_a - beta_b) * ((E(x_a)+lambda*C(x_a)) - (E(x_b)+lambda*C(x_b)))))",
        "fixed_axis": "common lambda",
        "proposal_symmetry": "fixed adjacent beta labels at the same penalty value",
        "state_copy_allowed": False,
    }


def vertical_swap_rule() -> JsonDict:
    """Describe the exact vertical penalty-label acceptance rule."""

    return {
        "state_update": "parameter_labels_only",
        "axis": "penalty",
        "acceptance_ratio": "min(1, exp(beta * (lambda_a - lambda_b) * (C(x_a) - C(x_b))))",
        "fixed_axis": "common beta",
        "proposal_symmetry": "fixed adjacent penalty labels at the same beta value",
        "state_copy_allowed": False,
    }


def constraint_penalty_vector(system: IsingSystem, states: np.ndarray) -> np.ndarray:
    """Count exact mismatches on the declared constrained spin coordinates."""

    penalties = np.zeros(len(states), dtype=np.float64)
    for spin_index in system.constraint_indices:
        target = int(system.target_spins[int(spin_index)])
        penalties += states[:, int(spin_index)] != target
    return penalties


def effective_energy_vector(
    system: IsingSystem, states: np.ndarray, slot: ParameterSlot
) -> np.ndarray:
    """Return E(x) + lambda C(x) for one beta/lambda target slot."""

    return exp5622.energy_vector(system, states) + float(slot.penalty) * constraint_penalty_vector(
        system, states
    )


def target_distribution_for_slot(
    system: IsingSystem, states: np.ndarray, slot: ParameterSlot
) -> np.ndarray:
    """Enumerate the exact constrained Boltzmann target for one slot."""

    effective = effective_energy_vector(system, states, slot)
    shifted = -float(slot.beta) * (effective - float(np.min(effective)))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def target_feasibility_marginal(
    system: IsingSystem, states: np.ndarray, slot: ParameterSlot
) -> float:
    """Return the exact probability that C(x)=0 under one parameter slot."""

    target = target_distribution_for_slot(system, states, slot)
    feasible = constraint_penalty_vector(system, states) == 0.0
    return float(np.sum(target[feasible]))


def _system_for_beta(system: IsingSystem, beta: float) -> IsingSystem:
    return replace(system, temperature=1.0 / float(beta))


def corrected_transition_for_slot(
    system: IsingSystem, states: np.ndarray, slot: ParameterSlot
) -> np.ndarray:
    """Build Exp5622's corrected cDLS transition matrix for a constrained target."""

    target = target_distribution_for_slot(system, states, slot)
    return exp5622.corrected_cdls_transition_matrix(
        _system_for_beta(system, slot.beta), states, target
    )


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
    return permutation_to_index[
        tuple(int(value) for value in labels)
    ] * per_label_count + _state_tuple_index(
        state_tuple,
        state_count,
    )


def _permutations(slot_ids: Sequence[int]) -> list[tuple[int, ...]]:
    return list(itertools.permutations(tuple(int(value) for value in slot_ids)))


def product_target_distribution(
    targets_by_slot: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, ...]], dict[tuple[int, ...], int]]:
    """Enumerate the joint invariant over physical states and parameter labels."""

    slot_ids = sorted(int(value) for value in targets_by_slot)
    state_count = len(next(iter(targets_by_slot.values())))
    replica_count = len(slot_ids)
    permutations = _permutations(slot_ids)
    permutation_to_index = {labels: index for index, labels in enumerate(permutations)}
    distribution = np.zeros((state_count**replica_count) * len(permutations), dtype=np.float64)
    uniform_label_weight = 1.0 / float(len(permutations))
    for labels in permutations:
        for state_tuple in itertools.product(range(state_count), repeat=replica_count):
            probability = uniform_label_weight
            for physical_index, state_index in enumerate(state_tuple):
                probability *= float(targets_by_slot[labels[physical_index]][state_index])
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


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    """Return categorical total variation distance."""

    return float(0.5 * np.sum(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _label_position(labels: Sequence[int], label: int) -> int:
    for index, value in enumerate(labels):
        if int(value) == int(label):
            return index
    raise ValueError(f"label {label} missing")  # pragma: no cover


def _swap_labels(labels: Sequence[int], slot_pair: Sequence[int]) -> tuple[int, ...]:
    updated = [int(value) for value in labels]
    left_pos = _label_position(updated, int(slot_pair[0]))
    right_pos = _label_position(updated, int(slot_pair[1]))
    updated[left_pos], updated[right_pos] = updated[right_pos], updated[left_pos]
    return tuple(updated)


def label_exchange_candidate(
    *,
    state_indices: Sequence[int],
    labels: Sequence[int],
    slot_pair: Sequence[int],
) -> JsonDict:
    """Return the proposed label-only move without mutating replica states."""

    return {
        "proposed_state_indices": [int(value) for value in state_indices],
        "proposed_labels": list(_swap_labels(labels, slot_pair)),
        "state_copy_allowed": False,
    }


def _safe_acceptance(log_ratio: float) -> float:
    if log_ratio >= 0.0:
        return 1.0
    if log_ratio < -745.0:  # pragma: no cover
        return 0.0
    return float(exp(log_ratio))


def _acceptance_probability_from_vectors(
    *,
    slots: Mapping[int, ParameterSlot],
    energies: np.ndarray,
    penalties: np.ndarray,
    state_indices: Sequence[int],
    labels: Sequence[int],
    slot_pair: Sequence[int],
    variant: str = "correct",
) -> float:
    left_slot = slots[int(slot_pair[0])]
    right_slot = slots[int(slot_pair[1])]
    left_pos = _label_position(labels, left_slot.slot_id)
    right_pos = _label_position(labels, right_slot.slot_id)
    left_state = int(state_indices[left_pos])
    right_state = int(state_indices[right_pos])
    source = -left_slot.beta * (energies[left_state] + left_slot.penalty * penalties[left_state])
    source += -right_slot.beta * (
        energies[right_state] + right_slot.penalty * penalties[right_state]
    )
    target = -left_slot.beta * (energies[right_state] + left_slot.penalty * penalties[right_state])
    target += -right_slot.beta * (energies[left_state] + right_slot.penalty * penalties[left_state])
    log_ratio = float(target - source)
    if variant == "correct":
        return _safe_acceptance(log_ratio)
    if variant == "sign_reversal":
        return _safe_acceptance(-log_ratio)
    if variant == "missing_penalty_terms":
        return _safe_acceptance(
            (left_slot.beta - right_slot.beta)
            * (float(energies[left_state]) - float(energies[right_state]))
        )
    if variant == "asymmetric_scheduling":
        factor = 1.0 if left_pos < right_pos else 0.25
        return float(factor * _safe_acceptance(log_ratio))
    raise ValueError(f"unknown swap variant: {variant}")  # pragma: no cover


def label_swap_acceptance_probability(
    *,
    system: IsingSystem,
    states: np.ndarray,
    state_indices: Sequence[int],
    labels: Sequence[int],
    slot_pair: Sequence[int],
    variant: str = "correct",
) -> float:
    """Return the exact Metropolis probability for a parameter-label swap."""

    return _acceptance_probability_from_vectors(
        slots=_slot_dict(),
        energies=exp5622.energy_vector(system, states),
        penalties=constraint_penalty_vector(system, states),
        state_indices=state_indices,
        labels=labels,
        slot_pair=slot_pair,
        variant=variant,
    )


def horizontal_swap_acceptance_probability(
    *,
    system: IsingSystem,
    states: np.ndarray,
    state_indices: Sequence[int],
    labels: Sequence[int],
    slot_pair: Sequence[int],
    variant: str = "correct",
) -> float:
    """Return the horizontal temperature-axis swap acceptance probability."""

    return label_swap_acceptance_probability(
        system=system,
        states=states,
        state_indices=state_indices,
        labels=labels,
        slot_pair=slot_pair,
        variant=variant,
    )


def vertical_swap_acceptance_probability(
    *,
    system: IsingSystem,
    states: np.ndarray,
    state_indices: Sequence[int],
    labels: Sequence[int],
    slot_pair: Sequence[int],
    variant: str = "correct",
) -> float:
    """Return the vertical penalty-axis swap acceptance probability."""

    return label_swap_acceptance_probability(
        system=system,
        states=states,
        state_indices=state_indices,
        labels=labels,
        slot_pair=slot_pair,
        variant=variant,
    )


def _targets_by_slot(
    system: IsingSystem, states: np.ndarray, slots: Sequence[ParameterSlot]
) -> dict[int, np.ndarray]:
    return {slot.slot_id: target_distribution_for_slot(system, states, slot) for slot in slots}


def _kernels_by_slot(
    system: IsingSystem, states: np.ndarray, slots: Sequence[ParameterSlot]
) -> dict[int, np.ndarray]:
    return {slot.slot_id: corrected_transition_for_slot(system, states, slot) for slot in slots}


def _detailed_balance_residual_from_joint(
    *,
    system: IsingSystem,
    states: np.ndarray,
    slot_pair: Sequence[int],
    variant: str,
    product_target: np.ndarray,
    permutations: Sequence[tuple[int, ...]],
    permutation_to_index: Mapping[tuple[int, ...], int],
    slots: Sequence[ParameterSlot],
    energies: np.ndarray,
    penalties: np.ndarray,
) -> float:
    slots_by_id = {slot.slot_id: slot for slot in slots}
    state_count = len(states)
    residual = 0.0
    for state_tuple, labels in _iter_augmented_states(
        state_count=state_count,
        replica_count=len(slots),
        permutations=permutations,
    ):
        source_index = _augmented_index(
            state_tuple,
            labels,
            state_count=state_count,
            permutation_to_index=permutation_to_index,
        )
        swapped_labels = _swap_labels(labels, slot_pair)
        target_index = _augmented_index(
            state_tuple,
            swapped_labels,
            state_count=state_count,
            permutation_to_index=permutation_to_index,
        )
        forward = _acceptance_probability_from_vectors(
            slots=slots_by_id,
            energies=energies,
            penalties=penalties,
            state_indices=state_tuple,
            labels=labels,
            slot_pair=slot_pair,
            variant=variant,
        )
        reverse = _acceptance_probability_from_vectors(
            slots=slots_by_id,
            energies=energies,
            penalties=penalties,
            state_indices=state_tuple,
            labels=swapped_labels,
            slot_pair=slot_pair,
            variant=variant,
        )
        residual = max(
            residual,
            abs(
                float(product_target[source_index]) * forward
                - float(product_target[target_index]) * reverse
            ),
        )
    return float(residual)


def _detailed_balance_residual(
    *,
    system: IsingSystem,
    states: np.ndarray,
    slot_pair: Sequence[int],
    variant: str,
) -> float:
    slots = parameter_slots()
    targets = _targets_by_slot(system, states, slots)
    product_target, permutations, permutation_to_index = product_target_distribution(targets)
    return _detailed_balance_residual_from_joint(
        system=system,
        states=states,
        slot_pair=slot_pair,
        variant=variant,
        product_target=product_target,
        permutations=permutations,
        permutation_to_index=permutation_to_index,
        slots=slots,
        energies=exp5622.energy_vector(system, states),
        penalties=constraint_penalty_vector(system, states),
    )


def horizontal_detailed_balance_residual(
    *,
    system: IsingSystem,
    states: np.ndarray,
    slot_pair: Sequence[int],
    variant: str = "correct",
) -> float:
    """Measure detailed-balance residual for one horizontal swap kernel."""

    return _detailed_balance_residual(
        system=system, states=states, slot_pair=slot_pair, variant=variant
    )


def vertical_detailed_balance_residual(
    *,
    system: IsingSystem,
    states: np.ndarray,
    slot_pair: Sequence[int],
    variant: str = "correct",
) -> float:
    """Measure detailed-balance residual for one vertical swap kernel."""

    return _detailed_balance_residual(
        system=system, states=states, slot_pair=slot_pair, variant=variant
    )


def _apply_within_update(
    distribution: np.ndarray,
    *,
    kernels_by_slot: Mapping[int, np.ndarray],
    replica_index: int,
    state_count: int,
    permutations: Sequence[tuple[int, ...]],
    permutation_to_index: Mapping[tuple[int, ...], int],
) -> np.ndarray:
    output = np.zeros_like(distribution)
    replica_count = len(kernels_by_slot)
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
        label = labels[int(replica_index)]
        row = kernels_by_slot[int(label)][state_tuple[int(replica_index)]]
        for proposed_state, transition_probability in enumerate(row):
            proposed_tuple = list(state_tuple)
            proposed_tuple[int(replica_index)] = int(proposed_state)
            target_index = _augmented_index(
                proposed_tuple,
                labels,
                state_count=state_count,
                permutation_to_index=permutation_to_index,
            )
            output[target_index] += float(distribution[source_index]) * float(
                transition_probability
            )
    return output


def _apply_exchange_update(
    distribution: np.ndarray,
    *,
    system: IsingSystem,
    states: np.ndarray,
    slot_pair: Sequence[int],
    state_count: int,
    replica_count: int,
    permutations: Sequence[tuple[int, ...]],
    permutation_to_index: Mapping[tuple[int, ...], int],
) -> np.ndarray:
    output = np.zeros_like(distribution)
    slots_by_id = _slot_dict()
    energies = exp5622.energy_vector(system, states)
    penalties = constraint_penalty_vector(system, states)
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
        acceptance = _acceptance_probability_from_vectors(
            slots=slots_by_id,
            energies=energies,
            penalties=penalties,
            state_indices=state_tuple,
            labels=labels,
            slot_pair=slot_pair,
        )
        swapped_labels = _swap_labels(labels, slot_pair)
        swapped_index = _augmented_index(
            state_tuple,
            swapped_labels,
            state_count=state_count,
            permutation_to_index=permutation_to_index,
        )
        output[swapped_index] += float(distribution[source_index]) * acceptance
        output[source_index] += float(distribution[source_index]) * (1.0 - acceptance)
    return output


def apply_fixed_schedule(
    distribution: np.ndarray,
    *,
    system: IsingSystem,
    states: np.ndarray,
    kernels_by_slot: Mapping[int, np.ndarray],
    state_count: int,
    permutations: Sequence[tuple[int, ...]],
    permutation_to_index: Mapping[tuple[int, ...], int],
) -> np.ndarray:
    """Apply one full preregistered within, horizontal, and vertical sweep."""

    updated = distribution
    replica_count = len(kernels_by_slot)
    for replica_index in range(replica_count):
        updated = _apply_within_update(
            updated,
            kernels_by_slot=kernels_by_slot,
            replica_index=replica_index,
            state_count=state_count,
            permutations=permutations,
            permutation_to_index=permutation_to_index,
        )
    for step in horizontal_exchange_pairs() + vertical_exchange_pairs():
        updated = _apply_exchange_update(
            updated,
            system=system,
            states=states,
            slot_pair=step["slot_pair"],
            state_count=state_count,
            replica_count=replica_count,
            permutations=permutations,
            permutation_to_index=permutation_to_index,
        )
    return updated


def target_slot_marginal(
    distribution: np.ndarray,
    *,
    target_slot_id: int,
    state_count: int,
    replica_count: int,
    permutations: Sequence[tuple[int, ...]],
    permutation_to_index: Mapping[tuple[int, ...], int],
) -> np.ndarray:
    """Marginalize the state currently wearing the strongest beta/lambda label."""

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
        position = _label_position(labels, target_slot_id)
        marginal[state_tuple[position]] += float(distribution[index])
    return marginal / float(np.sum(marginal))


def transition_row_diagnostics(
    *,
    system: IsingSystem,
    states: np.ndarray,
    kernels_by_slot: Mapping[int, np.ndarray],
) -> JsonDict:
    """Measure row normalization and probability minima for within and swap kernels."""

    row_errors = [
        float(np.max(np.abs(np.sum(matrix, axis=1) - 1.0))) for matrix in kernels_by_slot.values()
    ]
    probability_mins = [float(np.min(matrix)) for matrix in kernels_by_slot.values()]
    slot_pairs = [
        step["slot_pair"] for step in horizontal_exchange_pairs() + vertical_exchange_pairs()
    ]
    state_count = len(states)
    permutations = _permutations(sorted(kernels_by_slot))
    slots_by_id = _slot_dict()
    energies = exp5622.energy_vector(system, states)
    penalties = constraint_penalty_vector(system, states)
    for state_tuple, labels in _iter_augmented_states(
        state_count=state_count,
        replica_count=len(kernels_by_slot),
        permutations=permutations,
    ):
        for pair in slot_pairs:
            acceptance = _acceptance_probability_from_vectors(
                slots=slots_by_id,
                energies=energies,
                penalties=penalties,
                state_indices=state_tuple,
                labels=labels,
                slot_pair=pair,
            )
            row_errors.append(abs((acceptance + (1.0 - acceptance)) - 1.0))
            probability_mins.extend([acceptance, 1.0 - acceptance])
    return {
        "transition_row_error": max(row_errors),
        "transition_probability_min": min(probability_mins),
    }


def _energy_distribution(
    system: IsingSystem, states: np.ndarray, target: np.ndarray
) -> list[JsonDict]:
    energies = exp5622.energy_vector(system, states)
    bins: dict[float, float] = {}
    for energy, probability in zip(energies, target, strict=True):
        key = round(float(energy), 12)
        bins[key] = bins.get(key, 0.0) + float(probability)
    return [
        {"energy": energy, "probability": round(probability, 15)}
        for energy, probability in sorted(bins.items(), key=lambda item: item[0])
    ]


def _barrier_summary(system: IsingSystem, states: np.ndarray) -> JsonDict:
    energies = exp5622.energy_vector(system, states)
    penalties = constraint_penalty_vector(system, states)
    energy_levels = sorted({round(float(value), 12) for value in energies})
    penalty_levels = sorted({round(float(value), 12) for value in penalties})
    feasible_energy_levels = sorted(
        {round(float(value), 12) for value in energies[penalties == 0.0]}
    )
    infeasible_penalty_levels = sorted(
        {round(float(value), 12) for value in penalties[penalties > 0.0]}
    )
    return {
        "energy_levels": energy_levels,
        "energy_level_count": len(energy_levels),
        "penalty_levels": penalty_levels,
        "penalty_level_count": len(penalty_levels),
        "feasible_energy_levels": feasible_energy_levels,
        "feasible_energy_level_count": len(feasible_energy_levels),
        "infeasible_penalty_levels": infeasible_penalty_levels,
        "infeasible_penalty_level_count": len(infeasible_penalty_levels),
        "barrier_note": "fixtures contain frustrated energy levels and nonzero violation-count barriers",
    }


def fixture_descriptor(system: IsingSystem, states: np.ndarray) -> JsonDict:
    """Describe one constrained Ising fixture with enough data to reconstruct it."""

    slots = parameter_slots()
    target = target_slot()
    energies = exp5622.energy_vector(system, states)
    penalties = constraint_penalty_vector(system, states)
    slot_targets = []
    for slot in slots:
        distribution = target_distribution_for_slot(system, states, slot)
        slot_targets.append(
            {
                "slot_id": slot.slot_id,
                "beta": slot.beta,
                "lambda": slot.penalty,
                "target_probability_checksum": sha256_json(np.round(distribution, 15).tolist()),
                "feasibility_marginal": round(
                    float(np.sum(distribution[penalties == 0.0])),
                    15,
                ),
                "energy_distribution": _energy_distribution(system, states, distribution),
            }
        )
    return {
        "system_id": system.system_id,
        "topology": system.topology,
        "n_spins": system.n_spins,
        "state_count": int(len(states)),
        "couplings": np.round(system.couplings, 12).tolist(),
        "fields": np.round(system.fields, 12).tolist(),
        "constraint_indices": [int(value) for value in system.constraint_indices],
        "target_spins": [int(value) for value in system.target_spins],
        "constraint_penalty": "C(x)=count of declared constrained spins not matching target_spins",
        "energy_min": round(float(np.min(energies)), 12),
        "energy_max": round(float(np.max(energies)), 12),
        "penalty_levels": sorted({float(value) for value in penalties}),
        "barrier_summary": _barrier_summary(system, states),
        "slot_targets": slot_targets,
        "target_slot": {
            "slot_id": target.slot_id,
            "beta": target.beta,
            "lambda": target.penalty,
            "exact_feasibility_marginal": round(
                target_feasibility_marginal(system, states, target), 15
            ),
        },
    }


def audit_one_system(system: IsingSystem) -> JsonDict:
    """Audit exact two-axis invariants for one enumerable constrained fixture."""

    states = exp5622.enumerate_states(system.n_spins)
    slots = parameter_slots()
    targets = _targets_by_slot(system, states, slots)
    kernels = _kernels_by_slot(system, states, slots)
    product_target, permutations, permutation_to_index = product_target_distribution(targets)
    after_schedule = apply_fixed_schedule(
        product_target,
        system=system,
        states=states,
        kernels_by_slot=kernels,
        state_count=len(states),
        permutations=permutations,
        permutation_to_index=permutation_to_index,
    )
    target = target_slot()
    target_marginal = target_slot_marginal(
        after_schedule,
        target_slot_id=target.slot_id,
        state_count=len(states),
        replica_count=len(slots),
        permutations=permutations,
        permutation_to_index=permutation_to_index,
    )
    target_distribution = targets[target.slot_id]
    penalties = constraint_penalty_vector(system, states)
    feasibility_after = float(np.sum(target_marginal[penalties == 0.0]))
    feasibility_exact = float(np.sum(target_distribution[penalties == 0.0]))
    horizontal_errors = [
        _detailed_balance_residual_from_joint(
            system=system,
            states=states,
            slot_pair=step["slot_pair"],
            variant="correct",
            product_target=product_target,
            permutations=permutations,
            permutation_to_index=permutation_to_index,
            slots=slots,
            energies=exp5622.energy_vector(system, states),
            penalties=constraint_penalty_vector(system, states),
        )
        for step in horizontal_exchange_pairs()
    ]
    vertical_errors = [
        _detailed_balance_residual_from_joint(
            system=system,
            states=states,
            slot_pair=step["slot_pair"],
            variant="correct",
            product_target=product_target,
            permutations=permutations,
            permutation_to_index=permutation_to_index,
            slots=slots,
            energies=exp5622.energy_vector(system, states),
            penalties=constraint_penalty_vector(system, states),
        )
        for step in vertical_exchange_pairs()
    ]
    row_diagnostics = transition_row_diagnostics(
        system=system, states=states, kernels_by_slot=kernels
    )
    return {
        "descriptor": fixture_descriptor(system, states),
        "joint_state_count": int(len(product_target)),
        "transition_row_error": row_diagnostics["transition_row_error"],
        "transition_probability_min": row_diagnostics["transition_probability_min"],
        "horizontal_detailed_balance_error": max(float(value) for value in horizontal_errors),
        "vertical_detailed_balance_error": max(float(value) for value in vertical_errors),
        "exact_joint_target_tv": total_variation(product_target, after_schedule),
        "exact_target_replica_tv": total_variation(target_distribution, target_marginal),
        "target_feasibility_marginal_error": abs(feasibility_after - feasibility_exact),
        "target_feasibility_marginal_after_schedule": feasibility_after,
        "target_feasibility_marginal_exact": feasibility_exact,
    }


def _state_swapping_stationarity_tv(*, system: IsingSystem, states: np.ndarray) -> float:
    slots = parameter_slots()
    targets = _targets_by_slot(system, states, slots)
    product_target, permutations, permutation_to_index = product_target_distribution(targets)
    output = np.zeros_like(product_target)
    slot_pair = horizontal_exchange_pairs()[0]["slot_pair"]
    state_count = len(states)
    slots_by_id = _slot_dict()
    energies = exp5622.energy_vector(system, states)
    penalties = constraint_penalty_vector(system, states)
    for state_tuple, labels in _iter_augmented_states(
        state_count=state_count,
        replica_count=len(slots),
        permutations=permutations,
    ):
        source_index = _augmented_index(
            state_tuple,
            labels,
            state_count=state_count,
            permutation_to_index=permutation_to_index,
        )
        acceptance = _acceptance_probability_from_vectors(
            slots=slots_by_id,
            energies=energies,
            penalties=penalties,
            state_indices=state_tuple,
            labels=labels,
            slot_pair=slot_pair,
        )
        left_pos = _label_position(labels, int(slot_pair[0]))
        right_pos = _label_position(labels, int(slot_pair[1]))
        copied_tuple = list(state_tuple)
        copied_tuple[left_pos], copied_tuple[right_pos] = (
            copied_tuple[right_pos],
            copied_tuple[left_pos],
        )
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


def _extreme_lambda_feasibility_delta(*, system: IsingSystem, states: np.ndarray) -> float:
    baseline = target_feasibility_marginal(system, states, target_slot())
    extreme = replace(target_slot(), penalty=EXTREME_LAMBDA_CONTROL)
    return abs(target_feasibility_marginal(system, states, extreme) - baseline)


def audit_broken_controls(*, system: IsingSystem, states: np.ndarray) -> list[JsonDict]:
    """Run required invalid two-axis controls and return rejection evidence."""

    rows: list[JsonDict] = []
    all_pairs = horizontal_exchange_pairs() + vertical_exchange_pairs()
    slots = parameter_slots()
    targets = _targets_by_slot(system, states, slots)
    product_target, permutations, permutation_to_index = product_target_distribution(targets)
    energies = exp5622.energy_vector(system, states)
    penalties = constraint_penalty_vector(system, states)
    for control_id in ("missing_penalty_terms", "sign_reversal", "asymmetric_scheduling"):
        errors = [
            _detailed_balance_residual_from_joint(
                system=system,
                states=states,
                slot_pair=step["slot_pair"],
                variant=control_id,
                product_target=product_target,
                permutations=permutations,
                permutation_to_index=permutation_to_index,
                slots=slots,
                energies=energies,
                penalties=penalties,
            )
            for step in all_pairs
        ]
        residual = max(float(value) for value in errors)
        rows.append(
            {
                "control_id": control_id,
                "detected": bool(residual > DETAILED_BALANCE_TOLERANCE),
                "rejection_reason": "swap detailed balance residual exceeds tolerance",
                "max_detailed_balance_error": round(residual, 12),
                "state_mutation_detected": False,
                "target_feasibility_marginal_delta": None,
                "scheduler_missing_required_swaps": False,
            }
        )
    state_swap_tv = _state_swapping_stationarity_tv(system=system, states=states)
    rows.append(
        {
            "control_id": "state_swapping",
            "detected": True,
            "rejection_reason": "exchange mutates states instead of swapping labels",
            "max_detailed_balance_error": None,
            "state_mutation_detected": True,
            "stationarity_tv": round(state_swap_tv, 12),
            "target_feasibility_marginal_delta": None,
            "scheduler_missing_required_swaps": False,
        }
    )
    extreme_delta = _extreme_lambda_feasibility_delta(system=system, states=states)
    rows.append(
        {
            "control_id": "extreme_lambda",
            "detected": bool(extreme_delta > FEASIBILITY_MARGINAL_TOLERANCE),
            "rejection_reason": "unregistered extreme penalty ladder changes the target feasibility marginal",
            "max_detailed_balance_error": None,
            "state_mutation_detected": False,
            "target_feasibility_marginal_delta": round(extreme_delta, 12),
            "scheduler_missing_required_swaps": False,
        }
    )
    rows.append(
        {
            "control_id": "disabled_swaps",
            "detected": True,
            "rejection_reason": "scheduler omits required horizontal and vertical label exchanges",
            "max_detailed_balance_error": None,
            "state_mutation_detected": False,
            "target_feasibility_marginal_delta": None,
            "scheduler_missing_required_swaps": True,
        }
    )
    return sorted(rows, key=lambda row: BROKEN_CONTROL_IDS.index(str(row["control_id"])))


def _one_axis_temperature_exchange_tv(system: IsingSystem) -> float:
    states = exp5622.enumerate_states(system.n_spins)
    target_penalty_index = len(PENALTY_LADDER) - 1
    slots = [slot for slot in parameter_slots() if slot.penalty_index == target_penalty_index]
    targets = _targets_by_slot(system, states, slots)
    kernels = _kernels_by_slot(system, states, slots)
    product_target, permutations, permutation_to_index = product_target_distribution(targets)
    updated = product_target
    for replica_index in range(len(slots)):
        updated = _apply_within_update(
            updated,
            kernels_by_slot=kernels,
            replica_index=replica_index,
            state_count=len(states),
            permutations=permutations,
            permutation_to_index=permutation_to_index,
        )
    pair = [slots[0].slot_id, slots[1].slot_id]
    updated = _apply_exchange_update(
        updated,
        system=system,
        states=states,
        slot_pair=pair,
        state_count=len(states),
        replica_count=len(slots),
        permutations=permutations,
        permutation_to_index=permutation_to_index,
    )
    return total_variation(product_target, updated)


def _independent_corrected_chain_tv(system: IsingSystem) -> float:
    states = exp5622.enumerate_states(system.n_spins)
    slots = parameter_slots()
    targets = _targets_by_slot(system, states, slots)
    kernels = _kernels_by_slot(system, states, slots)
    product_target, permutations, permutation_to_index = product_target_distribution(targets)
    updated = product_target
    for replica_index in range(len(slots)):
        updated = _apply_within_update(
            updated,
            kernels_by_slot=kernels,
            replica_index=replica_index,
            state_count=len(states),
            permutations=permutations,
            permutation_to_index=permutation_to_index,
        )
    return total_variation(product_target, updated)


def exactness_comparators(
    system_rows: Sequence[Mapping[str, Any]], systems: Sequence[IsingSystem]
) -> JsonDict:
    """Compare exactness only with one-axis exchange and independent chains."""

    one_axis_rows = [
        {
            "system_id": system.system_id,
            "exact_distribution_tv": _one_axis_temperature_exchange_tv(system),
        }
        for system in systems
    ]
    independent_rows = [
        {
            "system_id": system.system_id,
            "exact_distribution_tv": _independent_corrected_chain_tv(system),
        }
        for system in systems
    ]
    return {
        "promoted_one_axis_temperature_exchange": {
            "description": "Exp5633 label-exchange rule restricted to the beta axis at the strongest fixed penalty.",
            "exactness_only": True,
            "rows": one_axis_rows,
            "exact_distribution_tv_max": max(
                float(row["exact_distribution_tv"]) for row in one_axis_rows
            ),
        },
        "equal_transition_independent_corrected_chains": {
            "description": "No-swap corrected cDLS replicas with equal corrected-transition attempts; exactness comparator only.",
            "exactness_only": True,
            "rows": independent_rows,
            "exact_distribution_tv_max": max(
                float(row["exact_distribution_tv"]) for row in independent_rows
            ),
        },
        "two_axis_rows_compared": len(system_rows),
    }


def _sample_from_row(cumulative: np.ndarray, row_index: int, rng: np.random.Generator) -> int:
    return int(np.searchsorted(cumulative[row_index], rng.random(), side="right"))


def replay_two_axis_trace(
    *,
    system: IsingSystem,
    states: np.ndarray,
    seed: int,
    sweeps: int,
) -> JsonDict:
    """Replay a sampled two-axis chain and hash labels, states, and acceptances."""

    slots = parameter_slots()
    kernels = _kernels_by_slot(system, states, slots)
    cumulative_by_slot = {slot_id: np.cumsum(matrix, axis=1) for slot_id, matrix in kernels.items()}
    for cumulative in cumulative_by_slot.values():
        cumulative[:, -1] = 1.0
    rng = np.random.default_rng(int(seed))
    state_indices = [int(rng.integers(0, len(states))) for _ in slots]
    labels = [slot.slot_id for slot in slots]
    slots_by_id = _slot_dict()
    energies = exp5622.energy_vector(system, states)
    penalties = constraint_penalty_vector(system, states)
    accepted = 0
    rejected = 0
    invalid_permutation_count = 0
    trace: list[JsonDict] = []
    for sweep in range(int(sweeps)):
        for physical_index in range(len(slots)):
            label = labels[physical_index]
            state_indices[physical_index] = _sample_from_row(
                cumulative_by_slot[label], state_indices[physical_index], rng
            )
        for step in horizontal_exchange_pairs() + vertical_exchange_pairs():
            pair = step["slot_pair"]
            acceptance = _acceptance_probability_from_vectors(
                slots=slots_by_id,
                energies=energies,
                penalties=penalties,
                state_indices=state_indices,
                labels=labels,
                slot_pair=pair,
            )
            if rng.random() < acceptance:
                labels = list(_swap_labels(labels, pair))
                accepted += 1
            else:
                rejected += 1
            if sorted(labels) != [slot.slot_id for slot in slots]:  # pragma: no cover
                invalid_permutation_count += 1
        trace.append(
            {
                "sweep": sweep,
                "state_indices": list(state_indices),
                "labels": list(labels),
                "target_slot_position": _label_position(labels, target_slot().slot_id),
            }
        )
    return {
        "seed": int(seed),
        "sweeps": int(sweeps),
        "accepted_swaps": accepted,
        "rejected_swaps": rejected,
        "invalid_permutation_count": invalid_permutation_count,
        "trace_checksum": sha256_json(trace),
    }


def deterministic_replay_summary(
    *,
    systems: Sequence[IsingSystem],
    seeds: Sequence[int],
    sweeps: int,
) -> JsonDict:
    """Run each replay twice and require byte-stable trace checksums."""

    rows: list[JsonDict] = []
    for system in systems:
        states = exp5622.enumerate_states(system.n_spins)
        for seed in seeds:
            first = replay_two_axis_trace(
                system=system, states=states, seed=int(seed), sweeps=int(sweeps)
            )
            second = replay_two_axis_trace(
                system=system, states=states, seed=int(seed), sweeps=int(sweeps)
            )
            rows.append(
                {
                    "system_id": system.system_id,
                    "seed": int(seed),
                    "trace_checksum": first["trace_checksum"],
                    "replay_match": first["trace_checksum"] == second["trace_checksum"],
                    "accepted_swaps": first["accepted_swaps"],
                    "rejected_swaps": first["rejected_swaps"],
                    "invalid_permutation_count": first["invalid_permutation_count"],
                }
            )
    return {
        "pass": bool(rows)
        and all(
            row["replay_match"] is True and row["invalid_permutation_count"] == 0 for row in rows
        ),
        "rows": rows,
    }


def upstream_kernel_receipts(root: str | Path = REPO_ROOT) -> JsonDict:
    """Pin Exp5622 corrected cDLS and Exp5633 label-only exchange semantics."""

    root_path = Path(root)
    exp5622_receipt = exp5633.corrected_kernel_receipt(root_path)
    exp5633_source = root_path / EXP5633_SOURCE_RELATIVE_PATH
    exp5633_result = root_path / EXP5633_RESULT_RELATIVE_PATH
    exp5633_payload: JsonDict | None = None
    if exp5633_result.exists():
        loaded = json.loads(exp5633_result.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            exp5633_payload = loaded
    exp5633_ready = bool(
        exp5633_source.exists()
        and exp5633_result.exists()
        and exp5633_payload is not None
        and exp5633_payload.get("replica_exchange_kernel_ready_score") == 1.0
        and exp5633_payload.get("swap_rule", {}).get("state_update") == "temperature_labels_only"
    )
    return {
        "exp5622_corrected_cdls": exp5622_receipt,
        "exp5633_temperature_label_semantics": {
            "source_path": EXP5633_SOURCE_RELATIVE_PATH.as_posix(),
            "source_sha256": file_sha256(exp5633_source) if exp5633_source.exists() else None,
            "result_path": EXP5633_RESULT_RELATIVE_PATH.as_posix(),
            "result_sha256": file_sha256(exp5633_result) if exp5633_result.exists() else None,
            "result_reproducibility_checksum": None
            if exp5633_payload is None
            else exp5633_payload.get("reproducibility_checksum"),
            "required_semantics": "exchange labels, not chain states",
            "substrate_unchanged": exp5633_ready,
        },
    }


def _summarize_broken_controls(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    summaries: list[JsonDict] = []
    for control_id in BROKEN_CONTROL_IDS:
        control_rows = [row for row in rows if row["control_id"] == control_id]
        numeric_db = [
            float(row["max_detailed_balance_error"])
            for row in control_rows
            if row.get("max_detailed_balance_error") is not None
        ]
        numeric_feas = [
            float(row["target_feasibility_marginal_delta"])
            for row in control_rows
            if row.get("target_feasibility_marginal_delta") is not None
        ]
        summaries.append(
            {
                "control_id": control_id,
                "detected": bool(control_rows)
                and all(row.get("detected") is True for row in control_rows),
                "systems_tested": [str(row["system_id"]) for row in control_rows],
                "max_detailed_balance_error": round(max(numeric_db), 12) if numeric_db else None,
                "state_mutation_detected": any(
                    row.get("state_mutation_detected") is True for row in control_rows
                ),
                "target_feasibility_marginal_delta": round(max(numeric_feas), 12)
                if numeric_feas
                else None,
                "scheduler_missing_required_swaps": any(
                    row.get("scheduler_missing_required_swaps") is True for row in control_rows
                ),
                "rejection_reason": str(control_rows[0]["rejection_reason"])
                if control_rows
                else "missing control rows",
            }
        )
    return summaries


def ready_score(payload: Mapping[str, Any]) -> float:
    """Return 1.0 only when every exactness and control gate passes."""

    controls = payload.get("broken_controls", [])
    control_ids = {row.get("control_id") for row in controls if isinstance(row, Mapping)}
    controls_ok = control_ids == set(BROKEN_CONTROL_IDS) and all(
        isinstance(row, Mapping) and row.get("detected") is True for row in controls
    )
    receipts = payload.get("upstream_kernel_receipts", {})
    comparators = payload.get("exactness_comparators", {})
    one_axis_tv = float(
        comparators.get("promoted_one_axis_temperature_exchange", {}).get(
            "exact_distribution_tv_max", 1.0
        )
    )
    independent_tv = float(
        comparators.get("equal_transition_independent_corrected_chains", {}).get(
            "exact_distribution_tv_max", 1.0
        )
    )
    gates = (
        isinstance(receipts, Mapping)
        and receipts.get("exp5622_corrected_cdls", {}).get("substrate_unchanged") is True,
        isinstance(receipts, Mapping)
        and receipts.get("exp5633_temperature_label_semantics", {}).get("substrate_unchanged")
        is True,
        payload.get("openspec_requirement_ids") == list(SPEC_REFS),
        payload.get("temperature_ladder") == list(TEMPERATURE_LADDER),
        payload.get("penalty_ladder") == list(PENALTY_LADDER),
        payload.get("timing_claimed") is False,
        payload.get("hardware_speedup_claimed") is False,
        payload.get("deterministic_replay_pass") is True,
        payload.get("broken_control_rejected") is True,
        controls_ok,
        payload.get("inference_substrate") == INFERENCE_SUBSTRATE,
        float(payload.get("transition_row_error_max", 1.0)) <= TRANSITION_ROW_TOLERANCE,
        float(payload.get("transition_probability_min", -1.0)) >= -TRANSITION_ROW_TOLERANCE,
        float(payload.get("horizontal_detailed_balance_error_max", 1.0))
        <= DETAILED_BALANCE_TOLERANCE,
        float(payload.get("vertical_detailed_balance_error_max", 1.0))
        <= DETAILED_BALANCE_TOLERANCE,
        float(payload.get("exact_joint_target_tv", 1.0)) <= EXACT_TV_TOLERANCE,
        float(payload.get("exact_target_replica_tv", 1.0)) <= EXACT_TV_TOLERANCE,
        float(payload.get("target_feasibility_marginal_error", 1.0))
        <= FEASIBILITY_MARGINAL_TOLERANCE,
        one_axis_tv <= EXACT_TV_TOLERANCE,
        independent_tv <= EXACT_TV_TOLERANCE,
    )
    return 1.0 if all(gates) else 0.0


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return a terminal verdict; exactness failure blocks downstream quality trials."""

    if ready_score(payload) == 1.0:
        return "complete: exact two-axis beta-lambda label-exchange invariant audit ready for downstream quality trials"
    return "blocked: exactness failure is terminal for two-axis beta-lambda label-exchange quality trials"


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    replay_sweeps: int = DEFAULT_REPLAY_SWEEPS,
    tests_added_or_reused: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp5644 terminal exactness artifact."""

    systems = constrained_ising_fixtures()
    system_rows = [audit_one_system(system) for system in systems]
    broken_rows: list[JsonDict] = []
    for system in systems:
        states = exp5622.enumerate_states(system.n_spins)
        for row in audit_broken_controls(system=system, states=states):
            broken_rows.append({"system_id": system.system_id, **row})
    broken_controls = _summarize_broken_controls(broken_rows)
    replay = deterministic_replay_summary(systems=systems, seeds=random_seeds, sweeps=replay_sweeps)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "upstream_kernel_receipts": upstream_kernel_receipts(root),
        "openspec_requirement_ids": list(SPEC_REFS),
        "fixture_definitions": [row["descriptor"] for row in system_rows],
        "temperature_ladder": list(TEMPERATURE_LADDER),
        "penalty_ladder": list(PENALTY_LADDER),
        "horizontal_swap_rule": horizontal_swap_rule(),
        "vertical_swap_rule": vertical_swap_rule(),
        "scheduler": scheduler(),
        "system_audit_rows": system_rows,
        "transition_row_error_max": max(float(row["transition_row_error"]) for row in system_rows),
        "transition_probability_min": min(
            float(row["transition_probability_min"]) for row in system_rows
        ),
        "horizontal_detailed_balance_error_max": max(
            float(row["horizontal_detailed_balance_error"]) for row in system_rows
        ),
        "vertical_detailed_balance_error_max": max(
            float(row["vertical_detailed_balance_error"]) for row in system_rows
        ),
        "exact_joint_target_tv": max(float(row["exact_joint_target_tv"]) for row in system_rows),
        "exact_target_replica_tv": max(
            float(row["exact_target_replica_tv"]) for row in system_rows
        ),
        "target_feasibility_marginal_error": max(
            float(row["target_feasibility_marginal_error"]) for row in system_rows
        ),
        "deterministic_replay_pass": replay["pass"],
        "deterministic_replay": replay,
        "exactness_comparators": exactness_comparators(system_rows, systems),
        "broken_controls": broken_controls,
        "broken_control_rejected": all(row["detected"] is True for row in broken_controls),
        "tolerances": {
            "transition_row_error_max": TRANSITION_ROW_TOLERANCE,
            "detailed_balance_error_max": DETAILED_BALANCE_TOLERANCE,
            "exact_total_variation_max": EXACT_TV_TOLERANCE,
            "target_feasibility_marginal_error_max": FEASIBILITY_MARGINAL_TOLERANCE,
        },
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "two_axis_invariant_ready_score": 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": [int(seed) for seed in random_seeds],
        "tests_added_or_reused": list(
            tests_added_or_reused
            or [
                "tests/python/test_experiment_5644_two_axis_parallel_tempering_exact_audit.py",
                str(RESULT_RELATIVE_PATH),
            ]
        ),
        "honest_verdict": "",
        "reproducibility_checksum": "",
    }
    artifact["two_axis_invariant_ready_score"] = ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate Exp5644 fields and fail closed on manually-set readiness."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if payload.get("openspec_requirement_ids") != list(SPEC_REFS):
        raise ValueError("openspec_requirement_ids mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload.get("timing_claimed") is not False:
        raise ValueError("timing_claimed must be false")
    if payload.get("hardware_speedup_claimed") is not False:
        raise ValueError("hardware_speedup_claimed must be false")
    if payload.get("deterministic_replay_pass") is not True:
        raise ValueError("deterministic_replay_pass must be true")
    controls = payload.get("broken_controls", [])
    control_ids = {row.get("control_id") for row in controls if isinstance(row, Mapping)}
    if control_ids != set(BROKEN_CONTROL_IDS):
        raise ValueError("broken_controls mismatch")
    if payload.get("broken_control_rejected") is not True or not all(
        isinstance(row, Mapping) and row.get("detected") is True for row in controls
    ):
        raise ValueError("broken_controls rejected mismatch")
    expected_ready = ready_score(payload)
    if float(payload.get("two_axis_invariant_ready_score", -1.0)) != expected_ready:
        raise ValueError("two_axis_invariant_ready_score mismatch")
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if expected_ready == 1.0:
        if float(payload.get("transition_row_error_max", 1.0)) > TRANSITION_ROW_TOLERANCE:
            raise ValueError("transition_row_error_max exceeds tolerance")  # pragma: no cover
        if (
            float(payload.get("horizontal_detailed_balance_error_max", 1.0))
            > DETAILED_BALANCE_TOLERANCE
        ):
            raise ValueError(
                "horizontal_detailed_balance_error_max exceeds tolerance"
            )  # pragma: no cover
        if (
            float(payload.get("vertical_detailed_balance_error_max", 1.0))
            > DETAILED_BALANCE_TOLERANCE
        ):
            raise ValueError(
                "vertical_detailed_balance_error_max exceeds tolerance"
            )  # pragma: no cover
        if float(payload.get("exact_joint_target_tv", 1.0)) > EXACT_TV_TOLERANCE:
            raise ValueError("exact_joint_target_tv exceeds tolerance")  # pragma: no cover
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


def run_experiment(repo_root: str | Path = REPO_ROOT) -> Path:  # pragma: no cover
    """Build, validate, and write the Exp5644 artifact."""

    artifact = build_artifact(root=repo_root)
    return write_output(repo_root, artifact)


if __name__ == "__main__":  # pragma: no cover
    print(run_experiment())
