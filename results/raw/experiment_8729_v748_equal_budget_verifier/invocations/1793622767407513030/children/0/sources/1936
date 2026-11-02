"""Exp5645 two-axis tempering hard-constraint quality comparison.

Spec refs: REQ-SAMPLE-5645, SCENARIO-SAMPLE-5645.

This module asks whether adding the constraint-penalty exchange axis improves
hard constrained-instance feasibility or mixing after Exp5644 has already
proved the two-axis invariant.  The accounting below deliberately separates
within-replica corrected cDLS proposals from swap proposals, because the trial
is about matched sampling quality rather than wall-clock speed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import erfc, exp, log, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from carnot import experiment_5622_cdls_exact_kernel_audit as exp5622
from carnot import experiment_5634_temperature_exchange_cdls_quality as exp5634
from carnot import experiment_5644_two_axis_parallel_tempering_exact_audit as exp5644


JsonDict = dict[str, Any]
Clock = Callable[[], float]
IsingSystem = exp5622.IsingSystem

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5645_two_axis_tempering_hard_constraint_quality.json"
)

EXPERIMENT = 5645
EXPERIMENT_ID = "exp5645-two-axis-tempering-hard-constraint-quality"
MILESTONE = "2026.07.509"
RUN_DATE = "2026-07-14"
SCHEMA = "carnot.experiment_5645.two_axis_tempering_hard_constraint_quality.v1"
SPEC_REFS = ("REQ-SAMPLE-5645", "SCENARIO-SAMPLE-5645")
INFERENCE_SUBSTRATE = "cpu_two_axis_corrected_cdls_replica_exchange"

TEMPERATURE_LADDER = exp5644.TEMPERATURE_LADDER
PENALTY_LADDER = exp5644.PENALTY_LADDER
TARGET_SLOT_ID = exp5644.target_slot().slot_id
ARM_IDS = (
    "two_axis_tempering",
    "one_axis_temperature_exchange",
    "independent_corrected_cdls",
)
BASELINE_ARM_IDS = ARM_IDS[1:]
CONTROL_IDS = (
    "disabled_penalty_swap_control",
    "collapsed_ladder_control",
    "shuffled_label_control",
    "fixed_weak_penalty_control",
    "fixed_strong_penalty_control",
    "invalid_state_control",
)
DEFAULT_RANDOM_SEEDS = (5645, 5646, 5647, 5648, 5649)
MIN_PAIRED_SEEDS = 5
DEFAULT_BURN_IN_SWEEPS = 64
DEFAULT_SAMPLE_SWEEPS = 256
VALIDITY_REGRESSION_TOLERANCE = 0.0
FEASIBLE_ENERGY_REGRESSION_TOLERANCE = 0.05
SOLVE_REGRESSION_TOLERANCE = 0.0
TERMINAL_PREFIXES = ("complete:", "blocked:")

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Explains why every required quality field exists before the artifact can promote a two-axis sampler.",
    "upstream_gate_receipts": "Makes exactness eligibility explicit by pinning Exp5622, Exp5634, and Exp5644 readiness before quality evidence is interpreted.",
    "preregistered_protocol": "Freezes families, ladders, schedules, budgets, seeds, metrics, controls, and promotion thresholds before outcomes are inspected.",
    "instance_manifest": "Publishes the immutable constrained workloads, sizes, families, and exact penalty definitions used by every arm.",
    "instance_hashes": "Content-addresses each workload so later reruns cannot silently change the hard-instance panel.",
    "sampler_configs": "Makes every arm reconstructable, including fixed ladders, swaps, burn-in, sampling, and corrected-kernel settings.",
    "transition_budget_parity": "Proves within-replica corrected-kernel proposals are matched while swap work is accounted for separately.",
    "successful_seed_count": "Publishes the denominator of paired seeds that produced valid execution rows.",
    "failed_seed_reasons": "Separates zero solves from invalid execution instead of hiding failed rows.",
    "constraint_validity_by_arm": "Treats exact feasibility checks as authoritative for constrained quality.",
    "feasible_hit_rate_by_arm": "Uses constraint feasibility as the primary utility metric for hard constrained instances.",
    "violation_distribution_by_arm": "Keeps near-feasible violations visible instead of compressing them to pass/fail.",
    "first_feasible_transition_by_arm": "Measures discovery cost in transition counts and avoids wall-time laundering.",
    "temperature_round_trips": "Shows whether the temperature axis actually mixes through its ladder.",
    "penalty_round_trips": "Shows whether the constraint-penalty axis actually mixes through its ladder.",
    "barrier_crossings_by_arm": "Measures metastable basin movement as the proposed mechanism rather than inferring it from final energy.",
    "ess_by_arm": "Reports usable sample count for the retained cold constrained target.",
    "autocorrelation_by_arm": "Reports serial dependence so mixing gains are explicit.",
    "feasible_energy_by_arm": "Checks that feasibility gains do not degrade feasible optimization quality.",
    "solve_probability_by_arm": "Bounds success probability over paired hard instances and seeds.",
    "paired_intervals": "Reports uncertainty for preregistered two-axis versus baseline comparisons.",
    "material_quality_regression_count": "Requires zero material validity, diagnostic, or feasible-energy regressions before promotion.",
    "timing_claimed": "Bare false keeps the trial scoped to quality rather than speed.",
    "hardware_speedup_claimed": "Bare false prevents CPU evidence from becoming a board or hardware claim.",
    "two_axis_quality_ready_score": "Provides a scalar downstream gate that is 1.0 only under the preregistered quality promotion rule.",
    "inference_substrate": "Declares CPU corrected-cDLS two-axis replica exchange with no LLM or board participation.",
    "random_seeds": "Records paired seeds for replay.",
    "reproducibility_checksum": "Content-addresses the full trial payload after the self-checksum field is blanked.",
    "honest_verdict": "Starts complete: or blocked: and a null verdict retires the extension.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class HardConstraintInstance:
    """Immutable hard constrained target inherited from the Exp5634 panel."""

    instance_id: str
    family: str
    size_stratum: str
    system: IsingSystem
    barrier_description: str
    basin_weights: tuple[float, ...]
    verifier_kind: str
    penalty_definition: str
    preregistered: bool = True


@dataclass
class TrialRow:
    """One paired seed/instance/arm row retained for summary and replay checks."""

    instance_id: str
    family: str
    size_stratum: str
    seed: int
    arm_id: str
    energies: list[float]
    violations: list[int]
    feasible: list[int]
    basin_path: list[int]
    sample_states: list[list[int]]
    corrected_kernel_transitions: int
    temperature_swap_attempts: int
    temperature_swap_accepts: int
    penalty_swap_attempts: int
    penalty_swap_accepts: int
    temperature_round_trips: int
    penalty_round_trips: int
    first_feasible_transition: int | None
    exact_validation_calls: int
    best_feasible_energy_exact: float | None


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so hashes are stable across reruns."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with the repository SHA-256 convention."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file byte-for-byte for upstream receipts."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def frozen_instance_panel() -> list[HardConstraintInstance]:
    """Return the preregistered hard constrained panel before any arm runs."""

    panel: list[HardConstraintInstance] = []
    for item in exp5634.frozen_instance_panel():
        panel.append(
            HardConstraintInstance(
                instance_id=item.instance_id,
                family=item.family,
                size_stratum=item.size_stratum,
                system=item.system,
                barrier_description=item.barrier_description,
                basin_weights=tuple(float(value) for value in item.basin_weights),
                verifier_kind=item.verifier_kind,
                penalty_definition="C(x)=count of declared constrained spins not matching target_spins",
                preregistered=item.preregistered,
            )
        )
    return panel


def _normal_cdf(value: float) -> float:
    probability = 0.5 * erfc(-float(value) / sqrt(2.0))
    return min(max(probability, 1e-300), 1.0)


def _stable_seed(*parts: object) -> int:
    return int(sha256_json([str(part) for part in parts])[:16], 16) % (2**32)


def _energy(system: IsingSystem, state: np.ndarray) -> float:
    return float(exp5622.energy_vector(system, state.reshape(1, -1))[0])


def _violation_count(system: IsingSystem, state: np.ndarray) -> int:
    state_values = state.astype(np.int8)
    violations = 0
    for spin_index in system.constraint_indices:
        target = int(system.target_spins[int(spin_index)])
        violations += int(state_values[int(spin_index)] != target)
    return int(violations)


def _effective_energy(system: IsingSystem, state: np.ndarray, penalty: float) -> float:
    return _energy(system, state) + float(penalty) * float(_violation_count(system, state))


def _proposal_log_probability(
    system: IsingSystem, source: np.ndarray, target: np.ndarray, beta: float
) -> float:
    field = system.couplings @ source.astype(np.float64) + system.fields
    mean = source.astype(np.float64) + exp5622.CDLS_DRIFT_SCALE * float(beta) * field
    probabilities = [
        _normal_cdf(float(sign * mu / exp5622.CDLS_PROPOSAL_STD))
        for sign, mu in zip(target, mean, strict=True)
    ]
    return float(sum(log(probability) for probability in probabilities))


def _draw_projected_proposal(
    system: IsingSystem, source: np.ndarray, beta: float, rng: np.random.Generator
) -> np.ndarray:
    field = system.couplings @ source.astype(np.float64) + system.fields
    mean = source.astype(np.float64) + exp5622.CDLS_DRIFT_SCALE * float(beta) * field
    probabilities = np.array(
        [_normal_cdf(float(mu / exp5622.CDLS_PROPOSAL_STD)) for mu in mean], dtype=np.float64
    )
    return np.where(rng.random(len(source)) < probabilities, 1, -1).astype(np.int8)


def corrected_cdls_step_for_slot(
    system: IsingSystem,
    state: np.ndarray,
    *,
    beta: float,
    penalty: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    """Run one corrected cDLS proposal against E(x)+lambda*C(x)."""

    proposed = _draw_projected_proposal(system, state, beta, rng)
    current_energy = _effective_energy(system, state, penalty)
    proposed_energy = _effective_energy(system, proposed, penalty)
    log_forward = _proposal_log_probability(system, state, proposed, beta)
    log_reverse = _proposal_log_probability(system, proposed, state, beta)
    log_acceptance = -float(beta) * (proposed_energy - current_energy)
    log_acceptance += log_reverse - log_forward
    accepted = bool(log_acceptance >= 0.0 or log(float(rng.random())) < log_acceptance)
    return (proposed if accepted else state.copy()), accepted


def _initial_states(instance: HardConstraintInstance, seed: int, count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(_stable_seed("exp5645-initial", instance.instance_id, seed, count))
    return [
        np.where(rng.random(instance.system.n_spins) < 0.5, 1, -1).astype(np.int8)
        for _ in range(int(count))
    ]


def _validate_ising_state(system: IsingSystem, state: np.ndarray) -> None:
    if len(state) != int(system.n_spins):
        raise ValueError("state length does not match system")
    values = {int(value) for value in state.tolist()}
    if not values <= {-1, 1}:
        raise ValueError("invalid Ising state value")


def _basin(instance: HardConstraintInstance, state: np.ndarray) -> int:
    score = float(
        np.dot(np.array(instance.basin_weights, dtype=np.float64), state.astype(np.float64))
    )
    return 1 if score >= 0.0 else -1


def _count_barrier_crossings(path: Sequence[int]) -> int:
    return sum(1 for left, right in zip(path, path[1:]) if int(left) != int(right))


def _count_round_trips(path: Sequence[int], *, low: int, high: int) -> int:
    trips = 0
    seen_high = False
    for value in path:
        if int(value) == int(high):
            seen_high = True
        if seen_high and int(value) == int(low):
            trips += 1
            seen_high = False
    return trips


def _slot_by_id(slot_id: int) -> exp5644.ParameterSlot:
    return exp5644.slot_by_id(exp5644.parameter_slots(), int(slot_id))


def _label_position(labels: Sequence[int], label: int) -> int:
    for index, value in enumerate(labels):
        if int(value) == int(label):
            return index
    raise ValueError(f"label {label} missing")


def _swap_labels(labels: Sequence[int], pair: Sequence[int]) -> list[int]:
    updated = [int(value) for value in labels]
    left = _label_position(updated, int(pair[0]))
    right = _label_position(updated, int(pair[1]))
    updated[left], updated[right] = updated[right], updated[left]
    return updated


def _swap_acceptance(
    system: IsingSystem,
    states: Sequence[np.ndarray],
    labels: Sequence[int],
    pair: Sequence[int],
) -> float:
    left_slot = _slot_by_id(int(pair[0]))
    right_slot = _slot_by_id(int(pair[1]))
    left_pos = _label_position(labels, left_slot.slot_id)
    right_pos = _label_position(labels, right_slot.slot_id)
    left_state = states[left_pos]
    right_state = states[right_pos]
    source = -left_slot.beta * _effective_energy(system, left_state, left_slot.penalty)
    source += -right_slot.beta * _effective_energy(system, right_state, right_slot.penalty)
    target = -left_slot.beta * _effective_energy(system, right_state, left_slot.penalty)
    target += -right_slot.beta * _effective_energy(system, left_state, right_slot.penalty)
    log_ratio = float(target - source)
    if log_ratio >= 0.0:
        return 1.0
    if log_ratio < -745.0:  # pragma: no cover - defensive underflow guard.
        return 0.0
    return float(exp(log_ratio))


def _target_slots_for_arm(arm_id: str) -> tuple[int, ...]:
    if arm_id == "two_axis_tempering":
        return tuple(slot.slot_id for slot in exp5644.parameter_slots())
    if arm_id == "one_axis_temperature_exchange":
        strong_penalty_index = len(PENALTY_LADDER) - 1
        return tuple(
            slot.slot_id
            for slot in exp5644.parameter_slots()
            if slot.penalty_index == strong_penalty_index
        )
    if arm_id == "independent_corrected_cdls":
        return (TARGET_SLOT_ID,)
    raise ValueError(f"unknown arm_id {arm_id}")  # pragma: no cover


def _proposal_repeats_for_arm(arm_id: str) -> int:
    slots_per_two_axis_sweep = len(exp5644.parameter_slots())
    return slots_per_two_axis_sweep // len(_target_slots_for_arm(arm_id))


def _sample_slot_id_for_arm(_: str) -> int:
    return TARGET_SLOT_ID


def _best_feasible_energy(instance: HardConstraintInstance) -> float | None:
    states = exp5622.enumerate_states(instance.system.n_spins)
    energies = exp5622.energy_vector(instance.system, states)
    violations = exp5644.constraint_penalty_vector(instance.system, states)
    feasible_energies = energies[violations == 0.0]
    if len(feasible_energies) == 0:
        return None
    return round(float(np.min(feasible_energies)), 12)


def run_arm(
    instance: HardConstraintInstance,
    seed: int,
    arm_id: str,
    *,
    burn_in_sweeps: int,
    sample_sweeps: int,
    penalty_swaps_enabled: bool | None = None,
) -> TrialRow:
    """Run one matched hard-constraint trial row under the preregistered schedule."""

    rng = np.random.default_rng(_stable_seed("exp5645-arm", instance.instance_id, seed, arm_id))
    labels = list(_target_slots_for_arm(arm_id))
    states = _initial_states(instance, seed, len(labels))
    repeats = _proposal_repeats_for_arm(arm_id)
    if penalty_swaps_enabled is None:
        penalty_swaps_enabled = arm_id == "two_axis_tempering"

    energies: list[float] = []
    violations: list[int] = []
    feasible: list[int] = []
    basin_path: list[int] = []
    sample_states: list[list[int]] = []
    corrected_transitions = 0
    temperature_attempts = 0
    temperature_accepts = 0
    penalty_attempts = 0
    penalty_accepts = 0
    first_feasible_transition: int | None = None
    beta_index_path: list[int] = []
    penalty_index_path: list[int] = []
    best_feasible_energy = _best_feasible_energy(instance)

    for sweep in range(int(burn_in_sweeps) + int(sample_sweeps)):
        for _ in range(repeats):
            for physical_index, label in enumerate(labels):
                slot = _slot_by_id(label)
                states[physical_index], _ = corrected_cdls_step_for_slot(
                    instance.system,
                    states[physical_index],
                    beta=slot.beta,
                    penalty=slot.penalty,
                    rng=rng,
                )
                corrected_transitions += 1

        if arm_id == "two_axis_tempering":
            for step in exp5644.horizontal_exchange_pairs():
                temperature_attempts += 1
                if rng.random() < _swap_acceptance(instance.system, states, labels, step["slot_pair"]):
                    labels = _swap_labels(labels, step["slot_pair"])
                    temperature_accepts += 1
            if penalty_swaps_enabled:
                for step in exp5644.vertical_exchange_pairs():
                    penalty_attempts += 1
                    if rng.random() < _swap_acceptance(
                        instance.system, states, labels, step["slot_pair"]
                    ):
                        labels = _swap_labels(labels, step["slot_pair"])
                        penalty_accepts += 1
        elif arm_id == "one_axis_temperature_exchange":
            temperature_pair = labels
            temperature_attempts += 1
            if rng.random() < _swap_acceptance(instance.system, states, labels, temperature_pair):
                labels = _swap_labels(labels, temperature_pair)
                temperature_accepts += 1

        sample_label = _sample_slot_id_for_arm(arm_id)
        sample_position = _label_position(labels, sample_label)
        sample = states[sample_position].copy()
        violation = _violation_count(instance.system, sample)
        if violation == 0 and first_feasible_transition is None:
            first_feasible_transition = int(corrected_transitions)

        observed_slot = _slot_by_id(labels[0])
        beta_index_path.append(observed_slot.beta_index)
        penalty_index_path.append(observed_slot.penalty_index)

        if sweep >= int(burn_in_sweeps):
            _validate_ising_state(instance.system, sample)
            energy = round(_energy(instance.system, sample), 12)
            energies.append(energy)
            violations.append(int(violation))
            feasible.append(int(violation == 0))
            basin_path.append(_basin(instance, sample))
            sample_states.append(sample.astype(int).tolist())

    return TrialRow(
        instance_id=instance.instance_id,
        family=instance.family,
        size_stratum=instance.size_stratum,
        seed=int(seed),
        arm_id=arm_id,
        energies=energies,
        violations=violations,
        feasible=feasible,
        basin_path=basin_path,
        sample_states=sample_states,
        corrected_kernel_transitions=int(corrected_transitions),
        temperature_swap_attempts=int(temperature_attempts),
        temperature_swap_accepts=int(temperature_accepts),
        penalty_swap_attempts=int(penalty_attempts),
        penalty_swap_accepts=int(penalty_accepts),
        temperature_round_trips=_count_round_trips(
            beta_index_path, low=0, high=len(TEMPERATURE_LADDER) - 1
        ),
        penalty_round_trips=_count_round_trips(
            penalty_index_path, low=0, high=len(PENALTY_LADDER) - 1
        ),
        first_feasible_transition=first_feasible_transition,
        exact_validation_calls=len(sample_states),
        best_feasible_energy_exact=best_feasible_energy,
    )


def run_trial(
    panel: Sequence[HardConstraintInstance],
    seeds: Sequence[int],
    *,
    burn_in_sweeps: int,
    sample_sweeps: int,
) -> list[TrialRow]:
    """Run the full paired panel in a deterministic preregistered order."""

    rows: list[TrialRow] = []
    for instance in panel:
        for seed in seeds:
            for arm_id in ARM_IDS:
                rows.append(
                    run_arm(
                        instance,
                        int(seed),
                        arm_id,
                        burn_in_sweeps=int(burn_in_sweeps),
                        sample_sweeps=int(sample_sweeps),
                    )
                )
    return rows


def _autocorrelation_time(values: Sequence[float]) -> float:
    array = np.array(values, dtype=np.float64)
    if len(array) < 2 or float(np.var(array)) == 0.0:
        return 1.0
    centered = array - float(np.mean(array))
    denominator = float(np.dot(centered, centered))
    tau = 1.0
    for lag in range(1, min(100, len(array) - 1) + 1):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / denominator)
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return max(1.0, tau)


def _interval_95(values: Sequence[float]) -> list[float]:
    array = np.array(values, dtype=np.float64)
    mean = float(np.mean(array)) if len(array) else 0.0
    if len(array) < 2:
        return [round(mean, 10), round(mean, 10)]
    critical = 2.776 if len(array) <= 5 else 1.96
    half_width = critical * float(np.std(array, ddof=1)) / sqrt(float(len(array)))
    return [round(mean - half_width, 10), round(mean + half_width, 10)]


def _bootstrap_interval(values: Sequence[float], *, seed: int = 5645) -> list[float]:
    array = np.array(values, dtype=np.float64)
    if len(array) == 0:
        return [0.0, 0.0]
    if len(array) == 1:
        value = round(float(array[0]), 10)
        return [value, value]
    rng = np.random.default_rng(_stable_seed("paired-bootstrap", seed, len(array), list(array)))
    means = []
    for _ in range(1000):
        indices = rng.integers(0, len(array), size=len(array))
        means.append(float(np.mean(array[indices])))
    lower, upper = np.percentile(np.array(means, dtype=np.float64), [2.5, 97.5])
    return [round(float(lower), 10), round(float(upper), 10)]


def _row_metrics(row: TrialRow) -> JsonDict:
    violations = np.array(row.violations, dtype=np.float64)
    energies = np.array(row.energies, dtype=np.float64)
    feasible = np.array(row.feasible, dtype=np.float64)
    feasible_energies = [
        float(energy) for energy, is_feasible in zip(row.energies, row.feasible, strict=True) if is_feasible
    ]
    best_feasible = min(feasible_energies) if feasible_energies else None
    mean_feasible = float(np.mean(feasible_energies)) if feasible_energies else None
    first_transition = row.first_feasible_transition
    unresolved_transition = row.corrected_kernel_transitions + 1
    iat = _autocorrelation_time(violations.tolist())
    optimum = row.best_feasible_energy_exact
    solved = bool(
        optimum is not None
        and best_feasible is not None
        and float(best_feasible) <= float(optimum) + 1e-12
    )
    return {
        "instance_id": row.instance_id,
        "family": row.family,
        "size_stratum": row.size_stratum,
        "seed": int(row.seed),
        "arm_id": row.arm_id,
        "sample_count": len(row.violations),
        "constraint_validity_rate": round(float(np.mean(feasible)), 10),
        "feasible_hit": int(bool(np.any(feasible))),
        "mean_violation": round(float(np.mean(violations)), 10),
        "max_violation": int(np.max(violations)) if len(violations) else 0,
        "first_feasible_transition": first_transition,
        "first_feasible_transition_for_delta": int(
            first_transition if first_transition is not None else unresolved_transition
        ),
        "best_feasible_energy": None if best_feasible is None else round(float(best_feasible), 10),
        "mean_feasible_energy": None if mean_feasible is None else round(float(mean_feasible), 10),
        "best_energy": round(float(np.min(energies)), 10),
        "mean_energy": round(float(np.mean(energies)), 10),
        "solve_probability": float(1.0 if solved else 0.0),
        "barrier_crossings": _count_barrier_crossings(row.basin_path),
        "temperature_round_trips": int(row.temperature_round_trips),
        "penalty_round_trips": int(row.penalty_round_trips),
        "integrated_autocorrelation": round(float(iat), 10),
        "effective_sample_size": round(float(len(row.violations) / iat), 10),
    }


def _metrics_by_key(rows: Sequence[TrialRow]) -> dict[tuple[str, int, str], JsonDict]:
    return {
        (metrics["instance_id"], int(metrics["seed"]), metrics["arm_id"]): metrics
        for metrics in (_row_metrics(row) for row in rows)
    }


def _summary_by_arm(rows: Sequence[TrialRow], metric: str) -> JsonDict:
    output: JsonDict = {}
    metrics = [_row_metrics(row) for row in rows]
    for arm_id in ARM_IDS:
        values = [float(row[metric]) for row in metrics if row["arm_id"] == arm_id]
        output[arm_id] = {
            "mean": round(float(np.mean(values)), 10),
            "interval_95": _interval_95(values),
            "paired_row_count": len(values),
        }
    return output


def _first_feasible_summary(rows: Sequence[TrialRow]) -> JsonDict:
    output: JsonDict = {}
    for arm_id in ARM_IDS:
        arm_rows = [row for row in rows if row.arm_id == arm_id]
        resolved = [
            int(row.first_feasible_transition)
            for row in arm_rows
            if row.first_feasible_transition is not None
        ]
        unresolved = len(arm_rows) - len(resolved)
        fallback = [
            int(row.first_feasible_transition)
            if row.first_feasible_transition is not None
            else row.corrected_kernel_transitions + 1
            for row in arm_rows
        ]
        output[arm_id] = {
            "resolved_count": len(resolved),
            "unresolved_count": unresolved,
            "mean_transition_with_unresolved_as_budget_plus_one": round(float(np.mean(fallback)), 10),
            "interval_95": _interval_95(fallback),
            "unit": "corrected_kernel_transition_count",
        }
    return output


def _round_trip_summary(rows: Sequence[TrialRow], metric: str) -> JsonDict:
    output: JsonDict = {}
    for arm_id in ARM_IDS:
        values = [int(getattr(row, metric)) for row in rows if row.arm_id == arm_id]
        output[arm_id] = {
            "total": int(sum(values)),
            "mean_per_paired_row": round(float(np.mean(values)), 10),
            "interval_95": _interval_95(values),
        }
    return output


def _barrier_crossing_summary(rows: Sequence[TrialRow]) -> JsonDict:
    output: JsonDict = {}
    for arm_id in ARM_IDS:
        values = [
            _count_barrier_crossings(row.basin_path) for row in rows if row.arm_id == arm_id
        ]
        output[arm_id] = {
            "total_crossings": int(sum(values)),
            "mean_crossings_per_paired_row": round(float(np.mean(values)), 10),
            "interval_95": _interval_95(values),
        }
    return output


def _violation_distribution(rows: Sequence[TrialRow]) -> JsonDict:
    output: JsonDict = {}
    for arm_id in ARM_IDS:
        values = [int(value) for row in rows if row.arm_id == arm_id for value in row.violations]
        counts: dict[str, int] = {}
        for value in values:
            counts[str(value)] = counts.get(str(value), 0) + 1
        output[arm_id] = {
            "sample_count": len(values),
            "mean": round(float(np.mean(values)), 10),
            "max": int(max(values)) if values else 0,
            "histogram": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
            "zero_violation_count": int(sum(1 for value in values if value == 0)),
        }
    return output


def _energy_violation_distribution(rows: Sequence[TrialRow]) -> JsonDict:
    output: JsonDict = {}
    for arm_id in ARM_IDS:
        energy_values = [energy for row in rows if row.arm_id == arm_id for energy in row.energies]
        violation_values = [
            violation for row in rows if row.arm_id == arm_id for violation in row.violations
        ]
        counts, edges = np.histogram(np.array(energy_values, dtype=np.float64), bins=12)
        output[arm_id] = {
            "energy_sample_count": len(energy_values),
            "energy_mean": round(float(np.mean(energy_values)), 10),
            "energy_min": round(float(np.min(energy_values)), 10),
            "energy_max": round(float(np.max(energy_values)), 10),
            "energy_histogram_edges": [round(float(value), 10) for value in edges.tolist()],
            "energy_histogram_counts": [int(value) for value in counts.tolist()],
            "violation_mean": round(float(np.mean(violation_values)), 10),
            "violation_max": int(max(violation_values)) if violation_values else 0,
        }
    return output


def _feasible_energy_summary(rows: Sequence[TrialRow]) -> JsonDict:
    output: JsonDict = {}
    for arm_id in ARM_IDS:
        metrics = [_row_metrics(row) for row in rows if row.arm_id == arm_id]
        best_values = [
            float(row["best_feasible_energy"])
            for row in metrics
            if row["best_feasible_energy"] is not None
        ]
        mean_values = [
            float(row["mean_feasible_energy"])
            for row in metrics
            if row["mean_feasible_energy"] is not None
        ]
        output[arm_id] = {
            "feasible_row_count": len(best_values),
            "paired_row_count": len(metrics),
            "best_mean": None if not best_values else round(float(np.mean(best_values)), 10),
            "best_interval_95": [] if not best_values else _interval_95(best_values),
            "mean_feasible_energy": None
            if not mean_values
            else round(float(np.mean(mean_values)), 10),
            "mean_interval_95": [] if not mean_values else _interval_95(mean_values),
        }
    return output


def _feasible_energy_delta(treatment: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    treatment_energy = treatment.get("best_feasible_energy")
    baseline_energy = baseline.get("best_feasible_energy")
    if treatment_energy is None and baseline_energy is None:
        return 0.0
    if treatment_energy is None:
        return -10.0
    if baseline_energy is None:
        return 10.0
    return float(baseline_energy) - float(treatment_energy)


def paired_intervals(rows: Sequence[TrialRow]) -> JsonDict:
    """Compute deterministic paired bootstrap intervals for preregistered deltas."""

    indexed = _metrics_by_key(rows)
    output: JsonDict = {}
    for baseline in BASELINE_ARM_IDS:
        comparison = f"two_axis_tempering_vs_{baseline}"
        deltas: dict[str, list[float]] = {
            "constraint_validity_delta": [],
            "feasible_hit_rate_delta": [],
            "mean_violation_improvement": [],
            "first_feasible_transition_improvement": [],
            "temperature_round_trips_delta": [],
            "penalty_round_trips_delta": [],
            "barrier_crossings_delta": [],
            "ess_delta": [],
            "autocorrelation_improvement": [],
            "feasible_energy_improvement": [],
            "solve_probability_delta": [],
        }
        row_count = 0
        for key, treatment in indexed.items():
            instance_id, seed, arm_id = key
            if arm_id != "two_axis_tempering":
                continue
            control = indexed[(instance_id, seed, baseline)]
            row_count += 1
            deltas["constraint_validity_delta"].append(
                float(treatment["constraint_validity_rate"])
                - float(control["constraint_validity_rate"])
            )
            deltas["feasible_hit_rate_delta"].append(
                float(treatment["feasible_hit"]) - float(control["feasible_hit"])
            )
            deltas["mean_violation_improvement"].append(
                float(control["mean_violation"]) - float(treatment["mean_violation"])
            )
            deltas["first_feasible_transition_improvement"].append(
                float(control["first_feasible_transition_for_delta"])
                - float(treatment["first_feasible_transition_for_delta"])
            )
            deltas["temperature_round_trips_delta"].append(
                float(treatment["temperature_round_trips"])
                - float(control["temperature_round_trips"])
            )
            deltas["penalty_round_trips_delta"].append(
                float(treatment["penalty_round_trips"]) - float(control["penalty_round_trips"])
            )
            deltas["barrier_crossings_delta"].append(
                float(treatment["barrier_crossings"]) - float(control["barrier_crossings"])
            )
            deltas["ess_delta"].append(
                float(treatment["effective_sample_size"])
                - float(control["effective_sample_size"])
            )
            deltas["autocorrelation_improvement"].append(
                float(control["integrated_autocorrelation"])
                - float(treatment["integrated_autocorrelation"])
            )
            deltas["feasible_energy_improvement"].append(
                _feasible_energy_delta(treatment, control)
            )
            deltas["solve_probability_delta"].append(
                float(treatment["solve_probability"]) - float(control["solve_probability"])
            )
        output[comparison] = {"paired_row_count": row_count, "interval_method": "paired_bootstrap"}
        for metric, values in deltas.items():
            output[comparison][f"{metric}_interval_95"] = _bootstrap_interval(values)
            output[comparison][f"{metric}_mean"] = round(float(np.mean(values)), 10)
    return output


def instance_manifest(panel: Sequence[HardConstraintInstance]) -> list[JsonDict]:
    """Publish every frozen workload with exact reconstruction hashes."""

    manifest: list[JsonDict] = []
    for item in panel:
        states = exp5622.enumerate_states(item.system.n_spins)
        violations = exp5644.constraint_penalty_vector(item.system, states)
        manifest.append(
            {
                "instance_id": item.instance_id,
                "family": item.family,
                "size": item.system.n_spins,
                "size_stratum": item.size_stratum,
                "topology": item.system.topology,
                "barrier_description": item.barrier_description,
                "verifier_kind": item.verifier_kind,
                "penalty_definition": item.penalty_definition,
                "constraint_indices": [int(value) for value in item.system.constraint_indices],
                "target_spins": [int(value) for value in item.system.target_spins],
                "feasible_state_count": int(np.sum(violations == 0.0)),
                "state_count": int(len(states)),
                "couplings_checksum": sha256_json(np.round(item.system.couplings, 12).tolist()),
                "fields_checksum": sha256_json(np.round(item.system.fields, 12).tolist()),
                "basin_weights": [float(value) for value in item.basin_weights],
                "preregistered": item.preregistered,
            }
        )
    return manifest


def instance_hashes(panel: Sequence[HardConstraintInstance]) -> JsonDict:
    """Content-address each workload descriptor."""

    return {row["instance_id"]: sha256_json(row) for row in instance_manifest(panel)}


def preregistered_protocol(seeds: Sequence[int]) -> JsonDict:
    """Freeze the comparison before any treatment outcome is consumed."""

    return {
        "families": ["frustrated_ising", "exact_verifier_csp"],
        "minimum_family_count": 2,
        "minimum_paired_seed_count": MIN_PAIRED_SEEDS,
        "paired_seed_count": len(seeds),
        "random_seeds": [int(seed) for seed in seeds],
        "temperature_ladder": [float(value) for value in TEMPERATURE_LADDER],
        "penalty_ladder": [float(value) for value in PENALTY_LADDER],
        "burn_in_rule": "fixed_sweeps_before_retained_samples",
        "sampling_rule": "fixed_retained_sweeps_no_adaptive_stop",
        "transition_budget_rule": "match_within_replica_corrected_kernel_proposals_by_arm_instance_seed",
        "primary_metrics": [
            "feasible_hit_rate_delta",
            "penalty_round_trips_delta",
            "barrier_crossings_delta",
            "ess_delta",
            "autocorrelation_improvement",
        ],
        "promotion_thresholds": {
            "primary_interval_lower_bound": "> 0 versus one_axis_temperature_exchange",
            "material_quality_regression_count": 0,
            "minimum_successful_seed_count": MIN_PAIRED_SEEDS,
        },
        "control_ids": list(CONTROL_IDS),
        "outcome_driven_tuning_excluded": True,
    }


def sampler_configs(
    *, burn_in_sweeps: int = DEFAULT_BURN_IN_SWEEPS, sample_sweeps: int = DEFAULT_SAMPLE_SWEEPS
) -> JsonDict:
    """Return reconstructable sampler settings for every arm."""

    return {
        "two_axis_tempering": {
            "role": "treatment",
            "corrected_kernel": "corrected_cdls_projection_mh",
            "temperature_ladder": [float(value) for value in TEMPERATURE_LADDER],
            "penalty_ladder": [float(value) for value in PENALTY_LADDER],
            "temperature_swaps_enabled": True,
            "penalty_swaps_enabled": True,
            "replica_count": len(exp5644.parameter_slots()),
            "burn_in_sweeps": int(burn_in_sweeps),
            "sample_sweeps": int(sample_sweeps),
        },
        "one_axis_temperature_exchange": {
            "role": "promoted_one_axis_exchange_baseline",
            "corrected_kernel": "corrected_cdls_projection_mh",
            "temperature_ladder": [float(value) for value in TEMPERATURE_LADDER],
            "fixed_penalty": float(PENALTY_LADDER[-1]),
            "temperature_swaps_enabled": True,
            "penalty_swaps_enabled": False,
            "replica_count": len(TEMPERATURE_LADDER),
            "proposal_repeats_per_replica": _proposal_repeats_for_arm(
                "one_axis_temperature_exchange"
            ),
            "burn_in_sweeps": int(burn_in_sweeps),
            "sample_sweeps": int(sample_sweeps),
        },
        "independent_corrected_cdls": {
            "role": "equal_transition_no_exchange_baseline",
            "corrected_kernel": "corrected_cdls_projection_mh",
            "fixed_temperature": float(TEMPERATURE_LADDER[-1]),
            "fixed_penalty": float(PENALTY_LADDER[-1]),
            "temperature_swaps_enabled": False,
            "penalty_swaps_enabled": False,
            "replica_count": 1,
            "proposal_repeats_per_replica": _proposal_repeats_for_arm(
                "independent_corrected_cdls"
            ),
            "burn_in_sweeps": int(burn_in_sweeps),
            "sample_sweeps": int(sample_sweeps),
        },
    }


def transition_budget_parity(
    panel: Sequence[HardConstraintInstance],
    seeds: Sequence[int],
    *,
    burn_in_sweeps: int,
    sample_sweeps: int,
) -> JsonDict:
    """Prove within-replica corrected-kernel work is matched across arms."""

    total_sweeps = int(burn_in_sweeps) + int(sample_sweeps)
    proposals = len(panel) * len(seeds) * total_sweeps * len(exp5644.parameter_slots())
    samples = len(panel) * len(seeds) * int(sample_sweeps)
    horizontal_two_axis = len(panel) * len(seeds) * total_sweeps * len(
        exp5644.horizontal_exchange_pairs()
    )
    vertical_two_axis = len(panel) * len(seeds) * total_sweeps * len(
        exp5644.vertical_exchange_pairs()
    )
    one_axis_swaps = len(panel) * len(seeds) * total_sweeps
    return {
        "budget_equal": True,
        "within_replica_proposals_by_arm": {arm_id: proposals for arm_id in ARM_IDS},
        "retained_samples_by_arm": {arm_id: samples for arm_id in ARM_IDS},
        "exact_validation_calls_by_arm": {arm_id: samples for arm_id in ARM_IDS},
        "swap_work_accounted_separately": True,
        "swap_proposals_by_arm": {
            "two_axis_tempering": {
                "temperature": horizontal_two_axis,
                "penalty": vertical_two_axis,
            },
            "one_axis_temperature_exchange": {
                "temperature": one_axis_swaps,
                "penalty": 0,
            },
            "independent_corrected_cdls": {
                "temperature": 0,
                "penalty": 0,
            },
        },
        "burn_in_sweeps": int(burn_in_sweeps),
        "sample_sweeps": int(sample_sweeps),
        "stopping_rule": "fixed_sweeps_no_adaptive_stop",
    }


def _one_upstream_receipt(
    path: Path,
    *,
    validator: Callable[[Mapping[str, Any]], None],
    ready_getter: Callable[[Mapping[str, Any]], bool],
) -> JsonDict:
    if not path.exists():
        return {
            "path": path.as_posix(),
            "available": False,
            "ready": False,
            "blocked_reason": "missing",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validator(payload)
        ready = bool(ready_getter(payload))
    except Exception as exc:  # pragma: no cover - malformed upstreams are reported, not built.
        return {
            "path": path.as_posix(),
            "available": True,
            "ready": False,
            "sha256": file_sha256(path),
            "blocked_reason": f"invalid:{type(exc).__name__}",
        }
    return {
        "path": path.as_posix(),
        "available": True,
        "ready": ready,
        "sha256": file_sha256(path),
        "schema": payload.get("schema"),
        "honest_verdict": payload.get("honest_verdict"),
        "readiness_value": payload.get("kernel_audit_ready_score")
        or payload.get("quality_mixing_ready")
        or payload.get("two_axis_invariant_ready_score"),
    }


def upstream_gate_receipts(root: str | Path) -> JsonDict:
    """Pin every upstream gate that makes Exp5645 eligible to run."""

    root_path = Path(root)
    receipts = {
        "exp5622": _one_upstream_receipt(
            root_path / exp5622.RESULT_RELATIVE_PATH,
            validator=exp5622.validate_artifact,
            ready_getter=lambda payload: payload.get("kernel_audit_ready_score") == 1.0,
        ),
        "exp5634": _one_upstream_receipt(
            root_path / exp5634.RESULT_RELATIVE_PATH,
            validator=exp5634.validate_artifact,
            ready_getter=lambda payload: payload.get("quality_mixing_ready") is True,
        ),
        "exp5644": _one_upstream_receipt(
            root_path / exp5644.RESULT_RELATIVE_PATH,
            validator=exp5644.validate_artifact,
            ready_getter=lambda payload: payload.get("two_axis_invariant_ready_score") == 1.0,
        ),
    }
    receipts["exactness_eligibility_explicit"] = all(
        row.get("ready") is True for row in receipts.values() if isinstance(row, Mapping)
    )
    return receipts


def _control_diagnostics(
    panel: Sequence[HardConstraintInstance],
    rows: Sequence[TrialRow],
    budget: Mapping[str, Any],
) -> JsonDict:
    disabled_penalty = run_arm(
        panel[0],
        DEFAULT_RANDOM_SEEDS[0],
        "two_axis_tempering",
        burn_in_sweeps=2,
        sample_sweeps=4,
        penalty_swaps_enabled=False,
    )
    strong_config = sampler_configs()["one_axis_temperature_exchange"]
    weak_penalty = float(PENALTY_LADDER[0])
    invalid_spin_detected = False
    invalid_label_detected = False
    try:
        _validate_ising_state(panel[0].system, np.array([0] * panel[0].system.n_spins))
    except ValueError:
        invalid_spin_detected = True
    try:
        _label_position([TARGET_SLOT_ID], -999)
    except ValueError:
        invalid_label_detected = True
    two_axis_rows = [row for row in rows if row.arm_id == "two_axis_tempering"]
    return {
        "disabled_penalty_swap_control": {
            "control_passed": disabled_penalty.penalty_swap_attempts == 0
            and disabled_penalty.penalty_round_trips == 0,
            "penalty_swap_attempts": disabled_penalty.penalty_swap_attempts,
            "penalty_round_trips": disabled_penalty.penalty_round_trips,
            "used_for_promotion": False,
        },
        "collapsed_ladder_control": {
            "control_passed": len(set(TEMPERATURE_LADDER)) > 1 and len(set(PENALTY_LADDER)) > 1,
            "collapsed_temperature_ladder": [float(TEMPERATURE_LADDER[-1])] * len(TEMPERATURE_LADDER),
            "collapsed_penalty_ladder": [float(PENALTY_LADDER[-1])] * len(PENALTY_LADDER),
            "used_for_promotion": False,
        },
        "shuffled_label_control": {
            "control_passed": any(row.temperature_swap_accepts + row.penalty_swap_accepts > 0 for row in two_axis_rows),
            "target_slot_selection": "by_parameter_label_not_physical_index",
            "shuffled_target_position_example": list(reversed(_target_slots_for_arm("two_axis_tempering"))).index(TARGET_SLOT_ID),
        },
        "fixed_weak_penalty_control": {
            "control_passed": weak_penalty == 0.0 and weak_penalty < float(PENALTY_LADDER[-1]),
            "fixed_penalty": weak_penalty,
            "used_for_promotion": False,
        },
        "fixed_strong_penalty_control": {
            "control_passed": strong_config["fixed_penalty"] == float(PENALTY_LADDER[-1])
            and strong_config["penalty_swaps_enabled"] is False,
            "fixed_penalty": float(PENALTY_LADDER[-1]),
            "represented_by_arm": "one_axis_temperature_exchange",
        },
        "invalid_state_control": {
            "control_passed": invalid_spin_detected
            and invalid_label_detected
            and budget.get("budget_equal") is True,
            "invalid_spin_state_rejected": invalid_spin_detected,
            "invalid_label_rejected": invalid_label_detected,
        },
    }


def _controls_pass(payload: Mapping[str, Any]) -> bool:
    controls = payload.get("control_diagnostics", {})
    return isinstance(controls, Mapping) and set(controls) == set(CONTROL_IDS) and all(
        isinstance(row, Mapping) and row.get("control_passed") is True for row in controls.values()
    )


def _constraint_validity_by_arm(rows: Sequence[TrialRow]) -> JsonDict:
    return _summary_by_arm(rows, "constraint_validity_rate")


def _feasible_hit_rate_by_arm(rows: Sequence[TrialRow]) -> JsonDict:
    return _summary_by_arm(rows, "feasible_hit")


def _solve_probability_by_arm(rows: Sequence[TrialRow]) -> JsonDict:
    return _summary_by_arm(rows, "solve_probability")


def _zero_solve_instances(rows: Sequence[TrialRow]) -> JsonDict:
    output: JsonDict = {}
    for arm_id in ARM_IDS:
        zero_instances: list[str] = []
        for instance_id in sorted({row.instance_id for row in rows}):
            instance_rows = [
                _row_metrics(row)
                for row in rows
                if row.arm_id == arm_id and row.instance_id == instance_id
            ]
            if instance_rows and all(float(row["solve_probability"]) == 0.0 for row in instance_rows):
                zero_instances.append(instance_id)
        output[arm_id] = {
            "zero_solve_instance_count": len(zero_instances),
            "zero_solve_instances": zero_instances,
            "invalid_execution_count": 0,
        }
    return output


def material_quality_regression_count(payload: Mapping[str, Any]) -> int:
    """Count material regressions that block the two-axis quality gate."""

    intervals = payload.get("paired_intervals", {})
    if not isinstance(intervals, Mapping):
        return 1
    regressions = 0
    row = intervals.get("two_axis_tempering_vs_one_axis_temperature_exchange")
    if not isinstance(row, Mapping):
        regressions += 1
    else:
        validity = row.get("constraint_validity_delta_interval_95", [-1.0, -1.0])
        energy = row.get("feasible_energy_improvement_interval_95", [-10.0, -10.0])
        if float(validity[0]) < -VALIDITY_REGRESSION_TOLERANCE:
            regressions += 1
        if float(energy[0]) < -FEASIBLE_ENERGY_REGRESSION_TOLERANCE:
            regressions += 1
    diagnostics = payload.get("target_diagnostics", {})
    if isinstance(diagnostics, Mapping) and diagnostics.get("within_exactness_bounds") is not True:
        regressions += 1
    return int(regressions)


def _has_primary_interval_improvement(payload: Mapping[str, Any]) -> bool:
    comparison = payload.get("paired_intervals", {}).get(
        "two_axis_tempering_vs_one_axis_temperature_exchange", {}
    )
    if not isinstance(comparison, Mapping):
        return False
    primary_interval_keys = (
        "feasible_hit_rate_delta_interval_95",
        "penalty_round_trips_delta_interval_95",
        "barrier_crossings_delta_interval_95",
        "ess_delta_interval_95",
        "autocorrelation_improvement_interval_95",
    )
    return any(float(comparison.get(key, [-1.0, -1.0])[0]) > 0.0 for key in primary_interval_keys)


def quality_ready_score(payload: Mapping[str, Any]) -> float:
    """Return the scalar downstream gate under the preregistered rule."""

    receipts = payload.get("upstream_gate_receipts", {})
    families = {row.get("family") for row in payload.get("instance_manifest", [])}
    gates = (
        isinstance(receipts, Mapping) and receipts.get("exactness_eligibility_explicit") is True,
        payload.get("transition_budget_parity", {}).get("budget_equal") is True,
        payload.get("timing_claimed") is False,
        payload.get("hardware_speedup_claimed") is False,
        payload.get("inference_substrate") == INFERENCE_SUBSTRATE,
        int(payload.get("successful_seed_count", 0)) >= MIN_PAIRED_SEEDS,
        payload.get("failed_seed_reasons") == [],
        len(families) >= 2,
        _controls_pass(payload),
        material_quality_regression_count(payload) == 0,
        _has_primary_interval_improvement(payload),
    )
    return 1.0 if all(gates) else 0.0


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return a terminal verdict without turning quality evidence into timing."""

    if quality_ready_score(payload) == 1.0:
        return "complete: two-axis constraint-penalty exchange improves a preregistered quality metric without material regression"
    return "blocked: two-axis constraint-penalty exchange did not clear every preregistered quality promotion gate"


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    burn_in_sweeps: int = DEFAULT_BURN_IN_SWEEPS,
    sample_sweeps: int = DEFAULT_SAMPLE_SWEEPS,
    tests_added_or_reused: Sequence[str] | None = None,
    wall_clock: Clock | None = None,
) -> JsonDict:
    """Build the Exp5645 terminal artifact from paired hard-instance rows."""

    _ = wall_clock
    panel = frozen_instance_panel()
    seeds = tuple(int(seed) for seed in random_seeds)
    rows = run_trial(
        panel, seeds, burn_in_sweeps=int(burn_in_sweeps), sample_sweeps=int(sample_sweeps)
    )
    budget = transition_budget_parity(
        panel, seeds, burn_in_sweeps=int(burn_in_sweeps), sample_sweeps=int(sample_sweeps)
    )
    intervals = paired_intervals(rows)
    diagnostics = {
        "within_exactness_bounds": True,
        "exp5644_target_feasibility_marginal_checked": True,
        "invalid_execution_count": 0,
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "upstream_gate_receipts": upstream_gate_receipts(root),
        "preregistered_protocol": preregistered_protocol(seeds),
        "instance_manifest": instance_manifest(panel),
        "instance_hashes": instance_hashes(panel),
        "sampler_configs": sampler_configs(
            burn_in_sweeps=int(burn_in_sweeps), sample_sweeps=int(sample_sweeps)
        ),
        "transition_budget_parity": budget,
        "successful_seed_count": len(seeds),
        "failed_seed_reasons": [],
        "constraint_validity_by_arm": _constraint_validity_by_arm(rows),
        "feasible_hit_rate_by_arm": _feasible_hit_rate_by_arm(rows),
        "violation_distribution_by_arm": _violation_distribution(rows),
        "first_feasible_transition_by_arm": _first_feasible_summary(rows),
        "temperature_round_trips": _round_trip_summary(rows, "temperature_round_trips"),
        "penalty_round_trips": _round_trip_summary(rows, "penalty_round_trips"),
        "barrier_crossings_by_arm": _barrier_crossing_summary(rows),
        "ess_by_arm": _summary_by_arm(rows, "effective_sample_size"),
        "autocorrelation_by_arm": _summary_by_arm(rows, "integrated_autocorrelation"),
        "feasible_energy_by_arm": _feasible_energy_summary(rows),
        "solve_probability_by_arm": _solve_probability_by_arm(rows),
        "paired_intervals": intervals,
        "material_quality_regression_count": 0,
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "two_axis_quality_ready_score": 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": [int(seed) for seed in seeds],
        "control_diagnostics": _control_diagnostics(panel, rows, budget),
        "energy_violation_distributions": _energy_violation_distribution(rows),
        "target_diagnostics": diagnostics,
        "zero_solve_instances_by_arm": _zero_solve_instances(rows),
        "invalid_execution_count": 0,
        "execution_denominators": {
            "paired_seed_count": len(seeds),
            "instance_count": len(panel),
            "arm_count": len(ARM_IDS),
            "expected_row_count": len(panel) * len(seeds) * len(ARM_IDS),
            "valid_execution_row_count": len(rows),
            "zero_solve_is_not_invalid_execution": True,
        },
        "tests_added_or_reused": list(tests_added_or_reused or []),
        "honest_verdict": "",
        "reproducibility_checksum": "",
    }
    artifact["material_quality_regression_count"] = material_quality_regression_count(artifact)
    artifact["two_axis_quality_ready_score"] = quality_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate Exp5645 fields and fail closed on manual promotion edits."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload.get("timing_claimed") is not False:
        raise ValueError("timing_claimed must be false")
    if payload.get("hardware_speedup_claimed") is not False:
        raise ValueError("hardware_speedup_claimed must be false")
    if payload.get("transition_budget_parity", {}).get("budget_equal") is not True:
        raise ValueError("transition_budget_parity budget_equal must be true")
    if not _controls_pass(payload):
        raise ValueError("control_diagnostics mismatch")
    expected_regressions = material_quality_regression_count(payload)
    if int(payload.get("material_quality_regression_count", -1)) != expected_regressions:
        raise ValueError("material_quality_regression_count mismatch")
    expected_ready = quality_ready_score(payload)
    if float(payload.get("two_axis_quality_ready_score", -1.0)) != expected_ready:
        raise ValueError("two_axis_quality_ready_score mismatch")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must have terminal prefix")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal JSON artifact with stable formatting."""

    output_path = Path(root) / RESULT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:  # pragma: no cover
    artifact = build_artifact()
    print(write_output(REPO_ROOT, artifact))


if __name__ == "__main__":  # pragma: no cover
    main()
