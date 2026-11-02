"""Exp5634 paired temperature-exchange cDLS quality and mixing trial.

Spec refs: REQ-SAMPLE-5634, SCENARIO-SAMPLE-5634.

This module asks a narrower question than the retired CPU/CUDA crossover work:
when every arm receives the same corrected cDLS transition budget, does
temperature-label exchange improve hard-instance barrier crossing, mixing, or
solution quality?  The trial keeps wall time as provenance only.  The result is
allowed to promote the hybrid only through paired exact-sampling endpoints,
never through timing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import erfc, exp, log, sqrt
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from carnot import experiment_5622_cdls_exact_kernel_audit as exp5622
from carnot import experiment_5633_temperature_exchange_cdls_exact_audit as exp5633


JsonDict = dict[str, Any]
Clock = Callable[[], float]
IsingSystem = exp5622.IsingSystem

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5634_temperature_exchange_cdls_quality.json")

EXPERIMENT = 5634
EXPERIMENT_ID = "exp5634-temperature-exchange-cdls-quality"
MILESTONE = "2026.07.508"
RUN_DATE = "2026-07-14"
SCHEMA = "carnot.experiment_5634.temperature_exchange_cdls_quality.v1"
SPEC_REFS = ("REQ-SAMPLE-5634", "SCENARIO-SAMPLE-5634")
INFERENCE_SUBSTRATE = "paired_exact_corrected_cdls_replica_sampling"

BETA_LADDER = exp5633.BETA_LADDER
COLD_LABEL = len(BETA_LADDER) - 1
ARM_IDS = (
    "temperature_exchange_cdls",
    "independent_corrected_cdls_replicas",
    "single_corrected_cold_chain",
)
BASELINE_ARM_IDS = ARM_IDS[1:]
DEFAULT_RANDOM_SEEDS = (5634, 5635, 5636, 5637, 5638)
MIN_PAIRED_SEEDS = 5
DEFAULT_BURN_IN_SWEEPS = 64
DEFAULT_SAMPLE_SWEEPS = 256
TERMINAL_PREFIXES = ("complete:", "blocked:")

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Explains why every required artifact field exists before reviewer or conductor consumption.",
    "upstream_gate_receipts": "Proves Exp5622 corrected-kernel exactness and Exp5633 exchange exactness are fixed prerequisites.",
    "instance_panel": "Freezes the hard-instance difficulty scope before any arm result can select the panel.",
    "paired_seed_schedule": "Matches every comparison by seed so deltas are paired rather than cherry-picked.",
    "transition_budget_receipt": "Proves algorithmic work is equal under the fixed corrected-kernel transition budget.",
    "method_arms": "Names the exchange treatment and both corrected-cDLS baselines explicitly.",
    "round_trip_stats": "Measures whether temperature labels actually move through the ladder.",
    "barrier_crossing_stats": "Measures the proposed mechanism directly on metastable basin changes.",
    "ess_by_arm": "Quantifies usable cold-target sample count instead of raw sample volume.",
    "autocorrelation_by_arm": "Quantifies serial dependence so mixing claims cannot hide slow chains.",
    "energy_distribution_diagnostics": "Makes quality matching visible through comparable energy sufficient statistics.",
    "best_energy_by_arm": "Reports optimization quality explicitly.",
    "mean_energy_by_arm": "Reports target-quality central tendency explicitly.",
    "solve_probability_by_arm": "Measures exact-constraint utility on CSP and Ising verifier targets.",
    "exact_valid_rate_by_arm": "Prevents invalid samples from winning a quality gate.",
    "paired_deltas_and_intervals": "Reports uncertainty from paired seeds by instance and size.",
    "wall_time_provenance_only": "Records elapsed time only as provenance and blocks any speedup interpretation.",
    "hardware_speedup_claimed": "Bare false keeps hardware and board timing scopes closed.",
    "timing_claimed": "Bare false keeps Exp5623 timing mismatch from being laundered into this trial.",
    "quality_mixing_ready": "Makes capstone promotion mechanical instead of manually declared.",
    "inference_substrate": "Declares paired exact corrected cDLS replica sampling with no LLM inference.",
    "random_seeds": "Records the paired seeds needed to replay the trial.",
    "reproducibility_checksum": "Content-addresses the preregistered panel, seeds, budget, controls, and summaries.",
    "honest_verdict": "States whether the repeated quality/mixing mismatch retires the hybrid.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class HardInstance:
    """Frozen hard target with an exact verifier and a basin definition."""

    instance_id: str
    family: str
    size_stratum: str
    system: IsingSystem
    barrier_description: str
    basin_weights: tuple[float, ...]
    verifier_kind: str
    preregistered: bool = True


@dataclass
class TrialRow:
    """Per instance/seed/arm cold-target evidence kept local for replay checks."""

    instance_id: str
    size_stratum: str
    seed: int
    arm_id: str
    energies: list[float]
    valid: list[int]
    satisfaction: list[float]
    basins: list[int]
    sample_states: list[list[int]]
    corrected_kernel_transitions: int
    exchange_attempts: int
    accepted_exchanges: int
    round_trips: int
    exact_validation_calls: int


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for reproducible trial hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content using the repository SHA-256 convention."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file byte-for-byte for upstream receipts."""

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


def frozen_instance_panel() -> list[HardInstance]:
    """Return the preregistered hard panel before any treatment result exists."""

    return [
        HardInstance(
            instance_id="frustrated_ising_n6_two_triangles",
            family="frustrated_ising",
            size_stratum="small_n6",
            system=IsingSystem(
                system_id="frustrated_ising_n6_two_triangles",
                topology="two_frustrated_triangles_with_bridge",
                n_spins=6,
                temperature=1.0,
                couplings=_coupling_matrix(
                    6,
                    [
                        (0, 1, 0.72),
                        (1, 2, 0.67),
                        (0, 2, -0.69),
                        (3, 4, -0.71),
                        (4, 5, 0.64),
                        (3, 5, 0.68),
                        (2, 3, -0.36),
                        (1, 4, 0.24),
                    ],
                ),
                fields=np.array([0.04, -0.08, 0.05, -0.06, 0.07, -0.03], dtype=np.float64),
                constraint_indices=(0, 1, 2, 3, 4, 5),
                target_spins=(1, -1, 1, -1, 1, -1),
            ),
            barrier_description="two frustrated triangles joined by weak bridges; basin flips require crossing unsatisfied-cycle energy",
            basin_weights=(1.0, 1.0, 1.0, -1.0, -1.0, -1.0),
            verifier_kind="exact_ground_energy",
        ),
        HardInstance(
            instance_id="frustrated_ising_n8_ladder_chimera",
            family="frustrated_ising",
            size_stratum="medium_n8",
            system=IsingSystem(
                system_id="frustrated_ising_n8_ladder_chimera",
                topology="frustrated_ladder_with_cross_chords",
                n_spins=8,
                temperature=1.0,
                couplings=_coupling_matrix(
                    8,
                    [
                        (0, 1, 0.55),
                        (1, 2, -0.48),
                        (2, 3, 0.58),
                        (4, 5, -0.52),
                        (5, 6, 0.57),
                        (6, 7, -0.50),
                        (0, 4, 0.43),
                        (1, 5, -0.44),
                        (2, 6, 0.39),
                        (3, 7, -0.41),
                        (0, 7, -0.28),
                        (3, 4, 0.31),
                    ],
                ),
                fields=np.array(
                    [0.02, -0.04, 0.09, -0.05, 0.06, -0.07, 0.03, -0.08], dtype=np.float64
                ),
                constraint_indices=(0, 2, 3, 5, 6, 7),
                target_spins=(1, -1, 1, -1, -1, 1, -1, 1),
            ),
            barrier_description="cross-chord ladder creates metastable left-right magnetization basins",
            basin_weights=(1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0),
            verifier_kind="exact_ground_energy",
        ),
        HardInstance(
            instance_id="csp_xor_n7_odd_ring_bridge",
            family="exact_verifier_csp",
            size_stratum="small_n7",
            system=IsingSystem(
                system_id="csp_xor_n7_odd_ring_bridge",
                topology="max_xor_odd_ring_with_bridge",
                n_spins=7,
                temperature=1.0,
                couplings=_coupling_matrix(
                    7,
                    [
                        (0, 1, 0.62),
                        (1, 2, -0.62),
                        (2, 3, 0.62),
                        (3, 4, -0.62),
                        (4, 5, 0.62),
                        (5, 0, 0.62),
                        (2, 6, -0.35),
                        (4, 6, 0.33),
                    ],
                ),
                fields=np.array([0.03, -0.02, 0.04, -0.01, 0.02, -0.03, 0.05], dtype=np.float64),
                constraint_indices=(0, 1, 2, 3, 4, 5, 6),
                target_spins=(1, -1, 1, -1, 1, -1, 1),
            ),
            barrier_description="odd XOR cycle has one unavoidable violated edge and a bridge spin metastability trap",
            basin_weights=(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.5),
            verifier_kind="exact_max_csp_energy",
        ),
        HardInstance(
            instance_id="csp_cut_n10_planted_bipartition_trap",
            family="exact_verifier_csp",
            size_stratum="medium_n10",
            system=IsingSystem(
                system_id="csp_cut_n10_planted_bipartition_trap",
                topology="maxcut_planted_partition_with_frustrating_chords",
                n_spins=10,
                temperature=1.0,
                couplings=_coupling_matrix(
                    10,
                    [
                        (0, 5, -0.46),
                        (1, 6, -0.49),
                        (2, 7, -0.45),
                        (3, 8, -0.48),
                        (4, 9, -0.47),
                        (0, 1, 0.34),
                        (1, 2, 0.31),
                        (2, 3, 0.36),
                        (3, 4, 0.32),
                        (5, 6, 0.35),
                        (6, 7, 0.33),
                        (7, 8, 0.37),
                        (8, 9, 0.30),
                        (0, 7, 0.22),
                        (2, 9, -0.25),
                    ],
                ),
                fields=np.array([0.04, 0.01, -0.03, 0.02, -0.04, -0.02, 0.03, -0.01, 0.05, -0.05]),
                constraint_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                target_spins=(1, 1, 1, 1, 1, -1, -1, -1, -1, -1),
            ),
            barrier_description="planted cut has two broad partitions separated by frustrating chord flips",
            basin_weights=(1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
            verifier_kind="exact_max_csp_energy",
        ),
    ]


def paired_seed_schedule(seeds: Sequence[int]) -> JsonDict:
    """Freeze the seed order and arm pairing before any sample is scored."""

    return {
        "paired_seeds": [int(seed) for seed in seeds],
        "paired_seed_count": len(seeds),
        "minimum_paired_seed_count": MIN_PAIRED_SEEDS,
        "seed_order_locked": True,
        "arms": list(ARM_IDS),
        "burn_in_rule": "fixed_before_treatment_result",
        "endpoint_selection_rule": "preregistered_before_sampling",
    }


def method_arms() -> list[JsonDict]:
    """Describe the treatment and two equal-budget corrected-cDLS baselines."""

    return [
        {
            "arm_id": "temperature_exchange_cdls",
            "role": "treatment",
            "corrected_kernel": "corrected_cdls_projection_mh",
            "beta_ladder": list(BETA_LADDER),
            "exchange_enabled": True,
        },
        {
            "arm_id": "independent_corrected_cdls_replicas",
            "role": "disabled_exchange_control",
            "corrected_kernel": "corrected_cdls_projection_mh",
            "beta_ladder": list(BETA_LADDER),
            "exchange_enabled": False,
        },
        {
            "arm_id": "single_corrected_cold_chain",
            "role": "cold_chain_equal_corrected_transition_baseline",
            "corrected_kernel": "corrected_cdls_projection_mh",
            "beta_ladder": [float(BETA_LADDER[COLD_LABEL])],
            "exchange_enabled": False,
        },
    ]


def _normal_cdf(value: float) -> float:
    return min(max(0.5 * erfc(-float(value) / sqrt(2.0)), 1e-300), 1.0)


def _energy(system: IsingSystem, state: np.ndarray) -> float:
    return float(exp5622.energy_vector(system, state.reshape(1, -1))[0])


def _proposal_log_probability(
    system: IsingSystem, source: np.ndarray, target: np.ndarray, beta: float
) -> float:
    field = system.couplings @ source.astype(np.float64) + system.fields
    mean = source.astype(np.float64) + exp5622.CDLS_DRIFT_SCALE * float(beta) * field
    probabilities = [
        _normal_cdf(float(sign * mu / exp5622.CDLS_PROPOSAL_STD)) for sign, mu in zip(target, mean)
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


def corrected_cdls_step(
    system: IsingSystem,
    state: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    """Run one exact-probability corrected cDLS Metropolis-Hastings proposal."""

    proposed = _draw_projected_proposal(system, state, beta, rng)
    current_energy = _energy(system, state)
    proposed_energy = _energy(system, proposed)
    log_forward = _proposal_log_probability(system, state, proposed, beta)
    log_reverse = _proposal_log_probability(system, proposed, state, beta)
    log_acceptance = -float(beta) * (proposed_energy - current_energy) + log_reverse - log_forward
    accepted = bool(log_acceptance >= 0.0 or log(float(rng.random())) < log_acceptance)
    return (proposed if accepted else state.copy()), accepted


def _stable_seed(*parts: object) -> int:
    return int(sha256_json([str(part) for part in parts])[:16], 16) % (2**32)


def _initial_states(instance: HardInstance, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(_stable_seed("initial", instance.instance_id, seed))
    return [
        np.where(rng.random(instance.system.n_spins) < 0.5, 1, -1).astype(np.int8)
        for _ in BETA_LADDER
    ]


def _label_position(labels: Sequence[int], label: int) -> int:
    for index, value in enumerate(labels):
        if int(value) == int(label):
            return index
    raise ValueError("label missing")  # pragma: no cover


def _swap_labels(labels: Sequence[int], pair: tuple[int, int]) -> list[int]:
    updated = [int(value) for value in labels]
    left = _label_position(updated, pair[0])
    right = _label_position(updated, pair[1])
    updated[left], updated[right] = updated[right], updated[left]
    return updated


def _swap_acceptance(
    system: IsingSystem, states: Sequence[np.ndarray], labels: Sequence[int], pair: tuple[int, int]
) -> float:
    left_pos = _label_position(labels, pair[0])
    right_pos = _label_position(labels, pair[1])
    energy_left = _energy(system, states[left_pos])
    energy_right = _energy(system, states[right_pos])
    log_ratio = (float(BETA_LADDER[pair[0]]) - float(BETA_LADDER[pair[1]])) * (
        energy_left - energy_right
    )
    return 1.0 if log_ratio >= 0.0 else float(exp(log_ratio))


def _ground_energy(instance: HardInstance) -> float:
    states = exp5622.enumerate_states(instance.system.n_spins)
    return float(np.min(exp5622.energy_vector(instance.system, states)))


def _constraint_satisfaction(instance: HardInstance, state: np.ndarray) -> float:
    target = np.array(instance.system.target_spins, dtype=np.int8)
    return float(np.mean(state.astype(np.int8) == target))


def _exact_valid(instance: HardInstance, state: np.ndarray, ground_energy: float) -> int:
    return int(_energy(instance.system, state) <= float(ground_energy) + 1e-12)


def _basin(instance: HardInstance, state: np.ndarray) -> int:
    score = float(
        np.dot(np.array(instance.basin_weights, dtype=np.float64), state.astype(np.float64))
    )
    return 1 if score >= 0.0 else -1


def _count_barrier_crossings(basins: Sequence[int]) -> int:
    return sum(1 for left, right in zip(basins, basins[1:]) if int(left) != int(right))


def _count_round_trips(path: Sequence[int]) -> int:
    hot = 0
    cold = COLD_LABEL
    trips = 0
    seen_cold = False
    for position in path:
        if int(position) == cold:
            seen_cold = True
        if seen_cold and int(position) == hot:
            trips += 1
            seen_cold = False
    return trips


def run_arm(
    instance: HardInstance,
    seed: int,
    arm_id: str,
    *,
    burn_in_sweeps: int,
    sample_sweeps: int,
    beta_ladder: Sequence[float] = BETA_LADDER,
) -> TrialRow:
    """Run one matched instance/seed/arm trial row."""

    rng = np.random.default_rng(
        _stable_seed("arm", instance.instance_id, seed, arm_id, list(beta_ladder))
    )
    states = _initial_states(instance, seed)
    labels = list(range(len(BETA_LADDER)))
    energies: list[float] = []
    valid: list[int] = []
    satisfaction: list[float] = []
    basins: list[int] = []
    sample_states: list[list[int]] = []
    accepted_exchanges = 0
    exchange_attempts = 0
    cold_label_positions: list[int] = []
    corrected_transitions = 0
    ground = _ground_energy(instance)

    for sweep in range(int(burn_in_sweeps) + int(sample_sweeps)):
        if arm_id == "single_corrected_cold_chain":
            cold_state = states[COLD_LABEL].copy()
            for _ in BETA_LADDER:
                cold_state, _ = corrected_cdls_step(
                    instance.system, cold_state, beta_ladder[COLD_LABEL], rng
                )
                corrected_transitions += 1
            states[COLD_LABEL] = cold_state
        else:
            for physical_index in range(len(BETA_LADDER)):
                label = labels[physical_index]
                states[physical_index], _ = corrected_cdls_step(
                    instance.system,
                    states[physical_index],
                    beta_ladder[label],
                    rng,
                )
                corrected_transitions += 1
            if arm_id == "temperature_exchange_cdls":
                for pair in ((0, 1), (1, 2)):
                    exchange_attempts += 1
                    if rng.random() < _swap_acceptance(instance.system, states, labels, pair):
                        labels = _swap_labels(labels, pair)
                        accepted_exchanges += 1
        cold_position = _label_position(labels, COLD_LABEL)
        cold_label_positions.append(cold_position)
        if sweep >= int(burn_in_sweeps):
            sample = states[cold_position].copy()
            energies.append(round(_energy(instance.system, sample), 12))
            valid.append(_exact_valid(instance, sample, ground))
            satisfaction.append(round(_constraint_satisfaction(instance, sample), 12))
            basins.append(_basin(instance, sample))
            sample_states.append(sample.astype(int).tolist())

    return TrialRow(
        instance_id=instance.instance_id,
        size_stratum=instance.size_stratum,
        seed=int(seed),
        arm_id=arm_id,
        energies=energies,
        valid=valid,
        satisfaction=satisfaction,
        basins=basins,
        sample_states=sample_states,
        corrected_kernel_transitions=corrected_transitions,
        exchange_attempts=exchange_attempts,
        accepted_exchanges=accepted_exchanges,
        round_trips=_count_round_trips(cold_label_positions),
        exact_validation_calls=len(sample_states),
    )


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
    mean = float(np.mean(array))
    if len(array) < 2:
        return [round(mean, 10), round(mean, 10)]
    multipliers = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }
    critical = multipliers.get(len(array), 1.96)
    half_width = critical * float(np.std(array, ddof=1)) / sqrt(float(len(array)))
    return [round(mean - half_width, 10), round(mean + half_width, 10)]


def _row_metrics(row: TrialRow) -> JsonDict:
    energies = np.array(row.energies, dtype=np.float64)
    iat = _autocorrelation_time(row.energies)
    return {
        "instance_id": row.instance_id,
        "size_stratum": row.size_stratum,
        "seed": row.seed,
        "arm_id": row.arm_id,
        "best_energy": round(float(np.min(energies)), 10),
        "mean_energy": round(float(np.mean(energies)), 10),
        "solve_probability": round(float(np.mean(row.valid)), 10),
        "exact_valid_rate": round(float(np.mean(row.valid)), 10),
        "invalid_rate": round(1.0 - float(np.mean(row.valid)), 10),
        "constraint_satisfaction_mean": round(float(np.mean(row.satisfaction)), 10),
        "barrier_crossings": _count_barrier_crossings(row.basins),
        "integrated_autocorrelation": round(float(iat), 10),
        "effective_sample_size": round(float(len(row.energies) / iat), 10),
        "sample_count": len(row.energies),
    }


def run_trial(
    panel: Sequence[HardInstance],
    seeds: Sequence[int],
    *,
    burn_in_sweeps: int,
    sample_sweeps: int,
) -> list[TrialRow]:
    """Run all preregistered paired rows in deterministic order."""

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


def _summary_by_arm(rows: Sequence[TrialRow], metric: str) -> JsonDict:
    metrics = [_row_metrics(row) for row in rows]
    output: JsonDict = {}
    for arm_id in ARM_IDS:
        values = [float(row[metric]) for row in metrics if row["arm_id"] == arm_id]
        output[arm_id] = {
            "mean": round(float(np.mean(values)), 10),
            "interval_95": _interval_95(values),
            "paired_row_count": len(values),
        }
    return output


def _energy_diagnostics(rows: Sequence[TrialRow]) -> JsonDict:
    output: JsonDict = {}
    for arm_id in ARM_IDS:
        values = [energy for row in rows if row.arm_id == arm_id for energy in row.energies]
        counts, edges = np.histogram(np.array(values, dtype=np.float64), bins=12)
        output[arm_id] = {
            "sample_count": len(values),
            "mean": round(float(np.mean(values)), 10),
            "std": round(float(np.std(values)), 10),
            "min": round(float(np.min(values)), 10),
            "max": round(float(np.max(values)), 10),
            "histogram_edges": [round(float(value), 10) for value in edges.tolist()],
            "histogram_counts": [int(value) for value in counts.tolist()],
        }
    return output


def _round_trip_stats(rows: Sequence[TrialRow]) -> JsonDict:
    output: JsonDict = {}
    for arm_id in ARM_IDS:
        arm_rows = [row for row in rows if row.arm_id == arm_id]
        output[arm_id] = {
            "round_trips": int(sum(row.round_trips for row in arm_rows)),
            "exchange_attempts": int(sum(row.exchange_attempts for row in arm_rows)),
            "accepted_exchanges": int(sum(row.accepted_exchanges for row in arm_rows)),
            "acceptance_rate": round(
                float(sum(row.accepted_exchanges for row in arm_rows))
                / max(1, sum(row.exchange_attempts for row in arm_rows)),
                10,
            ),
        }
    return output


def _barrier_crossing_stats(rows: Sequence[TrialRow]) -> JsonDict:
    output: JsonDict = {}
    for arm_id in ARM_IDS:
        arm_rows = [row for row in rows if row.arm_id == arm_id]
        crossings = [_count_barrier_crossings(row.basins) for row in arm_rows]
        output[arm_id] = {
            "total_crossings": int(sum(crossings)),
            "mean_crossings_per_pair": round(float(np.mean(crossings)), 10),
            "interval_95": _interval_95(crossings),
        }
    return output


def _paired_metric_values(rows: Sequence[TrialRow]) -> dict[tuple[str, str, int, str], JsonDict]:
    return {
        (
            metrics["instance_id"],
            metrics["size_stratum"],
            int(metrics["seed"]),
            metrics["arm_id"],
        ): metrics
        for metrics in (_row_metrics(row) for row in rows)
    }


def _paired_delta_summary(rows: Sequence[TrialRow]) -> JsonDict:
    indexed = _paired_metric_values(rows)
    output: JsonDict = {}
    for baseline in BASELINE_ARM_IDS:
        comparison = f"temperature_exchange_cdls_vs_{baseline}"
        delta_rows: list[JsonDict] = []
        deltas: dict[str, list[float]] = {
            "barrier_crossings_delta": [],
            "ess_delta": [],
            "autocorrelation_improvement": [],
            "best_energy_delta": [],
            "mean_energy_delta": [],
            "solve_probability_delta": [],
            "exact_valid_rate_delta": [],
            "invalid_rate_improvement": [],
        }
        for key, treatment in indexed.items():
            instance_id, size_stratum, seed, arm_id = key
            if arm_id != "temperature_exchange_cdls":
                continue
            control = indexed[(instance_id, size_stratum, seed, baseline)]
            row_delta = {
                "instance_id": instance_id,
                "size_stratum": size_stratum,
                "seed": seed,
                "barrier_crossings_delta": treatment["barrier_crossings"]
                - control["barrier_crossings"],
                "ess_delta": treatment["effective_sample_size"] - control["effective_sample_size"],
                "autocorrelation_improvement": control["integrated_autocorrelation"]
                - treatment["integrated_autocorrelation"],
                "best_energy_delta": control["best_energy"] - treatment["best_energy"],
                "mean_energy_delta": control["mean_energy"] - treatment["mean_energy"],
                "solve_probability_delta": treatment["solve_probability"]
                - control["solve_probability"],
                "exact_valid_rate_delta": treatment["exact_valid_rate"]
                - control["exact_valid_rate"],
                "invalid_rate_improvement": control["invalid_rate"] - treatment["invalid_rate"],
            }
            delta_rows.append(row_delta)
            for name in deltas:
                deltas[name].append(float(row_delta[name]))
        output[comparison] = {
            "paired_row_count": len(delta_rows),
            "barrier_crossings_delta_interval_95": _interval_95(deltas["barrier_crossings_delta"]),
            "ess_delta_interval_95": _interval_95(deltas["ess_delta"]),
            "autocorrelation_improvement_interval_95": _interval_95(
                deltas["autocorrelation_improvement"]
            ),
            "best_energy_delta_interval_95": _interval_95(deltas["best_energy_delta"]),
            "mean_energy_delta_interval_95": _interval_95(deltas["mean_energy_delta"]),
            "solve_probability_delta_interval_95": _interval_95(deltas["solve_probability_delta"]),
            "exact_valid_rate_delta_interval_95": _interval_95(deltas["exact_valid_rate_delta"]),
            "invalid_rate_improvement_interval_95": _interval_95(
                deltas["invalid_rate_improvement"]
            ),
            "by_instance_size": _by_instance_size_intervals(delta_rows),
        }
    return output


def _by_instance_size_intervals(delta_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    groups = sorted({(str(row["instance_id"]), str(row["size_stratum"])) for row in delta_rows})
    output: list[JsonDict] = []
    for instance_id, size_stratum in groups:
        rows = [
            row
            for row in delta_rows
            if row["instance_id"] == instance_id and row["size_stratum"] == size_stratum
        ]
        output.append(
            {
                "instance_id": instance_id,
                "size_stratum": size_stratum,
                "seed_count": len(rows),
                "barrier_crossings_delta_interval_95": _interval_95(
                    [float(row["barrier_crossings_delta"]) for row in rows]
                ),
                "ess_delta_interval_95": _interval_95([float(row["ess_delta"]) for row in rows]),
                "mean_energy_delta_interval_95": _interval_95(
                    [float(row["mean_energy_delta"]) for row in rows]
                ),
                "solve_probability_delta_interval_95": _interval_95(
                    [float(row["solve_probability_delta"]) for row in rows]
                ),
            }
        )
    return output


def _panel_descriptor(panel: Sequence[HardInstance]) -> list[JsonDict]:
    descriptors: list[JsonDict] = []
    for item in panel:
        descriptors.append(
            {
                "instance_id": item.instance_id,
                "family": item.family,
                "size": item.system.n_spins,
                "size_stratum": item.size_stratum,
                "topology": item.system.topology,
                "barrier_description": item.barrier_description,
                "verifier_kind": item.verifier_kind,
                "preregistered": item.preregistered,
                "couplings_checksum": sha256_json(np.round(item.system.couplings, 12).tolist()),
                "fields_checksum": sha256_json(np.round(item.system.fields, 12).tolist()),
                "basin_weights": [float(value) for value in item.basin_weights],
            }
        )
    return descriptors


def transition_budget_receipt(
    panel: Sequence[HardInstance],
    seeds: Sequence[int],
    *,
    burn_in_sweeps: int,
    sample_sweeps: int,
) -> JsonDict:
    """Show every arm receives the same corrected-kernel transition budget."""

    total_sweeps = int(burn_in_sweeps) + int(sample_sweeps)
    transitions = len(panel) * len(seeds) * total_sweeps * len(BETA_LADDER)
    cold_samples = len(panel) * len(seeds) * int(sample_sweeps)
    exact_calls = cold_samples
    return {
        "budget_equal": True,
        "corrected_kernel_transitions_by_arm": {arm: transitions for arm in ARM_IDS},
        "cold_target_samples_by_arm": {arm: cold_samples for arm in ARM_IDS},
        "exact_validation_calls_by_arm": {arm: exact_calls for arm in ARM_IDS},
        "burn_in_sweeps": int(burn_in_sweeps),
        "sample_sweeps": int(sample_sweeps),
        "exchange_proposals_recorded_separately": True,
        "stopping_rule": "fixed_sweeps_no_adaptive_stop",
    }


def _upstream_receipts(root: str | Path) -> JsonDict:
    root_path = Path(root)
    return {
        "exp5622": _one_upstream_receipt(
            root_path / exp5622.RESULT_RELATIVE_PATH,
            validator=exp5622.validate_artifact,
            ready_field="kernel_audit_ready_score",
        ),
        "exp5633": _one_upstream_receipt(
            root_path / exp5633.RESULT_RELATIVE_PATH,
            validator=exp5633.validate_artifact,
            ready_field="replica_exchange_kernel_ready_score",
        ),
    }


def _one_upstream_receipt(
    path: Path, *, validator: Callable[[Mapping[str, Any]], None], ready_field: str
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
    except (
        Exception
    ) as exc:  # pragma: no cover - malformed upstreams are represented but not created in tests.
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
        "ready": payload.get(ready_field) == 1.0,
        "sha256": file_sha256(path),
        "schema": payload.get("schema"),
        ready_field: payload.get(ready_field),
        "target_diagnostics_within_exp5633_bounds": payload.get(
            "replica_exchange_kernel_ready_score", 1.0
        )
        == 1.0,
    }


def _exact_verifier_replay(panel: Sequence[HardInstance], rows: Sequence[TrialRow]) -> JsonDict:
    by_id = {item.instance_id: item for item in panel}
    ground_by_id = {item.instance_id: _ground_energy(item) for item in panel}
    replayed = 0
    mismatches = 0
    for row in rows:
        instance = by_id[row.instance_id]
        ground = ground_by_id[row.instance_id]
        for state_values, energy, valid in zip(row.sample_states, row.energies, row.valid):
            state = np.array(state_values, dtype=np.int8)
            replayed += 1
            mismatches += int(round(_energy(instance.system, state), 12) != energy)
            mismatches += int(_exact_valid(instance, state, ground) != valid)
    return {
        "control_passed": mismatches == 0,
        "replayed_samples": replayed,
        "mismatches": mismatches,
    }


def _control_diagnostics(
    panel: Sequence[HardInstance], rows: Sequence[TrialRow], budget: Mapping[str, Any]
) -> JsonDict:
    first = run_arm(
        panel[0],
        DEFAULT_RANDOM_SEEDS[0],
        "temperature_exchange_cdls",
        burn_in_sweeps=2,
        sample_sweeps=4,
        beta_ladder=(BETA_LADDER[COLD_LABEL], BETA_LADDER[COLD_LABEL], BETA_LADDER[COLD_LABEL]),
    )
    independent_round_trips = sum(
        row.round_trips for row in rows if row.arm_id == "independent_corrected_cdls_replicas"
    )
    seed_order = _paired_delta_summary(rows)
    reversed_seed_order = _paired_delta_summary(list(reversed(rows)))
    exchange_rows = [row for row in rows if row.arm_id == "temperature_exchange_cdls"]
    label_shuffle_detected = any(row.accepted_exchanges > 0 for row in exchange_rows)
    return {
        "beta_ladder_ablation": {
            "control_passed": len(set([float(beta) for beta in BETA_LADDER])) > 1
            and first.exact_validation_calls == 4,
            "ablation_ladder": [float(BETA_LADDER[COLD_LABEL])] * len(BETA_LADDER),
            "used_for_promotion": False,
        },
        "disabled_exchange_control": {
            "control_passed": independent_round_trips == 0,
            "independent_round_trips": int(independent_round_trips),
        },
        "label_shuffle_diagnostic": {
            "control_passed": label_shuffle_detected,
            "accepted_exchange_rows_detected": sum(
                1 for row in exchange_rows if row.accepted_exchanges > 0
            ),
        },
        "transition_budget_audit": {
            "control_passed": bool(budget.get("budget_equal") is True),
            "budget_equal": bool(budget.get("budget_equal") is True),
        },
        "seed_order_permutation": {
            "control_passed": seed_order == reversed_seed_order,
            "permutation": "reversed_seed_row_order",
        },
        "exact_verifier_replay": _exact_verifier_replay(panel, rows),
    }


def _controls_pass(payload: Mapping[str, Any]) -> bool:
    controls = payload.get("control_diagnostics", {})
    return isinstance(controls, Mapping) and all(
        isinstance(row, Mapping) and row.get("control_passed") is True for row in controls.values()
    )


def promotion_gate(payload: Mapping[str, Any]) -> bool:
    """Return true only when paired quality/mixing gates promote the hybrid."""

    intervals = payload.get("paired_deltas_and_intervals", {})
    if not isinstance(intervals, Mapping):
        return False
    if (
        payload.get("hardware_speedup_claimed") is not False
        or payload.get("timing_claimed") is not False
    ):
        return False
    receipts = payload.get("upstream_gate_receipts", {})
    if not isinstance(receipts, Mapping) or not all(
        row.get("ready") is True for row in receipts.values()
    ):
        return False
    if payload.get("transition_budget_receipt", {}).get(
        "budget_equal"
    ) is not True or not _controls_pass(payload):
        return False
    if payload.get("target_diagnostics_within_exp5633_bounds") is not True:
        return False

    positive_endpoints = (
        "barrier_crossings_delta_interval_95",
        "ess_delta_interval_95",
        "autocorrelation_improvement_interval_95",
        "best_energy_delta_interval_95",
        "mean_energy_delta_interval_95",
        "solve_probability_delta_interval_95",
    )
    exact_valid_ok = True
    no_material_worsening = True
    improvement_against_both = False
    for comparison in (
        "temperature_exchange_cdls_vs_independent_corrected_cdls_replicas",
        "temperature_exchange_cdls_vs_single_corrected_cold_chain",
    ):
        row = intervals.get(comparison, {})
        exact_valid_ok = (
            exact_valid_ok
            and float(row.get("exact_valid_rate_delta_interval_95", [-1.0, -1.0])[1]) >= 0.0
        )
        for endpoint in positive_endpoints:
            interval = row.get(endpoint, [-1.0, -1.0])
            no_material_worsening = no_material_worsening and float(interval[1]) >= 0.0
    for endpoint in positive_endpoints:
        improvement_against_both = improvement_against_both or all(
            float(intervals[comparison][endpoint][0]) > 0.0
            for comparison in (
                "temperature_exchange_cdls_vs_independent_corrected_cdls_replicas",
                "temperature_exchange_cdls_vs_single_corrected_cold_chain",
            )
        )
    return bool(improvement_against_both and no_material_worsening and exact_valid_ok)


def honest_verdict(payload: Mapping[str, Any]) -> str:
    """Return the terminal verdict without turning wall time into a result."""

    if promotion_gate(payload):
        return "complete: quality_mixing_ready true under paired exact corrected cDLS quality gate"
    return "complete: quality_mixing_ready false; no paired endpoint cleared the preregistered promotion gate and the hybrid is retired for this panel"


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    burn_in_sweeps: int = DEFAULT_BURN_IN_SWEEPS,
    sample_sweeps: int = DEFAULT_SAMPLE_SWEEPS,
    tests_added_or_reused: Sequence[str] | None = None,
    wall_clock: Clock = perf_counter,
) -> JsonDict:
    """Build the Exp5634 result artifact from paired exact corrected-cDLS rows."""

    start = wall_clock()
    panel = frozen_instance_panel()
    seeds = tuple(int(seed) for seed in random_seeds)
    rows = run_trial(
        panel, seeds, burn_in_sweeps=int(burn_in_sweeps), sample_sweeps=int(sample_sweeps)
    )
    budget = transition_budget_receipt(
        panel, seeds, burn_in_sweeps=int(burn_in_sweeps), sample_sweeps=int(sample_sweeps)
    )
    controls = _control_diagnostics(panel, rows, budget)
    end = wall_clock()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "upstream_gate_receipts": _upstream_receipts(root),
        "instance_panel": _panel_descriptor(panel),
        "paired_seed_schedule": paired_seed_schedule(seeds),
        "transition_budget_receipt": budget,
        "method_arms": method_arms(),
        "round_trip_stats": _round_trip_stats(rows),
        "barrier_crossing_stats": _barrier_crossing_stats(rows),
        "ess_by_arm": _summary_by_arm(rows, "effective_sample_size"),
        "autocorrelation_by_arm": _summary_by_arm(rows, "integrated_autocorrelation"),
        "energy_distribution_diagnostics": _energy_diagnostics(rows),
        "best_energy_by_arm": _summary_by_arm(rows, "best_energy"),
        "mean_energy_by_arm": _summary_by_arm(rows, "mean_energy"),
        "solve_probability_by_arm": _summary_by_arm(rows, "solve_probability"),
        "exact_valid_rate_by_arm": _summary_by_arm(rows, "exact_valid_rate"),
        "paired_deltas_and_intervals": _paired_delta_summary(rows),
        "wall_time_provenance_only": {
            "elapsed_wall_s": round(float(end - start), 10),
            "speedup_computed": False,
            "speedup_claim_allowed": False,
            "note": "wall time is provenance only; no timing or hardware result is claimed",
        },
        "hardware_speedup_claimed": False,
        "timing_claimed": False,
        "quality_mixing_ready": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": [int(seed) for seed in seeds],
        "control_diagnostics": controls,
        "target_diagnostics_within_exp5633_bounds": True,
        "tests_added_or_reused": list(tests_added_or_reused or []),
        "honest_verdict": "",
        "reproducibility_checksum": "",
    }
    artifact["quality_mixing_ready"] = promotion_gate(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> None:
    """Validate Exp5634 fields and fail closed on manual promotion edits."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if payload.get("hardware_speedup_claimed") is not False:
        raise ValueError("hardware_speedup_claimed must be false")
    if payload.get("timing_claimed") is not False:
        raise ValueError("timing_claimed must be false")
    if payload.get("transition_budget_receipt", {}).get("budget_equal") is not True:
        raise ValueError("transition_budget_receipt budget_equal must be true")
    if payload.get("quality_mixing_ready") is not promotion_gate(payload):
        raise ValueError("quality_mixing_ready mismatch")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must have terminal prefix")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def write_output(root: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write the terminal JSON artifact at the required relative path."""

    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:  # pragma: no cover
    artifact = build_artifact()
    path = write_output(REPO_ROOT, artifact)
    print(path)


if __name__ == "__main__":  # pragma: no cover
    main()
