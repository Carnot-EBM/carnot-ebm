"""Compare temporal exchange with ordinary Gibbs under matched CPU work.

The headline panel uses exact finite Ising laws. The larger graph is a stress
diagnostic only. The source paper reports optimization and SPICE results, so
this experiment tests target-law fidelity independently before it grants any
sampler claim. It does not invoke or represent physical hardware.

Spec: REQ-SAMPLE-097, SCENARIO-SAMPLE-097, SCENARIO-SAMPLE-098,
SCENARIO-SAMPLE-099.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.sampling import temporal_exchange as te


JsonDict = dict[str, Any]
PreconditionGetter = Callable[[], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6793_temporal_exchange_ising_ab.json")
MODULE_PATH = Path("python/carnot/experiment_6793_temporal_exchange_ising_ab.py")
SCRIPT_PATH = Path("scripts/experiments/experiment_6793_temporal_exchange_ising_ab.py")
SPEC_PATH = Path("openspec/capabilities/training-inference/spec.md")

EXPERIMENT_ID = "experiment_6793_temporal_exchange_ising_ab"
SCHEMA_VERSION = "carnot.experiment_6793.temporal_exchange_ising_ab.v1"
INFERENCE_SUBSTRATE = "CPU exact-enumerable Ising simulation, no physical hardware"
CLAIM_BOUNDARY = (
    "Simulator fidelity and optimization diagnostics only; no FPGA, TSU, latency, power, "
    "energy-use, or hardware-availability claim"
)
SPEC_REFS = [
    "REQ-SAMPLE-097",
    "SCENARIO-SAMPLE-097",
    "SCENARIO-SAMPLE-098",
    "SCENARIO-SAMPLE-099",
]

ARMS = te.ARMS
TEMPERATURES = (0.75, 2.0)
COUPLING_GRID = (-0.08, 0.0, 0.08)
SEEDS = tuple(range(679300, 679320))
BURN_IN_SWEEPS = 128
COLLECTED_SAMPLES = 1024
SWEEPS_PER_SAMPLE = 1
MAX_EXACT_STATES = 4096
TARGET_LAW_NONINFERIORITY_MARGIN_TV = 0.03
CONFIDENCE_CRITICAL = 1.96
MAX_BIAS_ABS = 0.15
MAX_COUPLING_ABS = 0.60
MIN_TEMPERATURE = 0.50
MAX_TEMPERATURE = 2.50
REQUIRED_RAM_BYTES = 64 * 1024 * 1024
WALL_BUDGET_S = 600.0
SECONDS_PER_UPDATE_BUDGET = 50.0e-6
MAX_PLANNED_UPDATES = 10_000_000
STRESS_SEEDS = SEEDS[:4]
STRESS_BURN_IN_SWEEPS = 16
STRESS_SAMPLES = 128
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}

AGGREGATE_FIELDS = (
    "target_law_error_by_arm_family",
    "autocorrelation_by_arm_family",
    "effective_samples_per_update_by_arm_family",
    "optimum_hitting_updates_by_arm_family",
    "diversity_by_arm_family",
    "paired_efficiency_effects",
    "paired_efficiency_lcb",
    "target_law_noninferiority",
)


@dataclass(frozen=True)
class GraphFixture:
    """Store one fixed symmetric graph and its exact-enumeration boundary."""

    graph_id: str
    family: str
    biases: np.ndarray
    couplings: np.ndarray

    @property
    def n_spins(self) -> int:
        """Return the number of binary variables in the graph."""

        return int(self.biases.size)


class TemporalExchangeExperimentError(RuntimeError):
    """Stop when retained evidence cannot satisfy its declared schema."""


def canonical_json(value: Any) -> str:
    """Serialize finite evidence in one stable form for all digests."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TemporalExchangeExperimentError("evidence must be finite canonical JSON") from exc


def json_digest(value: Any) -> str:
    """Return a typed SHA-256 receipt for one canonical JSON value."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    """Keep a missing implementation distinct from a present empty file."""

    return (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
    )


def _rounded(value: float) -> float:
    """Keep stable JSON precision above every scientific gate tolerance."""

    result = float(f"{float(value):.15g}")
    return 0.0 if result == 0.0 else result


def _matrix(size: int, edges: Sequence[tuple[int, int, float]]) -> np.ndarray:
    """Build one symmetric matrix from an inspectable fixed edge list."""

    matrix = np.zeros((size, size), dtype=np.float64)
    for left, right, weight in edges:
        matrix[left, right] = matrix[right, left] = float(weight)
    return matrix


def frozen_graphs() -> tuple[GraphFixture, ...]:
    """Return three exact fixtures that were fixed before any sampler run."""

    return (
        GraphFixture(
            graph_id="ferromagnetic_ring_n6",
            family="ferromagnetic_ring",
            biases=np.asarray([0.12, -0.08, 0.05, -0.04, 0.09, -0.06]),
            couplings=_matrix(
                6,
                [
                    (0, 1, 0.55),
                    (1, 2, 0.50),
                    (2, 3, 0.45),
                    (3, 4, 0.55),
                    (4, 5, 0.50),
                    (5, 0, 0.45),
                    (0, 3, 0.20),
                ],
            ),
        ),
        GraphFixture(
            graph_id="frustrated_odd_cycle_n7",
            family="frustrated_odd_cycle",
            biases=np.asarray([0.10, -0.07, 0.04, -0.11, 0.08, -0.03, 0.06]),
            couplings=_matrix(
                7,
                [
                    (0, 1, -0.60),
                    (1, 2, -0.55),
                    (2, 3, -0.50),
                    (3, 4, -0.60),
                    (4, 5, -0.55),
                    (5, 6, -0.50),
                    (6, 0, -0.60),
                ],
            ),
        ),
        GraphFixture(
            graph_id="mixed_sparse_n8",
            family="mixed_sparse",
            biases=np.asarray([0.11, -0.09, 0.03, -0.05, 0.07, -0.12, 0.04, 0.08]),
            couplings=_matrix(
                8,
                [
                    (0, 1, 0.45),
                    (1, 2, -0.50),
                    (2, 3, 0.35),
                    (3, 0, -0.40),
                    (3, 4, 0.50),
                    (4, 5, -0.55),
                    (5, 6, 0.30),
                    (6, 7, -0.45),
                    (7, 4, 0.40),
                    (1, 6, 0.25),
                    (2, 5, -0.30),
                ],
            ),
        ),
    )


def stress_graph() -> GraphFixture:
    """Return one larger diagnostic graph with no exact target claim."""

    edges: list[tuple[int, int, float]] = []
    for node in range(32):
        ring_weight = 0.42 if node % 3 else -0.42
        chord_weight = -0.28 if node % 2 else 0.28
        edges.append((node, (node + 1) % 32, ring_weight))
        if node < 16:
            edges.append((node, node + 16, chord_weight))
    biases = np.asarray([0.03 * ((node % 7) - 3) for node in range(32)], dtype=np.float64)
    return GraphFixture(
        "mixed_sparse_n32_stress", "mixed_sparse_stress", biases, _matrix(32, edges)
    )


def _graph_receipt(graph: GraphFixture, *, exact: bool) -> JsonDict:
    """Serialize coefficients so graph identity does not depend on source names."""

    edges = [
        [left, right, _rounded(graph.couplings[left, right])]
        for left in range(graph.n_spins)
        for right in range(left + 1, graph.n_spins)
        if graph.couplings[left, right] != 0.0
    ]
    receipt: JsonDict = {
        "graph_id": graph.graph_id,
        "family": graph.family,
        "n_spins": graph.n_spins,
        "biases": [_rounded(value) for value in graph.biases],
        "edges": edges,
        "coefficient_range": {
            "minimum": _rounded(min([*graph.biases.tolist(), *graph.couplings.ravel().tolist()])),
            "maximum": _rounded(max([*graph.biases.tolist(), *graph.couplings.ravel().tolist()])),
        },
        "exact_target_enumerated": bool(exact),
    }
    receipt["graph_sha256"] = json_digest(receipt)
    return receipt


def coupling_for_temperature(temperature: float) -> float:
    """Apply the frozen AFM-low and FM-high schedule without graph tuning."""

    value = float(temperature)
    if value == TEMPERATURES[0]:
        return COUPLING_GRID[0]
    if value == TEMPERATURES[1]:
        return COUPLING_GRID[2]
    raise TemporalExchangeExperimentError(f"temperature is outside the frozen schedule: {value}")


def _sampler_seed(graph: GraphFixture, temperature: float, seed: int) -> int:
    """Derive one arm-independent RNG seed for a matched stratum cell."""

    token = f"exp6793:{graph.graph_id}:{temperature:.2f}:{int(seed)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def initial_state_pair(graph: GraphFixture, seed: int) -> JsonDict:
    """Freeze one explicit prior/current pair for all arms of a seed cell."""

    graph_token = int.from_bytes(
        hashlib.sha256(graph.graph_id.encode("ascii")).digest()[:4], "little"
    )
    generator = np.random.default_rng(np.random.SeedSequence([6793, graph_token, int(seed)]))
    previous = generator.choice(np.asarray([-1, 1], dtype=np.int8), size=graph.n_spins)
    current = generator.choice(np.asarray([-1, 1], dtype=np.int8), size=graph.n_spins)
    receipt = {
        "previous": previous.astype(int).tolist(),
        "current": current.astype(int).tolist(),
    }
    receipt["pair_sha256"] = json_digest(receipt)
    return receipt


def frozen_manifest() -> JsonDict:
    """Serialize every design choice that was fixed before metric reduction."""

    return {
        "design": "matched single-site updates with common random numbers",
        "headline_graph_ids": [graph.graph_id for graph in frozen_graphs()],
        "headline_spin_counts": [graph.n_spins for graph in frozen_graphs()],
        "temperatures": list(TEMPERATURES),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "coupling_grid": list(COUPLING_GRID),
        "coupling_selection_rule": "T=0.75 uses -0.08; T=2.0 uses +0.08",
        "schedule_tuned_on_headline_graphs": False,
        "burn_in_sweeps": BURN_IN_SWEEPS,
        "collected_samples": COLLECTED_SAMPLES,
        "sweeps_per_sample": SWEEPS_PER_SAMPLE,
        "target_law_noninferiority_margin_tv": TARGET_LAW_NONINFERIORITY_MARGIN_TV,
        "primary_efficiency_endpoint": "energy effective samples per attempted spin update",
        "confidence_method": "paired normal 95% interval within each graph-temperature stratum",
        "maximum_exact_states": MAX_EXACT_STATES,
        "coefficient_bounds": {"bias_abs": MAX_BIAS_ABS, "coupling_abs": MAX_COUPLING_ABS},
        "temperature_bounds": [MIN_TEMPERATURE, MAX_TEMPERATURE],
        "initial_state_pairs": [
            {"graph_id": graph.graph_id, "seed": seed, **initial_state_pair(graph, seed)}
            for graph in frozen_graphs()
            for seed in SEEDS
        ],
        "stress_panel": {
            "graph_id": stress_graph().graph_id,
            "n_spins": stress_graph().n_spins,
            "seeds": list(STRESS_SEEDS),
            "burn_in_sweeps": STRESS_BURN_IN_SWEEPS,
            "collected_samples": STRESS_SAMPLES,
            "headline_fidelity_eligible": False,
        },
    }


def _available_ram_bytes() -> int:
    """Read available physical pages without adding a platform dependency."""

    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 0


def _planned_update_count() -> int:
    """Count all headline and diagnostic attempted spin updates before launch."""

    headline = sum(
        graph.n_spins * (BURN_IN_SWEEPS + COLLECTED_SAMPLES * SWEEPS_PER_SAMPLE)
        for graph in frozen_graphs()
        for _temperature in TEMPERATURES
        for _seed in SEEDS
        for _arm in ARMS
    )
    stress = (
        stress_graph().n_spins
        * (STRESS_BURN_IN_SWEEPS + STRESS_SAMPLES)
        * len(TEMPERATURES)
        * len(STRESS_SEEDS)
        * len(ARMS)
    )
    return headline + stress


def check_preconditions() -> list[JsonDict]:
    """Check the owned CPU, finite-state, coefficient, clock, RAM, and wall gates."""

    module_spec = importlib.util.find_spec("carnot.sampling.temporal_exchange")
    largest_state_count = max(1 << graph.n_spins for graph in frozen_graphs())
    graph_values = [
        value
        for graph in frozen_graphs()
        for value in (*graph.biases.tolist(), *graph.couplings.ravel().tolist())
    ]
    coefficients_valid = bool(
        all(math.isfinite(value) for value in graph_values)
        and max(np.max(np.abs(graph.biases)) for graph in frozen_graphs()) <= MAX_BIAS_ABS
        and max(np.max(np.abs(graph.couplings)) for graph in frozen_graphs()) <= MAX_COUPLING_ABS
        and min(TEMPERATURES) >= MIN_TEMPERATURE
        and max(TEMPERATURES) <= MAX_TEMPERATURE
        and set(COUPLING_GRID) == {-0.08, 0.0, 0.08}
    )
    first_tick = time.perf_counter_ns()
    second_tick = time.perf_counter_ns()
    available_ram = _available_ram_bytes()
    planned_updates = _planned_update_count()
    estimated_wall_s = planned_updates * SECONDS_PER_UPDATE_BUDGET
    return [
        {
            "check": "cpu_sampler_importable",
            "passed": module_spec is not None,
            "expected": {"module": "carnot.sampling.temporal_exchange"},
            "observed": {"module_found": module_spec is not None},
        },
        {
            "check": "headline_exact_enumeration_bound",
            "passed": largest_state_count <= MAX_EXACT_STATES,
            "expected": {"maximum_states": MAX_EXACT_STATES},
            "observed": {"largest_state_count": largest_state_count},
        },
        {
            "check": "temperature_and_coefficient_ranges",
            "passed": coefficients_valid,
            "expected": {
                "temperature_range": [MIN_TEMPERATURE, MAX_TEMPERATURE],
                "maximum_bias_abs": MAX_BIAS_ABS,
                "maximum_coupling_abs": MAX_COUPLING_ABS,
            },
            "observed": {
                "temperatures": list(TEMPERATURES),
                "maximum_bias_abs": _rounded(
                    max(np.max(np.abs(graph.biases)) for graph in frozen_graphs())
                ),
                "maximum_coupling_abs": _rounded(
                    max(np.max(np.abs(graph.couplings)) for graph in frozen_graphs())
                ),
                "coupling_grid": list(COUPLING_GRID),
            },
        },
        {
            "check": "monotonic_timing",
            "passed": second_tick >= first_tick,
            "expected": {"second_tick_gte_first_tick": True},
            "observed": {"first_tick_ns": first_tick, "second_tick_ns": second_tick},
        },
        {
            "check": "ram_budget",
            "passed": available_ram >= REQUIRED_RAM_BYTES,
            "expected": {"available_bytes_at_least": REQUIRED_RAM_BYTES},
            "observed": {"available_bytes": available_ram},
        },
        {
            "check": "wall_budget",
            "passed": planned_updates <= MAX_PLANNED_UPDATES and estimated_wall_s <= WALL_BUDGET_S,
            "expected": {
                "maximum_planned_updates": MAX_PLANNED_UPDATES,
                "wall_budget_s": WALL_BUDGET_S,
            },
            "observed": {
                "planned_updates": planned_updates,
                "conservative_estimated_wall_s": _rounded(estimated_wall_s),
            },
        },
    ]


def _state_labels(states: np.ndarray) -> list[str]:
    """Encode states in spin-index order so marginals have visible support."""

    return ["".join("+" if spin > 0 else "-" for spin in state) for state in states]


def build_exact_targets() -> dict[tuple[str, float], JsonDict]:
    """Enumerate each headline graph-temperature spatial Boltzmann law."""

    targets: dict[tuple[str, float], JsonDict] = {}
    for graph in frozen_graphs():
        for temperature in TEMPERATURES:
            states, probabilities, energies = te.enumerate_target_distribution(
                graph.biases,
                graph.couplings,
                temperature,
                maximum_states=MAX_EXACT_STATES,
            )
            target: JsonDict = {
                "graph_id": graph.graph_id,
                "temperature": temperature,
                "n_spins": graph.n_spins,
                "exact": True,
                "state_order": _state_labels(states),
                "probabilities": [_rounded(value) for value in probabilities],
                "energies": [_rounded(value) for value in energies],
                "expected_energy": _rounded(float(probabilities @ energies)),
                "expected_magnetization": _rounded(float(probabilities @ np.mean(states, axis=1))),
                "optimum_energy": _rounded(float(np.min(energies))),
            }
            target["target_sha256"] = json_digest(target)
            targets[(graph.graph_id, temperature)] = target
    return targets


def _state_indices(samples: np.ndarray) -> np.ndarray:
    """Map bipolar rows to the same little-endian order as exact enumeration."""

    powers = (1 << np.arange(samples.shape[1], dtype=np.int64)).reshape((-1, 1))
    return ((samples > 0).astype(np.int64) @ powers).ravel()


def _autocorrelation(values: Sequence[float], maximum_lag: int = 100) -> JsonDict:
    """Estimate integrated autocorrelation until the first nonpositive lag."""

    array = np.asarray(values, dtype=np.float64)
    centered = array - float(np.mean(array))
    variance = float(np.dot(centered, centered) / array.size)
    if variance <= 1.0e-24:
        return {
            "integrated_time": _rounded(float(array.size)),
            "effective_samples": 1.0,
            "positive_lag_count": 0,
        }
    correlation_sum = 0.0
    positive_lags = 0
    for lag in range(1, min(int(maximum_lag), array.size - 1) + 1):
        correlation = float(np.dot(centered[:-lag], centered[lag:]) / (array.size - lag) / variance)
        if correlation <= 0.0:
            break
        correlation_sum += correlation
        positive_lags += 1
    integrated = min(float(array.size), max(1.0, 1.0 + 2.0 * correlation_sum))
    return {
        "integrated_time": _rounded(integrated),
        "effective_samples": _rounded(float(array.size) / integrated),
        "positive_lag_count": positive_lags,
    }


def _diversity(probabilities: np.ndarray) -> JsonDict:
    """Report observed support and entropy without treating them as fidelity."""

    nonzero = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    return {
        "unique_state_count": int(nonzero.size),
        "unique_state_rate": _rounded(float(nonzero.size) / float(probabilities.size)),
        "shannon_entropy_nats": _rounded(entropy),
        "effective_support": _rounded(math.exp(entropy)),
    }


def reproducible_row(row: Mapping[str, Any]) -> JsonDict:
    """Remove measured time and the self-digest from one row receipt."""

    return {
        key: deepcopy(value)
        for key, value in row.items()
        if key not in {"row_sha256", "wall_time_s"}
    }


def row_digest(row: Mapping[str, Any]) -> str:
    """Bind every deterministic input and output in one headline row."""

    return json_digest(reproducible_row(row))


def build_headline_row(
    graph: GraphFixture,
    temperature: float,
    seed: int,
    arm: str,
    target: Mapping[str, Any],
) -> JsonDict:
    """Run and serialize one exact-target graph-temperature-seed-arm cell."""

    pair = initial_state_pair(graph, seed)
    coupling = coupling_for_temperature(temperature) if arm == "temporal_exchange" else 0.0
    sampler_seed = _sampler_seed(graph, temperature, seed)
    result = te.sample_ising(
        graph.biases,
        graph.couplings,
        current=pair["current"],
        previous=pair["previous"],
        temperature=temperature,
        arm=arm,
        temporal_coupling=coupling,
        seed=sampler_seed,
        burn_in_sweeps=BURN_IN_SWEEPS,
        n_samples=COLLECTED_SAMPLES,
        sweeps_per_sample=SWEEPS_PER_SAMPLE,
        optimum_energy=float(target["optimum_energy"]),
    )
    counts = np.bincount(_state_indices(result.samples), minlength=1 << graph.n_spins)
    empirical = counts.astype(np.float64) / float(COLLECTED_SAMPLES)
    exact = np.asarray(target["probabilities"], dtype=np.float64)
    energy_autocorrelation = _autocorrelation(result.energy_trace)
    magnetization_autocorrelation = _autocorrelation(result.magnetization_trace)
    row: JsonDict = {
        "row_id": f"{graph.graph_id}:T={temperature:.2f}:seed={seed}:{arm}",
        "graph_id": graph.graph_id,
        "graph_family": graph.family,
        "n_spins": graph.n_spins,
        "temperature": temperature,
        "seed": int(seed),
        "sampler_seed": sampler_seed,
        "arm": arm,
        "temporal_coupling": coupling,
        "initial_state_pair": pair,
        "burn_in_sweeps": BURN_IN_SWEEPS,
        "collected_samples": COLLECTED_SAMPLES,
        "sweeps_per_sample": SWEEPS_PER_SAMPLE,
        "collection_update_counts": result.collection_update_counts.astype(int).tolist(),
        "update_count": result.update_count,
        "target_state_order": list(target["state_order"]),
        "target_marginal": list(target["probabilities"]),
        "empirical_marginal": [_rounded(value) for value in empirical],
        "exact_target_sha256": target["target_sha256"],
        "empirical_marginal_sha256": json_digest([_rounded(value) for value in empirical]),
        "trajectory_sha256": result.trajectory_sha256,
        "energy_trace": [_rounded(value) for value in result.energy_trace],
        "best_state": result.best_state.astype(int).tolist(),
        "best_energy": _rounded(te.ising_energy(result.best_state, graph.biases, graph.couplings)),
        "autocorrelation": {
            "energy": energy_autocorrelation,
            "magnetization": magnetization_autocorrelation,
        },
        "effective_samples": {
            "energy": energy_autocorrelation["effective_samples"],
            "magnetization": magnetization_autocorrelation["effective_samples"],
        },
        "effective_samples_per_update": _rounded(
            float(energy_autocorrelation["effective_samples"]) / result.update_count
        ),
        "optimum_hitting_updates": result.optimum_hitting_updates,
        "diversity": _diversity(empirical),
        "target_total_variation": _rounded(0.5 * float(np.sum(np.abs(empirical - exact)))),
        "magnetization_error": _rounded(
            abs(
                float(np.mean(result.magnetization_trace)) - float(target["expected_magnetization"])
            )
        ),
        "energy_error": _rounded(
            abs(float(np.mean(result.energy_trace)) - float(target["expected_energy"]))
        ),
        "wall_time_s": _rounded(result.wall_time_s),
        "headline_fidelity_eligible": True,
    }
    row["row_sha256"] = row_digest(row)
    return row


def build_headline_rows(targets: Mapping[tuple[str, float], Mapping[str, Any]]) -> list[JsonDict]:
    """Run the complete exact-target Cartesian panel in fixed order."""

    return [
        build_headline_row(graph, temperature, seed, arm, targets[(graph.graph_id, temperature)])
        for graph in frozen_graphs()
        for temperature in TEMPERATURES
        for seed in SEEDS
        for arm in ARMS
    ]


def _stress_row_digest(row: Mapping[str, Any]) -> str:
    """Bind a diagnostic row while excluding its measured wall time."""

    return json_digest(
        {key: value for key, value in row.items() if key not in {"row_sha256", "wall_time_s"}}
    )


def build_stress_rows() -> list[JsonDict]:
    """Run a separate larger panel only for work and dependence diagnostics."""

    graph = stress_graph()
    rows = []
    for temperature in TEMPERATURES:
        for seed in STRESS_SEEDS:
            pair = initial_state_pair(graph, seed)
            sampler_seed = _sampler_seed(graph, temperature, seed)
            for arm in ARMS:
                coupling = (
                    coupling_for_temperature(temperature) if arm == "temporal_exchange" else 0.0
                )
                result = te.sample_ising(
                    graph.biases,
                    graph.couplings,
                    current=pair["current"],
                    previous=pair["previous"],
                    temperature=temperature,
                    arm=arm,
                    temporal_coupling=coupling,
                    seed=sampler_seed,
                    burn_in_sweeps=STRESS_BURN_IN_SWEEPS,
                    n_samples=STRESS_SAMPLES,
                    optimum_energy=None,
                )
                unique_count = int(np.unique(result.samples, axis=0).shape[0])
                autocorrelation = _autocorrelation(result.energy_trace)
                row: JsonDict = {
                    "row_id": f"stress:{graph.graph_id}:T={temperature:.2f}:seed={seed}:{arm}",
                    "graph_id": graph.graph_id,
                    "graph_family": graph.family,
                    "n_spins": graph.n_spins,
                    "temperature": temperature,
                    "seed": seed,
                    "sampler_seed": sampler_seed,
                    "arm": arm,
                    "temporal_coupling": coupling,
                    "initial_state_pair": pair,
                    "burn_in_sweeps": STRESS_BURN_IN_SWEEPS,
                    "collected_samples": STRESS_SAMPLES,
                    "update_count": result.update_count,
                    "trajectory_sha256": result.trajectory_sha256,
                    "energy_trace": [_rounded(value) for value in result.energy_trace],
                    "best_state": result.best_state.astype(int).tolist(),
                    "best_energy": _rounded(
                        te.ising_energy(result.best_state, graph.biases, graph.couplings)
                    ),
                    "autocorrelation": {"energy": autocorrelation},
                    "effective_samples_per_update": _rounded(
                        float(autocorrelation["effective_samples"]) / result.update_count
                    ),
                    "optimum_hitting_updates": None,
                    "diversity": {
                        "unique_state_count": unique_count,
                        "unique_state_rate": _rounded(unique_count / STRESS_SAMPLES),
                    },
                    "target_marginal": None,
                    "empirical_marginal": None,
                    "target_total_variation": None,
                    "exact_target_available": False,
                    "headline_fidelity_eligible": False,
                    "wall_time_s": _rounded(result.wall_time_s),
                }
                row["row_sha256"] = _stress_row_digest(row)
                rows.append(row)
    return rows


def _mean_interval(values: Sequence[float]) -> JsonDict:
    """Compute a paired or unpaired normal interval from retained values."""

    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return {
        "count": int(array.size),
        "mean": _rounded(mean),
        "standard_error": _rounded(standard_error),
        "ci95_low": _rounded(mean - CONFIDENCE_CRITICAL * standard_error),
        "ci95_high": _rounded(mean + CONFIDENCE_CRITICAL * standard_error),
    }


def derive_aggregates(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce rows per graph-temperature stratum before any summary gate."""

    groups: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["graph_id"]), str(row["arm"]), float(row["temperature"]))].append(row)

    target_error = []
    autocorrelation = []
    effective = []
    hitting = []
    diversity = []
    for (graph_id, arm, temperature), group in sorted(groups.items()):
        tv_interval = _mean_interval([float(row["target_total_variation"]) for row in group])
        target_error.append(
            {
                "graph_id": graph_id,
                "temperature": temperature,
                "arm": arm,
                "mean_total_variation": tv_interval["mean"],
                "ci95_low": tv_interval["ci95_low"],
                "ci95_high": tv_interval["ci95_high"],
                "mean_magnetization_error": _rounded(
                    float(np.mean([float(row["magnetization_error"]) for row in group]))
                ),
                "mean_energy_error": _rounded(
                    float(np.mean([float(row["energy_error"]) for row in group]))
                ),
                "row_count": len(group),
            }
        )
        autocorrelation.append(
            {
                "graph_id": graph_id,
                "temperature": temperature,
                "arm": arm,
                "mean_energy_integrated_time": _rounded(
                    float(
                        np.mean(
                            [
                                float(row["autocorrelation"]["energy"]["integrated_time"])
                                for row in group
                            ]
                        )
                    )
                ),
                "mean_magnetization_integrated_time": _rounded(
                    float(
                        np.mean(
                            [
                                float(row["autocorrelation"]["magnetization"]["integrated_time"])
                                for row in group
                            ]
                        )
                    )
                ),
                "row_count": len(group),
            }
        )
        efficiency_interval = _mean_interval(
            [float(row["effective_samples_per_update"]) for row in group]
        )
        effective.append(
            {
                "graph_id": graph_id,
                "temperature": temperature,
                "arm": arm,
                "mean_energy_effective_samples_per_update": efficiency_interval["mean"],
                "ci95_low": efficiency_interval["ci95_low"],
                "ci95_high": efficiency_interval["ci95_high"],
                "row_count": len(group),
            }
        )
        budget = max(int(row["update_count"]) for row in group)
        observed_hits = [
            int(row["optimum_hitting_updates"])
            for row in group
            if row["optimum_hitting_updates"] is not None
        ]
        restricted = [
            int(row["optimum_hitting_updates"])
            if row["optimum_hitting_updates"] is not None
            else int(row["update_count"]) + 1
            for row in group
        ]
        hitting.append(
            {
                "graph_id": graph_id,
                "temperature": temperature,
                "arm": arm,
                "hit_count": len(observed_hits),
                "hit_rate": _rounded(len(observed_hits) / len(group)),
                "median_observed_hitting_updates": (
                    _rounded(float(np.median(observed_hits))) if observed_hits else None
                ),
                "restricted_mean_hitting_updates": _rounded(float(np.mean(restricted))),
                "censoring_update": budget + 1,
                "row_count": len(group),
            }
        )
        diversity.append(
            {
                "graph_id": graph_id,
                "temperature": temperature,
                "arm": arm,
                "mean_unique_state_count": _rounded(
                    float(np.mean([int(row["diversity"]["unique_state_count"]) for row in group]))
                ),
                "mean_effective_support": _rounded(
                    float(np.mean([float(row["diversity"]["effective_support"]) for row in group]))
                ),
                "row_count": len(group),
            }
        )

    index = {
        (str(row["graph_id"]), float(row["temperature"]), int(row["seed"]), str(row["arm"])): row
        for row in rows
    }
    paired_effects = []
    lcb_rows = []
    noninferiority_strata = []
    for graph in frozen_graphs():
        for temperature in TEMPERATURES:
            efficiency_deltas = []
            tv_deltas = []
            for seed in SEEDS:
                ordinary = index[(graph.graph_id, temperature, seed, "ordinary_gibbs")]
                temporal = index[(graph.graph_id, temperature, seed, "temporal_exchange")]
                efficiency_deltas.append(
                    float(temporal["effective_samples_per_update"])
                    - float(ordinary["effective_samples_per_update"])
                )
                tv_deltas.append(
                    float(temporal["target_total_variation"])
                    - float(ordinary["target_total_variation"])
                )
            efficiency_interval = _mean_interval(efficiency_deltas)
            tv_interval = _mean_interval(tv_deltas)
            paired_effects.append(
                {
                    "graph_id": graph.graph_id,
                    "temperature": temperature,
                    "endpoint": "temporal_minus_gibbs_energy_effective_samples_per_update",
                    "paired_seed_count": len(efficiency_deltas),
                    "paired_deltas": [_rounded(value) for value in efficiency_deltas],
                    **efficiency_interval,
                }
            )
            lcb_rows.append(
                {
                    "graph_id": graph.graph_id,
                    "temperature": temperature,
                    "paired_efficiency_lcb": efficiency_interval["ci95_low"],
                    "above_zero": efficiency_interval["ci95_low"] > 0.0,
                }
            )
            noninferiority_strata.append(
                {
                    "graph_id": graph.graph_id,
                    "temperature": temperature,
                    "mean_temporal_minus_gibbs_tv": tv_interval["mean"],
                    "upper_confidence_bound": tv_interval["ci95_high"],
                    "margin": TARGET_LAW_NONINFERIORITY_MARGIN_TV,
                    "passed": tv_interval["ci95_high"] <= TARGET_LAW_NONINFERIORITY_MARGIN_TV,
                }
            )
    family_rows = []
    for graph in frozen_graphs():
        graph_strata = [row for row in noninferiority_strata if row["graph_id"] == graph.graph_id]
        family_rows.append(
            {
                "graph_id": graph.graph_id,
                "family": graph.family,
                "temperature_strata": graph_strata,
                "passed": all(row["passed"] for row in graph_strata),
            }
        )
    minimum_lcb = min(float(row["paired_efficiency_lcb"]) for row in lcb_rows)
    return {
        "target_law_error_by_arm_family": target_error,
        "autocorrelation_by_arm_family": autocorrelation,
        "effective_samples_per_update_by_arm_family": effective,
        "optimum_hitting_updates_by_arm_family": hitting,
        "diversity_by_arm_family": diversity,
        "paired_efficiency_effects": paired_effects,
        "paired_efficiency_lcb": {
            "endpoint": "energy effective samples per attempted spin update",
            "by_stratum": lcb_rows,
            "minimum_stratum_lcb": _rounded(minimum_lcb),
            "passed": all(row["above_zero"] for row in lcb_rows),
        },
        "target_law_noninferiority": {
            "metric": "temporal minus ordinary total variation",
            "margin": TARGET_LAW_NONINFERIORITY_MARGIN_TV,
            "by_family": family_rows,
            "passed": all(row["passed"] for row in family_rows),
        },
    }


def _expected_grid() -> set[tuple[str, float, int, str]]:
    """Construct the headline Cartesian product without reading result rows."""

    return {
        (graph.graph_id, temperature, seed, arm)
        for graph in frozen_graphs()
        for temperature in TEMPERATURES
        for seed in SEEDS
        for arm in ARMS
    }


def _row_grid(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, float, int, str]]:
    """Reduce retained rows to the four matched design axes."""

    return {
        (str(row["graph_id"]), float(row["temperature"]), int(row["seed"]), str(row["arm"]))
        for row in rows
    }


def _zero_coupling_equivalent(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Check bit-identical ordinary and disabled trajectories for every cell."""

    index = {(row["graph_id"], row["temperature"], row["seed"], row["arm"]): row for row in rows}
    for graph in frozen_graphs():
        for temperature in TEMPERATURES:
            for seed in SEEDS:
                ordinary = index.get((graph.graph_id, temperature, seed, "ordinary_gibbs"))
                disabled = index.get(
                    (graph.graph_id, temperature, seed, "temporal_exchange_zero_coupling")
                )
                if ordinary is None or disabled is None:
                    return False
                if ordinary["trajectory_sha256"] != disabled["trajectory_sha256"]:
                    return False
    return True


def _arm_definitions() -> JsonDict:
    """Describe the only algorithmic difference between matched arms."""

    return {
        "ordinary_gibbs": {
            "temporal_coupling": 0.0,
            "previous_configuration_stored": True,
            "previous_configuration_used_in_conditional": False,
        },
        "temporal_exchange": {
            "temporal_coupling": "frozen temperature schedule",
            "previous_configuration_stored": True,
            "previous_configuration_used_in_conditional": True,
        },
        "temporal_exchange_zero_coupling": {
            "temporal_coupling": 0.0,
            "previous_configuration_stored": True,
            "previous_configuration_used_in_conditional": True,
            "purpose": "disabled-coupling equivalence control",
        },
    }


def _coupling_schedule() -> JsonDict:
    """Record the preregistered grid and its paper-derived sign rule."""

    return {
        "grid": list(COUPLING_GRID),
        "selected_by_temperature": {"0.75": -0.08, "2.0": 0.08},
        "low_temperature_sign": "antiferromagnetic",
        "high_temperature_sign": "ferromagnetic",
        "selection_basis": "paper sign direction plus fixed symmetric small grid",
        "headline_graph_metrics_used_for_selection": False,
    }


def _update_budget_by_arm(
    rows: Sequence[Mapping[str, Any]], stress_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Sum attempted spin updates without counting sweeps as one update."""

    return {
        arm: {
            "headline_updates": sum(int(row["update_count"]) for row in rows if row["arm"] == arm),
            "stress_updates": sum(
                int(row["update_count"]) for row in stress_rows if row["arm"] == arm
            ),
            "headline_row_count": sum(row["arm"] == arm for row in rows),
            "stress_row_count": sum(row["arm"] == arm for row in stress_rows),
        }
        for arm in ARMS
    }


def _primary_source_receipt() -> JsonDict:
    """Bind the paper version and distinguish its claims from this experiment."""

    return {
        "arxiv_id": "2608.21753",
        "title": "High-Efficiency Ising Machine with Time-Dimensional Exchange Coupling",
        "authors": ["Zhengyu Du", "Haijie Xu", "Kaiming Cai", "Zhe Yuan", "Yue Zhang"],
        "submitted": "2026-08-22",
        "abstract_url": "https://arxiv.org/abs/2608.21753",
        "pdf_url": "https://arxiv.org/pdf/2608.21753",
        "pdf_sha256": "sha256:05fbd5a982e38ca7973bfd57ed45561ffb0f5569f5631250014d7db9c337ff25",
        "implemented_term": "-J_v sum_i sigma_i(t) sigma_i(t-1)",
        "source_scope": "optimization simulation and SPICE feasibility",
        "carnot_scope": "CPU target-law and matched-update comparison",
    }


def _base_artifact(
    run_date: str, duration_s: float, preconditions: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Create the full schema before selecting blocked, null, or positive."""

    graph_receipts = [_graph_receipt(graph, exact=True) for graph in frozen_graphs()]
    graph_receipts.append(_graph_receipt(stress_graph(), exact=False))
    return {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "run_date": str(run_date),
        "status": "in_progress",
        "spec_refs": list(SPEC_REFS),
        "field_principles": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _rounded(duration_s),
        "random_seed": {
            "headline_seeds": list(SEEDS),
            "stress_seeds": list(STRESS_SEEDS),
            "sampler_seed_scheme": "sha256(exp6793:graph_id:temperature:seed)",
            "initial_state_seed_scheme": "SeedSequence(6793, graph_sha32, seed)",
        },
        "reproducibility_checksum": "pending",
        "primary_source_receipt": _primary_source_receipt(),
        "frozen_manifest": frozen_manifest(),
        "arm_definitions": _arm_definitions(),
        "graph_families": graph_receipts,
        "temperatures": list(TEMPERATURES),
        "coupling_schedule": _coupling_schedule(),
        "update_budget_by_arm": {arm: {} for arm in ARMS},
        "preconditions_checked": [dict(row) for row in preconditions],
        "rows": [],
        "exact_target_hashes": {},
        "target_law_error_by_arm_family": [],
        "autocorrelation_by_arm_family": [],
        "effective_samples_per_update_by_arm_family": [],
        "optimum_hitting_updates_by_arm_family": [],
        "diversity_by_arm_family": [],
        "paired_efficiency_effects": [],
        "paired_efficiency_lcb": {"by_stratum": [], "passed": False},
        "target_law_noninferiority": {
            "margin": TARGET_LAW_NONINFERIORITY_MARGIN_TV,
            "by_family": [],
            "passed": False,
        },
        "stress_rows_separate": True,
        "stress_rows_generated_after_headline_completed": False,
        "stress_rows": [],
        "physical_hardware_invoked": False,
        "temporal_exchange_comparison_completed": False,
        "gate_check_summary": [],
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "complete_partial: temporal exchange comparison did not finish",
        "claim_boundary": CLAIM_BOUNDARY,
        "implementation_receipt": {
            "module_path": MODULE_PATH.as_posix(),
            "module_sha256": file_digest(REPO_ROOT / MODULE_PATH),
            "sampler_module_path": "python/carnot/sampling/temporal_exchange.py",
            "sampler_module_sha256": file_digest(
                REPO_ROOT / "python/carnot/sampling/temporal_exchange.py"
            ),
        },
    }


def _field_principles(keys: Sequence[str]) -> JsonDict:
    """Explain why each top-level field is needed for later cold audit."""

    specific = {
        "inference_substrate": "The label prevents CPU simulation from becoming a hardware receipt.",
        "duration_s": "A monotonic measurement reports real CPU work without padding.",
        "random_seed": "Matched seed roles make every stochastic row replayable.",
        "reproducibility_checksum": "The digest binds deterministic inputs, rows, and reductions.",
        "primary_source_receipt": "The receipt fixes the paper version and its narrower evidence scope.",
        "frozen_manifest": "The manifest proves that graphs, schedules, and gates precede outcomes.",
        "rows": "Every graph, temperature, seed, and arm cell remains attributable.",
        "exact_target_hashes": "Exact-law hashes let Exp6794 recompute fidelity independently.",
        "paired_efficiency_lcb": "A lower bound prevents a noisy mean from earning positive credit.",
        "target_law_noninferiority": "Optimization speed cannot excuse a changed target law.",
        "stress_rows_separate": "The larger non-enumerated graph cannot establish sampler fidelity.",
        "physical_hardware_invoked": "Bare false forbids FPGA or TSU interpretation.",
        "temporal_exchange_comparison_completed": "Exp6794 consumes this exact completion gate.",
        "gate_check_summary": "A block names the failed check and its observed value.",
        "verifier_is_oracle": "False states that the sampler does not consume exact target outcomes.",
        "verdict_class": "A closed class separates measured nulls from blocks and positives.",
        "honest_verdict": "A terminal prefix preserves conductor classification.",
    }
    return {
        key: specific.get(key, f"The {key} field preserves attributable experiment evidence.")
        for key in keys
    }


def _normalized_rows(rows: Sequence[Mapping[str, Any]], *, stress: bool = False) -> list[JsonDict]:
    """Remove measured row time before the reproducibility digest is computed."""

    ignored = {"wall_time_s", "row_sha256"}
    normalized = []
    for row in rows:
        value = {key: deepcopy(item) for key, item in row.items() if key not in ignored}
        value["row_sha256"] = _stress_row_digest(row) if stress else row_digest(row)
        normalized.append(value)
    return normalized


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind deterministic evidence while excluding real wall-clock variation."""

    return json_digest(
        {
            "random_seed": artifact["random_seed"],
            "primary_source_receipt": artifact["primary_source_receipt"],
            "frozen_manifest": artifact["frozen_manifest"],
            "arm_definitions": artifact["arm_definitions"],
            "graph_families": artifact["graph_families"],
            "temperatures": artifact["temperatures"],
            "coupling_schedule": artifact["coupling_schedule"],
            "rows": _normalized_rows(artifact["rows"]),
            "stress_rows": _normalized_rows(artifact["stress_rows"], stress=True),
            "exact_target_hashes": artifact["exact_target_hashes"],
            **{field: artifact[field] for field in AGGREGATE_FIELDS},
            "temporal_exchange_comparison_completed": artifact[
                "temporal_exchange_comparison_completed"
            ],
            "verdict_class": artifact["verdict_class"],
        }
    )


def _finish_artifact(artifact: JsonDict) -> JsonDict:
    """Add principles and the deterministic digest only after all fields exist."""

    artifact["field_principles"] = _field_principles(list(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Preserve the complete schema when an owned launch gate fails."""

    artifact = _base_artifact(run_date, duration_s, preconditions)
    artifact["status"] = "complete_blocked_temporal_exchange_ab"
    artifact["gate_check_summary"] = [dict(row) for row in preconditions if not row.get("passed")]
    artifact["verdict_class"] = "blocked"
    artifact["honest_verdict"] = (
        "complete_blocked_temporal_exchange_ab: one or more sampler, bound, clock, RAM, or "
        "wall-budget preconditions failed"
    )
    return _finish_artifact(artifact)


def _completion_errors(
    rows: Sequence[Mapping[str, Any]],
    stress_rows: Sequence[Mapping[str, Any]],
    targets: Mapping[tuple[str, float], Mapping[str, Any]],
) -> list[JsonDict]:
    """Return failed row, receipt, work, and disabled-control checks."""

    errors = []
    expected = _expected_grid()
    observed = _row_grid(rows)
    if len(rows) != len(expected) or observed != expected:
        errors.append(
            {
                "check": "complete_headline_row_grid",
                "expected": {"row_count": len(expected)},
                "observed": {"row_count": len(rows), "unique_grid_count": len(observed)},
            }
        )
    invalid_hashes = [row.get("row_id") for row in rows if row.get("row_sha256") != row_digest(row)]
    if invalid_hashes:
        errors.append(
            {
                "check": "headline_row_hashes",
                "expected": {"invalid_count": 0},
                "observed": {"invalid_row_ids": invalid_hashes},
            }
        )
    if len(targets) != len(frozen_graphs()) * len(TEMPERATURES):
        errors.append(
            {
                "check": "exact_target_grid",
                "expected": {"target_count": len(frozen_graphs()) * len(TEMPERATURES)},
                "observed": {"target_count": len(targets)},
            }
        )
    if len(rows) == len(expected) and observed == expected and not _zero_coupling_equivalent(rows):
        errors.append(
            {
                "check": "disabled_coupling_equivalence",
                "expected": {"all_trajectory_hashes_equal": True},
                "observed": {"all_trajectory_hashes_equal": False},
            }
        )
    expected_stress = len(TEMPERATURES) * len(STRESS_SEEDS) * len(ARMS)
    invalid_stress = [
        row.get("row_id")
        for row in stress_rows
        if row.get("row_sha256") != _stress_row_digest(row)
        or row.get("headline_fidelity_eligible") is not False
    ]
    if len(stress_rows) != expected_stress or invalid_stress:
        errors.append(
            {
                "check": "separate_stress_rows",
                "expected": {"row_count": expected_stress, "invalid_count": 0},
                "observed": {"row_count": len(stress_rows), "invalid_row_ids": invalid_stress},
            }
        )
    return errors


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    stress_rows: Sequence[Mapping[str, Any]],
    targets: Mapping[tuple[str, float], Mapping[str, Any]],
    run_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Reduce a complete panel into one terminal positive, null, or partial result."""

    artifact = _base_artifact(run_date, duration_s, preconditions)
    artifact["rows"] = [dict(row) for row in rows]
    artifact["stress_rows"] = [dict(row) for row in stress_rows]
    artifact["stress_rows_generated_after_headline_completed"] = True
    artifact["exact_target_hashes"] = {
        f"{graph_id}:T={temperature:.2f}": target["target_sha256"]
        for (graph_id, temperature), target in sorted(targets.items())
    }
    artifact["update_budget_by_arm"] = _update_budget_by_arm(rows, stress_rows)
    artifact.update(derive_aggregates(rows))
    errors = _completion_errors(rows, stress_rows, targets)
    artifact["gate_check_summary"] = errors
    completed = not errors and all(row.get("passed") for row in preconditions)
    artifact["temporal_exchange_comparison_completed"] = completed
    positive = bool(
        completed
        and artifact["paired_efficiency_lcb"]["passed"]
        and artifact["target_law_noninferiority"]["passed"]
    )
    if not completed:
        artifact["status"] = "complete_partial"
        artifact["verdict_class"] = "partial"
        artifact["honest_verdict"] = (
            "complete_partial: one or more row, receipt, work, or control checks failed"
        )
    elif positive:
        artifact["status"] = "complete_positive"
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = (
            "complete: temporal exchange improved energy effective samples per update in every "
            "headline stratum within the frozen target-law margin; CPU simulation only"
        )
    else:
        artifact["status"] = "complete_null"
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = (
            "complete: temporal exchange did not pass both the paired efficiency and target-law "
            "gates; CPU simulation only"
        )
    return _finish_artifact(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute schema, row, aggregate, completion, and claim-boundary gates."""

    required = set(_base_artifact("19700101", 0.0, []))
    missing = required - set(artifact)
    if missing:
        return ["required_fields_missing"]
    errors = []
    if set(artifact["field_principles"]) != set(artifact):
        errors.append("field_principles_mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact["physical_hardware_invoked"] is not False:
        errors.append("physical_hardware_boundary_mismatch")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("oracle_boundary_mismatch")
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class_mismatch")
    if not str(artifact["honest_verdict"]).startswith(
        (
            "complete:",
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
        )
    ):
        errors.append("honest_verdict_prefix_mismatch")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    duration = artifact["duration_s"]
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0.0:
        errors.append("duration_invalid")

    blocked = artifact["status"] == "complete_blocked_temporal_exchange_ab"
    if blocked:
        if (
            artifact["verdict_class"] != "blocked"
            or artifact["temporal_exchange_comparison_completed"] is not False
            or artifact["rows"]
            or not artifact["gate_check_summary"]
            or not str(artifact["honest_verdict"]).startswith(
                "complete_blocked_temporal_exchange_ab"
            )
        ):
            errors.append("blocked_terminal_state_mismatch")
        return errors

    rows = artifact["rows"]
    if len(rows) != len(_expected_grid()) or _row_grid(rows) != _expected_grid():
        errors.append("row_grid_mismatch")
    if any(row.get("row_sha256") != row_digest(row) for row in rows):
        errors.append("row_hash_mismatch")
    derived = derive_aggregates(rows)
    if any(artifact[field] != derived[field] for field in AGGREGATE_FIELDS):
        errors.append("aggregate_mismatch")
    target_hashes = {
        f"{row['graph_id']}:T={float(row['temperature']):.2f}": row["exact_target_sha256"]
        for row in rows
    }
    if artifact["exact_target_hashes"] != dict(sorted(target_hashes.items())):
        errors.append("exact_target_hash_mismatch")
    if not _zero_coupling_equivalent(rows):
        errors.append("disabled_coupling_equivalence_mismatch")
    expected_stress = len(TEMPERATURES) * len(STRESS_SEEDS) * len(ARMS)
    if (
        artifact["stress_rows_separate"] is not True
        or len(artifact["stress_rows"]) != expected_stress
        or any(
            row.get("row_sha256") != _stress_row_digest(row)
            or row.get("headline_fidelity_eligible") is not False
            for row in artifact["stress_rows"]
        )
    ):
        errors.append("stress_row_mismatch")
    should_complete = not {
        "row_grid_mismatch",
        "row_hash_mismatch",
        "aggregate_mismatch",
        "exact_target_hash_mismatch",
        "disabled_coupling_equivalence_mismatch",
        "stress_row_mismatch",
    }.intersection(errors)
    if artifact["temporal_exchange_comparison_completed"] is not should_complete:
        errors.append("completion_mismatch")
    if should_complete:
        positive = bool(
            derived["paired_efficiency_lcb"]["passed"]
            and derived["target_law_noninferiority"]["passed"]
        )
        expected_class = "positive" if positive else "null"
        if artifact["verdict_class"] != expected_class:
            errors.append("verdict_class_mismatch")
    return errors


def _write_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the deliverable only after one complete JSON value exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(
    *,
    output_path: Path | str = RESULT_PATH,
    run_date: str,
    precondition_getter: PreconditionGetter = check_preconditions,
) -> JsonDict:
    """Check gates, run headline before stress, validate, and write once."""

    started = time.perf_counter()
    preconditions = precondition_getter()
    failed = [row for row in preconditions if not row.get("passed")]
    if failed:
        artifact = _blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions=preconditions,
        )
    else:
        targets = build_exact_targets()
        rows = build_headline_rows(targets)
        stress_rows = build_stress_rows()
        artifact = build_artifact(
            rows=rows,
            stress_rows=stress_rows,
            targets=targets,
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions=preconditions,
        )
    validation_errors = validate_artifact(artifact)
    if validation_errors:
        raise TemporalExchangeExperimentError(f"artifact validation failed: {validation_errors}")
    _write_atomic(Path(output_path), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dated simulator comparison from the required command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args(argv)
    artifact = run(output_path=args.output, run_date=args.date)
    print(
        canonical_json(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "temporal_exchange_comparison_completed": artifact[
                    "temporal_exchange_comparison_completed"
                ],
                "verdict_class": artifact["verdict_class"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the script wrapper.
    raise SystemExit(main())
