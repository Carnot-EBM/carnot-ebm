"""Benchmark corrected multiscale sampling under matched exact-energy budgets.

The benchmark compares bounded Python implementations only. Each arm receives
the same count of full-state energy evaluations for a cell and seed. Small
cells also use an independent enumerator as a direct finite-law reference.

Spec refs: REQ-SAMPLER-7134 and SCENARIO-SAMPLER-7134-*.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Mapping, Sequence

from carnot import experiment_6657_bounded_treewidth_ising_reference as exact_reference
from carnot import experiment_7133_v626_multiscale_sampler_prototype as prototype


JsonDict = dict[str, Any]
SpinState = tuple[int, ...]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7134_v626_multiscale_sampler_benchmark.json")
SOURCE_PATH = Path("python/carnot/experiment_7134_v626_multiscale_sampler_benchmark.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7134_v626_multiscale_sampler_benchmark.py")
TEST_PATH = Path("tests/python/test_experiment_7134_v626_multiscale_sampler_benchmark.py")
SAMPLER_SPEC_PATH = Path("openspec/capabilities/samplers/spec.md")
PROTOTYPE_SOURCE_PATH = Path("python/carnot/experiment_7133_v626_multiscale_sampler_prototype.py")
PROTOTYPE_ARTIFACT_PATH = Path("results/experiment_7133_v626_multiscale_sampler_prototype.json")
RUN_DATE = "20260908"
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator: matched-budget sampler benchmark"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
RANDOM_SEED = 713420260908
FROZEN_SEEDS = (7134001, 7134002, 7134003, 7134004, 7134005)
PROTOTYPE_ARTIFACT_HASH = "sha256:cec6342f98eb9a6844c99b1c28c92e32124c449d8f677140ed1565126d9542cb"
FLOAT_TOLERANCE = 1.0e-12
SOURCE_ARTIFACT_PATHS = (
    Path("research-program.md"),
    Path("results/experiment_6612_spectral_k_block_scale_rust_parity.json"),
    PROTOTYPE_ARTIFACT_PATH,
    PROTOTYPE_SOURCE_PATH,
    Path("python/carnot/analysis/pbit_sampler_portability.py"),
    Path("python/carnot/experiment_6657_bounded_treewidth_ising_reference.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("openspec/capabilities/ising-backend/spec.md"),
    SAMPLER_SPEC_PATH,
    Path("openspec/capabilities/research-reporting/spec.md"),
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "prototype_hash",
    "fixture_hashes",
    "rows",
    "seed_rows",
    "condition_rows",
    "arm_rows",
    "chain_rows",
    "budget_rows",
    "effective_sample_size_rows",
    "autocorrelation_rows",
    "acceptance_rows",
    "total_variation_rows",
    "energy_moment_rows",
    "mode_occupancy_rows",
    "wall_time_rows",
    "failure_rows",
    "matched_budget_verified",
    "finite_parity_verified",
    "hardware_execution_claimed",
    "asymptotic_scaling_claimed",
    "sampler_benchmark_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "A reason for every required field makes the evidence contract auditable.",
    "preconditions_checked": "Measured gates stop all chains when a required producer or host resource fails.",
    "run_date": "The fixed execution date identifies this evidence window.",
    "inference_substrate": "The declaration limits inference to the matched-budget CPU sampler benchmark.",
    "inference_substrate_class": "The closed class separates CPU simulation from a blocked no-run result.",
    "execution_venue": "The host venue prevents an attached-hardware interpretation.",
    "duration_s": "Monotonic elapsed time reports the complete host run without claiming speed portability.",
    "source_artifact_hashes": "Hashes bind each prior artifact, implementation, verifier, and specification input.",
    "prototype_hash": "The producer artifact hash binds this benchmark to the corrected Exp7133 result.",
    "fixture_hashes": "Condition hashes bind every size, coupling, field, and temperature.",
    "rows": "One summary row per seed, cell, and arm prevents aggregate-only claims.",
    "seed_rows": "Seed summaries prove that no stochastic unit was pooled or removed.",
    "condition_rows": "Condition rows expose the full frustration and temperature design.",
    "arm_rows": "Arm summaries show which methods were eligible in each condition.",
    "chain_rows": "Chain traces retain status, samples, exact-energy costs, and correction facts per unit.",
    "budget_rows": "Budget rows compare charged exact-energy evaluations with the frozen allowance.",
    "effective_sample_size_rows": "Observable-level effective sample sizes report mixing within the fixed budget.",
    "autocorrelation_rows": "Complete frozen lag windows prevent favorable truncation after outcomes are known.",
    "acceptance_rows": "Acceptance or state-change rates expose proposal behavior per unit.",
    "total_variation_rows": "Finite cells compare empirical state mass with an independent exact law.",
    "energy_moment_rows": "Energy means and variances expose distributional parity per unit.",
    "mode_occupancy_rows": "Signed magnetization occupancy exposes mode traversal per unit.",
    "wall_time_rows": "Host wall time is reported per unit without a hardware or scaling claim.",
    "failure_rows": "Every failed or timed-out unit remains visible to reducers.",
    "matched_budget_verified": "This flag passes only when every completed arm spends its exact declared budget.",
    "finite_parity_verified": "This flag passes only when each finite unit reports independent-law total variation.",
    "hardware_execution_claimed": "False records that no FPGA, TSU, or other hardware executed.",
    "asymptotic_scaling_claimed": "False prevents bounded sizes from supporting a scaling inference.",
    "sampler_benchmark_complete_score": "This binary score requires the full roster, budget, metrics, and claim checks.",
    "random_seed": "A fixed experiment seed binds deterministic stream separation and replay.",
    "reproducibility_checksum": "The canonical digest detects any change to published evidence.",
    "gate_check_summary": "The summary retains the exact failed producer field and values, or a complete pass.",
    "verifier_is_oracle": "False records that the benchmark statistic is not used to generate sampler transitions.",
    "verdict_class": "The closed terminal class distinguishes a measured advantage, null, block, or failure.",
    "honest_verdict": "The text states the bounded outcome and agrees with the terminal class.",
}


@dataclass(frozen=True)
class BenchmarkCell:
    """Store one fixed lattice condition and whether exact enumeration is allowed."""

    cell_id: str
    size: int
    frustration: str
    temperature: float
    edges: tuple[tuple[int, int, float], ...]
    fields: tuple[float, ...]
    enumerated: bool

    @property
    def n_spins(self) -> int:
        """Return the site count for this square lattice."""

        return self.size * self.size

    @property
    def fixture_hash(self) -> str:
        """Bind every value that changes this condition or its exact-law scope."""

        return sha256_json(
            {
                "cell_id": self.cell_id,
                "size": self.size,
                "frustration": self.frustration,
                "temperature": self.temperature,
                "edges": [list(edge) for edge in self.edges],
                "fields": list(self.fields),
                "enumerated": self.enumerated,
            }
        )


@dataclass(frozen=True)
class AnalysisPlan:
    """Freeze all choices that could otherwise move after arm outcomes are read."""

    burn_in_steps: int = 32
    thinning: int = 2
    lag_window: int = 32
    observables: tuple[str, ...] = (
        "energy",
        "magnetization",
        "positive_mode_indicator",
    )
    energy_evaluation_budget: int = 5064
    timeout_s: float = 10.0
    failure_policy: str = "retain_failed_and_timed_out_units_as_rows"


@dataclass(frozen=True)
class MultiscaleLaw:
    """Store a normalized coarse law and fixed conditional fine parameters."""

    blocks: tuple[tuple[int, int], ...]
    coarse_states: tuple[SpinState, ...]
    coarse_probabilities: tuple[float, ...]
    fine_coefficients: tuple[tuple[float, float], ...]


@dataclass
class EnergyCounter:
    """Count each exact energy call and stop an arm at its fixed allowance."""

    cell: BenchmarkCell
    budget: int
    evaluations: int = 0

    def evaluate(self, state: SpinState) -> float:
        """Charge one full-state energy call before computing its value."""

        if self.evaluations >= self.budget:
            raise RuntimeError("energy evaluation budget exceeded")
        self.evaluations += 1
        return _energy(self.cell, state)

    def verify_spent(self) -> None:
        """Reject an arm that leaves exact-energy work unused."""

        if self.evaluations != self.budget:
            raise RuntimeError("energy evaluation budget was not spent exactly")


def canonical_json(value: Any) -> bytes:
    """Encode stable JSON and reject nonfinite evidence before publication."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return a tagged digest for one canonical JSON value."""

    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one required file without giving a missing file a valid digest."""

    return (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
    )


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind all final fields except the digest field that stores this checksum."""

    material = dict(payload)
    material["reproducibility_checksum"] = ""
    return sha256_json(material)


def _grid_edges(size: int, frustration: str) -> tuple[tuple[int, int, float], ...]:
    edges: list[tuple[int, int, float]] = []
    for row in range(size):
        for column in range(size - 1):
            left = row * size + column
            sign = -1.0 if row == 0 and (frustration == "high" or column == 0) else 1.0
            magnitude = 0.78 + 0.03 * ((row + column) % 5)
            edges.append((left, left + 1, sign * magnitude))
    for row in range(size - 1):
        for column in range(size):
            top = row * size + column
            magnitude = 0.82 + 0.025 * ((2 * row + column) % 5)
            edges.append((top, top + size, magnitude))
    return tuple(edges)


def frozen_cells() -> tuple[BenchmarkCell, ...]:
    """Return the fixed size-by-frustration-by-temperature benchmark grid."""

    cells: list[BenchmarkCell] = []
    for size in (2, 4):
        for frustration in ("low", "high"):
            for temperature in (0.8, 1.6):
                fields = tuple(
                    (0.035 + 0.005 * (index % 4)) * (-1.0 if index % 2 else 1.0)
                    for index in range(size * size)
                )
                cells.append(
                    BenchmarkCell(
                        cell_id=f"square{size}x{size}_{frustration}_t{temperature:.1f}",
                        size=size,
                        frustration=frustration,
                        temperature=temperature,
                        edges=_grid_edges(size, frustration),
                        fields=fields,
                        enumerated=size == 2,
                    )
                )
    return tuple(cells)


def frozen_analysis_plan() -> AnalysisPlan:
    """Return a new immutable copy of the preregistered analysis choices."""

    return AnalysisPlan()


def replace_cell(cell: BenchmarkCell, **changes: Any) -> BenchmarkCell:
    """Create an immutable cell mutation for fail-closed validation tests."""

    return replace(cell, **changes)


def _expected_edges(size: int) -> set[tuple[int, int]]:
    expected: set[tuple[int, int]] = set()
    for row in range(size):
        for column in range(size - 1):
            site = row * size + column
            expected.add((site, site + 1))
    for row in range(size - 1):
        for column in range(size):
            site = row * size + column
            expected.add((site, site + size))
    return expected


def validate_cell(cell: BenchmarkCell) -> JsonDict:
    """Reject cells outside the fixed bounded square-lattice design."""

    if cell.size not in (2, 4):
        raise ValueError("size must be one of the supported bounded values")
    if not math.isfinite(float(cell.temperature)) or cell.temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if len(cell.fields) != cell.n_spins:
        raise ValueError("field count must match the square lattice")
    if not all(math.isfinite(float(value)) for value in cell.fields):
        raise ValueError("fields must be finite")
    edge_map: dict[tuple[int, int], float] = {}
    for left, right, coupling in cell.edges:
        if not 0 <= left < cell.n_spins or not 0 <= right < cell.n_spins:
            raise ValueError("edge endpoint lies outside the square lattice")
        if left == right:
            raise ValueError("self-loop is unsupported")
        edge = tuple(sorted((left, right)))
        if edge in edge_map:
            raise ValueError("duplicate edge is unsupported")
        if not math.isfinite(float(coupling)):
            raise ValueError("couplings must be finite")
        edge_map[edge] = float(coupling)
    if set(edge_map) != _expected_edges(cell.size):
        raise ValueError("edge set must equal the open square lattice")
    frustrated = 0
    for row in range(cell.size - 1):
        for column in range(cell.size - 1):
            top_left = row * cell.size + column
            top_right = top_left + 1
            bottom_left = top_left + cell.size
            bottom_right = bottom_left + 1
            loop = (
                edge_map[(top_left, top_right)]
                * edge_map[(top_left, bottom_left)]
                * edge_map[(top_right, bottom_right)]
                * edge_map[(bottom_left, bottom_right)]
            )
            frustrated += int(loop < 0.0)
    if frustrated == 0:
        raise ValueError("cell must contain at least one frustrated plaquette")
    if cell.frustration not in ("low", "high"):
        raise ValueError("frustration label must be low or high")
    if cell.enumerated is not (cell.size == 2):
        raise ValueError("enumerated scope must match the frozen size boundary")
    return {
        "passed": True,
        "n_spins": cell.n_spins,
        "edge_count": len(cell.edges),
        "frustrated_plaquettes": frustrated,
    }


def eligible_arms(cell: BenchmarkCell) -> tuple[str, ...]:
    """Return exact-law draws only when complete enumeration is permitted."""

    validate_cell(cell)
    return (
        ("multiscale_mh", "local_gibbs", "exact_law")
        if cell.enumerated
        else (
            "multiscale_mh",
            "local_gibbs",
        )
    )


def unit_id(cell_id: str, seed: int, arm: str) -> str:
    """Build the stable identity shared by every per-unit evidence table."""

    return f"{cell_id}::seed={int(seed)}::arm={arm}"


def _energy(cell: BenchmarkCell, state: Sequence[int]) -> float:
    spins = tuple(int(value) for value in state)
    if len(spins) != cell.n_spins or any(value not in (-1, 1) for value in spins):
        raise ValueError("state must contain one -1 or +1 value per site")
    favorable = sum(coupling * spins[left] * spins[right] for left, right, coupling in cell.edges)
    favorable += sum(field * spins[index] for index, field in enumerate(cell.fields))
    return -float(favorable)


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return tuple(value / total for value in weights)


def _multiscale_law(cell: BenchmarkCell) -> MultiscaleLaw:
    """Generalize the corrected paired coarse-to-fine proposal to bounded grids."""

    blocks = tuple(
        (row * cell.size + column, row * cell.size + column + 1)
        for row in range(cell.size)
        for column in range(0, cell.size, 2)
    )
    block_of = {site: index for index, block in enumerate(blocks) for site in block}
    edge_map = {tuple(sorted((left, right))): coupling for left, right, coupling in cell.edges}
    proposal_temperature = cell.temperature * prototype.PROPOSAL_TEMPERATURE_MULTIPLIER
    coarse_states = tuple(product((-1, 1), repeat=len(blocks)))
    coarse_scores: list[float] = []
    for coarse in coarse_states:
        score = sum(
            0.5 * (cell.fields[anchor] + cell.fields[fine]) * coarse[index]
            for index, (anchor, fine) in enumerate(blocks)
        )
        score += prototype.COARSE_INTERACTION_SHRINKAGE * sum(
            coupling * coarse[block_of[left]] * coarse[block_of[right]]
            for left, right, coupling in cell.edges
            if block_of[left] != block_of[right]
        )
        coarse_scores.append(score / proposal_temperature)
    coefficients = tuple(
        (edge_map[(anchor, fine)] + cell.fields[fine], edge_map[(anchor, fine)] - cell.fields[fine])
        for anchor, fine in blocks
    )
    return MultiscaleLaw(blocks, coarse_states, _softmax(coarse_scores), coefficients)


def _coarse_and_fine(law: MultiscaleLaw, state: SpinState) -> tuple[SpinState, SpinState]:
    coarse = tuple(state[anchor] for anchor, _fine in law.blocks)
    fine = tuple(state[anchor] * state[fine_site] for anchor, fine_site in law.blocks)
    return coarse, fine


def _fine_plus_probability(
    cell: BenchmarkCell, law: MultiscaleLaw, block_index: int, coarse_spin: int
) -> float:
    positive, negative = law.fine_coefficients[block_index]
    coefficient = positive if coarse_spin == 1 else negative
    scale = cell.temperature * prototype.PROPOSAL_TEMPERATURE_MULTIPLIER
    return 1.0 / (1.0 + math.exp(-2.0 * coefficient / scale))


def _proposal_log_probability(cell: BenchmarkCell, law: MultiscaleLaw, state: SpinState) -> float:
    coarse, fine = _coarse_and_fine(law, state)
    coarse_index = law.coarse_states.index(coarse)
    value = math.log(law.coarse_probabilities[coarse_index])
    for index, (coarse_spin, fine_spin) in enumerate(zip(coarse, fine, strict=True)):
        plus = _fine_plus_probability(cell, law, index, coarse_spin)
        value += math.log(plus if fine_spin == 1 else 1.0 - plus)
    return value


def _draw_multiscale(cell: BenchmarkCell, law: MultiscaleLaw, rng: random.Random) -> SpinState:
    draw = rng.random()
    running = 0.0
    coarse = law.coarse_states[-1]
    for candidate, probability in zip(law.coarse_states, law.coarse_probabilities, strict=True):
        running += probability
        if draw <= running:
            coarse = candidate
            break
    result = [0] * cell.n_spins
    for index, ((anchor, fine_site), coarse_spin) in enumerate(
        zip(law.blocks, coarse, strict=True)
    ):
        relative = 1 if rng.random() < _fine_plus_probability(cell, law, index, coarse_spin) else -1
        result[anchor] = coarse_spin
        result[fine_site] = coarse_spin * relative
    return tuple(result)


def _stream_seed(seed: int, cell_id: str, arm: str) -> int:
    digest = hashlib.sha256(f"exp7134:{seed}:{cell_id}:{arm}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _initial_state(seed: int, cell: BenchmarkCell) -> SpinState:
    rng = random.Random(_stream_seed(seed, cell.cell_id, "matched_initial_state"))
    return tuple(1 if rng.random() < 0.5 else -1 for _ in range(cell.n_spins))


def _chain_sample_count(plan: AnalysisPlan) -> int:
    remaining = plan.energy_evaluation_budget - 2 * plan.burn_in_steps
    per_sample = 2 * plan.thinning + 1
    if remaining <= 0 or remaining % per_sample:
        raise ValueError("energy budget must exactly fund the frozen chain plan")
    return remaining // per_sample


def _exact_instance(cell: BenchmarkCell, seed: int) -> exact_reference.IsingInstance:
    return exact_reference.IsingInstance(
        cell.cell_id,
        cell.n_spins,
        cell.edges,
        cell.fields,
        cell.temperature,
        seed,
        "frustrated_square",
    )


def _draw_weighted(
    states: Sequence[SpinState], probabilities: Sequence[float], rng: random.Random
) -> SpinState:
    draw = rng.random()
    running = 0.0
    for state, probability in zip(states, probabilities, strict=True):
        running += float(probability)
        if draw <= running:
            return tuple(state)
    return tuple(states[-1])


def run_arm(cell: BenchmarkCell, *, arm: str, seed: int, plan: AnalysisPlan) -> JsonDict:
    """Run one arm while charging every exact full-state energy evaluation."""

    validate_cell(cell)
    if arm not in ("multiscale_mh", "local_gibbs", "exact_law"):
        raise ValueError("arm is not recognized")
    if arm not in eligible_arms(cell):
        raise ValueError("exact_law is allowed only for enumerated cells")
    if plan != frozen_analysis_plan():
        raise ValueError("analysis plan must equal the frozen plan")
    started = time.monotonic()
    rng = random.Random(_stream_seed(seed, cell.cell_id, arm))
    counter = EnergyCounter(cell, plan.energy_evaluation_budget)

    accepted = 0
    attempted = 0
    states: list[SpinState] = []
    if arm == "exact_law":
        reference = exact_reference.brute_force_reference(_exact_instance(cell, seed))
        exact_states = tuple(tuple(int(value) for value in state) for state in reference["states"])
        probabilities = tuple(float(value) for value in reference["probabilities"])
        counter.evaluations += len(exact_states)
        sample_count = plan.energy_evaluation_budget - counter.evaluations
        if sample_count <= plan.lag_window:
            raise ValueError("exact-law budget does not fund the frozen lag window")
        states = [_draw_weighted(exact_states, probabilities, rng) for _ in range(sample_count)]
        energies = [counter.evaluate(state) for state in states]
        accepted = sample_count
        attempted = sample_count
    else:
        sample_count = _chain_sample_count(plan)
        state = _initial_state(seed, cell)
        law = _multiscale_law(cell) if arm == "multiscale_mh" else None
        transition_count = plan.burn_in_steps + plan.thinning * sample_count
        for step in range(transition_count):
            attempted += 1
            previous = state
            if arm == "multiscale_mh":
                assert law is not None
                proposed = _draw_multiscale(cell, law, rng)
                current_energy = counter.evaluate(state)
                proposed_energy = counter.evaluate(proposed)
                log_acceptance = (
                    (-proposed_energy + current_energy) / cell.temperature
                    + _proposal_log_probability(cell, law, state)
                    - _proposal_log_probability(cell, law, proposed)
                )
                if math.log(max(rng.random(), 1.0e-300)) < min(0.0, log_acceptance):
                    state = proposed
            else:
                site = step % cell.n_spins
                minus = list(state)
                plus = list(state)
                minus[site] = -1
                plus[site] = 1
                minus_state = tuple(minus)
                plus_state = tuple(plus)
                minus_energy = counter.evaluate(minus_state)
                plus_energy = counter.evaluate(plus_state)
                probability_plus = 1.0 / (
                    1.0 + math.exp((plus_energy - minus_energy) / cell.temperature)
                )
                state = plus_state if rng.random() < probability_plus else minus_state
            accepted += int(state != previous)
            if step >= plan.burn_in_steps and (step - plan.burn_in_steps + 1) % plan.thinning == 0:
                states.append(state)
        energies = [counter.evaluate(retained) for retained in states]
    counter.verify_spent()
    duration = max(time.monotonic() - started, 1.0e-9)
    status = "timed_out" if duration > plan.timeout_s else "complete"
    return {
        "status": status,
        "failure_reason": "wall_time_exceeded_frozen_timeout" if status == "timed_out" else None,
        "states": [list(state) for state in states],
        "energies": energies,
        "sample_count": len(states),
        "energy_evaluations": counter.evaluations,
        "wall_time_s": duration,
        "accepted_transitions": accepted,
        "attempted_transitions": attempted,
        "acceptance_rate": accepted / attempted if attempted else None,
        "trace_sha256": sha256_json([list(state) for state in states]),
        "stream_seed": _stream_seed(seed, cell.cell_id, arm),
        "uses_mh_correction": arm == "multiscale_mh",
        "uses_forward_probability": arm == "multiscale_mh",
        "uses_reverse_probability": arm == "multiscale_mh",
        "reference_source": "independent_exp6657_enumerator" if arm == "exact_law" else None,
        "burn_in_steps": 0 if arm == "exact_law" else plan.burn_in_steps,
        "thinning": 1 if arm == "exact_law" else plan.thinning,
    }


def autocorrelation_series(values: Sequence[float], lag_window: int) -> list[float]:
    """Return every preregistered lag, including zero and negative correlations."""

    if lag_window < 1 or len(values) <= lag_window:
        raise ValueError("lag window must be smaller than the sample count")
    numeric = [float(value) for value in values]
    mean = sum(numeric) / len(numeric)
    variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
    if variance <= 0.0:
        return [1.0] + [0.0] * lag_window
    result = [1.0]
    for lag in range(1, lag_window + 1):
        covariance = sum(
            (numeric[index] - mean) * (numeric[index + lag] - mean)
            for index in range(len(numeric) - lag)
        ) / (len(numeric) - lag)
        result.append(covariance / variance)
    return result


def _observables(states: Sequence[SpinState], energies: Sequence[float]) -> dict[str, list[float]]:
    magnetization = [sum(state) / len(state) for state in states]
    return {
        "energy": [float(value) for value in energies],
        "magnetization": magnetization,
        "positive_mode_indicator": [float(value > 0.0) for value in magnetization],
    }


def _integrated_autocorrelation(series: Sequence[float]) -> tuple[float, int]:
    positive: list[float] = []
    stop = 1
    for lag, value in enumerate(series[1:], start=1):
        stop = lag
        if value <= 0.0:
            break
        positive.append(float(value))
    return max(1.0, 1.0 + 2.0 * sum(positive)), stop


def _exact_law(cell: BenchmarkCell) -> tuple[tuple[SpinState, ...], tuple[float, ...]]:
    reference = exact_reference.brute_force_reference(_exact_instance(cell, RANDOM_SEED))
    states = tuple(tuple(int(value) for value in state) for state in reference["states"])
    probabilities = tuple(float(value) for value in reference["probabilities"])
    return states, probabilities


def compute_metrics(
    cell: BenchmarkCell,
    arm: str,
    seed: int,
    result: Mapping[str, Any],
    plan: AnalysisPlan,
) -> JsonDict:
    """Compute every success-only metric from one retained state trace."""

    if result.get("status") != "complete":
        raise ValueError("metrics require a complete chain")
    states = [tuple(int(value) for value in state) for state in result["states"]]
    energies = [float(value) for value in result["energies"]]
    uid = unit_id(cell.cell_id, seed, arm)
    observable_values = _observables(states, energies)
    autocorrelation_rows: list[JsonDict] = []
    ess_rows: list[JsonDict] = []
    for observable in plan.observables:
        correlations = autocorrelation_series(observable_values[observable], plan.lag_window)
        integrated, stop = _integrated_autocorrelation(correlations)
        ess = min(float(len(states)), len(states) / integrated)
        autocorrelation_rows.append(
            {
                "unit_id": uid,
                "cell_id": cell.cell_id,
                "seed": seed,
                "arm": arm,
                "observable": observable,
                "lag_window": plan.lag_window,
                "autocorrelations": correlations,
                "integration_stop_lag": stop,
                "integrated_autocorrelation": integrated,
            }
        )
        ess_rows.append(
            {
                "unit_id": uid,
                "cell_id": cell.cell_id,
                "seed": seed,
                "arm": arm,
                "observable": observable,
                "sample_count": len(states),
                "effective_sample_size": ess,
            }
        )
    mean_energy = sum(energies) / len(energies)
    second_moment = sum(value * value for value in energies) / len(energies)
    variance = max(0.0, second_moment - mean_energy * mean_energy)
    magnetizations = observable_values["magnetization"]
    negative = sum(value < 0.0 for value in magnetizations) / len(magnetizations)
    zero = sum(value == 0.0 for value in magnetizations) / len(magnetizations)
    positive = sum(value > 0.0 for value in magnetizations) / len(magnetizations)
    total_variation: float | None = None
    target_mean: float | None = None
    target_variance: float | None = None
    if cell.enumerated:
        exact_states, probabilities = _exact_law(cell)
        counts = {state: 0 for state in exact_states}
        for state in states:
            counts[state] += 1
        total_variation = 0.5 * sum(
            abs(counts[state] / len(states) - probability)
            for state, probability in zip(exact_states, probabilities, strict=True)
        )
        exact_energies = [_energy(cell, state) for state in exact_states]
        target_mean = sum(
            probability * energy
            for probability, energy in zip(probabilities, exact_energies, strict=True)
        )
        target_second = sum(
            probability * energy * energy
            for probability, energy in zip(probabilities, exact_energies, strict=True)
        )
        target_variance = max(0.0, target_second - target_mean * target_mean)
    return {
        "effective_sample_size_rows": ess_rows,
        "autocorrelation_rows": autocorrelation_rows,
        "acceptance_rows": [
            {
                "unit_id": uid,
                "cell_id": cell.cell_id,
                "seed": seed,
                "arm": arm,
                "accepted_transitions": result["accepted_transitions"],
                "attempted_transitions": result["attempted_transitions"],
                "acceptance_rate": result["acceptance_rate"],
            }
        ],
        "total_variation_rows": [
            {
                "unit_id": uid,
                "cell_id": cell.cell_id,
                "seed": seed,
                "arm": arm,
                "enumerated": cell.enumerated,
                "total_variation": total_variation,
            }
        ],
        "energy_moment_rows": [
            {
                "unit_id": uid,
                "cell_id": cell.cell_id,
                "seed": seed,
                "arm": arm,
                "mean_energy": mean_energy,
                "energy_second_moment": second_moment,
                "energy_variance": variance,
                "target_mean_energy": target_mean,
                "target_energy_variance": target_variance,
            }
        ],
        "mode_occupancy_rows": [
            {
                "unit_id": uid,
                "cell_id": cell.cell_id,
                "seed": seed,
                "arm": arm,
                "negative_fraction": negative,
                "zero_fraction": zero,
                "positive_fraction": positive,
            }
        ],
    }


def _plan_payload() -> JsonDict:
    payload = asdict(frozen_analysis_plan())
    payload["observables"] = list(payload["observables"])
    return payload


def _source_hashes(root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(root / path) for path in SOURCE_ARTIFACT_PATHS}


def collect_preconditions(root: Path) -> list[JsonDict]:
    """Verify producer identity, host resources, sealed laws, seeds, and paths."""

    artifact_path = root / PROTOTYPE_ARTIFACT_PATH
    try:
        producer = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        producer = {}
    artifact_hash = sha256_file(artifact_path)
    code_hash = sha256_file(root / PROTOTYPE_SOURCE_PATH)
    ram_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    cpu_count = os.cpu_count() or 0
    finite_receipts: list[JsonDict] = []
    try:
        for cell in (item for item in frozen_cells() if item.enumerated):
            reference = exact_reference.brute_force_reference(_exact_instance(cell, RANDOM_SEED))
            finite_receipts.append(
                {
                    "cell_id": cell.cell_id,
                    "state_count": len(reference["states"]),
                    "probability_mass": sum(reference["probabilities"]),
                    "sealed_law_hash": sha256_json(
                        {
                            "states": [list(state) for state in reference["states"]],
                            "probabilities": reference["probabilities"],
                        }
                    ),
                }
            )
        finite_ready = len(finite_receipts) == 4 and all(
            row["state_count"] == 16 and abs(row["probability_mass"] - 1.0) <= FLOAT_TOLERANCE
            for row in finite_receipts
        )
    except (TypeError, ValueError):
        finite_ready = False
    output_paths = [root / "results", root / "scripts/experiments", root / "python/carnot"]
    checks = [
        {
            "resource": "multiscale_sampler_ready_score",
            "producer_field": "multiscale_sampler_ready_score",
            "available": producer.get("multiscale_sampler_ready_score") == 1,
            "expected_value": 1,
            "observed_value": producer.get("multiscale_sampler_ready_score"),
        },
        {
            "resource": "prototype_artifact_hash",
            "producer_field": "artifact_sha256",
            "available": artifact_hash == PROTOTYPE_ARTIFACT_HASH,
            "expected_value": PROTOTYPE_ARTIFACT_HASH,
            "observed_value": artifact_hash,
        },
        {
            "resource": "prototype_code_hash",
            "producer_field": "code_hash",
            "available": code_hash == producer.get("code_hash"),
            "expected_value": producer.get("code_hash"),
            "observed_value": code_hash,
        },
        {
            "resource": "host_resources",
            "producer_field": "host_resources",
            "available": ram_bytes >= 1 << 30 and cpu_count >= 1,
            "expected_value": {"minimum_ram_bytes": 1 << 30, "minimum_cpu_count": 1},
            "observed_value": {"ram_bytes": ram_bytes, "cpu_count": cpu_count},
        },
        {
            "resource": "sealed_finite_laws",
            "producer_field": "sealed_finite_laws",
            "available": finite_ready,
            "expected_value": "four_normalized_16_state_laws",
            "observed_value": finite_receipts,
        },
        {
            "resource": "fixed_seeds",
            "producer_field": "fixed_seeds",
            "available": len(FROZEN_SEEDS) >= 5 and len(set(FROZEN_SEEDS)) == len(FROZEN_SEEDS),
            "expected_value": list(FROZEN_SEEDS),
            "observed_value": list(FROZEN_SEEDS),
        },
        {
            "resource": "output_paths",
            "producer_field": "output_paths",
            "available": all(path.is_dir() and os.access(path, os.W_OK) for path in output_paths),
            "expected_value": "all_required_paths_writable",
            "observed_value": {
                str(path.relative_to(root)): path.is_dir() and os.access(path, os.W_OK)
                for path in output_paths
            },
        },
    ]
    return checks


def _base_artifact(root: Path, run_date: str, checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": "carnot.experiment_7134.matched_budget_sampler_benchmark.v1",
        "experiment_id": 7134,
        "status": "building",
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": "host",
        "duration_s": 0.0,
        "source_artifact_hashes": _source_hashes(root),
        "prototype_hash": sha256_file(root / PROTOTYPE_ARTIFACT_PATH),
        "fixture_hashes": {cell.cell_id: cell.fixture_hash for cell in frozen_cells()},
        "rows": [],
        "seed_rows": [],
        "condition_rows": [],
        "arm_rows": [],
        "chain_rows": [],
        "budget_rows": [],
        "effective_sample_size_rows": [],
        "autocorrelation_rows": [],
        "acceptance_rows": [],
        "total_variation_rows": [],
        "energy_moment_rows": [],
        "mode_occupancy_rows": [],
        "wall_time_rows": [],
        "failure_rows": [],
        "matched_budget_verified": False,
        "finite_parity_verified": False,
        "hardware_execution_claimed": False,
        "asymptotic_scaling_claimed": False,
        "sampler_benchmark_complete_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_precondition_failed: no sampler chain ran",
        "analysis_plan": _plan_payload(),
        "pooled_comparison": {},
        "scope": {
            "language": "Python",
            "venue": "host",
            "bounded_sizes": [2, 4],
            "rust_parity": False,
            "fpga_execution": False,
            "tsu_execution": False,
            "power_claim": False,
            "logarithmic_scaling_inference": False,
        },
        "methodology": (
            "Each completed arm spends 5064 exact full-state energy evaluations. "
            "The plan was fixed before any chain outcome was read."
        ),
    }


def _failed_result(reason: str) -> JsonDict:
    return {
        "status": "failed",
        "failure_reason": reason,
        "states": [],
        "energies": [],
        "sample_count": 0,
        "energy_evaluations": 0,
        "wall_time_s": 0.0,
        "accepted_transitions": 0,
        "attempted_transitions": 0,
        "acceptance_rate": None,
        "trace_sha256": sha256_json([]),
        "stream_seed": None,
        "uses_mh_correction": False,
        "uses_forward_probability": False,
        "uses_reverse_probability": False,
        "reference_source": None,
        "burn_in_steps": None,
        "thinning": None,
    }


def _paired_comparison(ess_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    scores: dict[tuple[str, int, str], float] = {}
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for row in ess_rows:
        key = (str(row["cell_id"]), int(row["seed"]), str(row["arm"]))
        grouped.setdefault(key, []).append(float(row["effective_sample_size"]))
    for key, values in grouped.items():
        scores[key] = min(values)
    deltas: list[float] = []
    pair_rows: list[JsonDict] = []
    for cell in frozen_cells():
        for seed in FROZEN_SEEDS:
            treatment = scores.get((cell.cell_id, seed, "multiscale_mh"))
            baseline = scores.get((cell.cell_id, seed, "local_gibbs"))
            if treatment is None or baseline is None:
                continue
            delta = treatment - baseline
            deltas.append(delta)
            pair_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "seed": seed,
                    "multiscale_minimum_observable_ess": treatment,
                    "local_gibbs_minimum_observable_ess": baseline,
                    "paired_ess_delta": delta,
                }
            )
    wins = sum(value > FLOAT_TOLERANCE for value in deltas)
    losses = sum(value < -FLOAT_TOLERANCE for value in deltas)
    ties = len(deltas) - wins - losses
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    contradictory = bool(deltas and losses >= wins)
    return {
        "statistic": "mean paired difference in minimum observable ESS",
        "comparison_arm": "multiscale_mh",
        "baseline_arm": "local_gibbs",
        "paired_unit_count": len(deltas),
        "mean_paired_ess_delta": mean_delta,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "pair_rows": pair_rows,
        "row_consistency_findings": ["wins_not_exceeding_losses"] if contradictory else [],
        "positive_advantage": bool(
            mean_delta is not None and mean_delta > 0.0 and not contradictory
        ),
    }


def build_artifact(
    *,
    root: Path,
    run_date: str,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    forced_failures: Mapping[str, str] | None = None,
) -> JsonDict:
    """Build complete per-unit evidence or one terminal blocked artifact."""

    started = time.monotonic()
    checks = list(preconditions) if preconditions is not None else collect_preconditions(root)
    artifact = _base_artifact(root, run_date, checks)
    failed_check = next((row for row in checks if row.get("available") is not True), None)
    if failed_check is not None:
        artifact["status"] = f"blocked_{failed_check['resource']}"
        artifact["inference_substrate_class"] = "blocked_no_run"
        artifact["gate_check_summary"] = {
            "failed_check": failed_check["resource"],
            "producer_field": failed_check.get("producer_field", failed_check["resource"]),
            "expected_value": failed_check.get("expected_value"),
            "observed_value": failed_check.get("observed_value"),
            "passed": False,
        }
        artifact["honest_verdict"] = f"blocked_{failed_check['resource']}: no sampler chain ran"
        artifact["duration_s"] = max(time.monotonic() - started, 1.0e-9)
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact

    artifact["gate_check_summary"] = {
        "failed_check": None,
        "producer_field": None,
        "expected_value": "all_preconditions_available",
        "observed_value": "all_preconditions_available",
        "passed": True,
    }
    forced = dict(forced_failures or {})
    plan = frozen_analysis_plan()
    cells = frozen_cells()
    condition_rows: list[JsonDict] = []
    for cell in cells:
        receipt = validate_cell(cell)
        condition_rows.append(
            {
                "cell_id": cell.cell_id,
                "size": cell.size,
                "n_spins": cell.n_spins,
                "frustration": cell.frustration,
                "temperature": cell.temperature,
                "enumerated": cell.enumerated,
                "edges": [list(edge) for edge in cell.edges],
                "fields": list(cell.fields),
                "eligible_arms": list(eligible_arms(cell)),
                "fixture_hash": cell.fixture_hash,
                **receipt,
            }
        )
    artifact["condition_rows"] = condition_rows

    for cell in cells:
        for seed in FROZEN_SEEDS:
            for arm in eligible_arms(cell):
                uid = unit_id(cell.cell_id, seed, arm)
                result = (
                    _failed_result(forced[uid])
                    if uid in forced
                    else run_arm(cell, arm=arm, seed=seed, plan=plan)
                )
                chain = {
                    "unit_id": uid,
                    "cell_id": cell.cell_id,
                    "seed": seed,
                    "arm": arm,
                    **result,
                }
                artifact["chain_rows"].append(chain)
                artifact["budget_rows"].append(
                    {
                        "unit_id": uid,
                        "cell_id": cell.cell_id,
                        "seed": seed,
                        "arm": arm,
                        "status": result["status"],
                        "budget": plan.energy_evaluation_budget,
                        "energy_evaluations": result["energy_evaluations"],
                        "matched": result["status"] == "complete"
                        and result["energy_evaluations"] == plan.energy_evaluation_budget,
                    }
                )
                artifact["wall_time_rows"].append(
                    {
                        "unit_id": uid,
                        "cell_id": cell.cell_id,
                        "seed": seed,
                        "arm": arm,
                        "status": result["status"],
                        "wall_time_s": result["wall_time_s"],
                    }
                )
                artifact["failure_rows"].append(
                    {
                        "unit_id": uid,
                        "cell_id": cell.cell_id,
                        "seed": seed,
                        "arm": arm,
                        "failed": result["status"] != "complete",
                        "timed_out": result["status"] == "timed_out",
                        "failure_reason": result["failure_reason"],
                    }
                )
                row: JsonDict = {
                    "unit_id": uid,
                    "cell_id": cell.cell_id,
                    "seed": seed,
                    "arm": arm,
                    "status": result["status"],
                    "energy_evaluations": result["energy_evaluations"],
                    "wall_time_s": result["wall_time_s"],
                    "failed": result["status"] != "complete",
                }
                if result["status"] == "complete":
                    metrics = compute_metrics(cell, arm, seed, result, plan)
                    for field, metric_rows in metrics.items():
                        artifact[field].extend(metric_rows)
                    ess_values = [
                        item["effective_sample_size"]
                        for item in metrics["effective_sample_size_rows"]
                    ]
                    row.update(
                        {
                            "minimum_observable_ess": min(ess_values),
                            "maximum_integrated_autocorrelation": max(
                                item["integrated_autocorrelation"]
                                for item in metrics["autocorrelation_rows"]
                            ),
                            "acceptance_rate": result["acceptance_rate"],
                            "total_variation": metrics["total_variation_rows"][0][
                                "total_variation"
                            ],
                            "mean_energy": metrics["energy_moment_rows"][0]["mean_energy"],
                            "positive_mode_fraction": metrics["mode_occupancy_rows"][0][
                                "positive_fraction"
                            ],
                        }
                    )
                artifact["rows"].append(row)

    expected_per_seed = sum(len(eligible_arms(cell)) for cell in cells)
    artifact["seed_rows"] = [
        {
            "seed": seed,
            "unit_count": sum(row["seed"] == seed for row in artifact["chain_rows"]),
            "expected_unit_count": expected_per_seed,
            "failure_count": sum(
                row["seed"] == seed and row["failed"] for row in artifact["failure_rows"]
            ),
        }
        for seed in FROZEN_SEEDS
    ]
    artifact["arm_rows"] = [
        {
            "cell_id": cell.cell_id,
            "arm": arm,
            "seed_count": sum(
                row["cell_id"] == cell.cell_id and row["arm"] == arm
                for row in artifact["chain_rows"]
            ),
            "failure_count": sum(
                row["cell_id"] == cell.cell_id and row["arm"] == arm and row["failed"]
                for row in artifact["failure_rows"]
            ),
        }
        for cell in cells
        for arm in eligible_arms(cell)
    ]
    failures = sum(row["failed"] for row in artifact["failure_rows"])
    artifact["failure_rate"] = failures / len(artifact["failure_rows"])
    artifact["matched_budget_verified"] = failures == 0 and all(
        row["matched"] is True for row in artifact["budget_rows"]
    )
    artifact["finite_parity_verified"] = failures == 0 and all(
        row["total_variation"] is not None
        for row in artifact["total_variation_rows"]
        if next(cell for cell in cells if cell.cell_id == row["cell_id"]).enumerated
    )
    comparison = _paired_comparison(artifact["effective_sample_size_rows"])
    artifact["pooled_comparison"] = comparison
    complete = (
        failures == 0 and artifact["matched_budget_verified"] and artifact["finite_parity_verified"]
    )
    artifact["sampler_benchmark_complete_score"] = int(complete)
    if failures:
        artifact["status"] = "disqualified_failed_or_timed_out_units"
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = (
            "disqualified: one or more sampler units failed or timed out and remain in the rows"
        )
    elif comparison["positive_advantage"]:
        artifact["status"] = "complete_positive_matched_budget_comparison"
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = (
            "complete: corrected multiscale sampling improves the preregistered bounded host ESS statistic"
        )
    else:
        artifact["status"] = "complete_null_matched_budget_comparison"
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = (
            "null: corrected multiscale sampling has no preregistered bounded host ESS advantage"
        )
    artifact["duration_s"] = max(time.monotonic() - started, 1.0e-9)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _unit_roster() -> set[str]:
    return {
        unit_id(cell.cell_id, seed, arm)
        for cell in frozen_cells()
        for seed in FROZEN_SEEDS
        for arm in eligible_arms(cell)
    }


def _row_units(payload: Mapping[str, Any], field: str) -> list[Any]:
    rows = payload.get(field, [])
    return (
        [row.get("unit_id") if isinstance(row, Mapping) else None for row in rows]
        if isinstance(rows, list)
        else []
    )


def _close(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return stable errors for drifted, pooled, incomplete, or over-claimed evidence."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(payload)
    if missing:
        return ["missing_required_fields"]
    errors: list[str] = []
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("checksum_mismatch")
    principles = payload.get("field_principles")
    if (
        not isinstance(principles, Mapping)
        or set(principles) != set(REQUIRED_ARTIFACT_FIELDS)
        or any(not isinstance(value, str) or not value.strip() for value in principles.values())
    ):
        errors.append("field_principles_invalid")
    if payload.get("analysis_plan") != _plan_payload():
        errors.append("analysis_plan_drift")
    if (
        payload.get("execution_venue") != "host"
        or payload.get("hardware_execution_claimed") is not False
        or payload.get("asymptotic_scaling_claimed") is not False
        or payload.get("scope")
        != {
            "language": "Python",
            "venue": "host",
            "bounded_sizes": [2, 4],
            "rust_parity": False,
            "fpga_execution": False,
            "tsu_execution": False,
            "power_claim": False,
            "logarithmic_scaling_inference": False,
        }
    ):
        errors.append("claim_boundary_invalid")
    duration = payload.get("duration_s")
    if (
        not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or duration <= 0
    ):
        errors.append("duration_invalid")
    if payload.get("verdict_class") == "blocked":
        summary = payload.get("gate_check_summary")
        row_fields = (
            "rows",
            "seed_rows",
            "condition_rows",
            "arm_rows",
            "chain_rows",
            "budget_rows",
            "effective_sample_size_rows",
            "autocorrelation_rows",
            "acceptance_rows",
            "total_variation_rows",
            "energy_moment_rows",
            "mode_occupancy_rows",
            "wall_time_rows",
            "failure_rows",
        )
        if (
            payload.get("inference_substrate_class") != "blocked_no_run"
            or any(payload.get(field) != [] for field in row_fields)
            or payload.get("sampler_benchmark_complete_score") != 0
            or not str(payload.get("honest_verdict", "")).startswith("blocked_")
            or not isinstance(summary, Mapping)
            or summary.get("passed") is not False
            or summary.get("failed_check") is None
            or summary.get("producer_field") is None
            or summary.get("expected_value") is None
            or "observed_value" not in summary
        ):
            errors.append("blocked_terminal_state_invalid")
        return list(dict.fromkeys(errors))
    if (
        payload.get("inference_substrate") != INFERENCE_SUBSTRATE
        or payload.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS
    ):
        errors.append("substrate_class_invalid")
    roster = _unit_roster()
    chain_units = _row_units(payload, "chain_rows")
    if len(chain_units) != len(roster) or set(chain_units) != roster:
        errors.append("seed_pooling_or_row_loss")
    if (
        len(_row_units(payload, "rows")) != len(roster)
        or set(_row_units(payload, "rows")) != roster
    ):
        errors.append("seed_pooling_or_row_loss")
    for row in payload.get("rows", []):
        if not isinstance(row, Mapping):
            errors.append("seed_pooling_or_row_loss")
            break
        chain_by_id = next(
            (
                item
                for item in payload.get("chain_rows", [])
                if isinstance(item, Mapping) and item.get("unit_id") == row.get("unit_id")
            ),
            None,
        )
        if not isinstance(chain_by_id, Mapping) or any(
            row.get(field) != chain_by_id.get(field) for field in ("cell_id", "seed", "arm")
        ):
            errors.append("seed_pooling_or_row_loss")
            break
    for field, error in (
        ("budget_rows", "budget_rows_incomplete"),
        ("wall_time_rows", "wall_time_rows_incomplete"),
        ("failure_rows", "failure_rows_incomplete"),
    ):
        units = _row_units(payload, field)
        if len(units) != len(roster) or set(units) != roster:
            errors.append(error)
    chain_by_unit = {
        row.get("unit_id"): row for row in payload.get("chain_rows", []) if isinstance(row, Mapping)
    }
    budget_by_unit = {
        row.get("unit_id"): row
        for row in payload.get("budget_rows", [])
        if isinstance(row, Mapping)
    }
    failure_by_unit = {
        row.get("unit_id"): row
        for row in payload.get("failure_rows", [])
        if isinstance(row, Mapping)
    }
    completed = {
        uid
        for uid, row in chain_by_unit.items()
        if uid in roster and row.get("status") == "complete"
    }
    for uid in roster & set(chain_by_unit):
        chain = chain_by_unit[uid]
        budget = budget_by_unit.get(uid, {})
        failure = failure_by_unit.get(uid, {})
        failed = chain.get("status") != "complete"
        if failure.get("failed") is not failed:
            errors.append("failure_rows_incomplete")
        if not failed and (
            chain.get("energy_evaluations") != frozen_analysis_plan().energy_evaluation_budget
            or budget.get("energy_evaluations") != chain.get("energy_evaluations")
            or budget.get("budget") != frozen_analysis_plan().energy_evaluation_budget
            or budget.get("matched") is not True
        ):
            errors.append("budget_mismatch")
        if (
            chain.get("arm") == "multiscale_mh"
            and not failed
            and (
                chain.get("uses_mh_correction") is not True
                or chain.get("uses_forward_probability") is not True
                or chain.get("uses_reverse_probability") is not True
            )
        ):
            errors.append("uncorrected_multiscale_proposal")
        if (
            chain.get("arm") == "exact_law"
            and not failed
            and chain.get("reference_source") != "independent_exp6657_enumerator"
        ):
            errors.append("finite_reference_invalid")
    observable_roster = {
        (uid, observable) for uid in completed for observable in frozen_analysis_plan().observables
    }
    for field, error in (
        ("effective_sample_size_rows", "effective_sample_size_rows_incomplete"),
        ("autocorrelation_rows", "autocorrelation_truncated"),
    ):
        rows = payload.get(field, [])
        observed = [
            (row.get("unit_id"), row.get("observable")) for row in rows if isinstance(row, Mapping)
        ]
        if len(observed) != len(observable_roster) or set(observed) != observable_roster:
            errors.append(error)
    for row in payload.get("autocorrelation_rows", []):
        if (
            not isinstance(row, Mapping)
            or row.get("lag_window") != frozen_analysis_plan().lag_window
            or not isinstance(row.get("autocorrelations"), list)
            or len(row["autocorrelations"]) != frozen_analysis_plan().lag_window + 1
        ):
            errors.append("autocorrelation_truncated")
            break
    for field in (
        "acceptance_rows",
        "total_variation_rows",
        "energy_moment_rows",
        "mode_occupancy_rows",
    ):
        units = _row_units(payload, field)
        if len(units) != len(completed) or set(units) != completed:
            errors.append(f"{field}_incomplete")
    cells = {cell.cell_id: cell for cell in frozen_cells()}
    for row in payload.get("total_variation_rows", []):
        cell = cells.get(row.get("cell_id")) if isinstance(row, Mapping) else None
        if (
            cell is None
            or (cell.enumerated and row.get("total_variation") is None)
            or (not cell.enumerated and row.get("total_variation") is not None)
        ):
            errors.append("finite_parity_invalid")
            break
    comparison = _paired_comparison(payload.get("effective_sample_size_rows", []))
    declared = payload.get("pooled_comparison")
    if (
        not isinstance(declared, Mapping)
        or any(
            declared.get(field) != comparison[field]
            for field in (
                "statistic",
                "comparison_arm",
                "baseline_arm",
                "paired_unit_count",
                "wins",
                "losses",
                "ties",
                "row_consistency_findings",
                "positive_advantage",
            )
        )
        or not _close(
            declared.get("mean_paired_ess_delta") if isinstance(declared, Mapping) else None,
            comparison["mean_paired_ess_delta"],
        )
    ):
        errors.append("pooled_claim_mismatch")
    failure_count = sum(row.get("status") != "complete" for row in chain_by_unit.values())
    expected_complete = (
        len(chain_by_unit) == len(roster)
        and failure_count == 0
        and all(row.get("matched") is True for row in budget_by_unit.values())
    )
    if payload.get("matched_budget_verified") is not expected_complete:
        errors.append("budget_mismatch")
    expected_finite = expected_complete and all(
        row.get("total_variation") is not None
        for row in payload.get("total_variation_rows", [])
        if cells.get(row.get("cell_id"), BenchmarkCell("", 2, "low", 1.0, (), (), True)).enumerated
    )
    if payload.get("finite_parity_verified") is not expected_finite:
        errors.append("finite_parity_invalid")
    expected_score = int(expected_complete and expected_finite)
    if payload.get("sampler_benchmark_complete_score") != expected_score:
        errors.append("completion_score_invalid")
    verdict = payload.get("verdict_class")
    honest = str(payload.get("honest_verdict", ""))
    verdict_valid = False
    if failure_count:
        verdict_valid = verdict == "disqualified" and honest.startswith("disqualified:")
    elif comparison["positive_advantage"]:
        verdict_valid = verdict == "positive" and honest.startswith("complete:")
    else:
        verdict_valid = verdict == "null" and honest.startswith("null:")
    if not verdict_valid:
        errors.append("verdict_invalid")
    return list(dict.fromkeys(errors))


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Publish one complete JSON document through same-directory replacement."""

    encoded = canonical_json(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(path), "sha256": sha256_file(path), "atomic_replace": True}


def run_experiment(*, root: Path, output: Path, run_date: str) -> JsonDict:
    """Build, validate, and atomically publish one terminal benchmark artifact."""

    artifact = build_artifact(root=root, run_date=run_date)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp7134 artifact: {errors}")
    write_json_atomic(output, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp7134 or validate a caller-selected artifact."""

    args = _parse_args(argv)
    if args.validate is not None:
        try:
            payload = json.loads(args.validate.read_text(encoding="utf-8"))
            errors = validate_artifact(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors = [f"artifact_read_error:{type(exc).__name__}"]
        if errors:
            print(json.dumps({"errors": errors, "validated": False}, sort_keys=True))
            return 2
        print(f"validated {args.validate}")
        return 0
    artifact = run_experiment(root=REPO_ROOT, output=args.output, run_date=args.date)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sampler_benchmark_complete_score": artifact["sampler_benchmark_complete_score"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["sampler_benchmark_complete_score"] == 1 else 2


if __name__ == "__main__":  # pragma: no cover - the required wrapper owns direct execution.
    raise SystemExit(main())
