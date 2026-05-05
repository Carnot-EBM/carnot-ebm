"""CPU 2D parallel tempering for Ising constraint probes.

**Researcher summary:**
    This module adds a software-only temperature-replica sampler for the Exp
    1387 FoVer/KV260 feasibility check. It keeps the spin update rule close to
    Carnot's checkerboard Ising sampler, but runs multiple temperature replicas
    and exchanges adjacent replicas with the standard Metropolis criterion.

**Detailed explanation for engineers:**
    Parallel tempering reduces local-minimum freezing by running several copies
    of the same Ising problem at different temperatures. Hot replicas explore
    broadly, cold replicas refine low-energy states, and swaps let a useful
    state move through the temperature ladder. In FPGA papers this can be a
    wall-clock win because replicas update in parallel. This module measures
    convergence in sweeps only; it does not synthesize RTL or execute hardware.

Spec: REQ-ISING-021, SCENARIO-ISING-031
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class IsingConstraintProblem:
    """FoVer-derived boolean Ising problem used by the CPU PT probe.

    The arrays use Carnot's existing boolean convention: each spin is ``0`` or
    ``1`` and energy is ``-(b @ s + 0.5 * s @ J @ s)``. ``target_state`` is a
    planted low-energy state derived from the FoVer row hash, which lets the
    N=128 CPU probe use a known convergence threshold without brute forcing
    ``2**128`` assignments.

    Spec: REQ-ISING-021
    """

    name: str
    question_id: str
    label: str
    n_spins: int
    biases: np.ndarray
    coupling_matrix: np.ndarray
    target_state: np.ndarray
    ground_energy: float
    convergence_energy: float


@dataclass(frozen=True)
class IsingRunResult:
    """Single-temperature convergence result."""

    steps_to_convergence: int
    converged: bool
    best_energy: float
    final_energy: float
    best_state: tuple[int, ...]
    energy_trace: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe fields for experiment artifacts."""
        return {
            "steps_to_convergence": self.steps_to_convergence,
            "converged": self.converged,
            "best_energy": self.best_energy,
            "final_energy": self.final_energy,
            "best_state": list(self.best_state),
            "energy_trace": list(self.energy_trace),
        }


@dataclass(frozen=True)
class ParallelTemperingConfig:
    """Configuration for 2D parallel tempering over a temperature ladder.

    ``replica_count`` is the number of temperature replicas, not the number of
    CPU threads. The experiment uses the arXiv:2601.09037-inspired value of 15,
    but the CPU implementation is intentionally serial so it remains portable.

    Spec: REQ-ISING-021
    """

    replica_count: int = 15
    min_temperature: float = 0.5
    max_temperature: float = 5.0
    max_steps: int = 96
    swap_interval: int = 1

    def __post_init__(self) -> None:
        """Validate the ladder before the sampler starts drawing random states."""
        if self.replica_count < 2:
            raise ValueError("replica_count must be at least 2")
        if self.min_temperature <= 0.0 or self.max_temperature <= 0.0:
            raise ValueError("temperatures must be positive")
        if self.min_temperature >= self.max_temperature:
            raise ValueError("min_temperature must be lower than max_temperature")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.swap_interval < 1:
            raise ValueError("swap_interval must be at least 1")

    @property
    def temperatures(self) -> tuple[float, ...]:
        """Return the configured temperature ladder."""
        return make_temperature_schedule(
            replica_count=self.replica_count,
            min_temperature=self.min_temperature,
            max_temperature=self.max_temperature,
        )


@dataclass(frozen=True)
class ParallelTemperingRunResult:
    """2D parallel-tempering convergence result."""

    steps_to_convergence: int
    converged: bool
    best_energy: float
    final_cold_energy: float
    best_state: tuple[int, ...]
    energy_trace: tuple[float, ...]
    swap_attempts: int
    swap_acceptances: int
    temperature_schedule: tuple[float, ...]

    @property
    def swap_acceptance_rate(self) -> float:
        """Fraction of attempted adjacent swaps accepted by Metropolis."""
        if self.swap_attempts == 0:
            return 0.0
        return float(self.swap_acceptances / self.swap_attempts)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe fields for experiment artifacts."""
        return {
            "steps_to_convergence": self.steps_to_convergence,
            "converged": self.converged,
            "best_energy": self.best_energy,
            "final_cold_energy": self.final_cold_energy,
            "best_state": list(self.best_state),
            "energy_trace": list(self.energy_trace),
            "swap_attempts": self.swap_attempts,
            "swap_acceptances": self.swap_acceptances,
            "swap_acceptance_rate": self.swap_acceptance_rate,
            "temperature_schedule": list(self.temperature_schedule),
        }


def make_temperature_schedule(
    *,
    replica_count: int = 15,
    min_temperature: float = 0.5,
    max_temperature: float = 5.0,
) -> tuple[float, ...]:
    """Return a linear temperature ladder for the PT replicas.

    Spec: REQ-ISING-021
    """

    if replica_count < 2:
        raise ValueError("replica_count must be at least 2")
    if min_temperature <= 0.0 or max_temperature <= 0.0:
        raise ValueError("temperatures must be positive")
    if min_temperature >= max_temperature:
        raise ValueError("min_temperature must be lower than max_temperature")
    return tuple(
        float(value) for value in np.linspace(min_temperature, max_temperature, replica_count)
    )


def ising_energy(
    spins: np.ndarray,
    biases: np.ndarray,
    coupling_matrix: np.ndarray,
) -> float:
    """Compute Carnot's boolean Ising energy ``-(b@s + 0.5*s@J@s)``."""

    s = np.asarray(spins, dtype=np.float64)
    b = np.asarray(biases, dtype=np.float64)
    J = np.asarray(coupling_matrix, dtype=np.float64)
    return float(-(b @ s + 0.5 * s @ J @ s))


def metropolis_swap_acceptance_probability(
    *,
    energy_left: float,
    energy_right: float,
    beta_left: float,
    beta_right: float,
) -> float:
    """Return the adjacent-replica Metropolis swap probability.

    ``left`` and ``right`` are neighboring fixed temperature slots. The proposed
    move swaps their states. The target distribution ratio is
    ``exp((beta_left - beta_right) * (energy_left - energy_right))``.

    Spec: REQ-ISING-021
    """

    log_ratio = (beta_left - beta_right) * (energy_left - energy_right)
    if log_ratio >= 0.0:
        return 1.0
    return float(math.exp(max(log_ratio, -745.0)))


def run_single_temperature_ising(
    problem: IsingConstraintProblem,
    *,
    seed: int,
    temperature: float = 0.5,
    max_steps: int = 96,
) -> IsingRunResult:
    """Run one checkerboard Ising chain at a fixed temperature.

    This is the CPU baseline for Exp 1387: same problem, same convergence
    threshold, no replica ladder, and no adjacent-temperature swaps.

    Spec: REQ-ISING-021
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    rng = np.random.default_rng(seed)
    spins = (rng.random(problem.n_spins) < 0.5).astype(np.float64)
    beta = 1.0 / temperature
    best_state = spins.copy()
    best_energy = ising_energy(spins, problem.biases, problem.coupling_matrix)
    energy_trace: list[float] = []

    for step in range(1, max_steps + 1):
        spins = _checkerboard_gibbs_sweep(spins, problem.biases, problem.coupling_matrix, beta, rng)
        energy = ising_energy(spins, problem.biases, problem.coupling_matrix)
        energy_trace.append(energy)
        if energy < best_energy:
            best_energy = energy
            best_state = spins.copy()
        if best_energy <= problem.convergence_energy:
            return IsingRunResult(
                steps_to_convergence=step,
                converged=True,
                best_energy=best_energy,
                final_energy=energy,
                best_state=tuple(int(value) for value in best_state),
                energy_trace=tuple(float(value) for value in energy_trace),
            )

    return IsingRunResult(
        steps_to_convergence=max_steps,
        converged=False,
        best_energy=best_energy,
        final_energy=energy_trace[-1],
        best_state=tuple(int(value) for value in best_state),
        energy_trace=tuple(float(value) for value in energy_trace),
    )


class TwoDParallelTemperingSampler:
    """CPU 2D parallel tempering with adjacent-temperature exchanges.

    Spec: REQ-ISING-021
    """

    def __init__(self, config: ParallelTemperingConfig | None = None) -> None:
        """Store the immutable sampler configuration."""
        self.config = config or ParallelTemperingConfig()

    def run(self, problem: IsingConstraintProblem, *, seed: int) -> ParallelTemperingRunResult:
        """Run the PT ladder until the best replica reaches the threshold."""

        rng = np.random.default_rng(seed)
        temperatures = self.config.temperatures
        betas = np.asarray([1.0 / temp for temp in temperatures], dtype=np.float64)
        states = (rng.random((self.config.replica_count, problem.n_spins)) < 0.5).astype(np.float64)
        energies = np.asarray(
            [ising_energy(state, problem.biases, problem.coupling_matrix) for state in states],
            dtype=np.float64,
        )
        best_index = int(np.argmin(energies))
        best_energy = float(energies[best_index])
        best_state = states[best_index].copy()
        energy_trace: list[float] = []
        swap_attempts = 0
        swap_acceptances = 0

        for step in range(1, self.config.max_steps + 1):
            for replica_index, beta in enumerate(betas):
                states[replica_index] = _checkerboard_gibbs_sweep(
                    states[replica_index],
                    problem.biases,
                    problem.coupling_matrix,
                    float(beta),
                    rng,
                )
                energies[replica_index] = ising_energy(
                    states[replica_index],
                    problem.biases,
                    problem.coupling_matrix,
                )

            if step % self.config.swap_interval == 0:
                phase = (step // self.config.swap_interval) % 2
                for left in range(phase, self.config.replica_count - 1, 2):
                    right = left + 1
                    probability = metropolis_swap_acceptance_probability(
                        energy_left=float(energies[left]),
                        energy_right=float(energies[right]),
                        beta_left=float(betas[left]),
                        beta_right=float(betas[right]),
                    )
                    swap_attempts += 1
                    if rng.random() < probability:
                        states[[left, right]] = states[[right, left]]
                        energies[[left, right]] = energies[[right, left]]
                        swap_acceptances += 1

            current_best_index = int(np.argmin(energies))
            current_best_energy = float(energies[current_best_index])
            energy_trace.append(current_best_energy)
            if current_best_energy < best_energy:
                best_energy = current_best_energy
                best_state = states[current_best_index].copy()
            if best_energy <= problem.convergence_energy:
                return ParallelTemperingRunResult(
                    steps_to_convergence=step,
                    converged=True,
                    best_energy=best_energy,
                    final_cold_energy=float(energies[0]),
                    best_state=tuple(int(value) for value in best_state),
                    energy_trace=tuple(float(value) for value in energy_trace),
                    swap_attempts=swap_attempts,
                    swap_acceptances=swap_acceptances,
                    temperature_schedule=temperatures,
                )

        return ParallelTemperingRunResult(
            steps_to_convergence=self.config.max_steps,
            converged=False,
            best_energy=best_energy,
            final_cold_energy=float(energies[0]),
            best_state=tuple(int(value) for value in best_state),
            energy_trace=tuple(float(value) for value in energy_trace),
            swap_attempts=swap_attempts,
            swap_acceptances=swap_acceptances,
            temperature_schedule=temperatures,
        )


def load_fover_constraint_problems(
    *,
    repo_root: str | Path,
    limit: int = 5,
    n_spins: int = 128,
    fover_path: str | Path | None = None,
) -> list[IsingConstraintProblem]:
    """Load checked-in FoVer rows and derive planted dense Ising constraints.

    The planted construction is a benchmark proxy, not a new FoVer labeler. It
    ties each Ising instance to real local FoVer rows while preserving a known
    low-energy target at N=128, where exact brute force is impossible.

    Spec: REQ-ISING-021
    """

    if not 3 <= limit <= 5:
        raise ValueError("Exp 1387 must load between 3 and 5 FoVer problems")
    root = Path(repo_root)
    path = Path(fover_path) if fover_path is not None else root / "data" / "fover_corpus.jsonl"
    rows = _load_fover_rows(path, limit)
    return [
        _problem_from_fover_row(row, row_index=index, n_spins=n_spins)
        for index, row in enumerate(rows)
    ]


def estimate_kv260_lut_budget(
    *,
    replica_count: int = 15,
    sparsification_k_value: int = 16,
    lut_count_per_replica: int = 36_000,
    kv260_lut_budget: int = 117_000,
) -> dict[str, Any]:
    """Return the CPU-only KV260 LUT estimate for the replica ladder.

    Spec: REQ-ISING-021, SCENARIO-ISING-031
    """

    total_luts = int(replica_count * lut_count_per_replica)
    max_replicas_that_fit = int(kv260_lut_budget // lut_count_per_replica)
    return {
        "sparsification_k_value": int(sparsification_k_value),
        "estimated_kv260_lut_count_per_replica": int(lut_count_per_replica),
        "estimated_kv260_total_lut_count_15_replicas": int(total_luts),
        "kv260_lut_budget": int(kv260_lut_budget),
        "fits_15_replicas_kv260_budget": bool(total_luts <= kv260_lut_budget),
        "max_replicas_that_fit_kv260_budget": max_replicas_that_fit,
        "feasible_design_replica_count": max_replicas_that_fit,
        "lut_budget_feasible": bool(max_replicas_that_fit >= 2),
        "estimate_basis": (
            "arXiv:2503.01177 copy-node sparsification, Exp 950 K=16 sparse "
            "baseline, and research-hardware-wishlist.md ~36K LUT N=128 estimate"
        ),
    }


def run_fover_2d_parallel_tempering_probe(
    *,
    repo_root: str | Path,
    limit: int = 5,
    n_spins: int = 128,
    max_steps: int = 96,
    seeds: Sequence[int] = (0, 1, 2),
    run_date: str = "20260505",
    fover_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the Exp 1387 CPU probe and return the result payload.

    Spec: REQ-ISING-021, SCENARIO-ISING-031
    """

    problems = load_fover_constraint_problems(
        repo_root=repo_root,
        limit=limit,
        n_spins=n_spins,
        fover_path=fover_path,
    )
    config = ParallelTemperingConfig(replica_count=15, max_steps=max_steps)
    pt_sampler = TwoDParallelTemperingSampler(config)
    standard_temperature = config.min_temperature

    per_problem_results: list[dict[str, Any]] = []
    standard_steps_all: list[float] = []
    pt_steps_all: list[float] = []

    for problem in problems:
        standard_runs = [
            run_single_temperature_ising(
                problem,
                seed=int(seed) + 10_000,
                temperature=standard_temperature,
                max_steps=max_steps,
            )
            for seed in seeds
        ]
        pt_runs = [pt_sampler.run(problem, seed=int(seed) + 20_000) for seed in seeds]
        standard_steps = [run.steps_to_convergence for run in standard_runs]
        pt_steps = [run.steps_to_convergence for run in pt_runs]
        standard_steps_all.extend(float(value) for value in standard_steps)
        pt_steps_all.extend(float(value) for value in pt_steps)

        standard_mean = _mean(standard_steps)
        pt_mean = _mean(pt_steps)
        per_problem_results.append(
            {
                "problem": problem.name,
                "question_id": problem.question_id,
                "label": problem.label,
                "n_spins": problem.n_spins,
                "ground_energy": problem.ground_energy,
                "convergence_energy": problem.convergence_energy,
                "standard_mean_steps": standard_mean,
                "two_d_pt_mean_steps": pt_mean,
                "speedup": standard_mean / pt_mean if pt_mean > 0.0 else None,
                "standard_runs": [run.as_dict() for run in standard_runs],
                "two_d_pt_runs": [run.as_dict() for run in pt_runs],
            }
        )

    standard_mean_all = _mean(standard_steps_all)
    pt_mean_all = _mean(pt_steps_all)
    speedup = standard_mean_all / pt_mean_all if pt_mean_all > 0.0 else None
    lut_estimate = estimate_kv260_lut_budget(replica_count=config.replica_count)

    if speedup is not None and speedup > 1.0:
        verdict_prefix = "cpu_2d_pt_convergence_speedup_observed"
    else:
        verdict_prefix = "cpu_2d_pt_no_convergence_speedup_on_this_fover_slice"
    if lut_estimate["fits_15_replicas_kv260_budget"]:
        verdict = f"{verdict_prefix}_no_hardware_claim"
    else:
        verdict = f"{verdict_prefix}_15_replica_kv260_lut_over_budget_no_hardware_claim"

    return {
        "status": "complete",
        "run_date": run_date,
        "constraint_problems_tested": [problem.name for problem in problems],
        "replica_count": config.replica_count,
        "temperature_schedule": list(config.temperatures),
        "steps_to_convergence_standard_pt": {
            "mean_steps": standard_mean_all,
            "per_problem": [
                {
                    "problem": result["problem"],
                    "mean_steps": result["standard_mean_steps"],
                }
                for result in per_problem_results
            ],
            "temperature": standard_temperature,
            "sampler": "single-temperature checkerboard Gibbs CPU baseline",
        },
        "steps_to_convergence_2d_pt": {
            "mean_steps": pt_mean_all,
            "per_problem": [
                {
                    "problem": result["problem"],
                    "mean_steps": result["two_d_pt_mean_steps"],
                    "speedup": result["speedup"],
                }
                for result in per_problem_results
            ],
            "sampler": "15-replica CPU 2D parallel tempering with adjacent swaps",
        },
        "convergence_speedup_2d_pt": speedup,
        "sparsification_k_value": lut_estimate["sparsification_k_value"],
        "estimated_kv260_lut_count_per_replica": lut_estimate[
            "estimated_kv260_lut_count_per_replica"
        ],
        "estimated_kv260_total_lut_count_15_replicas": lut_estimate[
            "estimated_kv260_total_lut_count_15_replicas"
        ],
        "lut_budget_feasible": lut_estimate["lut_budget_feasible"],
        "hardware_claim_allowed": False,
        "kv260_claim_allowed": False,
        "honest_verdict": verdict,
        "per_problem_results": per_problem_results,
        "kv260_lut_estimate": lut_estimate,
        "metadata": {
            "experiment_id": 1387,
            "cpu_only": True,
            "hardware_execution_performed": False,
            "synthesis_performed": False,
            "n_spins_per_problem": n_spins,
            "max_steps": max_steps,
            "seeds": [int(seed) for seed in seeds],
            "fover_source": str(
                Path(fover_path)
                if fover_path is not None
                else Path(repo_root) / "data/fover_corpus.jsonl"
            ),
            "arxiv_2601_09037_source": "https://arxiv.org/abs/2601.09037",
            "arxiv_2503_01177_source": "https://arxiv.org/abs/2503.01177",
            "hardware_scope_note": (
                "CPU simulation only. The LUT estimate is arithmetic from the "
                "checked-in K=16 ~36K LUT baseline; no Vivado, bitfile, or KV260 "
                "board execution was performed."
            ),
        },
    }


def _checkerboard_gibbs_sweep(
    spins: np.ndarray,
    biases: np.ndarray,
    coupling_matrix: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run one two-phase checkerboard Gibbs sweep in NumPy."""

    next_spins = np.asarray(spins, dtype=np.float64).copy()
    even_mask = np.arange(next_spins.shape[0]) % 2 == 0
    odd_mask = ~even_mask
    for mask in (even_mask, odd_mask):
        fields = (
            np.asarray(biases, dtype=np.float64)
            + np.asarray(coupling_matrix, dtype=np.float64) @ next_spins
        )
        probs = _sigmoid(np.clip(2.0 * beta * fields, -60.0, 60.0))
        draws = (rng.random(next_spins.shape[0]) < probs).astype(np.float64)
        next_spins[mask] = draws[mask]
    return next_spins


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic helper for Gibbs probabilities."""

    return 1.0 / (1.0 + np.exp(-values))


def _load_fover_rows(path: Path, n_rows: int) -> list[dict[str, Any]]:
    """Load ``n_rows`` JSON objects from a local FoVer JSONL or JSON list."""

    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
                if len(rows) >= n_rows:
                    return rows
    else:
        with path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, list):
            rows = [row for row in loaded if isinstance(row, dict)]
            if len(rows) >= n_rows:
                return rows[:n_rows]
    raise ValueError(f"needed {n_rows} FoVer rows from {path}")


def _problem_from_fover_row(
    row: dict[str, Any],
    *,
    row_index: int,
    n_spins: int,
) -> IsingConstraintProblem:
    """Derive a planted dense Ising problem from one FoVer row."""

    row_text = str(row.get("step_text") or row.get("response") or row.get("question") or "")
    label = str(row.get("label") or row.get("is_correct") or "unknown")
    question_id = _stable_name_part(
        row.get("question_id") or row.get("question_index") or row_index
    )
    seed_material = f"{question_id}|{label}|{row_text}"
    digest = _digest_bytes(seed_material)
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)

    target = _target_bits(seed_material, n_spins).astype(np.float64)
    target_pm = 2.0 * target - 1.0
    biases = 0.70 * target_pm
    coupling_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    base_pair_strength = 0.18 / math.sqrt(float(n_spins))

    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            transformed_sign = target_pm[i] * target_pm[j]
            strength = base_pair_strength * float(rng.uniform(0.65, 1.35))
            coupling_matrix[i, j] = 4.0 * strength * transformed_sign
            coupling_matrix[j, i] = coupling_matrix[i, j]
            biases[i] += -2.0 * strength * transformed_sign
            biases[j] += -2.0 * strength * transformed_sign

    ground_energy = ising_energy(target, biases, coupling_matrix)
    convergence_energy = ground_energy + max(0.03 * abs(ground_energy), 0.50)
    return IsingConstraintProblem(
        name=f"fover_{question_id}_row{row_index}",
        question_id=question_id,
        label=label,
        n_spins=n_spins,
        biases=biases.astype(np.float64),
        coupling_matrix=coupling_matrix.astype(np.float64),
        target_state=target.astype(np.float64),
        ground_energy=float(ground_energy),
        convergence_energy=float(convergence_energy),
    )


def _target_bits(seed_material: str, n_bits: int) -> np.ndarray:
    """Generate deterministic target bits from repeated SHA-256 digests."""

    chunks: list[np.ndarray] = []
    counter = 0
    while sum(len(chunk) for chunk in chunks) < n_bits:
        digest = _digest_bytes(f"{seed_material}|target|{counter}")
        chunks.append(np.unpackbits(np.frombuffer(digest, dtype=np.uint8)))
        counter += 1
    return np.concatenate(chunks)[:n_bits]


def _digest_bytes(text: str) -> bytes:
    """Hash text into deterministic bytes."""

    return hashlib.sha256(text.encode("utf-8")).digest()


def _stable_name_part(raw: object) -> str:
    """Return a short JSON-safe identifier."""

    text = str(raw) if raw is not None else "unknown"
    return "".join(ch if ch.isalnum() else "_" for ch in text)[:24] or "unknown"


def _mean(values: Iterable[float]) -> float:
    """Return a JSON-friendly mean value."""

    vals = [float(value) for value in values]
    return float(np.mean(vals)) if vals else 0.0
