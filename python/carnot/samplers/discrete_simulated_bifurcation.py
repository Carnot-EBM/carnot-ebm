"""Discrete simulated bifurcation CPU probe for FoVer-derived Ising problems.

The simulator is intentionally small and deterministic because Exp 1399 is a
hardware-feasibility probe, not a production optimizer. It uses FoVer rows only
to seed repeatable dense Ising/QUBO instances, then compares a dSB-style
parallel sign update with a standard Gibbs baseline on the same planted
problems. The KV260 estimate is arithmetic only: it checks whether the dense
int8 J matrix and one update unit fit the checked-in KV260 resource budget.

Spec: REQ-ISING-022, SCENARIO-ISING-032
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DiscreteSBConstraintProblem:
    """FoVer-seeded dense Ising problem for the dSB CPU comparison.

    The problem stores spins in the bipolar Ising convention, ``{-1, +1}``.
    Dense int8 couplings make the memory estimate match the prospective KV260
    representation while still giving the CPU probe a known planted low-energy
    state. The planted construction is a benchmark proxy; it does not relabel
    or formally solve the FoVer row.

    Spec: REQ-ISING-022
    """

    name: str
    question_id: str
    label: str
    n_variables: int
    coupling_matrix: np.ndarray
    target_state: np.ndarray
    ground_energy: float
    convergence_energy: float


@dataclass(frozen=True)
class DiscreteSBConfig:
    """Configuration for the simplified dSB sign-pressure update.

    ``eta`` controls how strongly the current Ising field moves each spin.
    ``max_steps`` also defines the pressure schedule length: pressure starts at
    0 and linearly reaches 1 on the final configured step.

    Spec: REQ-ISING-022
    """

    max_steps: int = 128
    eta: float = 0.09
    pressure_start: float = 0.0
    pressure_end: float = 1.0

    def __post_init__(self) -> None:
        """Validate the small set of dSB controls before simulation starts."""

        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive")
        if self.pressure_start < 0.0 or self.pressure_end < 0.0:
            raise ValueError("pressure values must be non-negative")


@dataclass(frozen=True)
class IsingConvergenceRun:
    """Convergence result for either Gibbs or dSB on one problem and seed."""

    steps_to_convergence: int
    converged: bool
    best_energy: float
    final_energy: float
    best_state: tuple[int, ...]
    energy_trace: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe values for experiment artifacts."""

        return {
            "steps_to_convergence": self.steps_to_convergence,
            "converged": self.converged,
            "best_energy": self.best_energy,
            "final_energy": self.final_energy,
            "best_state": list(self.best_state),
            "energy_trace": list(self.energy_trace),
        }


def make_pressure_schedule(
    max_steps: int,
    *,
    pressure_start: float = 0.0,
    pressure_end: float = 1.0,
) -> tuple[float, ...]:
    """Return the linear dSB pressure schedule from 0 to 1.

    Spec: REQ-ISING-022
    """

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if pressure_start < 0.0 or pressure_end < 0.0:
        raise ValueError("pressure values must be non-negative")
    return tuple(float(value) for value in np.linspace(pressure_start, pressure_end, max_steps))


def bipolar_ising_energy(spins: np.ndarray, coupling_matrix: np.ndarray) -> float:
    """Compute ``E = -0.5 * s^T J s`` for bipolar Ising spins."""

    s = np.asarray(spins, dtype=np.float64)
    J = np.asarray(coupling_matrix, dtype=np.float64)
    return float(-0.5 * s @ J @ s)


def run_discrete_sb(
    problem: DiscreteSBConstraintProblem,
    *,
    seed: int,
    config: DiscreteSBConfig | None = None,
) -> IsingConvergenceRun:
    """Run the simplified dSB update until the convergence threshold is reached.

    Each step applies the experiment's required update rule:
    ``x_i(t+1) = sign(x_i(t) + eta * sum_j J_ij * x_j(t) - pressure(t))``.
    The coupling field uses the signed spin state, matching the dSB paper's
    use of ``sgn(x_j)`` inside the matrix-vector multiplication.

    Spec: REQ-ISING-022
    """

    cfg = config or DiscreteSBConfig()
    rng = np.random.default_rng(seed)
    position = rng.normal(loc=0.0, scale=0.01, size=problem.n_variables)
    spins = _sign_pm(position)
    best_state = spins.copy()
    best_energy = bipolar_ising_energy(spins, problem.coupling_matrix)
    energy_trace: list[float] = []
    schedule = make_pressure_schedule(
        cfg.max_steps,
        pressure_start=cfg.pressure_start,
        pressure_end=cfg.pressure_end,
    )

    for step, pressure in enumerate(schedule, start=1):
        field = np.asarray(problem.coupling_matrix, dtype=np.float64) @ spins
        position = _sign_pm(position + cfg.eta * field - pressure)
        spins = position
        energy = bipolar_ising_energy(spins, problem.coupling_matrix)
        energy_trace.append(energy)
        if energy < best_energy:
            best_energy = energy
            best_state = spins.copy()
        if best_energy <= problem.convergence_energy:
            return IsingConvergenceRun(
                steps_to_convergence=step,
                converged=True,
                best_energy=float(best_energy),
                final_energy=float(energy),
                best_state=tuple(int(value) for value in best_state),
                energy_trace=tuple(float(value) for value in energy_trace),
            )

    return IsingConvergenceRun(
        steps_to_convergence=cfg.max_steps,
        converged=False,
        best_energy=float(best_energy),
        final_energy=float(energy_trace[-1]),
        best_state=tuple(int(value) for value in best_state),
        energy_trace=tuple(float(value) for value in energy_trace),
    )


def run_gibbs_ising_baseline(
    problem: DiscreteSBConstraintProblem,
    *,
    seed: int,
    max_steps: int = 128,
    beta: float = 0.015,
) -> IsingConvergenceRun:
    """Run a sequential Gibbs Ising baseline on the same bipolar problem.

    One step is one full sweep over all variables in random order. The low
    default beta keeps the baseline from collapsing instantly on dense planted
    matrices, which makes step-count comparisons more informative.

    Spec: REQ-ISING-022
    """

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    rng = np.random.default_rng(seed)
    spins = rng.choice(np.asarray([-1.0, 1.0]), size=problem.n_variables)
    best_state = spins.copy()
    best_energy = bipolar_ising_energy(spins, problem.coupling_matrix)
    energy_trace: list[float] = []
    J = np.asarray(problem.coupling_matrix, dtype=np.float64)

    for step in range(1, max_steps + 1):
        for index in rng.permutation(problem.n_variables):
            field = float(J[index] @ spins)
            probability_up = _sigmoid_scalar(np.clip(2.0 * beta * field, -60.0, 60.0))
            spins[index] = 1.0 if rng.random() < probability_up else -1.0

        energy = bipolar_ising_energy(spins, J)
        energy_trace.append(energy)
        if energy < best_energy:
            best_energy = energy
            best_state = spins.copy()
        if best_energy <= problem.convergence_energy:
            return IsingConvergenceRun(
                steps_to_convergence=step,
                converged=True,
                best_energy=float(best_energy),
                final_energy=float(energy),
                best_state=tuple(int(value) for value in best_state),
                energy_trace=tuple(float(value) for value in energy_trace),
            )

    return IsingConvergenceRun(
        steps_to_convergence=max_steps,
        converged=False,
        best_energy=float(best_energy),
        final_energy=float(energy_trace[-1]),
        best_state=tuple(int(value) for value in best_state),
        energy_trace=tuple(float(value) for value in energy_trace),
    )


def load_fover_discrete_sb_problems(
    *,
    repo_root: str | Path,
    limit: int = 5,
    n_variable_schedule: Sequence[int] | None = None,
    fover_path: str | Path | None = None,
) -> list[DiscreteSBConstraintProblem]:
    """Load local FoVer rows and derive 3-5 dense int8 Ising problems.

    The default schedule spans N=64 through N=256, matching Exp 1399's hardware
    question. Tests can pass smaller schedules to keep unit runtime low.

    Spec: REQ-ISING-022
    """

    if not 3 <= limit <= 5:
        raise ValueError("Exp 1399 must load between 3 and 5 FoVer problems")

    schedule = (
        _default_variable_schedule(limit)
        if n_variable_schedule is None
        else tuple(int(value) for value in n_variable_schedule)
    )
    if len(schedule) != limit:
        raise ValueError("n_variable_schedule length must match limit")
    if any(value < 2 for value in schedule):
        raise ValueError("all variable counts must be at least 2")

    root = Path(repo_root)
    path = Path(fover_path) if fover_path is not None else root / "data" / "fover_corpus.jsonl"
    rows = _load_fover_rows(path, limit)
    return [
        _problem_from_fover_row(row, row_index=index, n_variables=int(schedule[index]))
        for index, row in enumerate(rows)
    ]


def estimate_kv260_discrete_sb_resources(
    *,
    n_variables: int = 256,
    bits_per_coupling: int = 8,
    bram36_blocks: int = 144,
    lut_estimate_per_update_unit: int = 2_000,
    kv260_lut_budget: int = 117_000,
) -> dict[str, Any]:
    """Return the KV260 BRAM/LUT arithmetic for a single dSB update unit.

    A BRAM_36 tile stores 36 Kb, so KV260's 144 tiles provide 5184 Kb or
    648 KB. The coupling matrix estimate uses the dense int8 J matrix only;
    extra buffering would need a later RTL-specific synthesis estimate.

    Spec: REQ-ISING-022, SCENARIO-ISING-032
    """

    if n_variables < 1:
        raise ValueError("n_variables must be positive")
    if bits_per_coupling < 1:
        raise ValueError("bits_per_coupling must be positive")

    matrix_bits = int(n_variables * n_variables * bits_per_coupling)
    bram_estimate_kb = matrix_bits / 8.0 / 1024.0
    kv260_bram_budget_kb = bram36_blocks * 36 / 8
    bram_budget_feasible = bram_estimate_kb < kv260_bram_budget_kb
    kv260_lut_budget_fits = int(lut_estimate_per_update_unit) <= int(kv260_lut_budget)
    return {
        "n_variables": int(n_variables),
        "bits_per_coupling": int(bits_per_coupling),
        "bram_estimate_kb_for_256var": float(bram_estimate_kb),
        "kv260_bram_budget_kb": int(kv260_bram_budget_kb),
        "bram_budget_feasible": bool(bram_budget_feasible),
        "lut_estimate_per_update_unit": int(lut_estimate_per_update_unit),
        "kv260_lut_budget": int(kv260_lut_budget),
        "kv260_lut_budget_fits": bool(kv260_lut_budget_fits),
        "estimate_basis": (
            "Dense int8 J matrix plus one dSB update unit; no Vivado synthesis "
            "or KV260 board execution performed in this CPU probe."
        ),
    }


def run_fover_discrete_sb_probe(
    *,
    repo_root: str | Path,
    limit: int = 5,
    n_variable_schedule: Sequence[int] | None = None,
    max_steps: int = 128,
    seeds: Sequence[int] = (0, 1, 2),
    run_date: str = "20260506",
    fover_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run Exp 1399 and return the complete JSON-ready artifact.

    Spec: REQ-ISING-022, SCENARIO-ISING-032
    """

    problems = load_fover_discrete_sb_problems(
        repo_root=repo_root,
        limit=limit,
        n_variable_schedule=n_variable_schedule,
        fover_path=fover_path,
    )
    config = DiscreteSBConfig(max_steps=max_steps)
    per_problem_results: list[dict[str, Any]] = []
    baseline_steps_all: list[float] = []
    dsb_steps_all: list[float] = []

    for problem in problems:
        baseline_runs = [
            run_gibbs_ising_baseline(problem, seed=int(seed) + 30_000, max_steps=max_steps)
            for seed in seeds
        ]
        dsb_runs = [
            run_discrete_sb(problem, seed=int(seed) + 40_000, config=config) for seed in seeds
        ]
        baseline_steps = [run.steps_to_convergence for run in baseline_runs]
        dsb_steps = [run.steps_to_convergence for run in dsb_runs]
        baseline_steps_all.extend(float(value) for value in baseline_steps)
        dsb_steps_all.extend(float(value) for value in dsb_steps)

        baseline_mean = _mean(baseline_steps)
        dsb_mean = _mean(dsb_steps)
        per_problem_results.append(
            {
                "problem": problem.name,
                "question_id": problem.question_id,
                "label": problem.label,
                "n_variables": problem.n_variables,
                "ground_energy": problem.ground_energy,
                "convergence_energy": problem.convergence_energy,
                "ising_baseline_mean_steps": baseline_mean,
                "discrete_sb_mean_steps": dsb_mean,
                "speedup": baseline_mean / dsb_mean if dsb_mean > 0.0 else None,
                "ising_baseline_runs": [run.as_dict() for run in baseline_runs],
                "discrete_sb_runs": [run.as_dict() for run in dsb_runs],
            }
        )

    baseline_mean_all = _mean(baseline_steps_all)
    dsb_mean_all = _mean(dsb_steps_all)
    speedup = baseline_mean_all / dsb_mean_all if dsb_mean_all > 0.0 else None
    resource_estimate = estimate_kv260_discrete_sb_resources()
    hardware_claim_allowed = bool(
        resource_estimate["bram_budget_feasible"] and resource_estimate["kv260_lut_budget_fits"]
    )
    if speedup is not None and speedup > 1.0:
        verdict_prefix = "cpu_discrete_sb_convergence_speedup_observed"
    else:
        verdict_prefix = "cpu_discrete_sb_no_convergence_speedup_on_this_fover_slice"
    verdict_suffix = (
        "kv260_bram_lut_budget_fits_estimate_only"
        if hardware_claim_allowed
        else "kv260_budget_not_feasible"
    )

    return {
        "status": "complete",
        "run_date": run_date,
        "algorithm": (
            "Discrete Simulated Bifurcation CPU sign-pressure model "
            "x_i(t+1)=sign(x_i(t)+eta*sum_j J_ij*x_j(t)-pressure(t))"
        ),
        "constraint_problems_tested": [problem.name for problem in problems],
        "n_variables": [problem.n_variables for problem in problems],
        "steps_to_convergence_ising_baseline": {
            "mean_steps": baseline_mean_all,
            "per_problem": [
                {
                    "problem": result["problem"],
                    "n_variables": result["n_variables"],
                    "mean_steps": result["ising_baseline_mean_steps"],
                }
                for result in per_problem_results
            ],
            "sampler": "sequential bipolar Gibbs Ising baseline",
        },
        "steps_to_convergence_discrete_sb": {
            "mean_steps": dsb_mean_all,
            "per_problem": [
                {
                    "problem": result["problem"],
                    "n_variables": result["n_variables"],
                    "mean_steps": result["discrete_sb_mean_steps"],
                    "speedup": result["speedup"],
                }
                for result in per_problem_results
            ],
            "sampler": "parallel dSB sign-pressure CPU update",
            "pressure_schedule": {
                "start": config.pressure_start,
                "end": config.pressure_end,
                "steps": config.max_steps,
            },
            "eta": config.eta,
        },
        "convergence_speedup_discrete_sb": speedup,
        "bram_estimate_kb_for_256var": resource_estimate["bram_estimate_kb_for_256var"],
        "kv260_bram_budget_kb": resource_estimate["kv260_bram_budget_kb"],
        "bram_budget_feasible": resource_estimate["bram_budget_feasible"],
        "lut_estimate_per_update_unit": resource_estimate["lut_estimate_per_update_unit"],
        "kv260_lut_budget_fits": resource_estimate["kv260_lut_budget_fits"],
        "hardware_claim_allowed": hardware_claim_allowed,
        "kv260_claim_allowed": hardware_claim_allowed,
        "honest_verdict": f"{verdict_prefix}_{verdict_suffix}",
        "per_problem_results": per_problem_results,
        "kv260_resource_estimate": resource_estimate,
        "metadata": {
            "experiment_id": 1399,
            "cpu_only": True,
            "hardware_execution_performed": False,
            "synthesis_performed": False,
            "fover_source": str(
                Path(fover_path)
                if fover_path is not None
                else Path(repo_root) / "data/fover_corpus.jsonl"
            ),
            "arxiv_2510_12407_source": "https://arxiv.org/abs/2510.12407",
            "exp1387_comparison": (
                "Exp 1387 was LUT-limited at 540K LUTs for 15 PT replicas; "
                "this dSB estimate fits one 2K-LUT update unit and a 64 KB "
                "N=256 int8 J matrix within the KV260 arithmetic budget."
            ),
        },
    }


def _default_variable_schedule(limit: int) -> tuple[int, ...]:
    """Return the default N schedule, preserving the 64..256 span for three rows."""

    if limit == 3:
        return (64, 128, 256)
    return (64, 96, 128, 192, 256)[:limit]


def _problem_from_fover_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    n_variables: int,
) -> DiscreteSBConstraintProblem:
    """Derive one dense int8 Ising problem from a FoVer row."""

    row_text = str(row.get("step_text") or row.get("response") or row.get("question") or "")
    label = str(row.get("label") or row.get("is_correct") or "unknown")
    question_id = _stable_name_part(
        row.get("question_id") or row.get("question_index") or row_index
    )
    seed_material = f"{question_id}|{label}|{row_text}|dsb"
    seed = int.from_bytes(_digest_bytes(seed_material)[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    target = _target_spins(seed_material, n_variables)
    coupling_matrix = np.zeros((n_variables, n_variables), dtype=np.int8)

    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            weight = int(rng.integers(1, 4))
            coupling = int(weight * target[i] * target[j])
            coupling_matrix[i, j] = coupling
            coupling_matrix[j, i] = coupling

    ground_energy = bipolar_ising_energy(target, coupling_matrix)
    convergence_energy = ground_energy + max(0.08 * abs(ground_energy), 1.0)
    return DiscreteSBConstraintProblem(
        name=f"fover_{question_id}_row{row_index}_n{n_variables}",
        question_id=question_id,
        label=label,
        n_variables=n_variables,
        coupling_matrix=coupling_matrix,
        target_state=target.astype(np.float64),
        ground_energy=float(ground_energy),
        convergence_energy=float(convergence_energy),
    )


def _load_fover_rows(path: Path, n_rows: int) -> list[dict[str, Any]]:
    """Load a small number of JSON objects from a local FoVer JSON/JSONL file."""

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


def _target_spins(seed_material: str, n_spins: int) -> np.ndarray:
    """Generate deterministic bipolar target spins from repeated SHA-256 digests."""

    chunks: list[np.ndarray] = []
    counter = 0
    while sum(len(chunk) for chunk in chunks) < n_spins:
        digest = _digest_bytes(f"{seed_material}|target|{counter}")
        chunks.append(np.unpackbits(np.frombuffer(digest, dtype=np.uint8)))
        counter += 1
    bits = np.concatenate(chunks)[:n_spins].astype(np.float64)
    return 2.0 * bits - 1.0


def _digest_bytes(text: str) -> bytes:
    """Hash text into deterministic bytes."""

    return hashlib.sha256(text.encode("utf-8")).digest()


def _stable_name_part(raw: object) -> str:
    """Return a short JSON-safe identifier for artifact problem names."""

    text = str(raw) if raw is not None else "unknown"
    return "".join(ch if ch.isalnum() else "_" for ch in text)[:24] or "unknown"


def _sign_pm(values: np.ndarray) -> np.ndarray:
    """Map real values to bipolar signs, using +1 for exact zero."""

    return np.where(np.asarray(values, dtype=np.float64) >= 0.0, 1.0, -1.0)


def _sigmoid_scalar(value: float) -> float:
    """Return a numerically stable scalar logistic probability."""

    return float(1.0 / (1.0 + math.exp(-float(value))))


def _mean(values: Iterable[float]) -> float:
    """Return a JSON-friendly mean value."""

    vals = [float(value) for value in values]
    return float(np.mean(vals)) if vals else 0.0
