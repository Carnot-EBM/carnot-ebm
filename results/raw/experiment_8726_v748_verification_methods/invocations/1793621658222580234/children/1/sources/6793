#!/usr/bin/env python3
"""Exp 1674: CPU-only PIPIM dense Ising ablation.

This experiment compares a synchronous p-bit Ising update with EMA inertia
against Carnot's sequential bipolar Gibbs baseline on the same deterministic
dense FoVer-derived Ising problems and seeds. It records time-to-energy and
sample-quality deltas only; it does not run RTL, accelerator, or board checks.

Spec: REQ-ISING-041, SCENARIO-ISING-041
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - import path guard
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.samplers.discrete_simulated_bifurcation import (  # noqa: E402
    run_gibbs_ising_baseline,
)


EXPERIMENT_ID = 1674
RUN_DATE = "20260510"
SPEC_REFS = ["REQ-ISING-041", "SCENARIO-ISING-041"]
DEFAULT_RESULT_PATH = REPO_ROOT / "results/experiment_1674_pipim.json"


@dataclass(frozen=True)
class DensePBitProblem:
    """Dense planted bipolar Ising problem used by the PIPIM ablation."""

    name: str
    question_id: str
    label: str
    n_variables: int
    coupling_matrix: np.ndarray
    target_state: np.ndarray
    target_energy: float
    convergence_energy: float


@dataclass(frozen=True)
class PIPIMConfig:
    """Controls for synchronous p-bit Ising dynamics with EMA inertia."""

    max_steps: int = 96
    beta: float = 1.4
    inertia_alpha: float = 0.6

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 <= self.inertia_alpha < 1.0:
            raise ValueError("inertia_alpha must be in [0, 1)")


@dataclass(frozen=True)
class PIPIMRun:
    """Single PIPIM run summary for one dense problem and seed."""

    steps_to_energy: int
    reached_energy: bool
    best_energy: float
    final_energy: float
    best_energy_gap: float
    target_overlap: float
    best_state: tuple[int, ...]
    energy_trace: tuple[float, ...]
    ema_norm_trace: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sampler": "synchronous_pbit_inertia_pipim",
            "steps_to_energy": self.steps_to_energy,
            "reached_energy": self.reached_energy,
            "best_energy": self.best_energy,
            "final_energy": self.final_energy,
            "best_energy_gap": self.best_energy_gap,
            "target_overlap": self.target_overlap,
            "best_state": list(self.best_state),
            "energy_trace": list(self.energy_trace),
            "ema_norm_trace": list(self.ema_norm_trace),
        }


def bipolar_energy(spins: np.ndarray, coupling_matrix: np.ndarray) -> float:
    """Compute bipolar Ising energy ``E = -0.5 * s.T J s``."""

    s = np.asarray(spins, dtype=np.float64)
    J = np.asarray(coupling_matrix, dtype=np.float64)
    return float(-0.5 * s @ J @ s)


def build_dense_pbit_problem(
    row: Mapping[str, Any],
    *,
    row_index: int,
    n_variables: int,
) -> DensePBitProblem:
    """Derive a deterministic dense planted Ising problem from one FoVer row."""

    if n_variables < 2:
        raise ValueError("n_variables must be at least 2")

    row_text = str(row.get("step_text") or row.get("response") or row.get("question") or "")
    label = str(row.get("label") or row.get("is_correct") or "unknown")
    question_id = _stable_name_part(row.get("question_id") or row.get("question_index") or row_index)
    seed_material = f"{question_id}|{label}|{row_text}|pipim"
    seed = int.from_bytes(_digest_bytes(seed_material)[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    target = _target_spins(seed_material, n_variables)

    weights = rng.uniform(0.85, 1.15, size=(n_variables, n_variables))
    weights = (weights + weights.T) / 2.0
    coupling_matrix = weights * np.outer(target, target) / float(n_variables)
    np.fill_diagonal(coupling_matrix, 0.0)

    target_energy = bipolar_energy(target, coupling_matrix)
    convergence_energy = target_energy + max(0.08 * abs(target_energy), 0.2)
    return DensePBitProblem(
        name=f"fover_{question_id}_row{row_index}_n{n_variables}",
        question_id=question_id,
        label=label,
        n_variables=int(n_variables),
        coupling_matrix=coupling_matrix.astype(np.float64),
        target_state=target.astype(np.float64),
        target_energy=float(target_energy),
        convergence_energy=float(convergence_energy),
    )


def load_dense_pbit_problems(
    *,
    repo_root: str | Path,
    limit: int = 3,
    n_variable_schedule: Sequence[int] | None = None,
    fover_path: str | Path | None = None,
) -> list[DensePBitProblem]:
    """Load 3 to 5 FoVer rows and map them to dense p-bit Ising problems."""

    if not 3 <= limit <= 5:
        raise ValueError("Exp 1674 must load between 3 and 5 dense problems")

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
        build_dense_pbit_problem(row, row_index=index, n_variables=int(schedule[index]))
        for index, row in enumerate(rows)
    ]


def run_pipim(
    problem: DensePBitProblem,
    *,
    seed: int,
    config: PIPIMConfig | None = None,
) -> PIPIMRun:
    """Run synchronous p-bit updates with EMA inertia until target energy."""

    cfg = config or PIPIMConfig()
    rng = np.random.default_rng(int(seed))
    spins = rng.choice(np.asarray([-1.0, 1.0]), size=problem.n_variables)
    ema_field = np.zeros(problem.n_variables, dtype=np.float64)
    best_state = spins.copy()
    best_energy = bipolar_energy(spins, problem.coupling_matrix)
    final_energy = best_energy
    energy_trace: list[float] = []
    ema_norm_trace: list[float] = []
    J = np.asarray(problem.coupling_matrix, dtype=np.float64)

    for step in range(1, cfg.max_steps + 1):
        local_field = J @ spins
        ema_field = cfg.inertia_alpha * ema_field + (1.0 - cfg.inertia_alpha) * local_field
        p_up = _sigmoid(np.clip(2.0 * cfg.beta * ema_field, -60.0, 60.0))
        spins = np.where(rng.random(problem.n_variables) < p_up, 1.0, -1.0)
        final_energy = bipolar_energy(spins, J)
        energy_trace.append(float(final_energy))
        ema_norm_trace.append(float(np.linalg.norm(ema_field)))
        if final_energy < best_energy:
            best_energy = final_energy
            best_state = spins.copy()
        if best_energy <= problem.convergence_energy:
            return _make_pipim_run(
                problem=problem,
                steps_to_energy=step,
                reached_energy=True,
                best_energy=best_energy,
                final_energy=final_energy,
                best_state=best_state,
                energy_trace=energy_trace,
                ema_norm_trace=ema_norm_trace,
            )

    return _make_pipim_run(
        problem=problem,
        steps_to_energy=cfg.max_steps,
        reached_energy=False,
        best_energy=best_energy,
        final_energy=final_energy,
        best_state=best_state,
        energy_trace=energy_trace,
        ema_norm_trace=ema_norm_trace,
    )


def run_experiment(
    *,
    output_path: Path = DEFAULT_RESULT_PATH,
    n_problems: int = 3,
    n_variable_schedule: Sequence[int] | None = None,
    max_steps: int = 96,
    seeds: Sequence[int] = (0, 1, 2),
    beta: float = 1.4,
    inertia_alpha: float = 0.6,
    fover_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the matched sequential-Gibbs vs PIPIM ablation and write JSON."""

    problems = load_dense_pbit_problems(
        repo_root=REPO_ROOT,
        limit=n_problems,
        n_variable_schedule=n_variable_schedule,
        fover_path=fover_path,
    )
    config = PIPIMConfig(max_steps=max_steps, beta=beta, inertia_alpha=inertia_alpha)
    per_problem_results: list[dict[str, Any]] = []
    gibbs_steps_all: list[float] = []
    pipim_steps_all: list[float] = []
    gibbs_best_energy_all: list[float] = []
    pipim_best_energy_all: list[float] = []
    gibbs_gap_all: list[float] = []
    pipim_gap_all: list[float] = []
    gibbs_overlap_all: list[float] = []
    pipim_overlap_all: list[float] = []

    for problem in problems:
        gibbs_runs = [
            run_gibbs_ising_baseline(
                problem,
                seed=int(seed),
                max_steps=max_steps,
                beta=beta,
            )
            for seed in seeds
        ]
        pipim_runs = [run_pipim(problem, seed=int(seed), config=config) for seed in seeds]
        gibbs_quality = [_gibbs_quality(problem, run) for run in gibbs_runs]
        pipim_quality = [_pipim_quality(run) for run in pipim_runs]

        gibbs_steps = [run.steps_to_convergence for run in gibbs_runs]
        pipim_steps = [run.steps_to_energy for run in pipim_runs]
        gibbs_steps_all.extend(float(value) for value in gibbs_steps)
        pipim_steps_all.extend(float(value) for value in pipim_steps)
        gibbs_best_energy_all.extend(float(run.best_energy) for run in gibbs_runs)
        pipim_best_energy_all.extend(float(run.best_energy) for run in pipim_runs)
        gibbs_gap_all.extend(float(item["best_energy_gap"]) for item in gibbs_quality)
        pipim_gap_all.extend(float(item["best_energy_gap"]) for item in pipim_quality)
        gibbs_overlap_all.extend(float(item["target_overlap"]) for item in gibbs_quality)
        pipim_overlap_all.extend(float(item["target_overlap"]) for item in pipim_quality)

        gibbs_mean_steps = _mean(gibbs_steps)
        pipim_mean_steps = _mean(pipim_steps)
        per_problem_results.append(
            {
                "problem": problem.name,
                "question_id": problem.question_id,
                "label": problem.label,
                "n_variables": problem.n_variables,
                "target_energy": problem.target_energy,
                "convergence_energy": problem.convergence_energy,
                "gibbs_mean_steps": gibbs_mean_steps,
                "pipim_mean_steps": pipim_mean_steps,
                "time_to_energy_delta_steps": gibbs_mean_steps - pipim_mean_steps,
                "time_to_energy_speedup": gibbs_mean_steps / pipim_mean_steps
                if pipim_mean_steps > 0.0
                else None,
                "gibbs_runs": [run.as_dict() for run in gibbs_runs],
                "pipim_runs": [run.as_dict() for run in pipim_runs],
                "gibbs_quality": gibbs_quality,
                "pipim_quality": pipim_quality,
            }
        )

    gibbs_mean_steps_all = _mean(gibbs_steps_all)
    pipim_mean_steps_all = _mean(pipim_steps_all)
    gibbs_mean_gap = _mean(gibbs_gap_all)
    pipim_mean_gap = _mean(pipim_gap_all)
    gibbs_mean_overlap = _mean(gibbs_overlap_all)
    pipim_mean_overlap = _mean(pipim_overlap_all)
    time_to_energy_speedup = (
        gibbs_mean_steps_all / pipim_mean_steps_all if pipim_mean_steps_all > 0.0 else 0.0
    )
    sample_quality_delta = {
        "best_energy_gap_reduction": gibbs_mean_gap - pipim_mean_gap,
        "target_overlap_delta": pipim_mean_overlap - gibbs_mean_overlap,
        "best_energy_delta_pipim_minus_gibbs": _mean(pipim_best_energy_all)
        - _mean(gibbs_best_energy_all),
    }
    artifact = {
        "status": "complete",
        "run_date": RUN_DATE,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "algorithm": (
            "PIPIM synchronous p-bit Ising update with EMA inertia over dense "
            "local fields"
        ),
        "baseline_algorithm": "Carnot sequential bipolar Gibbs Ising baseline",
        "dense_problems_tested": [problem.name for problem in problems],
        "n_variables": [problem.n_variables for problem in problems],
        "seeds": [int(seed) for seed in seeds],
        "time_to_energy_gibbs_baseline": {
            "mean_steps": gibbs_mean_steps_all,
            "per_problem": [
                {
                    "problem": item["problem"],
                    "n_variables": item["n_variables"],
                    "mean_steps": item["gibbs_mean_steps"],
                }
                for item in per_problem_results
            ],
        },
        "time_to_energy_pipim": {
            "mean_steps": pipim_mean_steps_all,
            "per_problem": [
                {
                    "problem": item["problem"],
                    "n_variables": item["n_variables"],
                    "mean_steps": item["pipim_mean_steps"],
                }
                for item in per_problem_results
            ],
            "beta": config.beta,
            "inertia_alpha": config.inertia_alpha,
            "max_steps": config.max_steps,
        },
        "time_to_energy_delta_steps": gibbs_mean_steps_all - pipim_mean_steps_all,
        "time_to_energy_speedup": float(time_to_energy_speedup),
        "sample_quality_gibbs_baseline": {
            "mean_best_energy": _mean(gibbs_best_energy_all),
            "mean_best_energy_gap": gibbs_mean_gap,
            "mean_target_overlap": gibbs_mean_overlap,
        },
        "sample_quality_pipim": {
            "mean_best_energy": _mean(pipim_best_energy_all),
            "mean_best_energy_gap": pipim_mean_gap,
            "mean_target_overlap": pipim_mean_overlap,
        },
        "sample_quality_delta": sample_quality_delta,
        "cpu_only": True,
        "simulator_only": True,
        "hardware_execution_performed": False,
        "hardware_claim_allowed": False,
        "honest_verdict": _verdict(
            time_to_energy_delta=gibbs_mean_steps_all - pipim_mean_steps_all,
            gap_reduction=sample_quality_delta["best_energy_gap_reduction"],
        ),
        "per_problem_results": per_problem_results,
        "metadata": {
            "fover_source": str(
                Path(fover_path)
                if fover_path is not None
                else REPO_ROOT / "data" / "fover_corpus.jsonl"
            ),
            "dense_problem_construction": (
                "FoVer text seeds a planted dense bipolar Ising graph; the "
                "ablation measures sampler dynamics, not FoVer label accuracy."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(path: Path = DEFAULT_RESULT_PATH) -> dict[str, Any]:
    """Write the bootstrap artifact before the CPU ablation starts."""

    marker = {
        "status": "in_progress",
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "cpu_only": True,
        "simulator_only": True,
        "hardware_execution_performed": False,
        "hardware_claim_allowed": False,
        "honest_verdict": "in_progress",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return marker


def main() -> None:
    """CLI entry point used by the research conductor."""

    write_in_progress_artifact(DEFAULT_RESULT_PATH)
    artifact = run_experiment()
    print(
        artifact["time_to_energy_delta_steps"],
        artifact["sample_quality_delta"]["best_energy_gap_reduction"],
        artifact["hardware_claim_allowed"],
        artifact["honest_verdict"],
    )


def _make_pipim_run(
    *,
    problem: DensePBitProblem,
    steps_to_energy: int,
    reached_energy: bool,
    best_energy: float,
    final_energy: float,
    best_state: np.ndarray,
    energy_trace: list[float],
    ema_norm_trace: list[float],
) -> PIPIMRun:
    return PIPIMRun(
        steps_to_energy=int(steps_to_energy),
        reached_energy=bool(reached_energy),
        best_energy=float(best_energy),
        final_energy=float(final_energy),
        best_energy_gap=float(best_energy - problem.target_energy),
        target_overlap=state_target_overlap(best_state, problem.target_state),
        best_state=tuple(int(value) for value in best_state),
        energy_trace=tuple(float(value) for value in energy_trace),
        ema_norm_trace=tuple(float(value) for value in ema_norm_trace),
    )


def state_target_overlap(spins: np.ndarray | Sequence[int], target_state: np.ndarray) -> float:
    """Return sign-invariant overlap with the planted target state."""

    spins_array = np.asarray(spins, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    direct = float(np.mean(spins_array == target))
    flipped = float(np.mean(spins_array == -target))
    return max(direct, flipped)


def _gibbs_quality(problem: DensePBitProblem, run: Any) -> dict[str, float]:
    return {
        "best_energy_gap": float(run.best_energy - problem.target_energy),
        "target_overlap": state_target_overlap(run.best_state, problem.target_state),
    }


def _pipim_quality(run: PIPIMRun) -> dict[str, float]:
    return {
        "best_energy_gap": float(run.best_energy_gap),
        "target_overlap": float(run.target_overlap),
    }


def _verdict(*, time_to_energy_delta: float, gap_reduction: float) -> str:
    if time_to_energy_delta > 0.0 and gap_reduction > 0.0:
        return "complete_pipim_time_and_quality_improved_cpu_simulator_only"
    if time_to_energy_delta > 0.0:
        return "complete_pipim_time_improved_quality_not_improved_cpu_simulator_only"
    if gap_reduction > 0.0:
        return "complete_pipim_quality_improved_time_not_improved_cpu_simulator_only"
    return "complete_pipim_no_improvement_observed_cpu_simulator_only"


def _load_fover_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    if len(rows) < limit:
        raise ValueError(f"needed {limit} FoVer rows from {path}, found {len(rows)}")
    return rows


def _default_variable_schedule(limit: int) -> tuple[int, ...]:
    return (32, 48, 64, 80, 96)[:limit]


def _target_spins(seed_material: str, n_variables: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    counter = 0
    while sum(chunk.size for chunk in chunks) < n_variables:
        digest = _digest_bytes(f"{seed_material}|target|{counter}")
        chunks.append(np.unpackbits(np.frombuffer(digest, dtype=np.uint8)))
        counter += 1
    bits = np.concatenate(chunks)[:n_variables].astype(np.float64)
    return 2.0 * bits - 1.0


def _digest_bytes(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _stable_name_part(raw: object) -> str:
    text = str(raw) if raw is not None else "unknown"
    return "".join(ch if ch.isalnum() else "_" for ch in text)[:24] or "unknown"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _mean(values: Sequence[float] | Sequence[int]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


if __name__ == "__main__":  # pragma: no cover
    main()
