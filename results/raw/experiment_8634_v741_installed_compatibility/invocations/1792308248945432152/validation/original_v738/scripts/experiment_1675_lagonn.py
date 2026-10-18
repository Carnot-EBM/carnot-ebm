#!/usr/bin/env python3
"""Exp 1675: LagONN toy Max-3-SAT constraint-satisfaction prototype.

This CPU-only experiment isolates the LagONN mechanism on a deterministic
Max-3-SAT instance. A fixed Ising-style soft penalty is kept below the bias that
prefers all variables true, so it stalls at an infeasible local optimum. The
LagONN variant updates one Lagrange multiplier per violated clause until the
augmented energy makes the satisfying flips favorable.

Spec: REQ-ISING-042, SCENARIO-ISING-042
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = 1675
RUN_DATE = "20260510"
SPEC_REFS = ["REQ-ISING-042", "SCENARIO-ISING-042"]
DEFAULT_RESULT_PATH = REPO_ROOT / "results/experiment_1675_lagonn.json"

Literal = tuple[int, bool]
Clause = tuple[Literal, Literal, Literal]


@dataclass(frozen=True)
class ToyMax3SatProblem:
    """Deterministic Max-3-SAT problem with exactly three literals per clause."""

    n_variables: int
    clauses: tuple[Clause, ...]

    def __post_init__(self) -> None:
        for clause in self.clauses:
            if len(clause) != 3:
                raise ValueError("every Max-3-SAT clause must contain exactly three literals")
            for variable, _positive in clause:
                if variable < 0 or variable >= self.n_variables:
                    raise ValueError(f"literal variable index {variable} out of range")

    def initial_assignment(self) -> np.ndarray:
        """Return the intentionally infeasible all-true assignment."""

        return np.ones(self.n_variables, dtype=np.int8)

    def violation_vector(self, assignment: np.ndarray) -> np.ndarray:
        """Return one binary violation indicator per clause."""

        return np.asarray(
            [0 if _clause_satisfied(assignment, clause) else 1 for clause in self.clauses],
            dtype=np.int8,
        )

    def violation_count(self, assignment: np.ndarray) -> int:
        """Count unsatisfied clauses."""

        return int(self.violation_vector(assignment).sum())

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-facing toy problem description."""

        return {
            "n_variables": self.n_variables,
            "clauses": [
                [_literal_to_text(literal) for literal in clause] for clause in self.clauses
            ],
        }


@dataclass(frozen=True)
class LagONNConfig:
    """Solver controls shared by the LagONN and fixed-penalty runs."""

    max_steps: int = 8
    one_bias: float = 1.25
    soft_penalty_weight: float = 0.75
    initial_lambda: float = 0.75
    dual_lr: float = 0.35
    lambda_decay: float = 0.98

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.one_bias <= 0.0:
            raise ValueError("one_bias must be positive")
        if self.soft_penalty_weight < 0.0:
            raise ValueError("soft_penalty_weight must be non-negative")
        if self.initial_lambda < 0.0:
            raise ValueError("initial_lambda must be non-negative")
        if self.dual_lr <= 0.0:
            raise ValueError("dual_lr must be positive")
        if not 0.0 <= self.lambda_decay < 1.0:
            raise ValueError("lambda_decay must be in [0, 1)")


@dataclass(frozen=True)
class SolverResult:
    """Summary of one solver trajectory."""

    method: str
    converged: bool
    steps_to_convergence: int
    final_assignment: list[int]
    final_violations: int
    final_energy: float
    trace: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable solver result."""

        return {
            "method": self.method,
            "converged": self.converged,
            "steps_to_convergence": self.steps_to_convergence,
            "final_assignment": self.final_assignment,
            "final_violations": self.final_violations,
            "final_energy": self.final_energy,
            "trace": self.trace,
        }


def build_toy_max3sat_problem() -> ToyMax3SatProblem:
    """Build the deterministic toy Max-3-SAT instance for Exp 1675."""

    clauses: tuple[Clause, ...] = (
        ((0, True), (1, True), (2, True)),
        ((3, True), (4, True), (5, True)),
        ((0, False), (1, False), (2, False)),
        ((3, False), (4, False), (5, False)),
        ((0, False), (3, True), (4, True)),
        ((5, False), (1, True), (2, True)),
    )
    return ToyMax3SatProblem(n_variables=6, clauses=clauses)


def ising_bias_energy(assignment: np.ndarray, *, one_bias: float) -> float:
    """Compute the Ising-style bias term that prefers true variables."""

    return float(-one_bias * np.asarray(assignment, dtype=np.float64).sum())


def fixed_soft_energy(
    problem: ToyMax3SatProblem,
    assignment: np.ndarray,
    *,
    one_bias: float,
    penalty_weight: float,
) -> float:
    """Compute fixed soft-penalty energy for the Max-3-SAT clauses."""

    return float(
        ising_bias_energy(assignment, one_bias=one_bias)
        + penalty_weight * problem.violation_count(assignment)
    )


def lagrangian_energy(
    problem: ToyMax3SatProblem,
    assignment: np.ndarray,
    *,
    one_bias: float,
    lambdas: np.ndarray,
) -> float:
    """Compute LagONN augmented energy with one multiplier per clause."""

    violations = problem.violation_vector(assignment).astype(np.float64)
    return float(ising_bias_energy(assignment, one_bias=one_bias) + lambdas @ violations)


def run_soft_penalty_baseline(
    problem: ToyMax3SatProblem,
    *,
    config: LagONNConfig,
) -> SolverResult:
    """Run fixed-weight Ising-style soft-penalty local search."""

    assignment = problem.initial_assignment()
    energy_fn = _fixed_energy_for_config(problem, config)
    trace = [_trace_record(0, assignment, problem, energy_fn(assignment), None, None)]

    for step in range(1, config.max_steps + 1):
        flip_index, _candidate_energy = _best_flip(problem, assignment, energy_fn)
        if flip_index is not None:
            assignment = assignment.copy()
            assignment[flip_index] = 1 - assignment[flip_index]
        trace.append(
            _trace_record(step, assignment, problem, energy_fn(assignment), None, flip_index)
        )
        if problem.violation_count(assignment) == 0:
            return _solver_result(
                "fixed_soft_penalty_ising",
                True,
                step,
                assignment,
                problem,
                energy_fn(assignment),
                trace,
            )

    return _solver_result(
        "fixed_soft_penalty_ising",
        False,
        config.max_steps,
        assignment,
        problem,
        energy_fn(assignment),
        trace,
    )


def run_lagonn_solver(
    problem: ToyMax3SatProblem,
    *,
    config: LagONNConfig,
) -> SolverResult:
    """Run Lagrange multiplier oscillator updates on the same toy instance."""

    assignment = problem.initial_assignment()
    lambdas = np.full(len(problem.clauses), config.initial_lambda, dtype=np.float64)
    energy_fn = _lagrangian_energy_for_config(problem, config, lambdas)
    trace = [_trace_record(0, assignment, problem, energy_fn(assignment), lambdas, None)]

    for step in range(1, config.max_steps + 1):
        current_violations = problem.violation_vector(assignment).astype(np.float64)
        lambdas = np.maximum(
            0.0,
            config.lambda_decay * lambdas + config.dual_lr * current_violations,
        )
        energy_fn = _lagrangian_energy_for_config(problem, config, lambdas)
        flip_index, _candidate_energy = _best_flip(problem, assignment, energy_fn)
        if flip_index is not None:
            assignment = assignment.copy()
            assignment[flip_index] = 1 - assignment[flip_index]
        trace.append(
            _trace_record(step, assignment, problem, energy_fn(assignment), lambdas, flip_index)
        )
        if problem.violation_count(assignment) == 0:
            return _solver_result(
                "lagonn_lagrange_multiplier",
                True,
                step,
                assignment,
                problem,
                energy_fn(assignment),
                trace,
            )

    return _solver_result(
        "lagonn_lagrange_multiplier",
        False,
        config.max_steps,
        assignment,
        problem,
        energy_fn(assignment),
        trace,
    )


def run_experiment(
    *,
    output_path: Path = DEFAULT_RESULT_PATH,
    config: LagONNConfig | None = None,
) -> dict[str, Any]:
    """Run the toy LagONN-vs-soft-penalty comparison and write JSON."""

    cfg = config or LagONNConfig()
    problem = build_toy_max3sat_problem()
    initial_assignment = problem.initial_assignment()
    soft_result = run_soft_penalty_baseline(problem, config=cfg)
    lagonn_result = run_lagonn_solver(problem, config=cfg)
    violation_delta = soft_result.final_violations - lagonn_result.final_violations
    speedup = soft_result.steps_to_convergence / max(lagonn_result.steps_to_convergence, 1)
    artifact = {
        "status": "complete",
        "run_date": RUN_DATE,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "algorithm": "LagONN clause-wise Lagrange multiplier oscillator local search",
        "baseline_algorithm": "fixed-weight Ising-style Max-3-SAT soft penalty",
        "toy_problem": problem.as_dict(),
        "initial_assignment": _assignment_list(initial_assignment),
        "config": {
            "max_steps": cfg.max_steps,
            "one_bias": cfg.one_bias,
            "soft_penalty_weight": cfg.soft_penalty_weight,
            "initial_lambda": cfg.initial_lambda,
            "dual_lr": cfg.dual_lr,
            "lambda_decay": cfg.lambda_decay,
        },
        "steps_to_convergence_lagonn": lagonn_result.steps_to_convergence,
        "steps_to_convergence_soft_penalty": soft_result.steps_to_convergence,
        "lagonn_converged": lagonn_result.converged,
        "soft_penalty_converged": soft_result.converged,
        "final_violations_lagonn": lagonn_result.final_violations,
        "final_violations_soft_penalty": soft_result.final_violations,
        "convergence_speedup_lagonn_over_soft_penalty": float(speedup),
        "lagrange_multiplier_trace": lagonn_result.trace,
        "soft_penalty_trace": soft_result.trace,
        "method_results": {
            "lagonn": lagonn_result.as_dict(),
            "soft_penalty": soft_result.as_dict(),
        },
        "cpu_only": True,
        "simulator_only": True,
        "hardware_execution_performed": False,
        "hardware_claim_allowed": False,
        "honest_verdict": _verdict(
            lagonn_converged=lagonn_result.converged,
            soft_converged=soft_result.converged,
            violation_delta=violation_delta,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(path: Path = DEFAULT_RESULT_PATH) -> dict[str, Any]:
    """Write the bootstrap artifact before the CPU-only run starts."""

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
        artifact["steps_to_convergence_lagonn"],
        artifact["steps_to_convergence_soft_penalty"],
        artifact["final_violations_lagonn"],
        artifact["final_violations_soft_penalty"],
        artifact["hardware_claim_allowed"],
        artifact["honest_verdict"],
    )


def _clause_satisfied(assignment: np.ndarray, clause: Clause) -> bool:
    return any(
        bool(assignment[index]) if positive else not bool(assignment[index])
        for index, positive in clause
    )


def _literal_to_text(literal: Literal) -> str:
    variable, positive = literal
    return f"x{variable}" if positive else f"~x{variable}"


def _fixed_energy_for_config(
    problem: ToyMax3SatProblem,
    config: LagONNConfig,
) -> Callable[[np.ndarray], float]:
    def energy_fn(state: np.ndarray) -> float:
        return fixed_soft_energy(
            problem,
            state,
            one_bias=config.one_bias,
            penalty_weight=config.soft_penalty_weight,
        )

    return energy_fn


def _lagrangian_energy_for_config(
    problem: ToyMax3SatProblem,
    config: LagONNConfig,
    lambdas: np.ndarray,
) -> Callable[[np.ndarray], float]:
    def energy_fn(state: np.ndarray) -> float:
        return lagrangian_energy(
            problem,
            state,
            one_bias=config.one_bias,
            lambdas=lambdas,
        )

    return energy_fn


def _best_flip(
    problem: ToyMax3SatProblem,
    assignment: np.ndarray,
    energy_fn: Callable[[np.ndarray], float],
) -> tuple[int | None, float]:
    base_energy = energy_fn(assignment)
    best_index: int | None = None
    best_energy = base_energy
    for index in range(problem.n_variables):
        candidate = assignment.copy()
        candidate[index] = 1 - candidate[index]
        candidate_energy = energy_fn(candidate)
        if candidate_energy < best_energy - 1e-12:
            best_index = index
            best_energy = candidate_energy
    return best_index, float(best_energy)


def _trace_record(
    step: int,
    assignment: np.ndarray,
    problem: ToyMax3SatProblem,
    energy: float,
    lambdas: np.ndarray | None,
    flipped_variable: int | None,
) -> dict[str, Any]:
    record = {
        "step": int(step),
        "assignment": _assignment_list(assignment),
        "violations": problem.violation_count(assignment),
        "energy": round(float(energy), 6),
        "flipped_variable": flipped_variable,
    }
    if lambdas is not None:
        record["lambda"] = [round(float(value), 6) for value in lambdas]
    return record


def _solver_result(
    method: str,
    converged: bool,
    steps_to_convergence: int,
    assignment: np.ndarray,
    problem: ToyMax3SatProblem,
    final_energy: float,
    trace: list[dict[str, Any]],
) -> SolverResult:
    return SolverResult(
        method=method,
        converged=converged,
        steps_to_convergence=int(steps_to_convergence),
        final_assignment=_assignment_list(assignment),
        final_violations=problem.violation_count(assignment),
        final_energy=round(float(final_energy), 6),
        trace=trace,
    )


def _assignment_list(assignment: Sequence[int] | np.ndarray) -> list[int]:
    return [int(value) for value in assignment]


def _verdict(*, lagonn_converged: bool, soft_converged: bool, violation_delta: int) -> str:
    if lagonn_converged and not soft_converged and violation_delta > 0:
        return "complete_lagonn_converged_soft_penalty_stalled_cpu_only"
    if lagonn_converged and soft_converged:
        return "complete_both_methods_converged_cpu_only"
    if not lagonn_converged and soft_converged:
        return "complete_soft_penalty_better_on_toy_cpu_only"
    return "complete_no_convergence_difference_cpu_only"


if __name__ == "__main__":  # pragma: no cover
    main()
