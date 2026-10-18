#!/usr/bin/env python3
"""Exp 1927: hard-CSP neural-solver reality check.

This CPU-only experiment evaluates a Lagrange-weighted neural-style local
search on a deterministic planted 3-SAT instance. The reported score is the
true clause satisfaction rate from direct 3-SAT evaluation, not the solver's
weighted surrogate.

Spec: REQ-ISING-044, SCENARIO-ISING-044
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = 1927
RUN_DATE = "20260512"
SPEC_REFS = ["REQ-ISING-044", "SCENARIO-ISING-044"]
DEFAULT_RESULT_PATH = REPO_ROOT / "results/experiment_1927_hard_csp_neural.json"

Clause = tuple[int, int, int]
Assignment = tuple[bool, ...]


@dataclass(frozen=True)
class HardCspNeuralConfig:
    """Controls for the hard 3-SAT reality-check run."""

    n_variables: int = 12
    n_clauses: int = 52
    attempts: int = 5
    max_steps: int = 96
    time_budget_s: float = 2.0
    seed: int = 1927
    multiplier_lr: float = 0.35
    multiplier_decay: float = 0.98

    def __post_init__(self) -> None:
        if self.n_variables < 3:
            raise ValueError("n_variables must be at least 3 for 3-SAT")
        if self.n_clauses < 1:
            raise ValueError("n_clauses must be positive")
        max_unique_clauses = math.comb(self.n_variables, 3) * 8
        if self.n_clauses > max_unique_clauses:
            raise ValueError("n_clauses exceeds unique planted 3-SAT clause capacity")
        if self.attempts < 1:
            raise ValueError("attempts must be at least 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.time_budget_s <= 0.0:
            raise ValueError("time_budget_s must be positive")
        if self.multiplier_lr <= 0.0:
            raise ValueError("multiplier_lr must be positive")
        if not 0.0 <= self.multiplier_decay < 1.0:
            raise ValueError("multiplier_decay must be in [0, 1)")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration record."""

        return {
            "n_variables": self.n_variables,
            "n_clauses": self.n_clauses,
            "attempts": self.attempts,
            "max_steps": self.max_steps,
            "time_budget_s": self.time_budget_s,
            "seed": self.seed,
            "multiplier_lr": self.multiplier_lr,
            "multiplier_decay": self.multiplier_decay,
        }


@dataclass(frozen=True)
class Hard3SatInstance:
    """Deterministic 3-SAT instance with a planted satisfying assignment."""

    n_variables: int
    clauses: tuple[Clause, ...]
    planted_assignment: Assignment
    seed: int

    def __post_init__(self) -> None:
        if len(self.planted_assignment) != self.n_variables:
            raise ValueError("planted assignment length must match n_variables")
        for clause in self.clauses:
            if len(clause) != 3:
                raise ValueError("every 3-SAT clause must contain exactly three literals")
            for literal in clause:
                if abs(literal) < 1 or abs(literal) > self.n_variables:
                    raise ValueError(f"literal {literal} out of range")

    @property
    def clause_density(self) -> float:
        """Return the clause-to-variable ratio."""

        return len(self.clauses) / self.n_variables

    def satisfied_constraints(self, assignment: Sequence[bool | int]) -> int:
        """Count true clauses under an assignment."""

        state = _assignment_tuple(assignment)
        return sum(1 for clause in self.clauses if _clause_satisfied(clause, state))

    def constraint_satisfaction_rate(self, assignment: Sequence[bool | int]) -> float:
        """Return true clause satisfaction rate in [0, 1]."""

        return self.satisfied_constraints(assignment) / len(self.clauses)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-facing hard-CSP description."""

        return {
            "family": "planted_3sat",
            "n_variables": self.n_variables,
            "n_clauses": len(self.clauses),
            "clause_density": round(self.clause_density, 6),
            "seed": self.seed,
            "clauses": [list(clause) for clause in self.clauses],
            "planted_assignment": [int(value) for value in self.planted_assignment],
            "planted_true_constraint_satisfaction_rate": self.constraint_satisfaction_rate(
                self.planted_assignment
            ),
        }


def build_hard_3sat_instance(config: HardCspNeuralConfig | None = None) -> Hard3SatInstance:
    """Build a deterministic high-density planted 3-SAT instance."""

    cfg = config or HardCspNeuralConfig()
    rng = random.Random(cfg.seed)
    planted = tuple(bool(rng.getrandbits(1)) for _ in range(cfg.n_variables))
    clauses: list[Clause] = []
    seen: set[Clause] = set()

    while len(clauses) < cfg.n_clauses:
        variables = rng.sample(range(1, cfg.n_variables + 1), 3)
        signs = [rng.choice((-1, 1)) for _ in variables]
        candidate = tuple(sign * variable for sign, variable in zip(signs, variables, strict=True))
        if not _clause_satisfied(candidate, planted):
            repair_index = rng.randrange(3)
            variable = variables[repair_index]
            signs[repair_index] = 1 if planted[variable - 1] else -1
            candidate = tuple(
                sign * variable for sign, variable in zip(signs, variables, strict=True)
            )
        clause = candidate  # type: ignore[assignment]
        if clause not in seen:
            seen.add(clause)
            clauses.append(clause)

    return Hard3SatInstance(
        n_variables=cfg.n_variables,
        clauses=tuple(clauses),
        planted_assignment=planted,
        seed=cfg.seed,
    )


def evaluate_neural_solver_on_hard_csp(
    config: HardCspNeuralConfig | None = None,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Evaluate the neural-style solver and return a terminal artifact payload."""

    cfg = config or HardCspNeuralConfig()
    instance = build_hard_3sat_instance(cfg)
    started = clock()
    deadline = started + cfg.time_budget_s
    attempts: list[dict[str, Any]] = []
    best_assignment: Assignment | None = None
    best_satisfied = -1
    assignments_evaluated = 0
    timeout_exceeded = False

    for attempt_index in range(cfg.attempts):
        if clock() >= deadline:
            timeout_exceeded = True
            break
        attempt = _run_neural_attempt(
            instance,
            cfg,
            attempt_index=attempt_index,
            deadline=deadline,
            clock=clock,
        )
        attempts.append(attempt)
        assignments_evaluated += int(attempt["assignments_evaluated"])
        if attempt["best_satisfied_constraints"] > best_satisfied:
            best_satisfied = int(attempt["best_satisfied_constraints"])
            best_assignment = tuple(bool(value) for value in attempt["best_assignment"])
        if attempt["timed_out"]:
            timeout_exceeded = True
            break
        if best_satisfied == len(instance.clauses):
            break

    wall_time_s = round(float(clock() - started), 6)
    total_constraints = len(instance.clauses)
    if best_assignment is None:
        rate = 0.0
        best_assignment_list: list[int] = []
        best_satisfied = 0
    else:
        rate = best_satisfied / total_constraints
        best_assignment_list = [int(value) for value in best_assignment]

    return {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "run_date": RUN_DATE,
        "solver_name": "lagrange_neural_clause_weight_local_search",
        "csp_family": "planted_3sat",
        "problem": instance.as_dict(),
        "config": cfg.as_dict(),
        "time_budget_s": cfg.time_budget_s,
        "wall_time_s": wall_time_s,
        "timeout_exceeded": timeout_exceeded,
        "assignments_evaluated": assignments_evaluated,
        "true_constraint_satisfaction_rate": round(float(rate), 6),
        "best_satisfied_constraints": best_satisfied,
        "total_constraints": total_constraints,
        "best_assignment": best_assignment_list,
        "attempts": attempts,
        "cpu_only": True,
        "hardware_execution_performed": False,
        "hardware_claim_allowed": False,
        "honest_verdict": _honest_verdict(
            rate=rate,
            timeout_exceeded=timeout_exceeded,
            assignments_evaluated=assignments_evaluated,
        ),
    }


def run_experiment(
    *,
    output_path: Path = DEFAULT_RESULT_PATH,
    config: HardCspNeuralConfig | None = None,
) -> dict[str, Any]:
    """Run Exp 1927 and write the terminal JSON artifact."""

    artifact = evaluate_neural_solver_on_hard_csp(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(path: Path = DEFAULT_RESULT_PATH) -> dict[str, Any]:
    """Write a bootstrap marker before bounded solver work starts."""

    marker = {
        "status": "in_progress",
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "run_date": RUN_DATE,
        "cpu_only": True,
        "hardware_execution_performed": False,
        "hardware_claim_allowed": False,
        "honest_verdict": "in_progress",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return marker


def main() -> None:
    """CLI entry point for the research conductor."""

    write_in_progress_artifact(DEFAULT_RESULT_PATH)
    artifact = run_experiment()
    print(
        artifact["true_constraint_satisfaction_rate"],
        artifact["best_satisfied_constraints"],
        artifact["total_constraints"],
        artifact["assignments_evaluated"],
        artifact["timeout_exceeded"],
        artifact["honest_verdict"],
    )


def _run_neural_attempt(
    instance: Hard3SatInstance,
    config: HardCspNeuralConfig,
    *,
    attempt_index: int,
    deadline: float,
    clock: Callable[[], float],
) -> dict[str, Any]:
    rng = random.Random(config.seed + 10_007 * (attempt_index + 1))
    assignment = tuple(bool(rng.getrandbits(1)) for _ in range(instance.n_variables))
    multipliers = [1.0 for _ in instance.clauses]
    best_assignment = assignment
    best_satisfied = instance.satisfied_constraints(assignment)
    assignments_evaluated = 1
    final_satisfied = best_satisfied
    steps_run = 0
    timed_out = False

    for step in range(1, config.max_steps + 1):
        if clock() >= deadline:
            timed_out = True
            break
        steps_run = step
        flags = [_clause_satisfied(clause, assignment) for clause in instance.clauses]
        final_satisfied = sum(1 for flag in flags if flag)
        if final_satisfied == len(instance.clauses):
            break

        for index, satisfied in enumerate(flags):
            if satisfied:
                multipliers[index] *= config.multiplier_decay
            else:
                multipliers[index] = (
                    multipliers[index] * config.multiplier_decay + config.multiplier_lr
                )

        current_weighted_score = _weighted_clause_score(instance, assignment, multipliers)
        chosen_assignment = assignment
        chosen_score = (float("-inf"), -1, 0)
        for variable_index in range(instance.n_variables):
            if clock() >= deadline:
                timed_out = True
                break
            candidate = _flip_assignment(assignment, variable_index)
            candidate_satisfied = instance.satisfied_constraints(candidate)
            assignments_evaluated += 1
            candidate_weighted_score = _weighted_clause_score(instance, candidate, multipliers)
            score = (
                candidate_weighted_score - current_weighted_score,
                candidate_satisfied,
                -variable_index,
            )
            if score > chosen_score:
                chosen_score = score
                chosen_assignment = candidate
                if candidate_satisfied > best_satisfied:
                    best_satisfied = candidate_satisfied
                    best_assignment = candidate
        if timed_out:
            break
        if chosen_score[0] <= 0.0:
            exploratory_index = (attempt_index + step) % instance.n_variables
            chosen_assignment = _flip_assignment(assignment, exploratory_index)
        assignment = chosen_assignment
        final_satisfied = instance.satisfied_constraints(assignment)
        assignments_evaluated += 1
        if final_satisfied == len(instance.clauses):
            break

    return {
        "attempt_index": attempt_index,
        "seed": config.seed + 10_007 * (attempt_index + 1),
        "steps_run": steps_run,
        "timed_out": timed_out,
        "assignments_evaluated": assignments_evaluated,
        "final_satisfied_constraints": final_satisfied,
        "final_assignment": [int(value) for value in assignment],
        "true_constraint_satisfaction_rate": round(final_satisfied / len(instance.clauses), 6),
        "best_satisfied_constraints": best_satisfied,
        "best_true_constraint_satisfaction_rate": round(
            best_satisfied / len(instance.clauses), 6
        ),
        "best_assignment": [int(value) for value in best_assignment],
        "max_multiplier": round(max(multipliers), 6),
    }


def _weighted_clause_score(
    instance: Hard3SatInstance,
    assignment: Assignment,
    multipliers: Sequence[float],
) -> float:
    return float(
        sum(
            weight
            for clause, weight in zip(instance.clauses, multipliers, strict=True)
            if _clause_satisfied(clause, assignment)
        )
    )


def _assignment_tuple(assignment: Sequence[bool | int]) -> Assignment:
    return tuple(bool(value) for value in assignment)


def _flip_assignment(assignment: Assignment, variable_index: int) -> Assignment:
    updated = list(assignment)
    updated[variable_index] = not updated[variable_index]
    return tuple(updated)


def _clause_satisfied(clause: Sequence[int], assignment: Assignment) -> bool:
    return any(_literal_satisfied(literal, assignment) for literal in clause)


def _literal_satisfied(literal: int, assignment: Assignment) -> bool:
    value = assignment[abs(literal) - 1]
    return value if literal > 0 else not value


def _honest_verdict(
    *,
    rate: float,
    timeout_exceeded: bool,
    assignments_evaluated: int,
) -> str:
    if assignments_evaluated == 0:
        return "timeout_before_any_assignment_cpu_only"
    if rate >= 1.0 and not timeout_exceeded:
        return "complete_neural_solver_found_satisfying_assignment_cpu_only"
    if timeout_exceeded:
        return "timeout_partial_constraint_satisfaction_cpu_only"
    return "complete_partial_constraint_satisfaction_cpu_only"


if __name__ == "__main__":  # pragma: no cover
    main()
