"""Tests for the Exp 1385 self-adaptive Ising FoVer probe.

Spec traces: REQ-VERIFY-1385, SCENARIO-VERIFY-1385
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from carnot.samplers.self_adaptive_ising import (
    FoVerArithmeticConstraintProblem,
    SelfAdaptiveIsingMachine,
    augmented_lagrangian_energy,
    lagrange_relaxation_update,
    load_fover_arithmetic_constraint_problems,
    run_self_adaptive_ising_probe,
)


def test_lagrange_update_uses_subgradient_rule() -> None:
    """REQ-VERIFY-1385-2: lambda grows by eta * normalized violation."""
    updated = lagrange_relaxation_update(
        lambda_value=0.1,
        raw_violation=4.0,
        eta=0.5,
        value_scale=8.0,
    )
    assert updated == pytest.approx(0.35)


def test_augmented_lagrangian_energy_has_linear_and_quadratic_terms() -> None:
    """REQ-VERIFY-1385-3: L = E + lambda*g + rho/2*g^2."""
    energy = augmented_lagrangian_energy(
        base_energy=2.0,
        normalized_violation=0.25,
        lambda_value=0.5,
        rho=0.2,
    )
    assert energy == pytest.approx(2.0 + 0.5 * 0.25 + 0.5 * 0.2 * 0.25**2)


def test_adaptive_solver_converges_when_static_penalty_does_not() -> None:
    """REQ-VERIFY-1385-5: adaptive lambda updates overcome the weak base penalty."""
    problem = FoVerArithmeticConstraintProblem(
        problem_id="unit:10",
        expression="7+3",
        target=10,
        bit_width=5,
        preferred_value=18,
        source="unit",
        row_label="correct",
        text_excerpt="7+3=10",
    )
    machine = SelfAdaptiveIsingMachine(
        problem=problem,
        objective_weight=8.0,
        rho=0.2,
        eta=1.5,
        max_steps=50,
    )

    static = machine.run_static_penalty()
    adaptive = machine.run_adaptive_lagrange()

    assert not static.converged
    assert adaptive.converged
    assert adaptive.convergence_steps < static.convergence_steps
    assert adaptive.final_constraint_violation == 0.0
    assert adaptive.lambda_updates > 0


def test_fover_loader_finds_local_arithmetic_equations() -> None:
    """REQ-VERIFY-1385-4: loader returns local FoVer arithmetic constraints."""
    repo_root = Path(__file__).resolve().parents[2]
    problems = load_fover_arithmetic_constraint_problems(repo_root=repo_root, limit=5)

    assert len(problems) == 5
    assert len({problem.problem_id for problem in problems}) == 5
    assert all(problem.source.startswith("data/fover_") for problem in problems)
    assert all(0 <= problem.target <= problem.max_value for problem in problems)


def test_probe_payload_contains_required_fields_and_viability_gate() -> None:
    """SCENARIO-VERIFY-1385: probe reports required fields and gated viability."""
    repo_root = Path(__file__).resolve().parents[2]
    payload = run_self_adaptive_ising_probe(repo_root=repo_root, limit=5, run_date="20260505")

    required = {
        "status",
        "constraint_problems_tested",
        "static_penalty_convergence_steps",
        "adaptive_lagrange_convergence_steps",
        "convergence_speedup",
        "constraint_violation_reduction",
        "lagrange_multiplier_iterations",
        "penalty_tuning_iterations_saved",
        "adaptive_ising_viable",
        "honest_verdict",
    }
    assert required <= payload.keys()
    assert payload["status"] == "complete"
    assert len(payload["constraint_problems_tested"]) == 5
    assert len(payload["static_penalty_convergence_steps"]) == 5
    assert len(payload["adaptive_lagrange_convergence_steps"]) == 5

    faster_count = sum(1 for result in payload["per_problem_results"] if result["speedup"] > 1.0)
    expected_viable = faster_count >= math.ceil(len(payload["per_problem_results"]) / 2)
    assert payload["adaptive_ising_viable"] is expected_viable
