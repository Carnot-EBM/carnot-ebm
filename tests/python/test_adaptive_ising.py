"""Tests for the Exp 2359 self-adaptive Lagrangian Ising sampler.

Spec traces: REQ-SAMPLE-2359, SCENARIO-SAMPLE-2359
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from carnot.samplers.adaptive_ising import (
    SelfAdaptiveIsingSampler,
    make_benchmark_problems,
    run_fixed_penalty_baseline,
    run_self_adaptive_ising_benchmark,
)


def test_sampler_initializes_lambdas_from_constraint_count() -> None:
    """REQ-SAMPLE-2359-1: one zero lambda is created per reported constraint."""
    sampler = SelfAdaptiveIsingSampler(
        J=np.zeros((4, 4)),
        h=np.zeros(4),
        constraint_fn=lambda state: [abs(float(state.sum() - 2)) / 4.0, 0.0],
        init_state=np.array([1, 1, 0, 0]),
        random_seed=42,
    )

    assert sampler.state.tolist() == [1, 1, 0, 0]
    assert sampler.lambdas.tolist() == [0.0, 0.0]


def test_lagrange_update_uses_violation_rule() -> None:
    """REQ-SAMPLE-2359-3: lambda update is lambdas += lr * violations."""
    sampler = SelfAdaptiveIsingSampler(
        J=np.zeros((3, 3)),
        h=np.zeros(3),
        constraint_fn=lambda state: [0.25],
        lr=0.4,
        init_state=np.array([1, 0, 1]),
        random_seed=42,
    )

    updated = sampler.lagrange_update(np.array([0.25]))

    assert updated.tolist() == pytest.approx([0.1])
    assert sampler.lambdas.tolist() == pytest.approx([0.1])


def test_sample_step_runs_binary_gibbs_sweep() -> None:
    """REQ-SAMPLE-2359-2: sample_step visits every spin and preserves binary state."""
    sampler = SelfAdaptiveIsingSampler(
        J=np.zeros((5, 5)),
        h=np.full(5, 5.0),
        constraint_fn=lambda state: [0.0],
        beta=20.0,
        init_state=np.zeros(5, dtype=int),
        random_seed=42,
    )

    state = sampler.sample_step()

    assert state.shape == (5,)
    assert set(state.tolist()) <= {0, 1}
    assert state.sum() == 5


def test_solve_records_first_feasible_outer_iteration() -> None:
    """REQ-SAMPLE-2359-4: solve reports convergence when the violation threshold is hit."""
    sampler = SelfAdaptiveIsingSampler(
        J=np.zeros((6, 6)),
        h=np.array([1.5, 1.4, 1.3, -1.5, -1.4, -1.3]),
        constraint_fn=lambda state: [abs(float(state.sum() - 3)) / 6.0],
        lr=2.0,
        beta=10.0,
        init_state=np.zeros(6, dtype=int),
        random_seed=42,
    )

    result = sampler.solve(n_outer=20, n_inner=3, threshold=0.1)

    assert result.feasible is True
    assert result.iterations_to_feasibility <= 20
    assert result.final_constraint_violation < 0.1
    assert len(result.violation_history) == result.iterations_to_feasibility


def test_fixed_penalty_baseline_uses_constant_lambda() -> None:
    """REQ-SAMPLE-2359-5: fixed baseline keeps all constraint weights at 1.0."""
    problem = make_benchmark_problems(random_seed=42)[0]
    result = run_fixed_penalty_baseline(problem, fixed_penalty=1.0, random_seed=42)

    assert result.lambda_history
    assert all(row == [1.0] for row in result.lambda_history)


def test_benchmark_payload_contains_required_artifact_fields() -> None:
    """SCENARIO-SAMPLE-2359: benchmark returns the required Exp 2359 fields."""
    payload = run_self_adaptive_ising_benchmark(random_seed=42)

    required = {
        "honest_verdict",
        "adaptive_ising_validated",
        "adaptive_speedup",
        "final_constraint_violation",
        "n_problems",
        "random_seed",
        "per_problem_results",
    }
    assert required <= payload.keys()
    assert payload["honest_verdict"].startswith(("complete:", "success:", "blocked:", "partial:"))
    assert payload["n_problems"] == 2
    assert payload["random_seed"] == 42
    assert len(payload["per_problem_results"]) == 2
    assert math.isfinite(payload["adaptive_speedup"])
    assert payload["final_constraint_violation"] < 0.1

    per_problem_speedups = [result["adaptive_speedup"] for result in payload["per_problem_results"]]
    expected_validated = any(speedup >= 1.5 for speedup in per_problem_speedups)
    assert payload["adaptive_ising_validated"] is expected_validated
