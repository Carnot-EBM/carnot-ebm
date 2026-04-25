"""Tests for Exp 862: LagrangeAdaptive Ising FR-11 self-learning.

Spec traces: REQ-FR11-020, SCENARIO-FR11-030
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.samplers.lagrange_adaptive import LagrangeAdaptiveIsingConstraints


# ---------------------------------------------------------------------------
# REQ-FR11-020: build_J weights constraints by lambdas
# ---------------------------------------------------------------------------


def test_build_j_weights_constraints_by_lambdas() -> None:
    """build_J must scale constraint coupling by lambda_k * sign * penalty.

    Spec: REQ-FR11-020
    """
    solver = LagrangeAdaptiveIsingConstraints(n_spins=4, n_constraints=2)
    solver.lambdas = np.array([2.0, 3.0])

    constraints = [
        {"spins": [0, 1], "sign": 1, "penalty": 1.0},
        {"spins": [2, 3], "sign": -1, "penalty": 0.5},
    ]
    J = solver.build_J(constraints)

    # Constraint 0: lambda=2.0, sign=+1, penalty=1.0 → coupling = 2.0
    assert J[0, 1] == pytest.approx(2.0)
    assert J[1, 0] == pytest.approx(2.0)  # symmetric

    # Constraint 1: lambda=3.0, sign=-1, penalty=0.5 → coupling = -1.5
    assert J[2, 3] == pytest.approx(-1.5)
    assert J[3, 2] == pytest.approx(-1.5)  # symmetric


def test_build_j_additive_on_j_base() -> None:
    """build_J adds constraint couplings to J_base, not overwriting it.

    Spec: REQ-FR11-020
    """
    solver = LagrangeAdaptiveIsingConstraints(n_spins=3, n_constraints=1)
    solver.J_base[0, 1] = 5.0
    solver.J_base[1, 0] = 5.0
    solver.lambdas = np.array([1.0])

    constraints = [{"spins": [0, 1], "sign": 1, "penalty": 1.0}]
    J = solver.build_J(constraints)

    # 5.0 (base) + 1.0 * 1 * 1.0 (constraint) = 6.0
    assert J[0, 1] == pytest.approx(6.0)
    assert J[1, 0] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# REQ-FR11-020: run_session returns violation_rate and updated lambdas
# ---------------------------------------------------------------------------


def test_run_session_returns_violation_rate_and_lambdas() -> None:
    """run_session must return a violation_rate float and updated lambdas list.

    Spec: REQ-FR11-020
    """
    solver = LagrangeAdaptiveIsingConstraints(
        n_spins=4, n_constraints=2, lambda_init=1.0, lambda_lr=0.1
    )
    constraints = [
        {"spins": [0, 1], "sign": 1, "penalty": 1.0},
        {"spins": [2, 3], "sign": 1, "penalty": 1.0},
    ]
    result = solver.run_session(constraints, n_sweeps=50, n_samples=5)

    assert "violation_rate" in result
    assert "lambdas" in result
    assert "per_constraint_violation_rates" in result

    vr = result["violation_rate"]
    assert isinstance(vr, float)
    assert 0.0 <= vr <= 1.0

    lambdas = result["lambdas"]
    assert len(lambdas) == 2
    # Lambdas must have grown (violation_rate >= 0, lambda_lr > 0)
    assert all(lam >= 1.0 for lam in lambdas)


def test_run_session_lambdas_increase_when_violated() -> None:
    """Lambdas must increase when constraints are violated.

    We force violation by using a J_base that strongly anti-aligns the spins
    for constraint 0, so that constraint is reliably violated.

    Spec: REQ-FR11-020
    """
    solver = LagrangeAdaptiveIsingConstraints(
        n_spins=4, n_constraints=2, lambda_init=1.0, lambda_lr=1.0  # large LR for measurable effect
    )
    # Force s0 and s1 to ANTI-align (violating the sign=+1 "agree" constraint)
    solver.J_base[0, 1] = -50.0
    solver.J_base[1, 0] = -50.0

    constraints = [
        {"spins": [0, 1], "sign": 1, "penalty": 1.0},  # will be violated
        {"spins": [2, 3], "sign": 1, "penalty": 1.0},  # unconstrained (J neutral)
    ]

    initial_lambdas = solver.lambdas.copy()
    result = solver.run_session(constraints, n_sweeps=100, n_samples=20)

    # Constraint 0 should have been violated, so lambda[0] should have increased.
    assert result["lambdas"][0] > float(initial_lambdas[0])


# ---------------------------------------------------------------------------
# REQ-FR11-020 / SCENARIO-FR11-030: delta_s1_to_s5 computation
# ---------------------------------------------------------------------------


def test_delta_s1_to_s5_decreases_over_sessions() -> None:
    """Five-session relay on adversarially-biased constraints must reduce violation rate.

    This is the SCENARIO-FR11-030 acceptance test: given 5 sessions on synthetic
    constraints with adversarial J_base, delta_s1_to_s5 > 0 (violations decrease).

    We use the same setup as the main experiment script so the test is a
    faithful integration check.

    Spec: REQ-FR11-020, SCENARIO-FR11-030
    """
    N_SPINS = 10
    N_CONSTRAINTS = 10
    N_ADVERSARIAL = 5

    # Build constraints: ring topology, all sign=+1 (agree).
    constraints = []
    for k in range(N_CONSTRAINTS):
        i = k % N_SPINS
        j = (k + 1) % N_SPINS
        constraints.append({"spins": [i, j], "sign": 1, "penalty": 1.0})

    # Adversarial J_base: bias first 5 constraints toward violation.
    J_base = np.zeros((N_SPINS, N_SPINS), dtype=np.float64)
    for k in range(N_ADVERSARIAL):
        i, j = constraints[k]["spins"]
        J_base[i, j] -= 2.0
        J_base[j, i] -= 2.0

    solver = LagrangeAdaptiveIsingConstraints(
        n_spins=N_SPINS,
        n_constraints=N_CONSTRAINTS,
        lambda_init=1.0,
        lambda_lr=0.2,
    )
    solver.J_base = J_base

    violation_rates: list[float] = []
    for _ in range(5):
        res = solver.run_session(constraints, n_sweeps=200, n_samples=20)
        violation_rates.append(res["violation_rate"])

    delta_s1_to_s5 = violation_rates[0] - violation_rates[-1]

    # FR-11: self-learning confirmed when delta > 0.
    assert delta_s1_to_s5 > 0, (
        f"Expected violation rate to decrease over 5 sessions, "
        f"but got violation_rates={violation_rates}, delta={delta_s1_to_s5:.4f}"
    )


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------


def test_count_violations_counts_correctly() -> None:
    """_count_violations must count violated constraints per sample.

    Spec: REQ-FR11-020
    """
    solver = LagrangeAdaptiveIsingConstraints(n_spins=4, n_constraints=2)
    # Two samples:
    #   sample 0: s = [+1, +1, +1, -1]
    #   sample 1: s = [+1, -1, +1, +1]
    samples = np.array([[1, 1, 1, -1], [1, -1, 1, 1]], dtype=np.float64)
    constraints = [
        {"spins": [0, 1], "sign": 1, "penalty": 1.0},  # agree: s0==s1?
        {"spins": [2, 3], "sign": 1, "penalty": 1.0},  # agree: s2==s3?
    ]
    counts = solver._count_violations(samples, constraints)

    # Sample 0: (0,1)=+1*+1=+1 (ok), (2,3)=+1*-1=-1 (violated) → 1 violation
    assert counts[0] == 1
    # Sample 1: (0,1)=+1*-1=-1 (violated), (2,3)=+1*+1=+1 (ok) → 1 violation
    assert counts[1] == 1


def test_constraint_violation_rate_is_fraction() -> None:
    """_constraint_violation must return a float in [0, 1].

    Spec: REQ-FR11-020
    """
    solver = LagrangeAdaptiveIsingConstraints(n_spins=4, n_constraints=1)
    # 3 samples: all violate constraint (s0 != s1 but sign=+1 requires s0==s1)
    samples = np.array([[1, -1, 1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64)
    constraint = {"spins": [0, 1], "sign": 1, "penalty": 1.0}
    rate = solver._constraint_violation(samples, constraint)

    assert rate == pytest.approx(1.0)


def test_constraint_violation_zero_when_all_satisfied() -> None:
    """_constraint_violation must return 0.0 when all samples satisfy the constraint.

    Spec: REQ-FR11-020
    """
    solver = LagrangeAdaptiveIsingConstraints(n_spins=4, n_constraints=1)
    # All samples have s0 == s1 (sign=+1 satisfied)
    samples = np.array([[1, 1, -1, 1], [-1, -1, 1, -1]], dtype=np.float64)
    constraint = {"spins": [0, 1], "sign": 1, "penalty": 1.0}
    rate = solver._constraint_violation(samples, constraint)

    assert rate == pytest.approx(0.0)


def test_per_constraint_violation_rates_length() -> None:
    """run_session must return per_constraint_violation_rates with one entry per constraint.

    Spec: REQ-FR11-020
    """
    solver = LagrangeAdaptiveIsingConstraints(n_spins=6, n_constraints=3)
    constraints = [
        {"spins": [0, 1], "sign": 1, "penalty": 1.0},
        {"spins": [2, 3], "sign": 1, "penalty": 1.0},
        {"spins": [4, 5], "sign": 1, "penalty": 1.0},
    ]
    result = solver.run_session(constraints, n_sweeps=20, n_samples=5)
    assert len(result["per_constraint_violation_rates"]) == 3
