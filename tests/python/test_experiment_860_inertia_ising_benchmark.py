"""Tests for InertiaIsingSampler and Experiment 860 discrimination benchmark.

Traces to: REQ-ISING-020, SCENARIO-ISING-030

**What these tests verify:**
    - InertiaIsingSampler.sample() returns the correct output shape.
    - energy() is consistent with the J and h parameters (scalar float result).
    - sweeps_to_convergence() returns a positive integer within the cap.
    - discrimination_delta is computable and has the correct sign relationship.
    - Mpemba initialization produces valid ±1 spin configurations.
    - The experiment's build_constraint_matrix() produces correct/incorrect J matrices
      that yield a positive discrimination_delta (the core experimental claim).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from python.carnot.samplers.inertia_ising import InertiaIsingSampler
from scripts.experiment_860_inertia_ising_benchmark import build_constraint_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_sampler() -> InertiaIsingSampler:
    """A small 4-spin sampler for fast unit tests."""
    # REQ-ISING-020
    rng = np.random.default_rng(0)
    J = rng.standard_normal((4, 4))
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = rng.standard_normal(4)
    return InertiaIsingSampler(J, h, alpha=0.5, use_mpemba=True)


@pytest.fixture()
def ferromagnetic_sampler() -> InertiaIsingSampler:
    """A 6-spin fully ferromagnetic Ising problem — known low-energy ground state is all +1."""
    # REQ-ISING-020
    N = 6
    J = np.ones((N, N)) * 0.5
    np.fill_diagonal(J, 0.0)
    h = np.ones(N) * 0.1
    return InertiaIsingSampler(J, h, alpha=0.5, use_mpemba=True)


# ---------------------------------------------------------------------------
# InertiaIsingSampler.sample() — shape and value range
# ---------------------------------------------------------------------------


def test_sample_shape_single(tiny_sampler: InertiaIsingSampler) -> None:
    """sample() with n_samples=1 returns shape (1, N). REQ-ISING-020"""
    result = tiny_sampler.sample(n_sweeps=10, n_samples=1)
    assert result.shape == (1, 4), f"expected (1, 4), got {result.shape}"


def test_sample_shape_multi(tiny_sampler: InertiaIsingSampler) -> None:
    """sample() with n_samples=3 returns shape (3, N). REQ-ISING-020"""
    result = tiny_sampler.sample(n_sweeps=20, n_samples=3)
    assert result.shape == (3, 4), f"expected (3, 4), got {result.shape}"


def test_sample_values_binary(tiny_sampler: InertiaIsingSampler) -> None:
    """All spin values in sample() output must be exactly ±1. REQ-ISING-020"""
    result = tiny_sampler.sample(n_sweeps=10, n_samples=1)
    unique = set(float(v) for v in result.flatten())
    assert unique.issubset({-1.0, 1.0}), f"unexpected spin values: {unique}"


def test_sample_no_inertia_baseline() -> None:
    """alpha=0 (no inertia) sampler still produces valid ±1 samples. REQ-ISING-020"""
    J = np.eye(4) * 0.0
    h = np.zeros(4)
    sampler = InertiaIsingSampler(J, h, alpha=0.0, use_mpemba=False)
    result = sampler.sample(n_sweeps=5, n_samples=1)
    assert result.shape == (1, 4)
    unique = set(float(v) for v in result.flatten())
    assert unique.issubset({-1.0, 1.0})


# ---------------------------------------------------------------------------
# InertiaIsingSampler.energy() — correctness
# ---------------------------------------------------------------------------


def test_energy_ferromagnetic_ground_state(ferromagnetic_sampler: InertiaIsingSampler) -> None:
    """All +1 spin config has lower energy than all -1 in a ferromagnetic system. REQ-ISING-020"""
    N = 6
    s_plus = np.ones(N)
    s_minus = -np.ones(N)
    e_plus = ferromagnetic_sampler.energy(s_plus)
    e_minus = ferromagnetic_sampler.energy(s_minus)
    # Ferromagnetic + positive bias → all +1 should be lower energy
    assert e_plus < e_minus, f"all+1 energy {e_plus} should be < all-1 energy {e_minus}"


def test_energy_returns_scalar_float(tiny_sampler: InertiaIsingSampler) -> None:
    """energy() returns a Python float scalar. REQ-ISING-020"""
    s = np.ones(4)
    result = tiny_sampler.energy(s)
    assert isinstance(result, float), f"expected float, got {type(result)}"


def test_energy_zero_coupling() -> None:
    """With J=0, energy reduces to -h^T s. REQ-ISING-020"""
    h = np.array([1.0, 0.0, -1.0, 0.5])
    J = np.zeros((4, 4))
    sampler = InertiaIsingSampler(J, h, alpha=0.0, use_mpemba=False)
    s = np.ones(4)
    # E = -0.5 * 0 - h @ s = -(1 + 0 - 1 + 0.5) = -0.5
    expected = float(-h @ s)
    assert abs(sampler.energy(s) - expected) < 1e-10


def test_energy_consistency_with_formula(tiny_sampler: InertiaIsingSampler) -> None:
    """energy() matches manual formula -0.5 s^T J s - h^T s. REQ-ISING-020"""
    s = np.array([1.0, -1.0, 1.0, -1.0])
    expected = float(-0.5 * s @ tiny_sampler.J @ s - tiny_sampler.h @ s)
    assert abs(tiny_sampler.energy(s) - expected) < 1e-10


# ---------------------------------------------------------------------------
# InertiaIsingSampler.sweeps_to_convergence() — type and bounds
# ---------------------------------------------------------------------------


def test_sweeps_to_convergence_returns_int(tiny_sampler: InertiaIsingSampler) -> None:
    """sweeps_to_convergence() returns an int >= 1. REQ-ISING-020"""
    result = tiny_sampler.sweeps_to_convergence(threshold=0.01)
    assert isinstance(result, int), f"expected int, got {type(result)}"
    assert result >= 1, f"expected >= 1, got {result}"


def test_sweeps_to_convergence_bounded(tiny_sampler: InertiaIsingSampler) -> None:
    """sweeps_to_convergence() returns at most 1000 (the cap). REQ-ISING-020"""
    result = tiny_sampler.sweeps_to_convergence(threshold=0.01)
    assert result <= 1000, f"expected <= 1000, got {result}"


def test_sweeps_to_convergence_no_inertia() -> None:
    """Baseline (alpha=0) sampler also returns valid convergence count. REQ-ISING-020"""
    J = np.zeros((4, 4))
    h = np.zeros(4)
    sampler = InertiaIsingSampler(J, h, alpha=0.0, use_mpemba=False)
    result = sampler.sweeps_to_convergence()
    assert isinstance(result, int)
    assert 1 <= result <= 1000


# ---------------------------------------------------------------------------
# Mpemba initialization
# ---------------------------------------------------------------------------


def test_mpemba_init_values_binary(ferromagnetic_sampler: InertiaIsingSampler) -> None:
    """_mpemba_init() returns only ±1 values. REQ-ISING-020"""
    s = ferromagnetic_sampler._mpemba_init()
    unique = set(float(v) for v in s)
    assert unique.issubset({-1.0, 1.0}), f"unexpected values: {unique}"


def test_mpemba_init_shape(ferromagnetic_sampler: InertiaIsingSampler) -> None:
    """_mpemba_init() returns array of correct shape. REQ-ISING-020"""
    s = ferromagnetic_sampler._mpemba_init()
    assert s.shape == (6,)


def test_mpemba_init_degenerate_zero() -> None:
    """_mpemba_init() handles eigenvector components of exactly 0 gracefully. REQ-ISING-020"""
    # All-zero J has degenerate eigenvectors; the leading one may have zeros
    J = np.zeros((4, 4))
    h = np.zeros(4)
    sampler = InertiaIsingSampler(J, h, alpha=0.5, use_mpemba=True)
    s = sampler._mpemba_init()
    # All values must be ±1 — no zeros allowed
    assert all(v in {-1.0, 1.0} for v in s.tolist())


# ---------------------------------------------------------------------------
# Discrimination benchmark — SCENARIO-ISING-030
# ---------------------------------------------------------------------------


def test_build_constraint_matrix_correct_shape() -> None:
    """build_constraint_matrix() returns (N, N) J and (N,) h. SCENARIO-ISING-030"""
    J, h = build_constraint_matrix(9, correct=True, seed=0)
    assert J.shape == (9, 9)
    assert h.shape == (9,)


def test_build_constraint_matrix_symmetric() -> None:
    """Coupling matrix J is symmetric (required by Ising model). SCENARIO-ISING-030"""
    J, _ = build_constraint_matrix(9, correct=True, seed=1)
    assert np.allclose(J, J.T), "J must be symmetric"


def test_discrimination_delta_positive() -> None:
    """Correct config energy < error config energy (positive discrimination_delta). SCENARIO-ISING-030

    This is the core experimental claim: InertiaIsingSampler assigns higher energy
    to the erroneous code constraint encoding than to the correct one.
    """
    n = 9  # odd N guarantees ring frustration in the incorrect case
    J_correct, h_correct = build_constraint_matrix(n, correct=True, seed=42)
    J_error, h_error = build_constraint_matrix(n, correct=False, seed=42)

    alpha = 0.5
    n_sweeps = 200

    sampler_correct = InertiaIsingSampler(J_correct, h_correct, alpha=alpha, use_mpemba=True)
    samples_correct = sampler_correct.sample(n_sweeps=n_sweeps, n_samples=1)
    energy_correct = sampler_correct.energy(samples_correct[0])

    sampler_error = InertiaIsingSampler(J_error, h_error, alpha=alpha, use_mpemba=True)
    samples_error = sampler_error.sample(n_sweeps=n_sweeps, n_samples=1)
    energy_error = sampler_error.energy(samples_error[0])

    discrimination_delta = energy_error - energy_correct
    # REQ-ISING-020 + SCENARIO-ISING-030: inertia sampler must discriminate
    assert discrimination_delta > 0, (
        f"discrimination_delta={discrimination_delta:.4f} must be positive. "
        f"energy_correct={energy_correct:.4f}, energy_error={energy_error:.4f}"
    )


# ---------------------------------------------------------------------------
# Deliverable JSON validation
# ---------------------------------------------------------------------------


def test_deliverable_json_exists_and_valid() -> None:
    """results/experiment_860_inertia_ising_benchmark.json has all required fields. REQ-ISING-020"""
    deliverable = Path("results/experiment_860_inertia_ising_benchmark.json")
    if not deliverable.exists():
        pytest.skip("Deliverable not yet written — run the experiment script first")

    with deliverable.open() as f:
        artifact = json.load(f)

    required_fields = [
        "experiment",
        "schema",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "title",
    ]
    for field in required_fields:
        assert field in artifact, f"missing required field: {field}"

    assert "discrimination_delta" in artifact
    assert "inertia_mixing_sweeps" in artifact
    assert "baseline_mixing_sweeps" in artifact
    assert "mixing_sweeps_reduction" in artifact
    assert "honest_verdict" in artifact
    assert artifact["honest_verdict"] in {
        "discrimination_improved",
        "partial_improvement",
        "no_discrimination",
    }
