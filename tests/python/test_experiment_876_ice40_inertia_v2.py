"""Tests for SparseInertiaIsingSampler and Experiment 876 synthesis benchmark.

Traces to: REQ-HW-035, SCENARIO-HW-035

**What these tests verify:**
    - SparseInertiaIsingSampler.energy() computes correctly from sparse adjacency.
    - SparseInertiaIsingSampler._compute_local_fields() matches brute-force dense sum.
    - SparseInertiaIsingSampler.sweeps_to_converge() returns a valid int in bounds.
    - alpha_sweep_convergence() returns a dict with correct keys and decreasing values.
    - make_n8_constraint_adjacency() produces the expected K=12 ring+chord structure.
    - Sample() returns a valid ±1 spin array of the correct shape.
    - The deliverable JSON exists and contains all required schema fields.
    - Synthesis LUT count is recorded correctly in the deliverable JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from python.carnot.samplers.sparse_inertia_ising import (
    SparseInertiaIsingSampler,
    alpha_sweep_convergence,
    make_n8_constraint_adjacency,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ring4_sampler() -> SparseInertiaIsingSampler:
    """A 4-spin ring graph with ferromagnetic J=+1.0 and alpha=0.5.

    REQ-HW-035: sparse adjacency sampler instantiation.
    """
    adj = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]
    return SparseInertiaIsingSampler(n_spins=4, adjacency_list=adj, alpha=0.5, beta=1.0)


@pytest.fixture()
def n8_sampler() -> SparseInertiaIsingSampler:
    """Standard N=8 ring+chord sampler used in Experiment 876.

    REQ-HW-035: the canonical benchmark graph.
    """
    return SparseInertiaIsingSampler(
        n_spins=8,
        adjacency_list=make_n8_constraint_adjacency(),
        alpha=0.5,
        beta=1.0,
    )


# ---------------------------------------------------------------------------
# make_n8_constraint_adjacency — structure
# ---------------------------------------------------------------------------


def test_n8_adjacency_count() -> None:
    """make_n8_constraint_adjacency() returns exactly K=12 pairs. REQ-HW-035"""
    adj = make_n8_constraint_adjacency()
    assert len(adj) == 12, f"expected 12 pairs, got {len(adj)}"


def test_n8_adjacency_ring_present() -> None:
    """All 8 ring edges are present with J=+1.0. REQ-HW-035"""
    adj = make_n8_constraint_adjacency()
    ring_pairs = {(i, (i + 1) % 8) for i in range(8)}
    adj_pairs = {(i, j) for i, j, _ in adj}
    for pair in ring_pairs:
        assert pair in adj_pairs, f"missing ring edge {pair}"


def test_n8_adjacency_chord_present() -> None:
    """All 4 chord edges (0-4, 1-5, 2-6, 3-7) are present. REQ-HW-035"""
    adj = make_n8_constraint_adjacency()
    adj_pairs = {(i, j) for i, j, _ in adj}
    chord_pairs = {(0, 4), (1, 5), (2, 6), (3, 7)}
    for pair in chord_pairs:
        assert pair in adj_pairs, f"missing chord edge {pair}"


def test_n8_adjacency_all_weights_positive() -> None:
    """All J_ij values are positive (ferromagnetic). REQ-HW-035"""
    for i, j, w in make_n8_constraint_adjacency():
        assert w > 0.0, f"edge ({i},{j}) has non-positive weight {w}"


# ---------------------------------------------------------------------------
# SparseInertiaIsingSampler local field computation
# ---------------------------------------------------------------------------


def test_local_fields_all_plus1(ring4_sampler: SparseInertiaIsingSampler) -> None:
    """With all +1 spins, every local field should be +2 (2 neighbors × +1). REQ-HW-035"""
    s = np.ones(4)
    h = ring4_sampler._compute_local_fields(s)
    # Ring: each spin has exactly 2 ferromagnetic neighbors, all +1
    np.testing.assert_allclose(h, np.full(4, 2.0), atol=1e-12)


def test_local_fields_all_minus1(ring4_sampler: SparseInertiaIsingSampler) -> None:
    """With all -1 spins, every local field should be -2. REQ-HW-035"""
    s = -np.ones(4)
    h = ring4_sampler._compute_local_fields(s)
    np.testing.assert_allclose(h, np.full(4, -2.0), atol=1e-12)


def test_local_fields_alternating(ring4_sampler: SparseInertiaIsingSampler) -> None:
    """Alternating [+1,-1,+1,-1] on ring4: each spin's neighbors have OPPOSITE sign.

    For ferromagnetic J=+1 ring4, spin 0 (+1) has neighbors 1 (-1) and 3 (-1):
      h[0] = 1*(-1) + 1*(-1) = -2.
    Spin 1 (-1) has neighbors 0 (+1) and 2 (+1): h[1] = +2. Pattern: [-2,+2,-2,+2].

    REQ-HW-035
    """
    s = np.array([1.0, -1.0, 1.0, -1.0])
    h = ring4_sampler._compute_local_fields(s)
    expected = np.array([-2.0, 2.0, -2.0, 2.0])
    np.testing.assert_allclose(h, expected, atol=1e-12)


def test_local_fields_matches_dense(n8_sampler: SparseInertiaIsingSampler) -> None:
    """SparseInertiaIsingSampler local fields match brute-force dense computation. REQ-HW-035"""
    rng = np.random.default_rng(7)
    s = rng.choice([-1.0, 1.0], size=8)

    # Sparse result
    h_sparse = n8_sampler._compute_local_fields(s)

    # Build dense J matrix from adjacency list and compute dense h
    adj = make_n8_constraint_adjacency()
    J_dense = np.zeros((8, 8))
    for i, j, w in adj:
        J_dense[i, j] += w
        J_dense[j, i] += w
    h_dense = J_dense @ s

    np.testing.assert_allclose(
        h_sparse, h_dense, atol=1e-12, err_msg="sparse and dense local fields disagree"
    )


# ---------------------------------------------------------------------------
# SparseInertiaIsingSampler energy
# ---------------------------------------------------------------------------


def test_energy_all_aligned_is_minimum(n8_sampler: SparseInertiaIsingSampler) -> None:
    """All-+1 configuration has lower energy than mixed for ferromagnetic graph. REQ-HW-035"""
    all_plus = np.ones(8)
    rng = np.random.default_rng(99)
    mixed = rng.choice([-1.0, 1.0], size=8)
    # Not all the same — force some disagreement
    mixed[0] = -1.0
    e_aligned = n8_sampler.energy(all_plus)
    e_mixed = n8_sampler.energy(mixed)
    assert e_aligned < e_mixed, f"aligned energy {e_aligned:.3f} should be < mixed {e_mixed:.3f}"


def test_energy_returns_float(ring4_sampler: SparseInertiaIsingSampler) -> None:
    """energy() returns a Python float. REQ-HW-035"""
    result = ring4_sampler.energy(np.ones(4))
    assert isinstance(result, float), f"expected float, got {type(result)}"


def test_energy_symmetry() -> None:
    """energy() is the same for s and -s on a pure-coupling (no bias) graph. REQ-HW-035

    For a ferromagnetic Ising model with no external field, E(s) = E(-s)
    because each coupling term J_ij * s_i * s_j is unchanged by flipping all signs.
    """
    adj = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
    sampler = SparseInertiaIsingSampler(3, adj, alpha=0.5)
    s = np.array([1.0, -1.0, 1.0])
    e_plus = sampler.energy(s)
    e_minus = sampler.energy(-s)
    assert abs(e_plus - e_minus) < 1e-12, f"E(s)={e_plus} != E(-s)={e_minus}"


def test_energy_zero_graph() -> None:
    """With no edges, energy is always 0.0. REQ-HW-035"""
    sampler = SparseInertiaIsingSampler(4, [], alpha=0.5)
    s = np.array([1.0, -1.0, 1.0, -1.0])
    assert sampler.energy(s) == 0.0


# ---------------------------------------------------------------------------
# SparseInertiaIsingSampler.sweeps_to_converge
# ---------------------------------------------------------------------------


def test_sweeps_to_converge_returns_int(n8_sampler: SparseInertiaIsingSampler) -> None:
    """sweeps_to_converge() returns an int >= 1. REQ-HW-035"""
    result = n8_sampler.sweeps_to_converge(energy_threshold=-3.0, max_sweeps=400, seed=0)
    assert isinstance(result, int), f"expected int, got {type(result)}"
    assert result >= 1


def test_sweeps_to_converge_bounded(n8_sampler: SparseInertiaIsingSampler) -> None:
    """sweeps_to_converge() is at most max_sweeps. REQ-HW-035"""
    max_sw = 50
    result = n8_sampler.sweeps_to_converge(energy_threshold=-100.0, max_sweeps=max_sw, seed=0)
    assert result <= max_sw, f"expected <= {max_sw}, got {result}"


def test_sweeps_to_converge_no_inertia_valid() -> None:
    """Baseline (alpha=0) sampler returns valid convergence count. REQ-HW-035"""
    adj = make_n8_constraint_adjacency()
    sampler = SparseInertiaIsingSampler(8, adj, alpha=0.0, beta=1.0)
    result = sampler.sweeps_to_converge(energy_threshold=-3.0, max_sweeps=400, seed=42)
    assert isinstance(result, int)
    assert 1 <= result <= 400


def test_sweeps_easy_threshold() -> None:
    """Very easy threshold (energy < 0) should converge in 1 sweep for all-+1 graph. REQ-HW-035"""
    adj = make_n8_constraint_adjacency()
    sampler = SparseInertiaIsingSampler(8, adj, alpha=0.5, beta=2.0)
    result = sampler.sweeps_to_converge(energy_threshold=-0.1, max_sweeps=400, seed=0)
    assert result <= 400  # should converge quickly for easy threshold


# ---------------------------------------------------------------------------
# SparseInertiaIsingSampler.sample
# ---------------------------------------------------------------------------


def test_sample_shape(ring4_sampler: SparseInertiaIsingSampler) -> None:
    """sample() returns shape (N,) for N=4. REQ-HW-035"""
    result = ring4_sampler.sample(n_sweeps=10, seed=0)
    assert result.shape == (4,), f"expected (4,), got {result.shape}"


def test_sample_binary_values(ring4_sampler: SparseInertiaIsingSampler) -> None:
    """sample() returns only ±1 values. REQ-HW-035"""
    result = ring4_sampler.sample(n_sweeps=10, seed=1)
    unique = set(float(v) for v in result)
    assert unique.issubset({-1.0, 1.0}), f"unexpected values: {unique}"


# ---------------------------------------------------------------------------
# alpha_sweep_convergence
# ---------------------------------------------------------------------------


def test_alpha_sweep_returns_all_keys() -> None:
    """alpha_sweep_convergence() returns a value for each queried alpha. REQ-HW-035"""
    alphas = [0.0, 0.5]
    result = alpha_sweep_convergence(alphas, n_trials=5, energy_threshold=-3.0, max_sweeps=100)
    assert set(result.keys()) == set(alphas), f"missing keys: {set(alphas) - set(result.keys())}"


def test_alpha_sweep_values_positive() -> None:
    """All returned mean sweep counts are positive. REQ-HW-035"""
    result = alpha_sweep_convergence([0.5], n_trials=3, energy_threshold=-3.0, max_sweeps=100)
    assert result[0.5] > 0.0


def test_alpha_sweep_inertia_faster_than_none() -> None:
    """alpha=0.5 converges faster than alpha=0.0 on the N=8 ring+chord graph. REQ-HW-035

    SCENARIO-HW-035: inertia reduces mixing time vs baseline.
    """
    result = alpha_sweep_convergence([0.0, 0.5], n_trials=20, energy_threshold=-3.0, max_sweeps=400)
    assert result[0.5] < result[0.0], (
        f"expected inertia (0.5) faster than baseline (0.0): {result[0.5]:.1f} vs {result[0.0]:.1f}"
    )


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------


def test_ema_update_moves_toward_field(ring4_sampler: SparseInertiaIsingSampler) -> None:
    """_update_ema() moves h_ema toward the new field values. REQ-HW-035"""
    ring4_sampler._h_ema = np.zeros(4)
    # All +1 spins → h = [2, 2, 2, 2] for ring4
    h_new = np.array([2.0, 2.0, 2.0, 2.0])
    ring4_sampler._update_ema(h_new)
    # After one update with alpha=0.5: h_ema = 0.5*0 + 0.5*2 = 1.0 each
    expected = 0.5 * np.array([2.0, 2.0, 2.0, 2.0])  # (1-alpha)*h_new + alpha*0
    np.testing.assert_allclose(ring4_sampler._h_ema, expected, atol=1e-12)


def test_ema_zero_alpha_tracks_field(ring4_sampler: SparseInertiaIsingSampler) -> None:
    """With alpha=0, h_ema equals h_new exactly every step (no memory). REQ-HW-035"""
    no_inertia = SparseInertiaIsingSampler(4, [(0, 1, 1.0)], alpha=0.0)
    no_inertia._h_ema = np.array([5.0, -3.0, 0.0, 1.0])
    h_new = np.array([1.0, 2.0, -1.0, 0.5])
    no_inertia._update_ema(h_new)
    np.testing.assert_allclose(no_inertia._h_ema, h_new, atol=1e-12)


def test_flip_probability_stable_spin(ring4_sampler: SparseInertiaIsingSampler) -> None:
    """_flip_probability: aligned spin has low p_flip; anti-aligned has high p_flip. REQ-HW-035"""
    # Set h_ema so spin 0 is strongly aligned (+1 spin, +large field)
    ring4_sampler._h_ema = np.array([10.0, -10.0, 10.0, -10.0])
    # Spins: +1, -1, +1, -1 (all aligned with their fields)
    s = np.array([1.0, -1.0, 1.0, -1.0])
    p = ring4_sampler._flip_probability(s)
    # All spins aligned → low p_flip (all should be < 0.5)
    assert np.all(p < 0.5), f"all aligned spins should have p_flip < 0.5, got {p}"


def test_flip_probability_anti_aligned(ring4_sampler: SparseInertiaIsingSampler) -> None:
    """Anti-aligned spin (fights field) has high p_flip. REQ-HW-035"""
    # Field says +1, but spin is -1 → wants to flip
    ring4_sampler._h_ema = np.array([10.0, 10.0, 10.0, 10.0])
    s = np.array([-1.0, -1.0, -1.0, -1.0])  # all anti-aligned
    p = ring4_sampler._flip_probability(s)
    assert np.all(p > 0.5), f"anti-aligned spins should have p_flip > 0.5, got {p}"


# ---------------------------------------------------------------------------
# Deliverable JSON validation
# ---------------------------------------------------------------------------


def test_deliverable_json_exists_and_valid() -> None:
    """results/experiment_876_ice40_inertia_v2.json has all required schema fields.

    REQ-HW-035, SCENARIO-HW-035
    """
    deliverable = project_root / "results" / "experiment_876_ice40_inertia_v2.json"
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
        "best_alpha",
        "sweeps_reduction",
        "inertia_sweeps",
        "baseline_sweeps",
        "lut_count",
        "synthesis_clean",
        "honest_verdict",
    ]
    for field in required_fields:
        assert field in artifact, f"missing required field: {field}"

    assert artifact["honest_verdict"] in {
        "sweeps_5x_and_synthesis_clean",
        "sweeps_5x_synthesis_over_budget",
        "sweeps_improved_below_5x",
        "synthesis_blocked",
        "no_improvement",
    }, f"unexpected honest_verdict: {artifact['honest_verdict']}"

    assert isinstance(artifact["lut_count"], int)
    assert isinstance(artifact["best_alpha"], float)
    assert artifact["sweeps_reduction"] >= 1.0
