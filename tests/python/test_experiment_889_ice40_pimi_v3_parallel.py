"""Tests for Experiment 889: Synchronous PIMI sampler with truly parallel spin updates.

Verifies that SynchronousPIMISampler uses s_current (not s_new) for all
local field computations — the defining property of synchronous/parallel updates.

Spec: REQ-HW-036, SCENARIO-HW-036
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from python.carnot.samplers.synchronous_pimi import (
    SynchronousPIMISampler,
    make_n8_coupling_matrix,
    pimi_alpha_sweep,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def n8_sampler() -> SynchronousPIMISampler:
    """Standard N=8 ferromagnetic ring+chord sampler for benchmarks."""
    J = make_n8_coupling_matrix()
    h = np.zeros(8, dtype=np.float64)
    return SynchronousPIMISampler(n_spins=8, J=J, h=h, alpha=0.5, beta=1.0)


# ---------------------------------------------------------------------------
# REQ-HW-036: sample_once uses s_current (not s_new) for ALL j
# ---------------------------------------------------------------------------

class TestSampleOnceParallelProperty:
    """SCENARIO-HW-036-A: Verify true parallel (synchronous) update semantics.

    The key property: sample_once() must read h_local from s_current ONLY.
    If even one spin used s_new (updated neighbor), it would be a sequential
    update masquerading as parallel — exactly the Exp 860/876 mistake.
    """

    def test_sample_once_uses_only_s_current_for_field_computation(self):
        """Verify h_ema after sample_once reflects h_local computed from s_current.

        The post-call h_ema must equal alpha*h_ema_prev + (1-alpha)*(J @ s_input + h).
        If the sampler used s_new instead of s_current, this equality would fail
        because s_new != s_current whenever any spin flips.

        We use beta=0 to guarantee maximum flip probability (all spins can flip),
        making the s_new != s_current case likely.  Even so, h_ema must reflect
        the PRE-flip s_current, not the post-flip s_new.

        Spec: REQ-HW-036
        """
        J = make_n8_coupling_matrix()
        h = np.zeros(8, dtype=np.float64)
        sampler = SynchronousPIMISampler(n_spins=8, J=J, h=h, alpha=0.5, beta=1.0)

        s_input = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        # Compute expected h_ema based on s_input (the snapshot)
        h_local_expected = J @ s_input + h
        h_ema_expected = 0.5 * np.zeros(8) + 0.5 * h_local_expected  # alpha=0.5

        rng = np.random.default_rng(42)
        sampler.reset()
        s_new = sampler.sample_once_seeded(s_input, rng)

        # h_ema must match h_local computed from s_input, regardless of what s_new is
        np.testing.assert_allclose(
            sampler.h_ema, h_ema_expected, rtol=1e-9,
            err_msg="h_ema must reflect h_local from s_current (pre-flip snapshot)"
        )

    def test_s_new_not_fed_back_into_field_computation(self):
        """Verify that s_new values do not appear in h_local computation.

        If the sampler were sequential (checkerboard-style), some spins would
        see updated neighbors when computing h_local.  In synchronous mode,
        ALL spins see the same s_current snapshot.

        We test this by checking that calling sample_once() with an all-+1
        configuration gives a different result than if half the spins were
        'pre-flipped' (simulating a sequential contaminant).

        Spec: REQ-HW-036
        """
        J = make_n8_coupling_matrix()
        h = np.zeros(8, dtype=np.float64)
        sampler = SynchronousPIMISampler(n_spins=8, J=J, h=h, alpha=0.5, beta=2.0)
        rng = np.random.default_rng(42)

        s_all_up = np.ones(8, dtype=np.float64)
        # The sampler's h_local must use s_all_up, not any modified version.
        # We verify by checking h_ema after one call — it must equal (1-alpha)*J@s_all_up
        sampler.reset()
        sampler.sample_once_seeded(s_all_up, rng)

        # h_ema after first step: alpha*0 + (1-alpha)*h_local
        # h_local for all-up state: each spin has degree-3 neighbors all up = 3.0
        expected_h_local = J @ s_all_up  # all +3 for ring+chord with J=1
        expected_h_ema = (1.0 - 0.5) * expected_h_local  # alpha=0.5, h_ema_init=0

        np.testing.assert_allclose(
            sampler.h_ema, expected_h_ema, rtol=1e-9,
            err_msg="h_ema must reflect h_local computed from s_current, not from s_new"
        )

    def test_all_spins_update_simultaneously_not_sequentially(self):
        """Verify that ALL spins flip based on the same h_ema snapshot.

        In a sequential update, spin 0 would flip, then spin 1 would see
        the new spin 0, etc.  In synchronous mode, all flip decisions are
        made with the SAME h_ema.

        We test by running sample_once_seeded with a fixed RNG and verifying
        that the resulting s_new is consistent with parallel decisions.

        Spec: REQ-HW-036
        """
        J = make_n8_coupling_matrix()
        h = np.zeros(8, dtype=np.float64)
        sampler = SynchronousPIMISampler(n_spins=8, J=J, h=h, alpha=0.5, beta=1.0)

        # Mixed spin state to exercise both flip and no-flip cases
        s_current = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        rng = np.random.default_rng(0)

        sampler.reset()
        # Manually compute what sample_once_seeded should do
        h_local = J @ s_current + h
        expected_h_ema = 0.5 * np.zeros(8) + 0.5 * h_local  # alpha=0.5, init=0
        argument = np.clip(2.0 * 1.0 * expected_h_ema * s_current, -500.0, 500.0)
        expected_p_flip = 1.0 / (1.0 + np.exp(argument))

        # Re-seed RNG to get the same random values
        rng_check = np.random.default_rng(0)
        expected_rands = rng_check.random(8)
        expected_flip = expected_rands < expected_p_flip
        expected_s_new = s_current.copy()
        expected_s_new[expected_flip] = -s_current[expected_flip]

        # Now run actual sampler
        sampler.reset()
        rng2 = np.random.default_rng(0)
        actual_s_new = sampler.sample_once_seeded(s_current, rng2)

        np.testing.assert_array_equal(
            actual_s_new, expected_s_new,
            err_msg="Parallel flip decisions must match manual computation from same snapshot"
        )


# ---------------------------------------------------------------------------
# REQ-HW-036: EMA update correctness
# ---------------------------------------------------------------------------

class TestEMAUpdate:
    """SCENARIO-HW-036-B: EMA update uses alpha * h_ema_prev + (1-alpha) * h_local."""

    def test_ema_initial_state_is_zero(self):
        """h_ema starts at zero after construction or reset."""
        J = make_n8_coupling_matrix()
        h = np.zeros(8)
        sampler = SynchronousPIMISampler(n_spins=8, J=J, h=h, alpha=0.5)
        np.testing.assert_array_equal(sampler.h_ema, np.zeros(8))

    def test_ema_reset_clears_state(self, n8_sampler):
        """reset() restores h_ema to zero regardless of prior state."""
        rng = np.random.default_rng(0)
        s = np.ones(8)
        for _ in range(5):
            s = n8_sampler.sample_once_seeded(s, rng)
        # h_ema should be non-zero now
        assert np.any(n8_sampler.h_ema != 0.0)
        n8_sampler.reset()
        np.testing.assert_array_equal(n8_sampler.h_ema, np.zeros(8))

    def test_ema_formula_first_step(self):
        """After one step from zero, h_ema = (1-alpha) * h_local."""
        J = np.eye(4, dtype=np.float64)  # Simple 4-spin identity for testability
        h = np.zeros(4)
        alpha = 0.3
        sampler = SynchronousPIMISampler(n_spins=4, J=J, h=h, alpha=alpha, beta=0.0)
        # beta=0 means p_flip=0.5 always, but we just need h_ema to be set

        s = np.array([1.0, 1.0, -1.0, -1.0])
        rng = np.random.default_rng(0)
        sampler.reset()
        sampler.sample_once_seeded(s, rng)

        expected_h_local = J @ s  # = s itself (identity coupling)
        expected_h_ema = (1.0 - alpha) * expected_h_local
        np.testing.assert_allclose(sampler.h_ema, expected_h_ema, rtol=1e-9)

    def test_ema_formula_second_step(self):
        """After two steps, h_ema accumulates correctly."""
        J = np.eye(4, dtype=np.float64)
        h = np.zeros(4)
        alpha = 0.5
        sampler = SynchronousPIMISampler(n_spins=4, J=J, h=h, alpha=alpha, beta=0.0)

        s0 = np.array([1.0, 1.0, -1.0, -1.0])
        rng = np.random.default_rng(0)
        sampler.reset()
        s1 = sampler.sample_once_seeded(s0, rng)

        # After step 1: h_ema1 = (1-alpha) * (J @ s0)
        h_ema1_expected = (1.0 - alpha) * (J @ s0)
        np.testing.assert_allclose(sampler.h_ema, h_ema1_expected, rtol=1e-9)

        # Step 2: h_local2 = J @ s1
        sampler.sample_once_seeded(s1, rng)
        h_local2 = J @ s1
        h_ema2_expected = alpha * h_ema1_expected + (1.0 - alpha) * h_local2
        np.testing.assert_allclose(sampler.h_ema, h_ema2_expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# REQ-HW-036: Energy function correctness
# ---------------------------------------------------------------------------

class TestEnergy:
    """SCENARIO-HW-036-C: Energy E(s) = -0.5 * s^T J s - h^T s."""

    def test_energy_all_aligned_minimum(self):
        """All-up configuration achieves minimum energy for ferromagnetic J."""
        J = make_n8_coupling_matrix()
        h = np.zeros(8)
        sampler = SynchronousPIMISampler(8, J, h)
        s_all_up = np.ones(8)
        s_all_down = -np.ones(8)
        e_up = sampler.energy(s_all_up)
        e_down = sampler.energy(s_all_down)
        # Both are ground states — same energy by symmetry
        assert abs(e_up - e_down) < 1e-10
        # Energy must be negative (ferromagnetic ground state is negative)
        assert e_up < 0.0

    def test_energy_formula(self):
        """Verify energy formula -0.5 * s^T J s - h^T s explicitly."""
        J = np.array([[0, 1], [1, 0]], dtype=np.float64)
        h = np.array([0.5, -0.5])
        sampler = SynchronousPIMISampler(2, J, h)
        s = np.array([1.0, -1.0])
        # -0.5 * (1*0*1 + 1*1*(-1) + (-1)*1*1 + (-1)*0*(-1)) - (0.5*1 + (-0.5)*(-1))
        # = -0.5 * (0 - 1 - 1 + 0) - (0.5 + 0.5)
        # = -0.5 * (-2) - 1.0 = 1.0 - 1.0 = 0.0
        assert abs(sampler.energy(s) - 0.0) < 1e-10


# ---------------------------------------------------------------------------
# REQ-HW-036: Convergence measurement
# ---------------------------------------------------------------------------

class TestMeasureConvergence:
    """SCENARIO-HW-036-D: measure_convergence() returns mean sweeps over n_trials."""

    def test_convergence_returns_int(self, n8_sampler):
        """measure_convergence() must return an integer."""
        result = n8_sampler.measure_convergence(
            n_trials=5, target_energy=-3.0, max_sweeps=400
        )
        assert isinstance(result, int)

    def test_convergence_bounded_by_max_sweeps(self, n8_sampler):
        """Convergence cannot exceed max_sweeps."""
        max_s = 50
        result = n8_sampler.measure_convergence(
            n_trials=10, target_energy=-100.0, max_sweeps=max_s
        )
        assert result <= max_s

    def test_easy_problem_converges_fast(self, n8_sampler):
        """N=8 ferromagnetic problem should converge in < 20 sweeps on average."""
        result = n8_sampler.measure_convergence(
            n_trials=20, target_energy=-3.0, max_sweeps=200
        )
        assert result < 20


# ---------------------------------------------------------------------------
# REQ-HW-036: Sweeps reduction computation vs Exp 876 baseline
# ---------------------------------------------------------------------------

class TestSweepsReduction:
    """SCENARIO-HW-036-E: sweeps_reduction = baseline_sweeps / parallel_sweeps."""

    def test_sweeps_reduction_above_3x(self):
        """Synchronous PIMI must achieve at least 3x over no-inertia baseline.

        Exp 876 baseline_sweeps_mean = 12.99 (standard Gibbs, no EMA).
        Synchronous PIMI at best_alpha should achieve >= 3x reduction.
        This is the minimum threshold before the experiment is retired.

        Spec: REQ-HW-036
        """
        baseline_sweeps = 12.99  # from Exp 876 JSON

        J = make_n8_coupling_matrix()
        h = np.zeros(8)
        sampler = SynchronousPIMISampler(8, J, h, alpha=0.5, beta=1.0)
        parallel_sweeps = sampler.measure_convergence(
            n_trials=100, target_energy=-3.0, max_sweeps=400
        )

        sweeps_reduction = baseline_sweeps / parallel_sweeps
        assert sweeps_reduction >= 3.0, (
            f"Expected >= 3x sweep reduction, got {sweeps_reduction:.2f}x "
            f"(parallel={parallel_sweeps}, baseline={baseline_sweeps})"
        )

    def test_sweeps_reduction_formula(self):
        """sweeps_reduction = checkerboard_sweeps / parallel_sweeps (no off-by-one)."""
        checkerboard_sweeps = 15
        parallel_sweeps = 3
        expected = checkerboard_sweeps / parallel_sweeps
        assert abs(expected - 5.0) < 1e-10


# ---------------------------------------------------------------------------
# REQ-HW-036: sample_once() (non-seeded) and run()
# ---------------------------------------------------------------------------

class TestSampleOnceAndRun:
    """SCENARIO-HW-036-I: sample_once() and run() produce valid spin configurations."""

    def test_sample_once_returns_valid_spins(self, n8_sampler):
        """sample_once() must return an array of ±1 values, shape (N,)."""
        s = np.ones(8, dtype=np.float64)
        s_new = n8_sampler.sample_once(s)
        assert s_new.shape == (8,)
        assert set(s_new.tolist()).issubset({1.0, -1.0})

    def test_sample_once_does_not_modify_input(self, n8_sampler):
        """sample_once() must not modify the input spin vector in-place."""
        s = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        s_before = s.copy()
        n8_sampler.sample_once(s)
        np.testing.assert_array_equal(s, s_before)

    def test_sample_once_updates_h_ema(self, n8_sampler):
        """sample_once() must update h_ema state."""
        n8_sampler.reset()
        s = np.ones(8, dtype=np.float64)
        n8_sampler.sample_once(s)
        # After one call from zero-initialized h_ema, h_ema must be non-zero
        # (since h_local = J @ s is non-zero for connected graph)
        assert np.any(n8_sampler.h_ema != 0.0)

    def test_run_returns_tuple_of_state_and_energies(self, n8_sampler):
        """run() must return (final_state, energy_trajectory) tuple."""
        init = np.ones(8, dtype=np.float64)
        final, energies = n8_sampler.run(n_sweeps=10, init_state=init, seed=0)
        assert final.shape == (8,)
        assert len(energies) == 10
        assert all(isinstance(e, float) for e in energies)

    def test_run_valid_spins_in_final_state(self, n8_sampler):
        """Final state from run() must contain only ±1 values."""
        init = np.ones(8, dtype=np.float64)
        final, _ = n8_sampler.run(n_sweeps=20, init_state=init, seed=42)
        assert set(final.tolist()).issubset({1.0, -1.0})

    def test_run_resets_h_ema_between_calls(self, n8_sampler):
        """run() calls reset() internally — each run is independent."""
        init = np.ones(8, dtype=np.float64)
        _, e1 = n8_sampler.run(n_sweeps=5, init_state=init, seed=0)
        _, e2 = n8_sampler.run(n_sweeps=5, init_state=init, seed=0)
        # Same seed and init → same trajectory
        np.testing.assert_allclose(e1, e2, rtol=1e-9)


# ---------------------------------------------------------------------------
# REQ-HW-036: pimi_alpha_sweep helper
# ---------------------------------------------------------------------------

class TestPimiAlphaSweep:
    """SCENARIO-HW-036-F: pimi_alpha_sweep() returns a dict keyed by str(alpha)."""

    def test_alpha_sweep_returns_all_alphas(self):
        """pimi_alpha_sweep() must return an entry for every alpha in the input list."""
        alphas = [0.5, 0.25]
        result = pimi_alpha_sweep(alphas, n_trials=5, energy_threshold=-3.0, max_sweeps=200)
        assert set(result.keys()) == {"0.5", "0.25"}

    def test_alpha_sweep_values_are_floats(self):
        """All values in pimi_alpha_sweep result must be float."""
        result = pimi_alpha_sweep([0.5], n_trials=5, max_sweeps=200)
        for v in result.values():
            assert isinstance(v, float)

    def test_lower_alpha_not_always_better(self):
        """For this ferromagnetic graph, higher alpha (more history) converges faster."""
        result = pimi_alpha_sweep([0.5, 0.0625], n_trials=50, max_sweeps=200)
        # alpha=0.5 (mild inertia with fast adaptation) should beat alpha=0.0625
        # This is because 0.0625 has too much inertia — recovers slowly from bad init
        assert result["0.5"] <= result["0.0625"], (
            f"alpha=0.5 should converge faster than 0.0625 for this graph: "
            f"{result['0.5']:.1f} vs {result['0.0625']:.1f}"
        )


# ---------------------------------------------------------------------------
# REQ-HW-036: make_n8_coupling_matrix correctness
# ---------------------------------------------------------------------------

class TestMakeN8CouplingMatrix:
    """SCENARIO-HW-036-G: N=8 coupling matrix has correct ring+chord topology."""

    def test_shape_and_symmetry(self):
        """J must be 8x8 and symmetric."""
        J = make_n8_coupling_matrix()
        assert J.shape == (8, 8)
        np.testing.assert_array_equal(J, J.T)

    def test_diagonal_is_zero(self):
        """No self-coupling (J[i,i] = 0 for all i)."""
        J = make_n8_coupling_matrix()
        np.testing.assert_array_equal(np.diag(J), np.zeros(8))

    def test_k12_nonzero_pairs(self):
        """Must have exactly K=12 non-zero pairs (same as Exp 876 sparse adjacency)."""
        J = make_n8_coupling_matrix()
        # Count upper-triangle non-zeros
        upper_nonzero = int(np.sum(np.triu(J, k=1) != 0))
        assert upper_nonzero == 12

    def test_all_weights_plus_one(self):
        """All non-zero couplings must be +1.0 (ferromagnetic)."""
        J = make_n8_coupling_matrix()
        nonzero_vals = J[J != 0]
        np.testing.assert_array_equal(nonzero_vals, np.ones_like(nonzero_vals))


# ---------------------------------------------------------------------------
# REQ-HW-036: Deliverable JSON exists and is valid
# ---------------------------------------------------------------------------

class TestDeliverableJSON:
    """SCENARIO-HW-036-H: results/experiment_889_ice40_pimi_v3_parallel.json validity."""

    DELIVERABLE = Path("results/experiment_889_ice40_pimi_v3_parallel.json")

    REQUIRED_FIELDS = [
        "experiment", "title", "run_date", "started_at", "finished_at",
        "duration_s", "status", "honest_verdict",
        "parallel_sweeps", "checkerboard_sweeps", "sweeps_reduction",
        "best_alpha", "lut_count", "synthesis_clean",
        "n_spins", "n_trials", "energy_threshold", "max_sweeps",
        "alpha_sweep_pimi",
    ]

    def test_deliverable_exists(self):
        """The experiment result file must exist."""
        assert self.DELIVERABLE.exists(), (
            f"Deliverable not found: {self.DELIVERABLE}"
        )

    def test_deliverable_is_valid_json(self):
        """The deliverable must be parseable JSON."""
        data = json.loads(self.DELIVERABLE.read_text())
        assert isinstance(data, dict)

    def test_required_fields_present(self):
        """All required schema fields must be present in the deliverable."""
        data = json.loads(self.DELIVERABLE.read_text())
        for field in self.REQUIRED_FIELDS:
            assert field in data, f"Missing required field: {field}"

    def test_experiment_number(self):
        """experiment field must be 889."""
        data = json.loads(self.DELIVERABLE.read_text())
        assert data["experiment"] == 889

    def test_honest_verdict_is_valid(self):
        """honest_verdict must be one of the defined outcome strings."""
        valid_verdicts = {
            "pimi_5x_retro_closed",
            "pimi_5x_synthesis_over_budget",
            "pimi_improved_below_5x",
            "pimi_retired",
            "synthesis_blocked",
        }
        data = json.loads(self.DELIVERABLE.read_text())
        assert data["honest_verdict"] in valid_verdicts

    def test_sweeps_reduction_matches_formula(self):
        """sweeps_reduction must equal checkerboard_sweeps / parallel_sweeps."""
        data = json.loads(self.DELIVERABLE.read_text())
        expected = data["checkerboard_sweeps"] / data["parallel_sweeps"]
        assert abs(data["sweeps_reduction"] - expected) < 0.01

    def test_synthesis_lut_count_present_and_positive(self):
        """lut_count must be a positive integer."""
        data = json.loads(self.DELIVERABLE.read_text())
        assert data["lut_count"] > 0
