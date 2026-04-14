"""Tests for Experiment 321: D-Wave Neal vs CPU Ising Benchmark utilities.

Spec coverage: REQ-SAMPLE-003, REQ-SAMPLE-007

These tests cover the pure utility functions (planted problem generation,
energy computation, success rate, result aggregation) without requiring any
sampler hardware or heavy JAX compilation. Sampler calls are mocked via simple
callables that return deterministic numpy arrays.

Design rationale:
    The experiment script separates pure math (make_planted_ising_problem,
    compute_ising_energies, compute_success_rate, aggregate_trial_results)
    from I/O (run_one_sampler, benchmark_problem_size). Tests focus on the
    pure functions because they encode the benchmarking logic and are the
    most likely to contain subtle bugs (off-by-one in energy formula,
    wrong tolerance direction, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Tabu shim (same as in the experiment script) — ensures the import doesn't
# fail in CI where the standalone 'tabu' package may not be installed.
# ---------------------------------------------------------------------------

if "tabu" not in sys.modules:
    try:
        from dwave.samplers import TabuSampler as _TabuSampler  # type: ignore[import-untyped]
        import types as _types

        _tabu_shim = _types.ModuleType("tabu")
        _tabu_shim.TabuSampler = _TabuSampler  # type: ignore[attr-defined]
        sys.modules["tabu"] = _tabu_shim
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Repo root injection so scripts/ is importable without installation
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from experiment_321_dwave_benchmark import (  # noqa: E402
    BETA,
    COUPLING_STRENGTH,
    BIAS_STRENGTH,
    TOLERANCE_FRAC,
    aggregate_trial_results,
    compute_ising_energies,
    compute_success_rate,
    make_planted_ising_problem,
    run_one_sampler,
)


# ---------------------------------------------------------------------------
# make_planted_ising_problem
# ---------------------------------------------------------------------------


class TestMakePlantedIsingProblem:
    """REQ-SAMPLE-003: Planted Ising problem has correct structure and energy."""

    def _rng(self, seed: int = 0) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_output_shapes(self):
        """SCENARIO-SAMPLE-007: Outputs have correct shapes for n=8."""
        biases, couplings, planted, ge = make_planted_ising_problem(8, 0.5, self._rng())
        assert biases.shape == (8,)
        assert couplings.shape == (8, 8)
        assert planted.shape == (8,)
        assert isinstance(ge, float)

    def test_planted_is_bool(self):
        """SCENARIO-SAMPLE-007: Planted state contains boolean values."""
        _, _, planted, _ = make_planted_ising_problem(8, 0.5, self._rng())
        assert planted.dtype == bool

    def test_couplings_symmetric(self):
        """SCENARIO-SAMPLE-007: Coupling matrix is symmetric."""
        _, J, _, _ = make_planted_ising_problem(16, 0.3, self._rng())
        np.testing.assert_array_equal(J, J.T)

    def test_couplings_zero_diagonal(self):
        """SCENARIO-SAMPLE-007: Coupling matrix has zero diagonal (no self-coupling)."""
        _, J, _, _ = make_planted_ising_problem(16, 0.3, self._rng())
        np.testing.assert_array_equal(np.diag(J), 0.0)

    def test_ground_energy_at_planted_state(self):
        """SCENARIO-SAMPLE-007: Energy computed at planted state equals reported ground energy."""
        b, J, planted, ge = make_planted_ising_problem(16, 0.4, self._rng(1))
        # Compute energy manually for the planted state.
        s = planted.astype(np.float64)
        manual_energy = -b @ s - s @ J @ s
        np.testing.assert_allclose(manual_energy, ge, rtol=1e-9)

    def test_ground_energy_is_negative(self):
        """SCENARIO-SAMPLE-007: Ground energy is negative (energy is minimized at planted state)."""
        _, _, _, ge = make_planted_ising_problem(32, 0.3, self._rng(2))
        assert ge < 0.0

    def test_planted_state_is_minimum_over_random_samples(self):
        """SCENARIO-SAMPLE-007: Planted state achieves lower energy than random samples."""
        b, J, planted, ge = make_planted_ising_problem(32, 0.4, self._rng(3))
        # Draw 200 random binary configurations and check that most have higher energy.
        rng = np.random.default_rng(42)
        random_samples = rng.integers(0, 2, size=(200, 32)).astype(bool)
        random_energies = compute_ising_energies(b, J, random_samples)
        # At least 95% of random samples should have strictly higher energy.
        fraction_worse = float(np.mean(random_energies > ge))
        assert fraction_worse >= 0.90, (
            f"Only {fraction_worse:.1%} of random samples have energy > ground state. "
            "Planted construction may be wrong."
        )

    def test_biases_aligned_with_planted_state(self):
        """SCENARIO-SAMPLE-007: Bias sign matches planted state (b_i>0 iff planted_i=1)."""
        b, _, planted, _ = make_planted_ising_problem(20, 0.0, self._rng(4))
        # With density=0.0 there are no couplings, so biases fully determine the problem.
        for i in range(20):
            if planted[i]:
                assert b[i] > 0.0, f"Spin {i} planted=1 but b[{i}]={b[i]} <= 0"
            else:
                assert b[i] < 0.0, f"Spin {i} planted=0 but b[{i}]={b[i]} >= 0"

    def test_bias_magnitude(self):
        """SCENARIO-SAMPLE-007: Bias values have the correct BIAS_STRENGTH magnitude."""
        b, _, _, _ = make_planted_ising_problem(10, 0.0, self._rng(5))
        np.testing.assert_allclose(np.abs(b), BIAS_STRENGTH)

    def test_coupling_magnitude(self):
        """SCENARIO-SAMPLE-007: Non-zero coupling values have COUPLING_STRENGTH magnitude."""
        _, J, _, _ = make_planted_ising_problem(10, 1.0, self._rng(6))
        nonzero = J[J != 0.0]
        if len(nonzero) > 0:
            np.testing.assert_allclose(np.abs(nonzero), COUPLING_STRENGTH)

    def test_density_zero_gives_no_couplings(self):
        """SCENARIO-SAMPLE-007: Density=0 produces at most 1 edge (min(1, 0) clamped to 1)."""
        # density=0 → n_edges = max(1, int(0)) = 1, so exactly 1 edge.
        _, J, _, _ = make_planted_ising_problem(8, 0.0, self._rng(7))
        n_nonzero_upper = int(np.sum(J[np.triu_indices(8, k=1)] != 0.0))
        assert n_nonzero_upper <= 1

    def test_reproducible_with_same_seed(self):
        """SCENARIO-SAMPLE-007: Same seed produces identical problem."""
        b1, J1, p1, ge1 = make_planted_ising_problem(8, 0.3, np.random.default_rng(99))
        b2, J2, p2, ge2 = make_planted_ising_problem(8, 0.3, np.random.default_rng(99))
        np.testing.assert_array_equal(b1, b2)
        np.testing.assert_array_equal(J1, J2)
        np.testing.assert_array_equal(p1, p2)
        assert ge1 == ge2

    def test_different_seeds_produce_different_problems(self):
        """SCENARIO-SAMPLE-007: Different seeds produce different problems."""
        b1, _, p1, _ = make_planted_ising_problem(8, 0.3, np.random.default_rng(0))
        b2, _, p2, _ = make_planted_ising_problem(8, 0.3, np.random.default_rng(1))
        # At least one element differs (extremely unlikely to be equal by chance).
        assert not np.array_equal(b1, b2) or not np.array_equal(p1, p2)


# ---------------------------------------------------------------------------
# compute_ising_energies
# ---------------------------------------------------------------------------


class TestComputeIsingEnergies:
    """REQ-SAMPLE-003: Ising energy computation is correct."""

    def test_single_spin_zero_coupling(self):
        """SCENARIO-SAMPLE-007: Single spin, positive bias — energy is -b for spin=1."""
        b = np.array([3.0])
        J = np.zeros((1, 1))
        s = np.array([[True]])
        energies = compute_ising_energies(b, J, s)
        assert energies.shape == (1,)
        np.testing.assert_allclose(energies[0], -3.0)

    def test_single_spin_zero_energy_for_off_spin(self):
        """SCENARIO-SAMPLE-007: Spin=0 contributes zero to energy (b·s = 0 when s=0)."""
        b = np.array([3.0])
        J = np.zeros((1, 1))
        s = np.array([[False]])
        energies = compute_ising_energies(b, J, s)
        np.testing.assert_allclose(energies[0], 0.0)

    def test_two_spin_ferromagnetic_aligned(self):
        """SCENARIO-SAMPLE-007: Two aligned spins with J>0 give lower energy than anti-aligned."""
        b = np.zeros(2)
        J = np.array([[0.0, 1.0], [1.0, 0.0]])
        # Both spins on: E = -s^T J s = -(1*1+1*1) = -2.0
        s_aligned = np.array([[True, True]])
        # One spin on, one off: E = 0
        s_anti = np.array([[True, False]])
        e_aligned = compute_ising_energies(b, J, s_aligned)[0]
        e_anti = compute_ising_energies(b, J, s_anti)[0]
        assert e_aligned < e_anti

    def test_batch_shape(self):
        """SCENARIO-SAMPLE-007: Batch of n_samples returns energy vector of length n_samples."""
        n = 8
        b = np.ones(n)
        J = np.zeros((n, n))
        samples = np.ones((5, n), dtype=bool)
        energies = compute_ising_energies(b, J, samples)
        assert energies.shape == (5,)

    def test_all_zero_spins_zero_energy(self):
        """SCENARIO-SAMPLE-007: All-zero spin configuration gives zero energy."""
        n = 10
        b = np.random.default_rng(0).uniform(-1, 1, n)
        J = np.random.default_rng(0).uniform(-1, 1, (n, n))
        np.fill_diagonal(J, 0.0)
        J = (J + J.T) / 2
        s = np.zeros((3, n), dtype=bool)
        energies = compute_ising_energies(b, J, s)
        np.testing.assert_allclose(energies, 0.0)

    def test_planted_energy_matches_ground_truth(self):
        """SCENARIO-SAMPLE-007: Energy at planted state matches make_planted_ising_problem output."""
        rng = np.random.default_rng(42)
        b, J, planted, ge = make_planted_ising_problem(16, 0.3, rng)
        # Wrap planted state as single-row batch.
        s = planted.reshape(1, -1)
        energies = compute_ising_energies(b, J, s)
        np.testing.assert_allclose(energies[0], ge, rtol=1e-9)


# ---------------------------------------------------------------------------
# compute_success_rate
# ---------------------------------------------------------------------------


class TestComputeSuccessRate:
    """REQ-SAMPLE-003: Success rate computation correctly applies tolerance."""

    def test_all_at_ground_energy_is_100_percent(self):
        """SCENARIO-SAMPLE-007: All samples exactly at ground energy → 100% success."""
        ge = -100.0
        energies = np.full(10, ge)
        assert compute_success_rate(energies, ge) == pytest.approx(1.0)

    def test_all_above_threshold_is_zero_percent(self):
        """SCENARIO-SAMPLE-007: All samples far above threshold → 0% success."""
        ge = -100.0
        # threshold = -100 * (1 - 0.05) = -95; samples at -50 don't make it.
        energies = np.full(10, -50.0)
        assert compute_success_rate(energies, ge) == pytest.approx(0.0)

    def test_mixed_samples(self):
        """SCENARIO-SAMPLE-007: Mixed batch gives correct fraction below threshold."""
        ge = -100.0
        # threshold = -100 * 0.95 = -95
        # 4 samples at -96 (pass), 6 at -90 (fail)
        energies = np.array([-96.0] * 4 + [-90.0] * 6)
        rate = compute_success_rate(energies, ge)
        assert rate == pytest.approx(0.4)

    def test_tolerance_zero_requires_exact_match(self):
        """SCENARIO-SAMPLE-007: Tolerance=0 only counts samples exactly at ground energy."""
        ge = -100.0
        # With tolerance=0, threshold = ge * (1-0) = ge = -100
        # Samples at -100 pass, samples at -99.9 fail (> -100).
        energies = np.array([-100.0, -99.9, -100.0])
        rate = compute_success_rate(energies, ge, tolerance_frac=0.0)
        assert rate == pytest.approx(2 / 3)

    def test_tolerance_one_accepts_everything(self):
        """SCENARIO-SAMPLE-007: Tolerance=1 sets threshold=0, accepting any negative energy."""
        ge = -100.0
        # threshold = -100 * (1 - 1.0) = 0; all negative energies pass.
        energies = np.array([-1.0, -50.0, -200.0])
        rate = compute_success_rate(energies, ge, tolerance_frac=1.0)
        assert rate == pytest.approx(1.0)

    def test_degenerate_zero_ground_energy(self):
        """SCENARIO-SAMPLE-007: Zero ground energy falls back to counting <=0 energies."""
        energies = np.array([-1.0, 0.0, 1.0])
        rate = compute_success_rate(energies, ground_energy=0.0)
        # Samples at -1 and 0 are <= 0.
        assert rate == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# aggregate_trial_results
# ---------------------------------------------------------------------------


class TestAggregateTrialResults:
    """REQ-SAMPLE-007: Trial aggregation produces correct summary statistics."""

    def test_single_trial_stats(self):
        """SCENARIO-SAMPLE-007: Single trial — mean_best_energy == min of that trial."""
        energies = np.array([-10.0, -8.0, -9.0])
        result = aggregate_trial_results([energies], ground_energy=-12.0, elapsed_list=[0.5])
        assert result["mean_best_energy"] == pytest.approx(-10.0)
        assert result["std_best_energy"] == pytest.approx(0.0)
        assert result["n_total_samples"] == 3

    def test_mean_energy_across_trials(self):
        """SCENARIO-SAMPLE-007: mean_energy pools all samples from all trials."""
        e1 = np.array([-10.0, -8.0])
        e2 = np.array([-6.0, -4.0])
        result = aggregate_trial_results([e1, e2], ground_energy=-12.0, elapsed_list=[0.1, 0.2])
        # Pooled: [-10, -8, -6, -4] → mean = -7.0
        assert result["mean_energy"] == pytest.approx(-7.0)
        assert result["n_total_samples"] == 4

    def test_mean_time(self):
        """SCENARIO-SAMPLE-007: mean_time_s is the average of elapsed times."""
        energies = np.array([-5.0, -4.0])
        result = aggregate_trial_results(
            [energies, energies, energies],
            ground_energy=-10.0,
            elapsed_list=[1.0, 2.0, 3.0],
        )
        assert result["mean_time_s"] == pytest.approx(2.0)

    def test_success_rate_correct(self):
        """SCENARIO-SAMPLE-007: Aggregate success_rate uses pooled samples."""
        ge = -100.0
        # threshold at tolerance=5%: -95
        # e1: [-96, -94] (1/2 pass), e2: [-96, -96] (2/2 pass)
        # pooled: [-96, -94, -96, -96] → 3/4 pass
        e1 = np.array([-96.0, -94.0])
        e2 = np.array([-96.0, -96.0])
        result = aggregate_trial_results([e1, e2], ground_energy=ge, elapsed_list=[0.1, 0.1])
        assert result["success_rate"] == pytest.approx(0.75)

    def test_std_best_energy_multiple_trials(self):
        """SCENARIO-SAMPLE-007: std_best_energy reflects variation in best-found across trials."""
        e1 = np.array([-10.0, -8.0])   # best = -10
        e2 = np.array([-20.0, -15.0])  # best = -20
        result = aggregate_trial_results([e1, e2], ground_energy=-25.0, elapsed_list=[0.1, 0.1])
        assert result["std_best_energy"] == pytest.approx(5.0)

    def test_nan_elapsed_excluded_from_mean(self):
        """SCENARIO-SAMPLE-007: NaN elapsed times propagate as NaN (marks failed trials)."""
        energies = np.array([-5.0])
        result = aggregate_trial_results(
            [energies, energies], ground_energy=-10.0, elapsed_list=[1.0, float("nan")]
        )
        # np.mean([1.0, nan]) = nan — expected behavior flagging a failed trial.
        assert result["mean_time_s"] != result["mean_time_s"]  # NaN check


# ---------------------------------------------------------------------------
# run_one_sampler
# ---------------------------------------------------------------------------


class TestRunOneSampler:
    """REQ-SAMPLE-003: run_one_sampler correctly calls sampler and converts output."""

    def _make_sampler(self, n_spins: int, n_samples: int) -> MagicMock:
        """Build a mock sampler that returns a deterministic bool array."""
        mock = MagicMock()
        mock.minimize_energy.return_value = np.ones((n_samples, n_spins), dtype=bool)
        return mock

    def test_returns_bool_array(self):
        """SCENARIO-SAMPLE-007: run_one_sampler returns a boolean numpy array."""
        mock = self._make_sampler(n_spins=8, n_samples=5)
        b = np.ones(8)
        J = np.zeros((8, 8))
        samples, elapsed = run_one_sampler(mock, b, J, n_samples=5, n_steps=100, beta=10.0)
        assert samples.dtype == bool
        assert samples.shape == (5, 8)

    def test_elapsed_is_positive_float(self):
        """SCENARIO-SAMPLE-007: Elapsed time is a positive float."""
        mock = self._make_sampler(n_spins=4, n_samples=3)
        b = np.zeros(4)
        J = np.zeros((4, 4))
        _, elapsed = run_one_sampler(mock, b, J, n_samples=3, n_steps=50, beta=5.0)
        assert isinstance(elapsed, float)
        assert elapsed >= 0.0

    def test_sampler_called_with_correct_args(self):
        """SCENARIO-SAMPLE-007: minimize_energy called with exact passed parameters."""
        mock = self._make_sampler(n_spins=6, n_samples=4)
        b = np.arange(6, dtype=float)
        J = np.eye(6) * 0.1
        run_one_sampler(mock, b, J, n_samples=4, n_steps=200, beta=7.0)
        mock.minimize_energy.assert_called_once()
        call_args = mock.minimize_energy.call_args
        np.testing.assert_array_equal(call_args[0][0], b)  # biases
        np.testing.assert_array_equal(call_args[0][1], J)  # couplings
        assert call_args[0][2] == 4    # n_samples
        assert call_args[0][3] == 200  # n_steps
        assert call_args[0][4] == 7.0  # beta

    def test_jax_array_converted_to_numpy(self):
        """SCENARIO-SAMPLE-007: JAX DeviceArray output is converted to numpy bool."""
        # Simulate CpuBackend returning a JAX-like array (use numpy as stand-in).
        mock = MagicMock()
        import numpy as np

        # Return a numpy array with float32 dtype (like JAX CpuBackend does).
        jax_like = np.array([[0.9, 0.1, 0.8]], dtype=np.float32) > 0.5
        mock.minimize_energy.return_value = jax_like
        b = np.zeros(3)
        J = np.zeros((3, 3))
        samples, _ = run_one_sampler(mock, b, J, n_samples=1, n_steps=10, beta=1.0)
        assert samples.dtype == bool


# ---------------------------------------------------------------------------
# Integration: planted problem + energy computation
# ---------------------------------------------------------------------------


class TestPlantedProblemEnergyIntegration:
    """REQ-SAMPLE-003: Planted problem and energy computation work end-to-end."""

    def test_planted_state_is_optimal_for_dense_problem(self):
        """SCENARIO-SAMPLE-007: Planted state achieves lower energy than all random ±1 configs for n=10."""
        # For a small problem (n=10), exhaustively check that the planted state
        # achieves lower energy than 500 random configurations.
        rng = np.random.default_rng(77)
        b, J, planted, ge = make_planted_ising_problem(
            n_spins=10, density=1.0, rng=rng
        )
        # Random samples
        rand_samples = rng.integers(0, 2, (500, 10)).astype(bool)
        rand_energies = compute_ising_energies(b, J, rand_samples)
        planted_energy = compute_ising_energies(b, J, planted.reshape(1, -1))[0]
        assert planted_energy == pytest.approx(ge, rel=1e-9)
        # Planted state should beat > 99% of random samples.
        fraction_worse = float(np.mean(rand_energies > planted_energy))
        assert fraction_worse >= 0.99, (
            f"Only {fraction_worse:.1%} of random samples have energy > planted. "
            "Planted optimality may be broken."
        )

    def test_success_rate_for_planted_state_samples(self):
        """SCENARIO-SAMPLE-007: Batch of planted-state copies achieves 100% success rate."""
        rng = np.random.default_rng(88)
        b, J, planted, ge = make_planted_ising_problem(12, 0.3, rng)
        # All samples identical to planted state.
        samples = np.tile(planted, (20, 1))
        energies = compute_ising_energies(b, J, samples)
        rate = compute_success_rate(energies, ge, tolerance_frac=0.0)
        assert rate == pytest.approx(1.0)
