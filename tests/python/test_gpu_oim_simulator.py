"""Tests for GPUOscillatorIsingSimulator, OIMSpeedupResult, JEPARetrainResult.

Spec coverage: REQ-SAMPLE-017, REQ-SAMPLE-018, REQ-LEARN-036,
               SCENARIO-SAMPLE-030, SCENARIO-SAMPLE-031, SCENARIO-LEARN-064
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from carnot.samplers.gpu_oim_simulator import (
    GPUOscillatorIsingSimulator,
    JEPARetrainResult,
    OIMSpeedupResult,
)


# ---------------------------------------------------------------------------
# OIMSpeedupResult tests
# ---------------------------------------------------------------------------


class TestOIMSpeedupResult:
    """REQ-SAMPLE-018: OIMSpeedupResult speedup and production-ready flag."""

    def test_speedup_calculation(self):
        """SCENARIO-SAMPLE-031: speedup = cpu_ms / gpu_ms."""
        r = OIMSpeedupResult(n_spins=128, gpu_ms=0.1, cpu_ms=10.0)
        assert abs(r.speedup - 100.0) < 1e-6

    def test_is_production_ready_true(self):
        """SCENARIO-SAMPLE-031: >=10x speedup is production-ready."""
        r = OIMSpeedupResult(n_spins=128, gpu_ms=0.1, cpu_ms=10.0)
        assert r.is_production_ready is True

    def test_is_production_ready_exactly_10x(self):
        """SCENARIO-SAMPLE-031: exactly 10x is production-ready."""
        r = OIMSpeedupResult(n_spins=128, gpu_ms=1.0, cpu_ms=10.0)
        assert r.is_production_ready is True

    def test_is_production_ready_false(self):
        """SCENARIO-SAMPLE-031: <10x speedup is not production-ready."""
        r = OIMSpeedupResult(n_spins=128, gpu_ms=5.0, cpu_ms=10.0)
        assert r.is_production_ready is False

    def test_speedup_zero_gpu_ms(self):
        """SCENARIO-SAMPLE-031: zero gpu_ms returns infinity."""
        r = OIMSpeedupResult(n_spins=128, gpu_ms=0.0, cpu_ms=10.0)
        assert r.speedup == float("inf")

    def test_attributes(self):
        """OIMSpeedupResult stores attributes correctly."""
        r = OIMSpeedupResult(n_spins=64, gpu_ms=2.5, cpu_ms=50.0)
        assert r.n_spins == 64
        assert r.gpu_ms == 2.5
        assert r.cpu_ms == 50.0


# ---------------------------------------------------------------------------
# JEPARetrainResult tests
# ---------------------------------------------------------------------------


class TestJEPARetrainResult:
    """REQ-LEARN-036: JEPARetrainResult AUC improvement and target flag."""

    def test_auc_improvement_positive(self):
        """SCENARIO-LEARN-064: improvement = after - before."""
        r = JEPARetrainResult(n_pairs=200, before_auc=0.571, after_auc=0.720)
        assert abs(r.auc_improvement - 0.149) < 1e-6

    def test_auc_improvement_negative(self):
        """SCENARIO-LEARN-064: negative improvement when AUC drops."""
        r = JEPARetrainResult(n_pairs=100, before_auc=0.700, after_auc=0.650)
        assert r.auc_improvement < 0.0

    def test_target_met_true(self):
        """SCENARIO-LEARN-064: target_met when after_auc > 0.700."""
        r = JEPARetrainResult(n_pairs=200, before_auc=0.571, after_auc=0.720)
        assert r.target_met is True

    def test_target_met_false(self):
        """SCENARIO-LEARN-064: target_met False when after_auc <= 0.700."""
        r = JEPARetrainResult(n_pairs=200, before_auc=0.571, after_auc=0.700)
        assert r.target_met is False

    def test_target_met_below(self):
        """SCENARIO-LEARN-064: target_met False when after_auc < 0.700."""
        r = JEPARetrainResult(n_pairs=50, before_auc=0.457, after_auc=0.650)
        assert r.target_met is False

    def test_attributes(self):
        """JEPARetrainResult stores attributes correctly."""
        r = JEPARetrainResult(n_pairs=257, before_auc=0.571, after_auc=0.714)
        assert r.n_pairs == 257
        assert r.before_auc == 0.571
        assert r.after_auc == 0.714


# ---------------------------------------------------------------------------
# GPUOscillatorIsingSimulator tests
# ---------------------------------------------------------------------------


class TestGPUOscillatorIsingSimulator:
    """REQ-SAMPLE-017: GPUOscillatorIsingSimulator produces valid spin samples."""

    def _make_J(self, n: int) -> jnp.ndarray:
        """Create a simple ferromagnetic coupling matrix with zero diagonal."""
        J = jnp.ones((n, n)) * 0.1
        J = J - jnp.diag(jnp.diag(J))
        return J

    def test_sample_output_shape(self):
        """SCENARIO-SAMPLE-030: sample() returns shape (n_samples, n_spins)."""
        sim = GPUOscillatorIsingSimulator(n_spins=8, n_steps=10, device="cpu")
        J = self._make_J(8)
        result = sim.sample(J, n_samples=5)
        assert result.shape == (5, 8)

    def test_sample_is_boolean(self):
        """SCENARIO-SAMPLE-030: sample() returns boolean values."""
        sim = GPUOscillatorIsingSimulator(n_spins=8, n_steps=10, device="cpu")
        J = self._make_J(8)
        result = sim.sample(J, n_samples=5)
        assert result.dtype == jnp.bool_

    def test_sample_larger_problem(self):
        """SCENARIO-SAMPLE-030: sample() works at n_spins=128."""
        sim = GPUOscillatorIsingSimulator(n_spins=128, n_steps=5, device="cpu")
        J = self._make_J(128)
        result = sim.sample(J, n_samples=4)
        assert result.shape == (4, 128)

    def test_sample_single_sample(self):
        """SCENARIO-SAMPLE-030: sample() works with n_samples=1."""
        sim = GPUOscillatorIsingSimulator(n_spins=4, n_steps=5, device="cpu")
        J = self._make_J(4)
        result = sim.sample(J, n_samples=1)
        assert result.shape == (1, 4)

    def test_benchmark_returns_positive_float(self):
        """SCENARIO-SAMPLE-031: benchmark() returns positive ms-per-sample."""
        sim = GPUOscillatorIsingSimulator(n_spins=8, n_steps=10, device="cpu")
        J = self._make_J(8)
        ms = sim.benchmark(J, n_samples=10)
        assert isinstance(ms, float)
        assert ms > 0.0

    def test_device_fallback_to_cpu(self):
        """GPUOscillatorIsingSimulator falls back to CPU if 'gpu' unavailable."""
        # 'nonexistent_device' should fall back to CPU without raising.
        sim = GPUOscillatorIsingSimulator(n_spins=4, n_steps=5, device="nonexistent_device")
        J = self._make_J(4)
        result = sim.sample(J, n_samples=2)
        assert result.shape == (2, 4)

    def test_default_n_steps(self):
        """GPUOscillatorIsingSimulator default n_steps is 1000."""
        sim = GPUOscillatorIsingSimulator(n_spins=4)
        assert sim.n_steps == 1000

    def test_default_device(self):
        """GPUOscillatorIsingSimulator default device is 'cpu'."""
        sim = GPUOscillatorIsingSimulator(n_spins=4)
        assert sim.device == "cpu"

    def test_no_nan_in_output(self):
        """SCENARIO-SAMPLE-030: sample() produces no NaN values."""
        sim = GPUOscillatorIsingSimulator(n_spins=16, n_steps=20, device="cpu")
        J = self._make_J(16)
        result = sim.sample(J, n_samples=10)
        # Boolean array cannot contain NaN; ensure no exception was raised
        assert result.shape == (10, 16)

    def test_antiferromagnetic_coupling(self):
        """SCENARIO-SAMPLE-030: works with negative (antiferromagnetic) couplings."""
        n = 8
        J = -jnp.ones((n, n)) * 0.1
        J = J - jnp.diag(jnp.diag(J))
        sim = GPUOscillatorIsingSimulator(n_spins=n, n_steps=10, device="cpu")
        result = sim.sample(J, n_samples=4)
        assert result.shape == (4, n)
