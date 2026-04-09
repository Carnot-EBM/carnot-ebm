"""Tests for sampler backend abstraction layer.

Spec coverage: REQ-SAMPLE-003
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from carnot.samplers.backend import (
    CpuBackend,
    SamplerBackend,
    TsuBackend,
    get_backend,
)


# --- Protocol conformance ---


class TestProtocolConformance:
    """REQ-SAMPLE-003: Both backends satisfy the SamplerBackend protocol."""

    def test_cpu_is_sampler_backend(self):
        """SCENARIO-SAMPLE-006: CpuBackend conforms to SamplerBackend."""
        assert isinstance(CpuBackend(), SamplerBackend)

    def test_tsu_is_sampler_backend(self):
        """SCENARIO-SAMPLE-006: TsuBackend conforms to SamplerBackend."""
        assert isinstance(TsuBackend(), SamplerBackend)


# --- CpuBackend ---


class TestCpuBackend:
    """REQ-SAMPLE-003: CpuBackend wraps ParallelIsingSampler correctly."""

    def _ferromagnetic_problem(self, n: int = 10):
        """Create a simple ferromagnetic Ising problem.

        All biases positive, all couplings positive — the ground state is
        all-ones. A correct sampler at low temperature should return mostly
        ones.
        """
        biases = np.ones(n, dtype=np.float32) * 2.0
        couplings = np.ones((n, n), dtype=np.float32) * 0.5
        np.fill_diagonal(couplings, 0.0)
        return biases, couplings

    def test_backend_name(self):
        """SCENARIO-SAMPLE-006: CpuBackend reports correct name."""
        assert CpuBackend().backend_name == "cpu"

    def test_minimize_energy_shape(self):
        """SCENARIO-SAMPLE-006: minimize_energy returns correct shape."""
        b, J = self._ferromagnetic_problem(10)
        backend = CpuBackend(seed=0)
        samples = backend.minimize_energy(b, J, n_samples=5, n_steps=100, beta=10.0)
        assert samples.shape == (5, 10)
        assert samples.dtype == bool

    def test_minimize_energy_finds_ground_state(self):
        """SCENARIO-SAMPLE-007: Annealing on ferromagnet finds mostly-ones."""
        b, J = self._ferromagnetic_problem(8)
        backend = CpuBackend(seed=123)
        samples = backend.minimize_energy(b, J, n_samples=10, n_steps=500, beta=15.0)
        # At high beta with strong ferromagnetic coupling, most spins should be 1.
        mean_magnetization = samples.mean()
        assert mean_magnetization > 0.7, f"Expected mostly-ones, got mean={mean_magnetization}"

    def test_sample_shape(self):
        """SCENARIO-SAMPLE-006: sample returns correct shape."""
        b, J = self._ferromagnetic_problem(10)
        backend = CpuBackend(seed=0)
        samples = backend.sample(b, J, n_samples=5, config={"beta": 5.0})
        assert samples.shape == (5, 10)
        assert samples.dtype == bool

    def test_sample_fixed_temperature(self):
        """SCENARIO-SAMPLE-007: sample at high beta on ferromagnet is biased."""
        b, J = self._ferromagnetic_problem(8)
        backend = CpuBackend(seed=42)
        samples = backend.sample(
            b, J, n_samples=20, config={"beta": 15.0, "n_warmup": 500}
        )
        mean_magnetization = samples.mean()
        assert mean_magnetization > 0.7, f"Expected high magnetization, got {mean_magnetization}"


# --- TsuBackend ---


class TestTsuBackend:
    """REQ-SAMPLE-003: TsuBackend stub returns correct shapes and logs calls."""

    def test_backend_name(self):
        """SCENARIO-SAMPLE-006: TsuBackend reports correct name."""
        assert TsuBackend().backend_name == "tsu"

    def test_minimize_energy_shape(self):
        """SCENARIO-SAMPLE-008: Stub returns correct shape."""
        b = np.zeros(12, dtype=np.float32)
        J = np.zeros((12, 12), dtype=np.float32)
        backend = TsuBackend(seed=0)
        samples = backend.minimize_energy(b, J, n_samples=7, n_steps=100, beta=5.0)
        assert samples.shape == (7, 12)
        assert samples.dtype == bool

    def test_sample_shape(self):
        """SCENARIO-SAMPLE-008: Stub sample returns correct shape."""
        b = np.zeros(15, dtype=np.float32)
        J = np.zeros((15, 15), dtype=np.float32)
        backend = TsuBackend(seed=0)
        samples = backend.sample(b, J, n_samples=3, config={"beta": 1.0})
        assert samples.shape == (3, 15)
        assert samples.dtype == bool

    def test_call_logging(self):
        """SCENARIO-SAMPLE-008: Stub logs all calls for test inspection."""
        b = np.zeros(5, dtype=np.float32)
        J = np.zeros((5, 5), dtype=np.float32)
        backend = TsuBackend()

        backend.minimize_energy(b, J, n_samples=2, n_steps=50, beta=3.0)
        backend.sample(b, J, n_samples=4, config={"beta": 2.0})

        assert len(backend.call_log) == 2
        assert backend.call_log[0]["method"] == "minimize_energy"
        assert backend.call_log[0]["n_samples"] == 2
        assert backend.call_log[0]["beta"] == 3.0
        assert backend.call_log[1]["method"] == "sample"
        assert backend.call_log[1]["n_samples"] == 4

    def test_reproducible_with_seed(self):
        """SCENARIO-SAMPLE-008: Same seed produces same stub output."""
        b = np.zeros(10, dtype=np.float32)
        J = np.zeros((10, 10), dtype=np.float32)
        s1 = TsuBackend(seed=99).minimize_energy(b, J, 5, 10, 1.0)
        s2 = TsuBackend(seed=99).minimize_energy(b, J, 5, 10, 1.0)
        np.testing.assert_array_equal(s1, s2)


# --- Factory function ---


class TestGetBackend:
    """REQ-SAMPLE-003: Factory function resolves backends correctly."""

    def test_default_is_cpu(self):
        """SCENARIO-SAMPLE-006: Default backend is cpu."""
        # Clear env var to test true default.
        env_backup = os.environ.pop("CARNOT_BACKEND", None)
        try:
            backend = get_backend()
            assert backend.backend_name == "cpu"
            assert isinstance(backend, CpuBackend)
        finally:
            if env_backup is not None:
                os.environ["CARNOT_BACKEND"] = env_backup

    def test_explicit_cpu(self):
        """SCENARIO-SAMPLE-006: Explicit 'cpu' returns CpuBackend."""
        assert isinstance(get_backend("cpu"), CpuBackend)

    def test_explicit_tsu(self):
        """SCENARIO-SAMPLE-006: Explicit 'tsu' returns TsuBackend."""
        assert isinstance(get_backend("tsu"), TsuBackend)

    def test_env_var_override(self, monkeypatch):
        """SCENARIO-SAMPLE-008: CARNOT_BACKEND env var selects backend."""
        monkeypatch.setenv("CARNOT_BACKEND", "tsu")
        backend = get_backend()
        assert backend.backend_name == "tsu"
        assert isinstance(backend, TsuBackend)

    def test_unknown_backend_raises(self):
        """SCENARIO-SAMPLE-006: Unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sampler backend"):
            get_backend("quantum")
