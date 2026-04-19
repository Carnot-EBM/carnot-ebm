"""Tests for carnot.hardware.fpga_backend: SparsifiedIsingConfig and FpgaBackend.

Spec coverage:
    REQ-HARDWARE-013, REQ-HARDWARE-014, REQ-HARDWARE-015,
    SCENARIO-HARDWARE-013, SCENARIO-HARDWARE-014, SCENARIO-HARDWARE-015
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.hardware.fpga_backend import FpgaBackend, SparsifiedIsingConfig

# ---------------------------------------------------------------------------
# SparsifiedIsingConfig tests
# ---------------------------------------------------------------------------


class TestSparsifiedIsingConfig:
    """REQ-HARDWARE-013: Sparsified 128-spin coupling matrix."""

    def test_coupling_matrix_shape(self) -> None:
        """SCENARIO-HARDWARE-013: coupling_matrix() returns (n_spins, n_spins) array."""
        cfg = SparsifiedIsingConfig(n_spins=16, sparsity=0.5, seed=0)
        J = cfg.coupling_matrix()
        assert J.shape == (16, 16)

    def test_coupling_matrix_dtype(self) -> None:
        """SCENARIO-HARDWARE-013: coupling_matrix() returns float32."""
        cfg = SparsifiedIsingConfig(n_spins=8, sparsity=0.5, seed=1)
        J = cfg.coupling_matrix()
        assert J.dtype == jnp.float32

    def test_coupling_matrix_zero_diagonal(self) -> None:
        """SCENARIO-HARDWARE-013: diagonal of coupling_matrix() is zero (no self-coupling)."""
        cfg = SparsifiedIsingConfig(n_spins=16, sparsity=0.5, seed=2)
        J = cfg.coupling_matrix()
        diag = np.diag(np.array(J))
        np.testing.assert_array_equal(diag, np.zeros(16))

    def test_coupling_matrix_symmetric(self) -> None:
        """SCENARIO-HARDWARE-013: coupling_matrix() is symmetric (J[i,j] == J[j,i])."""
        cfg = SparsifiedIsingConfig(n_spins=16, sparsity=0.5, seed=3)
        J = np.array(cfg.coupling_matrix())
        np.testing.assert_allclose(J, J.T, atol=1e-6)

    def test_sparsity_fraction_approx(self) -> None:
        """SCENARIO-HARDWARE-013: sparsity fraction of off-diagonal couplings are zero.

        At sparsity=0.9, approximately 90% of off-diagonal entries should be zero.
        We check that the zero fraction is between 0.80 and 0.99 (10% tolerance
        for the stochastic mask).
        """
        cfg = SparsifiedIsingConfig(n_spins=64, sparsity=0.9, seed=10)
        J = np.array(cfg.coupling_matrix())
        # Count off-diagonal zeros.
        mask = ~np.eye(64, dtype=bool)
        off_diag = J[mask]
        zero_frac = np.sum(off_diag == 0.0) / len(off_diag)
        assert 0.80 <= zero_frac <= 0.99, f"sparsity={zero_frac:.3f} out of expected [0.80, 0.99]"

    def test_sparsity_zero_gives_dense(self) -> None:
        """SCENARIO-HARDWARE-013: sparsity=0 yields a fully-connected (dense) matrix."""
        cfg = SparsifiedIsingConfig(n_spins=8, sparsity=0.0, seed=5)
        J = np.array(cfg.coupling_matrix())
        mask = ~np.eye(8, dtype=bool)
        off_diag = J[mask]
        # With sparsity=0, all off-diagonal entries should be non-zero.
        assert np.all(off_diag != 0.0)

    def test_n_edges_type(self) -> None:
        """REQ-HARDWARE-013: n_edges() returns an int."""
        cfg = SparsifiedIsingConfig(n_spins=16, sparsity=0.5, seed=7)
        assert isinstance(cfg.n_edges(), int)

    def test_n_edges_upper_bound(self) -> None:
        """REQ-HARDWARE-013: n_edges() <= n_spins*(n_spins-1)/2 (upper triangle only)."""
        n = 16
        cfg = SparsifiedIsingConfig(n_spins=n, sparsity=0.0, seed=8)
        assert cfg.n_edges() <= n * (n - 1) // 2

    def test_n_edges_sparse(self) -> None:
        """REQ-HARDWARE-013: n_edges() is less than dense count when sparsity > 0."""
        n = 32
        cfg_dense = SparsifiedIsingConfig(n_spins=n, sparsity=0.0, seed=9)
        cfg_sparse = SparsifiedIsingConfig(n_spins=n, sparsity=0.9, seed=9)
        assert cfg_sparse.n_edges() < cfg_dense.n_edges()

    def test_default_params(self) -> None:
        """REQ-HARDWARE-013: default SparsifiedIsingConfig has n_spins=128, sparsity=0.9."""
        cfg = SparsifiedIsingConfig()
        assert cfg.n_spins == 128
        assert cfg.sparsity == 0.9
        assert isinstance(cfg.seed, int)

    def test_deterministic_with_same_seed(self) -> None:
        """SCENARIO-HARDWARE-013: same seed produces identical coupling matrix."""
        cfg1 = SparsifiedIsingConfig(n_spins=16, sparsity=0.5, seed=42)
        cfg2 = SparsifiedIsingConfig(n_spins=16, sparsity=0.5, seed=42)
        np.testing.assert_array_equal(np.array(cfg1.coupling_matrix()), np.array(cfg2.coupling_matrix()))

    def test_different_seeds_differ(self) -> None:
        """SCENARIO-HARDWARE-013: different seeds produce different coupling matrices."""
        cfg1 = SparsifiedIsingConfig(n_spins=16, sparsity=0.5, seed=1)
        cfg2 = SparsifiedIsingConfig(n_spins=16, sparsity=0.5, seed=2)
        J1 = np.array(cfg1.coupling_matrix())
        J2 = np.array(cfg2.coupling_matrix())
        assert not np.allclose(J1, J2)


# ---------------------------------------------------------------------------
# FpgaBackend construction tests
# ---------------------------------------------------------------------------


class TestFpgaBackendConstruction:
    """REQ-HARDWARE-014: FpgaBackend defaults to simulation when no bitfile."""

    def test_no_bitfile_forces_simulation(self) -> None:
        """SCENARIO-HARDWARE-014: FpgaBackend(bitfile_path=None) sets simulation_mode=True."""
        backend = FpgaBackend(bitfile_path=None)
        assert backend.simulation_mode is True

    def test_explicit_simulation_mode_true(self) -> None:
        """SCENARIO-HARDWARE-014: FpgaBackend(simulation_mode=True) is in simulation mode."""
        backend = FpgaBackend(simulation_mode=True)
        assert backend.simulation_mode is True

    def test_env_var_absent_defaults_simulation(self) -> None:
        """SCENARIO-HARDWARE-014: when CARNOT_KV260_BITFILE is not set, simulation is default."""
        env_backup = os.environ.pop("CARNOT_KV260_BITFILE", None)
        try:
            backend = FpgaBackend()
            assert backend.simulation_mode is True
        finally:
            if env_backup is not None:
                os.environ["CARNOT_KV260_BITFILE"] = env_backup

    def test_bitfile_path_none_overrides_simulation_false(self) -> None:
        """SCENARIO-HARDWARE-014: even simulation_mode=False is overridden when bitfile_path=None."""
        backend = FpgaBackend(bitfile_path=None, simulation_mode=False)
        assert backend.simulation_mode is True


# ---------------------------------------------------------------------------
# FpgaBackend.sample() tests
# ---------------------------------------------------------------------------


class TestFpgaBackendSample:
    """REQ-HARDWARE-013/014: sample() returns correct shape and dtype."""

    def test_sample_returns_correct_shape(self) -> None:
        """SCENARIO-HARDWARE-014: FpgaBackend(simulation_mode=True).sample() shape is (n_samples, n_spins)."""
        backend = FpgaBackend(simulation_mode=True)
        cfg = SparsifiedIsingConfig(n_spins=8, sparsity=0.5, seed=0)
        samples = backend.sample(cfg, n_samples=10)
        assert samples.shape == (10, 8)

    def test_sample_returns_bool_dtype(self) -> None:
        """SCENARIO-HARDWARE-014: sample() returns boolean array."""
        backend = FpgaBackend(simulation_mode=True)
        cfg = SparsifiedIsingConfig(n_spins=8, sparsity=0.5, seed=1)
        samples = backend.sample(cfg, n_samples=5)
        assert samples.dtype == jnp.bool_

    def test_sample_n_samples_1(self) -> None:
        """SCENARIO-HARDWARE-014: sample() works for n_samples=1."""
        backend = FpgaBackend(simulation_mode=True)
        cfg = SparsifiedIsingConfig(n_spins=4, sparsity=0.5, seed=2)
        samples = backend.sample(cfg, n_samples=1)
        assert samples.shape == (1, 4)

    def test_sample_128_spins_default_config(self) -> None:
        """SCENARIO-HARDWARE-013: sample() works with default 128-spin config."""
        backend = FpgaBackend(simulation_mode=True)
        cfg = SparsifiedIsingConfig()
        samples = backend.sample(cfg, n_samples=3)
        assert samples.shape == (3, 128)


# ---------------------------------------------------------------------------
# FpgaBackend.update_couplings() tests
# ---------------------------------------------------------------------------


class TestUpdateCouplings:
    """REQ-HARDWARE-015: update_couplings() writes atomically (POSIX rename)."""

    def test_update_couplings_creates_file(self, tmp_path: Path) -> None:
        """SCENARIO-HARDWARE-015: update_couplings writes a .npy file to the cache path."""
        cache_path = tmp_path / "test_coupling.npy"
        old_env = os.environ.get("CARNOT_KV260_COUPLING_CACHE")
        os.environ["CARNOT_KV260_COUPLING_CACHE"] = str(cache_path)
        try:
            backend = FpgaBackend(simulation_mode=True)
            J = jnp.zeros((4, 4), dtype=jnp.float32)
            backend.update_couplings(J)
            assert cache_path.exists()
        finally:
            if old_env is None:
                os.environ.pop("CARNOT_KV260_COUPLING_CACHE", None)
            else:
                os.environ["CARNOT_KV260_COUPLING_CACHE"] = old_env

    def test_update_couplings_roundtrip(self, tmp_path: Path) -> None:
        """SCENARIO-HARDWARE-015: coupling matrix survives write-then-read."""
        cache_path = tmp_path / "coupling_rt.npy"
        old_env = os.environ.get("CARNOT_KV260_COUPLING_CACHE")
        os.environ["CARNOT_KV260_COUPLING_CACHE"] = str(cache_path)
        try:
            backend = FpgaBackend(simulation_mode=True)
            J_in = np.array([[0.0, 0.5, -0.3, 0.0],
                              [0.5, 0.0, 0.1,  0.0],
                              [-0.3, 0.1, 0.0, 0.2],
                              [0.0, 0.0, 0.2,  0.0]], dtype=np.float32)
            backend.update_couplings(jnp.array(J_in))
            J_out = np.load(str(cache_path))
            np.testing.assert_allclose(J_out, J_in, atol=1e-6)
        finally:
            if old_env is None:
                os.environ.pop("CARNOT_KV260_COUPLING_CACHE", None)
            else:
                os.environ["CARNOT_KV260_COUPLING_CACHE"] = old_env

    def test_update_couplings_uses_atomic_rename(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """SCENARIO-HARDWARE-015: update_couplings uses os.rename (atomic write contract)."""
        cache_path = tmp_path / "coupling_atomic.npy"
        old_env = os.environ.get("CARNOT_KV260_COUPLING_CACHE")
        os.environ["CARNOT_KV260_COUPLING_CACHE"] = str(cache_path)

        rename_calls: list[tuple[str, str]] = []
        original_rename = os.rename

        def mock_rename(src: str, dst: str) -> None:
            rename_calls.append((src, dst))
            original_rename(src, dst)

        monkeypatch.setattr(os, "rename", mock_rename)

        try:
            backend = FpgaBackend(simulation_mode=True)
            J = jnp.zeros((4, 4), dtype=jnp.float32)
            backend.update_couplings(J)
            # Must have called os.rename exactly once (atomic temp→target).
            assert len(rename_calls) == 1
            src_path, dst_path = rename_calls[0]
            # Destination is our cache path.
            assert dst_path == str(cache_path)
            # Source is a temp file in the same directory (same filesystem = atomic).
            assert Path(src_path).parent == cache_path.parent
        finally:
            if old_env is None:
                os.environ.pop("CARNOT_KV260_COUPLING_CACHE", None)
            else:
                os.environ["CARNOT_KV260_COUPLING_CACHE"] = old_env


# ---------------------------------------------------------------------------
# FpgaBackend.benchmark() tests
# ---------------------------------------------------------------------------


class TestFpgaBackendBenchmark:
    """REQ-HARDWARE-013: benchmark() returns ms per sample."""

    def test_benchmark_returns_float(self) -> None:
        """SCENARIO-HARDWARE-013: benchmark() returns a float (ms per sample)."""
        backend = FpgaBackend(simulation_mode=True)
        result = backend.benchmark(n_samples=5)
        assert isinstance(result, float)

    def test_benchmark_positive(self) -> None:
        """SCENARIO-HARDWARE-013: benchmark() returns a positive number."""
        backend = FpgaBackend(simulation_mode=True)
        result = backend.benchmark(n_samples=5)
        assert result > 0.0


# ---------------------------------------------------------------------------
# carnot.hardware.__init__ export test
# ---------------------------------------------------------------------------


class TestHardwareExports:
    """REQ-HARDWARE-013/014: FpgaBackend and SparsifiedIsingConfig are exported from carnot.hardware."""

    def test_imports_from_carnot_hardware(self) -> None:
        """SCENARIO-HARDWARE-013: FpgaBackend and SparsifiedIsingConfig importable from carnot.hardware."""
        from carnot.hardware import FpgaBackend as FB
        from carnot.hardware import SparsifiedIsingConfig as SIC
        assert FB is FpgaBackend
        assert SIC is SparsifiedIsingConfig
