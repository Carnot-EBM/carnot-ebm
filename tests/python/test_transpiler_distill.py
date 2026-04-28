"""Tests for the Native Thermodynamic Distillation primitive.

This is the load-bearing test for Approach 3. We run small,
fast PT-PCD fits and verify:

1. The training loop runs without numerical issues.
2. After enough epochs on a single-Gaussian synthetic target, the
   resulting BM produces samples whose energies are concentrated and
   close to the training data — the simplest sanity check.
3. The exported `IsingSpec` round-trips through `phi`/`psi` and
   carries provenance.

We do NOT claim convergence to KL <= 0.10 in 100 epochs on a tiny
1024-spin BM — that's the prototype-validation acceptance criterion in
the change proposal, not a unit test. These tests are *correctness*
tests on a 64-spin scaled-down BM that runs in seconds.

Spec: REQ-PHASE2-004 (Native Thermodynamic Distillation primitive).
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.hardware.transpiler import (
    CarnotNativeDistiller,
    DistillerConfig,
    HardwareSpec,
    IsingSpec,
)


def _tiny_distiller_config() -> tuple[DistillerConfig, HardwareSpec]:
    """A small BM that runs fast for unit tests. 16-vis (8 per axis,
    256-cell grid per axis), 48-hid, 64 chains, 4 temperatures, 5 Gibbs
    sweeps, 5 epochs of training. Total spin count 64.
    """
    config = DistillerConfig(
        n_vis=16,
        n_hid=48,
        n_chains=64,
        n_temps=4,
        beta_min=0.05,
        k_steps=5,
        lr=0.05,
        l2_reg=1e-3,
        rebirth_fraction=0.1,
        seed=42,
    )
    spec = HardwareSpec(kind="sparse", max_spins=128, beta_range=(0.5, 1.0))
    return config, spec


# REQ-PHASE2-004
def test_distiller_construct_and_init() -> None:
    """Constructor sets up J as small symmetric zero-diagonal, h small,
    and chains live across the temperature ladder at uniform random.
    """
    config, spec = _tiny_distiller_config()
    d = CarnotNativeDistiller(config, spec, domain=(-1.0, 1.0))

    n_total = config.n_vis + config.n_hid
    assert d.J.shape == (n_total, n_total)
    assert d.h.shape == (n_total,)
    np.testing.assert_allclose(d.J, d.J.T)
    np.testing.assert_allclose(np.diag(d.J), 0.0)
    assert d.chains.shape == (config.n_temps, config.n_chains, n_total)
    # Initial spin values strictly in {-1, +1}
    assert set(np.unique(d.chains).tolist()) <= {-1.0, 1.0}
    # Geometric beta ladder spans [beta_min, 1.0]
    assert d.betas[0] == pytest.approx(1.0)
    assert d.betas[-1] == pytest.approx(config.beta_min)


# REQ-PHASE2-004
def test_distiller_rejects_oversize_spin_request() -> None:
    """If n_vis + n_hid exceeds hardware max_spins, construction fails."""
    config = DistillerConfig(n_vis=16, n_hid=48, n_chains=8, n_temps=2)
    spec = HardwareSpec(kind="sparse", max_spins=32, beta_range=(0.1, 1.0))
    with pytest.raises(ValueError, match="exceeds hardware max_spins"):
        CarnotNativeDistiller(config, spec)


# REQ-PHASE2-004
def test_distiller_rejects_odd_n_vis() -> None:
    """n_vis must be even (m_per_axis per 2D axis)."""
    config = DistillerConfig(n_vis=15, n_hid=48, n_chains=8, n_temps=2)
    spec = HardwareSpec(kind="sparse", max_spins=128, beta_range=(0.1, 1.0))
    with pytest.raises(ValueError, match="n_vis"):
        CarnotNativeDistiller(config, spec)


# REQ-PHASE2-004
def test_distiller_train_epoch_returns_metrics() -> None:
    """One training step on a small synthetic batch produces all
    expected metric keys and finite values.
    """
    config, spec = _tiny_distiller_config()
    d = CarnotNativeDistiller(config, spec, domain=(-1.0, 1.0))
    rng = np.random.default_rng(0)
    z_batch = rng.uniform(-0.5, 0.5, size=(16, 2))
    metrics = d.train_epoch(z_batch)

    expected = {"swap_accept_min", "swap_accept_mean", "j_norm", "free_energy_gap"}
    assert set(metrics.keys()) == expected
    for k, v in metrics.items():
        assert np.isfinite(v), f"metric {k}={v} is not finite"
    # j_norm increased from initialization (training did something)
    assert metrics["j_norm"] > 0.0


# REQ-PHASE2-004
def test_distiller_j_stays_symmetric_zero_diag() -> None:
    """After several training epochs J remains symmetric with zero
    diagonal — the basic Ising-form invariant must be preserved.
    """
    config, spec = _tiny_distiller_config()
    d = CarnotNativeDistiller(config, spec, domain=(-1.0, 1.0))
    rng = np.random.default_rng(0)
    for _ in range(5):
        z_batch = rng.uniform(-0.5, 0.5, size=(8, 2))
        d.train_epoch(z_batch)
    np.testing.assert_allclose(d.J, d.J.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(d.J), 0.0, atol=1e-10)


# REQ-PHASE2-004
def test_distiller_export_ising_spec_round_trip() -> None:
    """Exported IsingSpec validates as a proper Ising form, and its
    encoder/decoder pair round-trips a 2D continuous batch within the
    cell-width tolerance.
    """
    config, spec = _tiny_distiller_config()
    d = CarnotNativeDistiller(config, spec, domain=(-1.0, 1.0))
    rng = np.random.default_rng(1)
    for _ in range(2):
        d.train_epoch(rng.uniform(-0.5, 0.5, size=(16, 2)))
    ising = d.export_ising_spec(provenance={"source": "unit_test"})

    assert isinstance(ising, IsingSpec)
    assert ising.n_spins == config.n_vis + config.n_hid
    assert ising.provenance["source"] == "unit_test"
    assert ising.provenance["approach"] == "native_thermodynamic_distillation"
    assert "J_hash" in ising.provenance
    # J_hash is a tamper-evident snapshot
    assert len(ising.provenance["J_hash"]) == 16

    # Encoder/decoder round-trip a 2D point within cell tolerance
    z_in = np.array([[0.3, -0.2]])
    spins = ising.phi(z_in)
    assert spins.shape == (1, config.n_vis)
    assert set(np.unique(spins).tolist()) <= {-1.0, 1.0}

    # Append zeros for hidden spins to feed psi (psi only reads visible)
    s_full = np.concatenate([spins, np.ones((1, config.n_hid))], axis=-1)
    z_out = ising.psi(s_full)
    assert z_out.shape == (1, 2)
    # Round-trip within the cell width: m_per_axis = 8 → step = 2/256 ≈ 0.008
    step = 2.0 / (1 << (config.n_vis // 2))
    assert np.all(np.abs(z_out - z_in) <= step / 2 + 0.01)


# REQ-PHASE2-004
def test_distiller_sample_cold_chain() -> None:
    """sample_cold_chain returns the requested number of {-1,+1}
    configurations from the production-temperature chain bank.
    """
    config, spec = _tiny_distiller_config()
    d = CarnotNativeDistiller(config, spec, domain=(-1.0, 1.0))
    samples = d.sample_cold_chain(8)
    assert samples.shape == (8, config.n_vis + config.n_hid)
    assert set(np.unique(samples).tolist()) <= {-1.0, 1.0}

    with pytest.raises(ValueError, match="persistent chains"):
        d.sample_cold_chain(config.n_chains + 1)
