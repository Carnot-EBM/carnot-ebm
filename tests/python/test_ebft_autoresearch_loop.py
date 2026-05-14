"""Tests for the EBFT Autoresearch Loop.

Spec: REQ-LOOP-001, SCENARIO-LOOP-001
"""

import jax.numpy as jnp
import pytest

from carnot.pipeline.ebft_autoresearch_loop import EBFTAutoResearchLoop


def _make_loop(**kwargs) -> EBFTAutoResearchLoop:
    """Create a fast loop for tests (few steps, small batches)."""
    defaults = dict(
        model_spec="surrogate",
        n_train_steps=5,
        batch_size=4,
        lr=0.01,
        seed=42,
    )
    defaults.update(kwargs)
    return EBFTAutoResearchLoop(**defaults)


def test_init_stores_params():
    """REQ-LOOP-001-1: Constructor stores all hyperparameters."""
    loop = _make_loop(n_train_steps=3, lr=0.05)
    assert loop.model_spec == "surrogate"
    assert loop.n_train_steps == 3
    assert loop.batch_size == 4
    assert loop.lr == 0.05
    assert loop.seed == 42


def test_build_dataset_shapes():
    """REQ-LOOP-001-2: build_dataset returns (train, val) arrays of correct shape."""
    loop = _make_loop()
    train_seqs, val_seqs = loop.build_dataset()
    # Shapes: (N, feature_dim) where N >= batch_size
    assert train_seqs.ndim == 2
    assert val_seqs.ndim == 2
    assert train_seqs.shape[1] == val_seqs.shape[1]
    assert train_seqs.shape[0] >= loop.batch_size
    assert val_seqs.shape[0] >= loop.batch_size


def test_build_dataset_deterministic():
    """REQ-LOOP-001-2: Same seed -> identical dataset on repeated calls."""
    loop = _make_loop(seed=7)
    train_a, val_a = loop.build_dataset()
    train_b, val_b = loop.build_dataset()
    assert jnp.allclose(train_a, train_b)
    assert jnp.allclose(val_a, val_b)


def test_measure_energy_returns_scalar():
    """REQ-LOOP-001-3: measure_energy returns a non-negative Python float scalar."""
    loop = _make_loop()
    _, val_seqs = loop.build_dataset()
    params = loop._init_params()
    energy = loop.measure_energy(params, val_seqs)
    assert isinstance(energy, float)
    # E(x; W) = ||Wx||²/2 is always non-negative
    assert energy >= 0.0


def test_run_returns_required_fields():
    """REQ-LOOP-001-4: run() dict has all required artifact fields."""
    loop = _make_loop()
    result = loop.run()
    for key in ("baseline_energy", "final_energy", "energy_delta", "acceptance_gate_passed"):
        assert key in result, f"Missing key: {key}"


def test_run_energy_delta_sign():
    """REQ-LOOP-001-4: energy_delta == baseline_energy - final_energy."""
    loop = _make_loop()
    result = loop.run()
    expected_delta = result["baseline_energy"] - result["final_energy"]
    assert abs(result["energy_delta"] - expected_delta) < 1e-6


def test_run_acceptance_gate_consistent():
    """REQ-LOOP-001-4: acceptance_gate_passed is True iff energy_delta > 0."""
    loop = _make_loop()
    result = loop.run()
    assert result["acceptance_gate_passed"] == (result["energy_delta"] > 0)


def test_run_acceptance_gate_passes_after_training():
    """SCENARIO-LOOP-001: Training on expert trajectories decreases energy.

    With SGD lr=0.2 and experts at mean=2.0 (far from origin), the EBFT
    gradient ∂L/∂W ∝ W * C_expert pushes W toward 0.  Each step multiplies
    W by (1 - lr*λ) where λ ≈ 4 for the dominant eigenvalue of C_expert,
    giving decay factor 0.2 per step.  After 20 steps, W ≈ W_0 * 0.2^20 ≈ 0
    and validation energy (of expert-like sequences) drops to near zero.
    """
    # Stability requires lr < 2/(4d) where d=8 → lr < 0.0625.
    # lr=0.01 gives per-step factor (1 - 0.01*32) = 0.68; energy decays as 0.68^(2*steps).
    loop = _make_loop(n_train_steps=20, lr=0.01, seed=0)
    result = loop.run()
    assert result["acceptance_gate_passed"], (
        f"Expected energy to decrease; got delta={result['energy_delta']:.6f}"
    )
