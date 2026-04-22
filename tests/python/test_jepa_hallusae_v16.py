"""Tests for JEPAHalluSAEv16 module.

Coverage target: 100% of python/carnot/models/jepa_hallusae_v16.py.

REQ-LEARN-055, SCENARIO-LEARN-090, SCENARIO-LEARN-091
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from carnot.models.hallusal_sparse_ae import FEATURE_DIM, SparseAutoEncoder
from carnot.models.jepa_hallusae_v16 import JEPAHalluSAEv16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_frozen_sae() -> tuple[SparseAutoEncoder, dict]:
    """Create a small SparseAutoEncoder with random init params for testing.

    Using hidden_dim=512 (matches production) so SAE_DIM constant is exercised.
    """
    sae = SparseAutoEncoder(input_dim=FEATURE_DIM, hidden_dim=512, sparsity_weight=0.01)
    rng = jax.random.PRNGKey(7)
    dummy = jnp.ones((1, FEATURE_DIM), dtype=jnp.float32)
    params = sae.init(rng, dummy)
    return sae, params


@pytest.fixture()
def sae_and_params():
    return _make_frozen_sae()


@pytest.fixture()
def jepa(sae_and_params):
    sae, params = sae_and_params
    return JEPAHalluSAEv16(sae=sae, sae_params=params, seed=42)


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-090: encode() returns correct shape and sparsity
# ---------------------------------------------------------------------------


def test_encode_shape(jepa):
    """SCENARIO-LEARN-090: encode returns (512,) shaped vector. REQ-LEARN-055-3"""
    vec = jepa.encode("The answer is 42. COMPUTE: 6 * 7 = 42.")
    assert vec.shape == (512,), f"Expected (512,) got {vec.shape}"


def test_encode_top1_sparsity(jepa):
    """SCENARIO-LEARN-090: top-1 sparsity means exactly one non-zero dim. REQ-LEARN-055-3"""
    vec = jepa.encode("compute carefully: 3 + 4 = 7")
    nonzero = int(np.sum(vec != 0.0))
    # Top-1 sparsity: at most 1 non-zero per sample.
    assert nonzero <= 1, f"Expected <=1 non-zero dims, got {nonzero}"


def test_encode_is_numpy(jepa):
    """encode() must return a numpy array, not a JAX array, for MLP compatibility."""
    vec = jepa.encode("step text here")
    assert isinstance(vec, np.ndarray)


def test_encode_dtype_float32(jepa):
    """encode() output must be float32."""
    vec = jepa.encode("step text")
    assert vec.dtype == np.float32


# ---------------------------------------------------------------------------
# score() tests
# ---------------------------------------------------------------------------


def test_score_range(jepa):
    """score() returns a float in [0, 1] (sigmoid output). REQ-LEARN-055-3"""
    s = jepa.score("The answer is correct.")
    assert 0.0 <= s <= 1.0, f"score out of [0,1]: {s}"


def test_score_is_float(jepa):
    """score() must return a Python float, not a numpy scalar."""
    s = jepa.score("text")
    assert isinstance(s, float)


# ---------------------------------------------------------------------------
# train() tests
# ---------------------------------------------------------------------------


def test_train_returns_dict(jepa):
    """train() returns dict with required keys. REQ-LEARN-055-2"""
    texts = ["correct step", "wrong step", "another correct step", "another wrong step"]
    labels = [1.0, 0.0, 1.0, 0.0]
    result = jepa.train(texts, labels, n_epochs=5)
    assert "train_losses" in result
    assert "n_train_pairs" in result
    assert "sae_sparsity_rate" in result


def test_train_loss_count(jepa):
    """train() returns one loss value per epoch."""
    texts = ["a", "b", "c", "d"]
    labels = [1.0, 0.0, 1.0, 0.0]
    result = jepa.train(texts, labels, n_epochs=10)
    assert len(result["train_losses"]) == 10


def test_train_n_train_pairs(jepa):
    """n_train_pairs equals len(texts)."""
    texts = ["a", "b", "c"]
    labels = [1.0, 0.0, 1.0]
    result = jepa.train(texts, labels, n_epochs=2)
    assert result["n_train_pairs"] == 3


def test_train_sparsity_rate_bounds(jepa):
    """sae_sparsity_rate is in [0, 1]."""
    result = jepa.train(["x", "y"], [1.0, 0.0], n_epochs=2)
    rate = result["sae_sparsity_rate"]
    assert 0.0 <= rate <= 1.0


def test_train_mutates_weights(sae_and_params):
    """Training changes MLP weights (SAE weights unchanged). REQ-LEARN-055-2"""
    sae, params = sae_and_params
    jepa = JEPAHalluSAEv16(sae=sae, sae_params=params, seed=42)
    w1_before = jepa._W1.copy()
    jepa.train(["step a", "step b"], [1.0, 0.0], n_epochs=5)
    assert not np.allclose(jepa._W1, w1_before), "W1 should change after training"


def test_sae_params_frozen(sae_and_params):
    """SAE params dict is not mutated during JEPA training. REQ-LEARN-055-2"""
    sae, params = sae_and_params
    # Take a copy of the encoder weight to compare
    import jax
    enc_weight_before = jax.device_get(params["params"]["encoder"]["kernel"]).copy()

    jepa = JEPAHalluSAEv16(sae=sae, sae_params=params, seed=42)
    jepa.train(["step a", "step b"], [1.0, 0.0], n_epochs=5)

    enc_weight_after = jax.device_get(params["params"]["encoder"]["kernel"])
    assert np.allclose(enc_weight_before, enc_weight_after), "SAE params must not be mutated"


# ---------------------------------------------------------------------------
# save() / load() round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(sae_and_params):
    """save() then load() restores MLP weights exactly. REQ-LEARN-055-1"""
    sae, params = sae_and_params
    jepa = JEPAHalluSAEv16(sae=sae, sae_params=params, seed=42)
    # Train briefly so weights differ from init
    jepa.train(["a", "b"], [1.0, 0.0], n_epochs=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test_jepa.npz")
        jepa.save(path)

        # Create a fresh instance and load
        jepa2 = JEPAHalluSAEv16(sae=sae, sae_params=params, seed=99)  # different seed
        jepa2.load(path)

        assert np.allclose(jepa._W1, jepa2._W1)
        assert np.allclose(jepa._W2, jepa2._W2)
        assert np.allclose(jepa._W3, jepa2._W3)
        assert np.allclose(jepa._b1, jepa2._b1)
        assert np.allclose(jepa._b2, jepa2._b2)
        assert np.allclose(jepa._b3, jepa2._b3)


def test_save_produces_npz(sae_and_params):
    """save() writes a .npz file with expected keys."""
    sae, params = sae_and_params
    jepa = JEPAHalluSAEv16(sae=sae, sae_params=params)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "weights.npz")
        jepa.save(path)
        data = np.load(path)
        for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
            assert key in data, f"Missing key {key} in saved npz"


# ---------------------------------------------------------------------------
# SAE_DIM constant
# ---------------------------------------------------------------------------


def test_sae_dim_constant():
    """SAE_DIM class constant matches SparseAutoEncoder hidden_dim default. REQ-LEARN-055-3"""
    assert JEPAHalluSAEv16.SAE_DIM == 512


# ---------------------------------------------------------------------------
# _relu / _sigmoid helpers
# ---------------------------------------------------------------------------


def test_relu_positive(jepa):
    """_relu returns input for positive values."""
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(jepa._relu(x), x)


def test_relu_negative(jepa):
    """_relu clips negative values to zero."""
    x = np.array([-1.0, -2.0, 0.0])
    assert np.allclose(jepa._relu(x), np.array([0.0, 0.0, 0.0]))


def test_sigmoid_midpoint(jepa):
    """_sigmoid(0) == 0.5."""
    assert abs(jepa._sigmoid(np.array([0.0]))[0] - 0.5) < 1e-6


def test_sigmoid_large_positive(jepa):
    """_sigmoid clips to near 1.0 for large positive input."""
    assert jepa._sigmoid(np.array([100.0]))[0] > 0.999


def test_sigmoid_large_negative(jepa):
    """_sigmoid clips to near 0.0 for large negative input."""
    assert jepa._sigmoid(np.array([-100.0]))[0] < 0.001
