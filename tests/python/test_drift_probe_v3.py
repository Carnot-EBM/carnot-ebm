"""Unit tests for DRIFTProbeV3 — 100% coverage of drift_probe_v3.py.

SPEC COVERAGE: REQ-PROBE-010, SCENARIO-PROBE-015
"""

import pytest
import numpy as np
from numpy.typing import NDArray
from typing import List

from python.carnot.pipeline.drift_probe_v3 import (
    _cosine_drift_per_layer,
    _relu,
    _sigmoid,
    DRIFTProbeV3,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_layer_hs(
    n_layers: int, seq_len: int, hidden_dim: int, noise_scale: float = 0.1, seed: int = 0
) -> List[NDArray]:
    """Return one sample: a list of n_layers activation arrays [seq_len, hidden_dim]."""
    rng = np.random.default_rng(seed)
    return [
        rng.normal(0, noise_scale, (seq_len, hidden_dim)).astype(np.float32)
        for _ in range(n_layers)
    ]


def _make_dataset(
    n_correct: int, n_incorrect: int, n_layers: int = 6, seq_len: int = 8, hidden_dim: int = 16
) -> tuple:
    """Generate (X_layers, y) with controlled drift pattern.

    Incorrect samples have large noise on even layers; correct samples have small noise.
    Returns (X_list, y_list).
    """
    rng = np.random.default_rng(42)
    samples = []
    labels = []

    for i in range(n_correct):
        layers = []
        base = rng.normal(0, 0.05, (seq_len, hidden_dim)).astype(np.float32)
        for _ in range(n_layers):
            layers.append(base + rng.normal(0, 0.05, (seq_len, hidden_dim)).astype(np.float32))
        samples.append(layers)
        labels.append(0)

    for i in range(n_incorrect):
        layers = []
        base = rng.normal(0, 0.05, (seq_len, hidden_dim)).astype(np.float32)
        for j in range(n_layers):
            scale = 2.0 if j % 2 == 0 else 0.05
            layers.append(base + rng.normal(0, scale, (seq_len, hidden_dim)).astype(np.float32))
        samples.append(layers)
        labels.append(1)

    return samples, labels


# ---------------------------------------------------------------------------
# Tests: utility functions
# ---------------------------------------------------------------------------


def test_relu_positive():
    """REQ-PROBE-010: _relu passes positive values unchanged."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = _relu(x)
    np.testing.assert_array_equal(result, x)


def test_relu_negative_zeroed():
    """REQ-PROBE-010: _relu zeros out negative values."""
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    result = _relu(x)
    expected = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_sigmoid_mid():
    """REQ-PROBE-010: _sigmoid(0) == 0.5."""
    assert abs(_sigmoid(np.array([0.0]))[0] - 0.5) < 1e-6


def test_sigmoid_large_positive():
    """REQ-PROBE-010: _sigmoid(large) approaches 1."""
    assert _sigmoid(np.array([100.0]))[0] > 0.999


def test_sigmoid_large_negative():
    """REQ-PROBE-010: _sigmoid(-large) approaches 0."""
    assert _sigmoid(np.array([-100.0]))[0] < 0.001


def test_cosine_drift_all_same():
    """REQ-PROBE-010: identical tokens give zero drift."""
    # All token vectors identical → cosine sim = 1.0 → drift = 0.
    layer = np.ones((10, 8), dtype=np.float32)
    drift = _cosine_drift_per_layer([layer])
    assert drift.shape == (1,)
    assert abs(float(drift[0])) < 1e-5


def test_cosine_drift_opposite():
    """REQ-PROBE-010: alternating +-1 vectors give maximum drift (~2.0)."""
    seq = np.zeros((4, 4), dtype=np.float32)
    seq[0] = [1, 0, 0, 0]
    seq[1] = [-1, 0, 0, 0]
    seq[2] = [1, 0, 0, 0]
    seq[3] = [-1, 0, 0, 0]
    drift = _cosine_drift_per_layer([seq])
    # cosine sim between +1 and -1 is -1.0 → drift = 1 - (-1) = 2.0
    assert abs(float(drift[0]) - 2.0) < 1e-5


def test_cosine_drift_short_seq():
    """REQ-PROBE-010: single-token layer returns 0 drift without error."""
    layer = np.ones((1, 4), dtype=np.float32)
    drift = _cosine_drift_per_layer([layer])
    assert drift.shape == (1,)
    assert float(drift[0]) == 0.0


def test_cosine_drift_zero_vector():
    """REQ-PROBE-010: zero-norm vectors don't cause div-by-zero."""
    layer = np.zeros((4, 4), dtype=np.float32)
    drift = _cosine_drift_per_layer([layer])
    assert np.isfinite(float(drift[0]))


def test_cosine_drift_multi_layer():
    """REQ-PROBE-010: returns one scalar per layer in the input list."""
    layers = [np.random.randn(8, 16).astype(np.float32) for _ in range(5)]
    drift = _cosine_drift_per_layer(layers)
    assert drift.shape == (5,)
    assert np.all(np.isfinite(drift))


# ---------------------------------------------------------------------------
# Tests: DRIFTProbeV3 class
# ---------------------------------------------------------------------------


def test_predict_proba_before_fit_raises():
    """REQ-PROBE-010: predict_proba before fit raises RuntimeError."""
    probe = DRIFTProbeV3()
    X = [_make_layer_hs(4, 8, 16)]
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        probe.predict_proba(X)


def test_layer_attention_weights_before_fit_raises():
    """REQ-PROBE-010: layer_attention_weights before fit raises RuntimeError."""
    probe = DRIFTProbeV3()
    with pytest.raises(RuntimeError, match="not fitted"):
        probe.layer_attention_weights()


def test_fit_returns_self():
    """REQ-PROBE-010: fit() returns self for chaining."""
    probe = DRIFTProbeV3(n_iter=5)
    X, y = _make_dataset(4, 4, n_layers=4, seq_len=4, hidden_dim=8)
    result = probe.fit(X, y)
    assert result is probe


def test_predict_proba_shape():
    """REQ-PROBE-010: predict_proba returns 1-D array with len == len(X)."""
    X, y = _make_dataset(10, 10, n_layers=4, seq_len=4, hidden_dim=8)
    probe = DRIFTProbeV3(n_iter=10)
    probe.fit(X[:16], y[:16])
    proba = probe.predict_proba(X[16:])
    assert proba.ndim == 1
    assert len(proba) == len(X[16:])


def test_predict_proba_range():
    """REQ-PROBE-010: all probabilities are in [0, 1]."""
    X, y = _make_dataset(10, 10, n_layers=4, seq_len=4, hidden_dim=8)
    probe = DRIFTProbeV3(n_iter=20)
    probe.fit(X[:16], y[:16])
    proba = probe.predict_proba(X[16:])
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_layer_attention_weights_sum_to_one():
    """REQ-PROBE-010: layer_attention_weights() sums to 1 and is non-negative."""
    X, y = _make_dataset(10, 10, n_layers=6, seq_len=4, hidden_dim=8)
    probe = DRIFTProbeV3(n_iter=10)
    probe.fit(X, y)
    w = probe.layer_attention_weights()
    assert w.shape == (6,)
    assert np.all(w >= 0.0)
    assert abs(w.sum() - 1.0) < 1e-5


def test_layer_attention_weights_zero_w1_fallback():
    """REQ-PROBE-010: layer_attention_weights returns uniform when W1 is all-zero."""
    X, y = _make_dataset(4, 4, n_layers=4, seq_len=4, hidden_dim=8)
    probe = DRIFTProbeV3(n_iter=1, lr=0.0)  # lr=0 → weights never update from init
    probe.fit(X, y)
    # Force W1 to all-zero to hit the fallback branch.
    probe._W1[:] = 0.0
    w = probe.layer_attention_weights()
    expected = np.ones(4, dtype=np.float32) / 4
    np.testing.assert_allclose(w, expected, atol=1e-5)


def test_extract_drift_matrix_shape():
    """REQ-PROBE-010: _extract_drift_matrix returns [N, n_layers] matrix."""
    probe = DRIFTProbeV3()
    X = [_make_layer_hs(6, 8, 16, seed=i) for i in range(5)]
    matrix = probe._extract_drift_matrix(X)
    assert matrix.shape == (5, 6)


def test_above_random_auc_synthetic():
    """SCENARIO-PROBE-015: probe AUROC > 0.50 on synthetic drift-labeled dataset.

    This validates the core claim: a learned probe can extract above-random signal
    from a layer stack where incorrect responses have large drift at even layers.
    """
    from sklearn.metrics import roc_auc_score

    X, y = _make_dataset(100, 100, n_layers=6, seq_len=16, hidden_dim=32)
    # Shuffle so eval split has both classes.
    rng = np.random.default_rng(7)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]
    n_train = 160
    probe = DRIFTProbeV3(hidden_dim=32, lr=0.05, n_iter=300)
    probe.fit(X[:n_train], y[:n_train])
    proba = probe.predict_proba(X[n_train:])
    auc = roc_auc_score(y[n_train:], proba)
    assert auc > 0.50, f"Expected AUROC > 0.50, got {auc:.4f}"


def test_non_uniform_layer_weights_after_training():
    """SCENARIO-PROBE-015: layer attention weights are non-uniform after training on skewed data."""
    X, y = _make_dataset(80, 80, n_layers=6, seq_len=16, hidden_dim=32)
    rng = np.random.default_rng(9)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]
    probe = DRIFTProbeV3(hidden_dim=32, lr=0.05, n_iter=300)
    probe.fit(X, y)
    w = probe.layer_attention_weights()
    # Non-uniform: max weight should differ from min weight by > 5%.
    assert float(w.max() - w.min()) > 0.01, f"Expected non-uniform weights, got: {w.tolist()}"


def test_fit_with_all_same_label():
    """REQ-PROBE-010: fit does not crash when all labels are the same class."""
    X, _ = _make_dataset(10, 0, n_layers=4, seq_len=4, hidden_dim=8)
    y = [0] * 10
    probe = DRIFTProbeV3(n_iter=5)
    probe.fit(X, y)  # Should not raise.
    proba = probe.predict_proba(X[:3])
    assert len(proba) == 3
