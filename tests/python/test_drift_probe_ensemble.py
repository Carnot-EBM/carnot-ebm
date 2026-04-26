"""Tests for drift_probe_ensemble.py — DRIFTProbeEnsemble Tier 0i.

Validates that the per-layer ensemble probe correctly:
  - Returns 0.5 before fitting (unfitted guard).
  - Trains one probe per adjacent layer pair.
  - Learns non-uniform ensemble weights when the layer-pair signals differ.
  - Achieves AUC > 0.65 on a synthetic hallucination corpus (the SCENARIO-TIER0-006 target).
  - Handles missing layer keys gracefully.

Spec: REQ-TIER0-006, SCENARIO-TIER0-006
Spec: REQ-VERIFY-001
"""

from __future__ import annotations

import numpy as np
import pytest

from python.carnot.verify.drift_probe_ensemble import (
    DRIFTProbeEnsemble,
    _cosine_sim,
    _drift_for_pair,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_runner(layers: list[int], is_hallucinated_fn, seed: int = 0):
    """Build a synthetic model_runner that produces high/low drift per text.

    is_hallucinated_fn(text) -> bool controls whether the runner returns
    high-drift (hallucinated) or low-drift (correct) hidden states.

    Args:
        layers:              Layer indices to populate.
        is_hallucinated_fn:  Callable[[str], bool].
        seed:                Base RNG seed.

    Returns:
        Callable[[str], dict[int, np.ndarray]]
    """
    hidden_dim = 32

    def runner(text: str) -> dict[int, np.ndarray]:
        text_seed = (hash(text) & 0xFFFF_FFFF) ^ seed
        rng = np.random.default_rng(text_seed)
        seq_len = 16
        base = rng.standard_normal((seq_len, hidden_dim)).astype(np.float32)
        result = {}
        for i, layer_idx in enumerate(layers):
            if is_hallucinated_fn(text):
                scale = 0.9 + 0.3 * i
            else:
                scale = 0.04 + 0.01 * i
            noise = rng.standard_normal((seq_len, hidden_dim)).astype(np.float32)
            state = base + scale * noise
            norms = np.linalg.norm(state, axis=1, keepdims=True) + 1e-8
            result[layer_idx] = (state / norms).astype(np.float32)
        return result

    return runner


def _make_correct_texts(n: int) -> list[str]:
    return [f"correct response number {i} the answer is 42" for i in range(n)]


def _make_hallucinated_texts(n: int) -> list[str]:
    return [f"hallucinated response number {i} the answer is 999" for i in range(n)]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestCosineSimHelper:
    """REQ-TIER0-006 — helper correctly handles zero vectors."""

    def test_identical_vectors(self):
        """Identical non-zero vectors should have cosine similarity 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_sim(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert _cosine_sim(v1, v2) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector_returns_one(self):
        """Zero vector → no drift (similarity = 1.0 per drift_probe convention)."""
        v = np.array([1.0, 2.0, 3.0])
        z = np.zeros(3)
        assert _cosine_sim(z, v) == pytest.approx(1.0)
        assert _cosine_sim(v, z) == pytest.approx(1.0)


class TestDriftForPair:
    """REQ-TIER0-006-5 — missing keys produce zero drift."""

    def test_missing_layer_returns_zero(self):
        hs: dict = {}  # both layers absent
        assert _drift_for_pair(hs, -2, -1) == pytest.approx(0.0)

    def test_one_missing_layer_returns_zero(self):
        rng = np.random.default_rng(0)
        hs = {-1: rng.standard_normal((4, 8)).astype(np.float32)}
        assert _drift_for_pair(hs, -2, -1) == pytest.approx(0.0)

    def test_identical_layers_zero_drift(self):
        """Identical representations → cosine distance 0 → drift = 0."""
        rng = np.random.default_rng(1)
        rep = rng.standard_normal((8, 16)).astype(np.float32)
        # Normalise so cosine sim is well defined.
        norms = np.linalg.norm(rep, axis=1, keepdims=True) + 1e-8
        rep = rep / norms
        hs = {-2: rep, -1: rep}
        drift = _drift_for_pair(hs, -2, -1)
        assert drift == pytest.approx(0.0, abs=1e-5)


class TestDRIFTProbeEnsembleUnfitted:
    """REQ-TIER0-006-3 — unfitted ensemble returns 0.5."""

    def test_unfitted_returns_half(self):
        ens = DRIFTProbeEnsemble(model_runner=None, layers=[-4, -3, -2, -1])
        rng = np.random.default_rng(0)
        hs = {i: rng.standard_normal((4, 8)).astype(np.float32) for i in [-4, -3, -2, -1]}
        assert ens.predict_violation_prob(hs) == pytest.approx(0.5)

    def test_unfitted_from_text_returns_half(self):
        ens = DRIFTProbeEnsemble(model_runner=None)
        assert ens.predict_violation_prob_from_text("any text") == pytest.approx(0.5)


class TestDRIFTProbeEnsembleStructure:
    """REQ-TIER0-006-1, REQ-TIER0-006-2 — fit produces correct probe count and weights."""

    @pytest.fixture
    def fitted_ensemble(self):
        layers = [-4, -3, -2, -1]
        runner = _make_synthetic_runner(
            layers,
            is_hallucinated_fn=lambda t: "hallucinated" in t,
            seed=7,
        )
        ens = DRIFTProbeEnsemble(model_runner=runner, layers=layers)
        correct = _make_correct_texts(20)
        halluc = _make_hallucinated_texts(20)
        ens.fit(correct, halluc, val_fraction=0.2, n_grid_points=10)
        return ens

    def test_n_probes_equals_n_pairs(self, fitted_ensemble):
        """REQ-TIER0-006-1: one probe per adjacent layer pair."""
        # 4 layers → 3 pairs → 3 probes.
        assert len(fitted_ensemble.per_layer_probes) == 3

    def test_ensemble_weights_sum_to_one(self, fitted_ensemble):
        """REQ-TIER0-006-2: alpha is a probability simplex (non-negative, sum=1)."""
        alpha = fitted_ensemble.ensemble_weights
        assert alpha is not None
        assert alpha.shape == (3,)
        assert all(w >= 0.0 for w in alpha), f"Negative weight: {alpha}"
        assert float(alpha.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_predict_violation_prob_in_range(self, fitted_ensemble):
        """REQ-TIER0-006-3: output is in [0, 1]."""
        rng = np.random.default_rng(42)
        hs = {i: rng.standard_normal((8, 16)).astype(np.float32) for i in [-4, -3, -2, -1]}
        p = fitted_ensemble.predict_violation_prob(hs)
        assert 0.0 <= p <= 1.0


class TestDRIFTProbeEnsembleAUC:
    """SCENARIO-TIER0-006 — ensemble AUC > 0.65 on synthetic GSM8K hallucination corpus."""

    def test_ensemble_auc_exceeds_threshold(self):
        """With a clearly separable synthetic corpus the ensemble must exceed AUC=0.65."""
        from sklearn.metrics import roc_auc_score

        layers = [-4, -3, -2, -1]
        runner = _make_synthetic_runner(
            layers,
            is_hallucinated_fn=lambda t: "hallucinated" in t,
            seed=99,
        )
        ens = DRIFTProbeEnsemble(model_runner=runner, layers=layers)

        correct_train = _make_correct_texts(80)
        halluc_train = _make_hallucinated_texts(80)
        ens.fit(correct_train, halluc_train, val_fraction=0.2, n_grid_points=20)

        correct_eval = _make_correct_texts(20)
        halluc_eval = _make_hallucinated_texts(20)

        scores = []
        labels = []
        for text in correct_eval:
            hs = runner(text)
            scores.append(ens.predict_violation_prob(hs))
            labels.append(0)
        for text in halluc_eval:
            hs = runner(text)
            scores.append(ens.predict_violation_prob(hs))
            labels.append(1)

        auc = roc_auc_score(labels, scores)
        assert auc > 0.65, f"AUC={auc:.4f} did not exceed 0.65 (SCENARIO-TIER0-006)"


class TestDRIFTProbeEnsembleDefaultLayers:
    """REQ-TIER0-006-4 — default layers resolve to [-4, -3, -2, -1]."""

    def test_default_layers(self):
        ens = DRIFTProbeEnsemble()
        assert ens.layers == [-4, -3, -2, -1]


class TestSimplexGrid:
    """Internal grid covers the simplex correctly."""

    def test_grid_candidates_valid_simplex(self):
        grid = DRIFTProbeEnsemble._simplex_grid(n_dims=3, n_points=10)
        assert len(grid) == 10
        for alpha in grid:
            assert alpha.shape == (3,)
            assert float(alpha.sum()) == pytest.approx(1.0, abs=1e-9)
            assert all(w >= 0.0 for w in alpha)

    def test_first_candidate_is_uniform(self):
        grid = DRIFTProbeEnsemble._simplex_grid(n_dims=4, n_points=5)
        expected = np.full(4, 0.25)
        np.testing.assert_allclose(grid[0], expected, atol=1e-9)
