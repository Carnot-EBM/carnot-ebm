"""Tests for NUPProbeV3 — CLAPFeatureExtractor, CLAPFeatures, NUPProbeV3.

Spec: REQ-VERIFY-104, REQ-VERIFY-105, REQ-VERIFY-106,
      SCENARIO-VERIFY-137, SCENARIO-VERIFY-138, SCENARIO-VERIFY-139
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.pipeline.nup_probe_v3 import (
    CLAPFeatureExtractor,
    CLAPFeatures,
    NUPProbeV3,
)


# ---------------------------------------------------------------------------
# CLAPFeatureExtractor tests
# ---------------------------------------------------------------------------


class TestCLAPFeatureExtractor:
    """REQ-VERIFY-104, REQ-VERIFY-105"""

    def test_extract_features_returns_clap_features(self):
        """SCENARIO-VERIFY-137: extract_features on (4, 20, 768) returns CLAPFeatures."""
        rng = np.random.default_rng(0)
        acts = rng.normal(size=(4, 20, 768)).astype(np.float32)
        extractor = CLAPFeatureExtractor(n_layers=4, n_heads=8)
        result = extractor.extract_features(acts)
        assert isinstance(result, CLAPFeatures)
        assert result.per_token_entropy.shape == (20,)
        assert result.topk_concentration.shape == (20,)
        assert result.cross_layer_variance.shape == (20,)

    def test_per_token_entropy_non_negative(self):
        """Entropy of any distribution is >= 0."""
        rng = np.random.default_rng(1)
        acts = rng.normal(size=(4, 10, 64))
        extractor = CLAPFeatureExtractor(n_layers=4)
        result = extractor.extract_features(acts)
        assert np.all(result.per_token_entropy >= 0.0)

    def test_topk_concentration_in_range(self):
        """Concentration ratio must be in (0, 1]."""
        rng = np.random.default_rng(2)
        acts = rng.normal(size=(4, 10, 64))
        extractor = CLAPFeatureExtractor(n_layers=4, topk=5)
        result = extractor.extract_features(acts)
        assert np.all(result.topk_concentration > 0.0)
        assert np.all(result.topk_concentration <= 1.0 + 1e-6)

    def test_cross_layer_variance_non_negative(self):
        """Variance is always >= 0."""
        rng = np.random.default_rng(3)
        acts = rng.normal(size=(4, 10, 64))
        extractor = CLAPFeatureExtractor(n_layers=4)
        result = extractor.extract_features(acts)
        assert np.all(result.cross_layer_variance >= 0.0)

    def test_raises_on_wrong_ndim(self):
        """Should raise ValueError for non-3D input."""
        extractor = CLAPFeatureExtractor(n_layers=4)
        with pytest.raises(ValueError, match="3D"):
            extractor.extract_features(np.zeros((4, 10)))

    def test_raises_on_layer_mismatch(self):
        """Should raise ValueError when tensor n_layers != self.n_layers."""
        extractor = CLAPFeatureExtractor(n_layers=4)
        with pytest.raises(ValueError, match="n_layers"):
            extractor.extract_features(np.zeros((2, 10, 64)))

    def test_invalid_n_layers(self):
        """Constructor rejects n_layers < 1."""
        with pytest.raises(ValueError):
            CLAPFeatureExtractor(n_layers=0)

    def test_invalid_topk(self):
        """Constructor rejects topk < 1."""
        with pytest.raises(ValueError):
            CLAPFeatureExtractor(topk=0)

    def test_small_hidden_dim(self):
        """Works when hidden_dim < topk."""
        extractor = CLAPFeatureExtractor(n_layers=2, topk=10)
        acts = np.random.default_rng(7).normal(size=(2, 5, 3))
        result = extractor.extract_features(acts)
        assert result.per_token_entropy.shape == (5,)

    def test_single_token(self):
        """Works with n_tokens=1."""
        extractor = CLAPFeatureExtractor(n_layers=4)
        acts = np.ones((4, 1, 16), dtype=np.float64)
        result = extractor.extract_features(acts)
        assert result.per_token_entropy.shape == (1,)

    def test_uniform_activations_high_entropy(self):
        """Uniform activation → softmax → uniform distribution → maximum entropy."""
        extractor = CLAPFeatureExtractor(n_layers=2)
        acts = np.zeros((2, 5, 64))  # all zeros → uniform softmax → max entropy
        result = extractor.extract_features(acts)
        max_entropy = float(np.log(64))
        assert np.all(result.per_token_entropy > max_entropy * 0.99)

    def test_peaked_activations_low_concentration(self):
        """Highly peaked activations → concentration near 1.0."""
        extractor = CLAPFeatureExtractor(n_layers=2, topk=5)
        acts = np.zeros((2, 3, 32), dtype=np.float64)
        # Make first element very large at index 0 for each token
        acts[:, :, 0] = 100.0
        result = extractor.extract_features(acts)
        # Top-1 should dominate → concentration ≈ 1
        assert np.all(result.topk_concentration > 0.99)


# ---------------------------------------------------------------------------
# CLAPFeatures.to_feature_vector tests
# ---------------------------------------------------------------------------


class TestCLAPFeaturesToFeatureVector:
    """REQ-VERIFY-105"""

    def test_to_feature_vector_length(self):
        """to_feature_vector returns length 3 * n_tokens."""
        n_tokens = 10
        features = CLAPFeatures(
            per_token_entropy=np.ones(n_tokens),
            topk_concentration=np.ones(n_tokens) * 0.5,
            cross_layer_variance=np.ones(n_tokens) * 2.0,
        )
        vec = features.to_feature_vector()
        assert vec.shape == (3 * n_tokens,)

    def test_to_feature_vector_dtype(self):
        """Output is float64."""
        features = CLAPFeatures(
            per_token_entropy=np.ones(5, dtype=np.float32),
            topk_concentration=np.ones(5, dtype=np.float32),
            cross_layer_variance=np.ones(5, dtype=np.float32),
        )
        vec = features.to_feature_vector()
        assert vec.dtype == np.float64 or np.issubdtype(vec.dtype, np.floating)

    def test_to_feature_vector_constant_input_zero_mean(self):
        """Z-scoring a constant array yields all zeros (std=0 path)."""
        features = CLAPFeatures(
            per_token_entropy=np.full(5, 2.0),
            topk_concentration=np.full(5, 0.5),
            cross_layer_variance=np.full(5, 1.0),
        )
        vec = features.to_feature_vector()
        assert np.allclose(vec, 0.0), f"Expected zeros, got {vec}"

    def test_to_feature_vector_no_nan(self):
        """No NaN in output for arbitrary input."""
        rng = np.random.default_rng(9)
        features = CLAPFeatures(
            per_token_entropy=rng.uniform(0, 5, size=20),
            topk_concentration=rng.uniform(0.2, 1.0, size=20),
            cross_layer_variance=rng.uniform(0, 100, size=20),
        )
        vec = features.to_feature_vector()
        assert not np.any(np.isnan(vec))


# ---------------------------------------------------------------------------
# NUPProbeV3 tests
# ---------------------------------------------------------------------------


class TestNUPProbeV3:
    """REQ-VERIFY-106, SCENARIO-VERIFY-138, SCENARIO-VERIFY-139"""

    def _make_pairs(self, n: int, n_layers: int = 4, n_tokens: int = 10,
                    hidden_dim: int = 64, seed: int = 0):
        """Generate n synthetic (activations, label) pairs."""
        rng = np.random.default_rng(seed)
        pairs = []
        labels = []
        for i in range(n):
            acts = rng.normal(size=(n_layers, n_tokens, hidden_dim))
            label = int(rng.integers(0, 2))
            pairs.append(acts)
            labels.append(label)
        return pairs, labels

    def test_fit_converges_on_20_pairs(self):
        """SCENARIO-VERIFY-138: NUPProbeV3.fit on 20 synthetic pairs converges."""
        n_features = 3 * 10  # 3 * n_tokens
        probe = NUPProbeV3(n_features=n_features)
        pairs, labels = self._make_pairs(20)
        probe.fit(pairs, labels)
        assert probe._is_fitted
        assert probe._weights is not None
        assert probe._weights.shape == (n_features,)

    def test_evaluate_returns_auroc_key(self):
        """SCENARIO-VERIFY-139: evaluate returns auroc key in [0, 1]."""
        n_features = 3 * 10
        probe = NUPProbeV3(n_features=n_features)
        pairs, labels = self._make_pairs(20, seed=1)
        probe.fit(pairs[:16], labels[:16])
        result = probe.evaluate(pairs[16:], labels[16:])
        assert "auroc" in result
        assert 0.0 <= result["auroc"] <= 1.0

    def test_predict_returns_float_in_01(self):
        """predict() output must be in [0, 1]."""
        n_features = 3 * 10
        probe = NUPProbeV3(n_features=n_features)
        pairs, labels = self._make_pairs(10)
        probe.fit(pairs, labels)
        acts = np.random.default_rng(42).normal(size=(4, 10, 64))
        extractor = CLAPFeatureExtractor(n_layers=4)
        features = extractor.extract_features(acts)
        prob = probe.predict(features)
        assert 0.0 <= prob <= 1.0

    def test_predict_before_fit_returns_05(self):
        """predict() before fit returns 0.5 (uninformative)."""
        probe = NUPProbeV3(n_features=30)
        features = CLAPFeatures(
            per_token_entropy=np.ones(10),
            topk_concentration=np.ones(10) * 0.5,
            cross_layer_variance=np.ones(10),
        )
        assert probe.predict(features) == 0.5

    def test_evaluate_with_fewer_than_2_pairs(self):
        """evaluate() with < 2 pairs returns auroc=0.5."""
        probe = NUPProbeV3(n_features=30)
        acts = np.zeros((4, 10, 64))
        result = probe.evaluate([(acts, 1)])
        assert result["auroc"] == 0.5

    def test_evaluate_all_same_label(self):
        """evaluate() with all-positive returns auroc=0.5 (cannot discriminate)."""
        n_features = 3 * 10
        probe = NUPProbeV3(n_features=n_features)
        pairs, _ = self._make_pairs(10)
        labels = [1] * 10
        probe.fit(pairs[:8], labels[:8])
        result = probe.evaluate(pairs[8:], labels[8:])
        assert result["auroc"] == 0.5

    def test_fit_with_combined_pairs(self):
        """fit() also works when pairs are (activations, label) tuples."""
        n_features = 3 * 10
        probe = NUPProbeV3(n_features=n_features)
        pairs, labels = self._make_pairs(10)
        combined = list(zip(pairs, labels))
        probe.fit(combined)  # no separate labels arg
        assert probe._is_fitted

    def test_fit_empty_pairs(self):
        """fit() on empty input is a no-op."""
        probe = NUPProbeV3(n_features=30)
        probe.fit([])  # should not raise
        assert not probe._is_fitted

    def test_evaluate_n_pairs_key(self):
        """evaluate() result contains n_pairs key."""
        n_features = 3 * 10
        probe = NUPProbeV3(n_features=n_features)
        pairs, labels = self._make_pairs(10)
        probe.fit(pairs[:8], labels[:8])
        result = probe.evaluate(pairs[8:], labels[8:])
        assert "n_pairs" in result
        assert result["n_pairs"] == 2

    def test_feature_padding(self):
        """Short feature vector is padded to n_features."""
        n_features = 300  # larger than 3 * 10 = 30
        probe = NUPProbeV3(n_features=n_features)
        pairs, labels = self._make_pairs(10)
        probe.fit(pairs, labels)
        # Should not raise despite n_features > 3 * n_tokens
        acts = np.random.default_rng(0).normal(size=(4, 10, 64))
        features = CLAPFeatureExtractor(n_layers=4).extract_features(acts)
        prob = probe.predict(features)
        assert 0.0 <= prob <= 1.0

    def test_feature_truncation(self):
        """Long feature vector is truncated to n_features."""
        n_features = 6  # smaller than 3 * 10 = 30
        probe = NUPProbeV3(n_features=n_features)
        pairs, labels = self._make_pairs(10)
        probe.fit(pairs, labels)
        acts = np.random.default_rng(0).normal(size=(4, 10, 64))
        features = CLAPFeatureExtractor(n_layers=4).extract_features(acts)
        prob = probe.predict(features)
        assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# Integration: full pipeline test
# ---------------------------------------------------------------------------


class TestCLAPPipeline:
    """Integration test: CLAPFeatureExtractor -> CLAPFeatures -> NUPProbeV3."""

    def test_full_pipeline_ci_mode(self):
        """CI stub: synthetic (4, 10, 64) activations → feature extraction → probe."""
        extractor = CLAPFeatureExtractor(n_layers=4, n_heads=8)
        rng = np.random.default_rng(99)
        n_train = 20
        n_test = 5
        n_features = 3 * 10

        train_acts = [rng.normal(size=(4, 10, 64)) for _ in range(n_train)]
        train_labels = [int(i % 2) for i in range(n_train)]
        test_acts = [rng.normal(size=(4, 10, 64)) for _ in range(n_test)]
        test_labels = [int(i % 2) for i in range(n_test)]

        probe = NUPProbeV3(n_features=n_features, extractor=extractor)
        probe.fit(train_acts, train_labels)

        result = probe.evaluate(test_acts, test_labels)
        assert "auroc" in result
        assert 0.0 <= result["auroc"] <= 1.0

    def test_clap_features_match_shape_contract(self):
        """CLAPFeatures.to_feature_vector length = 3 * n_tokens for any n_tokens."""
        for n_tokens in [1, 5, 20, 100]:
            extractor = CLAPFeatureExtractor(n_layers=4)
            acts = np.random.default_rng(n_tokens).normal(size=(4, n_tokens, 32))
            features = extractor.extract_features(acts)
            vec = features.to_feature_vector()
            assert vec.shape[0] == 3 * n_tokens, (
                f"Expected {3 * n_tokens}, got {vec.shape[0]} for n_tokens={n_tokens}"
            )
