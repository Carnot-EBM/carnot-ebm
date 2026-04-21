"""Tests for HALPProbe — pre-generative hallucination detection.

All tests trace to REQ-VERIFY-155, SCENARIO-VERIFY-209, SCENARIO-VERIFY-210.

**Coverage target:** 100% of python/carnot/pipeline/halp_probe.py.
"""

from __future__ import annotations

import pytest

from carnot.pipeline.halp_probe import HALPProbe, HALPProbeResult


# ---------------------------------------------------------------------------
# HALPProbeResult tests
# ---------------------------------------------------------------------------


class TestHALPProbeResult:
    """Spec: REQ-VERIFY-155-4"""

    def test_fields_accessible(self) -> None:
        """HALPProbeResult holds all four required fields."""
        # REQ-VERIFY-155-4
        r = HALPProbeResult(
            question="What is 2+2?",
            hidden_state_dim=64,
            hallucination_score=0.8,
            predicted_hallucinated=True,
        )
        assert r.question == "What is 2+2?"
        assert r.hidden_state_dim == 64
        assert r.hallucination_score == 0.8
        assert r.predicted_hallucinated is True

    def test_not_hallucinated(self) -> None:
        """predicted_hallucinated can be False."""
        # REQ-VERIFY-155-4
        r = HALPProbeResult(
            question="simple",
            hidden_state_dim=4,
            hallucination_score=0.2,
            predicted_hallucinated=False,
        )
        assert r.predicted_hallucinated is False


# ---------------------------------------------------------------------------
# HALPProbe.__init__ tests
# ---------------------------------------------------------------------------


class TestHALPProbeInit:
    """Spec: REQ-VERIFY-155-1"""

    def test_defaults(self) -> None:
        """Default hyperparameters match spec."""
        # REQ-VERIFY-155-1
        probe = HALPProbe()
        assert probe.n_features == 64
        assert probe.hidden_dim == 32
        assert probe.threshold == 0.5
        assert probe.weights is None

    def test_custom_params(self) -> None:
        """Custom hyperparameters are stored correctly."""
        # REQ-VERIFY-155-1
        probe = HALPProbe(n_features=8, hidden_dim=16, threshold=0.7)
        assert probe.n_features == 8
        assert probe.hidden_dim == 16
        assert probe.threshold == 0.7


# ---------------------------------------------------------------------------
# HALPProbe._extract_features tests
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    """Spec: REQ-VERIFY-155-2"""

    def test_output_shape(self) -> None:
        """Feature vector has length n_features."""
        # REQ-VERIFY-155-2
        probe = HALPProbe(n_features=8)
        feats = probe._extract_features("what is two plus two")
        assert len(feats) == 8

    def test_empty_question(self) -> None:
        """Empty question yields zero feature vector."""
        # REQ-VERIFY-155-2
        probe = HALPProbe(n_features=4)
        feats = probe._extract_features("")
        import jax.numpy as jnp
        assert float(jnp.sum(feats)) == 0.0

    def test_word_length_normalised(self) -> None:
        """Feature values are word_length / 20.0 (bounded by longest word in last n_features)."""
        # REQ-VERIFY-155-2
        probe = HALPProbe(n_features=4)
        # Single 10-character word -> last slot gets 10/20 = 0.5
        feats = probe._extract_features("helloworld")
        import jax.numpy as jnp
        assert abs(float(feats[-1]) - 0.5) < 1e-5

    def test_more_words_than_features(self) -> None:
        """When question has more words than n_features, only last n_features words are used."""
        # REQ-VERIFY-155-2
        probe = HALPProbe(n_features=2)
        # Three words; only last 2 should contribute
        feats = probe._extract_features("a bb ccc")
        import jax.numpy as jnp
        # Slot 0 = 'bb' (len=2 -> 0.1), slot 1 = 'ccc' (len=3 -> 0.15)
        assert abs(float(feats[0]) - 2.0 / 20.0) < 1e-5
        assert abs(float(feats[1]) - 3.0 / 20.0) < 1e-5

    def test_fewer_words_than_features(self) -> None:
        """When question has fewer words than n_features, leading slots remain zero."""
        # REQ-VERIFY-155-2
        probe = HALPProbe(n_features=8)
        feats = probe._extract_features("hi")  # one word, len=2
        import jax.numpy as jnp
        # All slots except the last should be zero
        assert float(jnp.sum(feats[:-1])) == 0.0
        assert abs(float(feats[-1]) - 2.0 / 20.0) < 1e-5


# ---------------------------------------------------------------------------
# HALPProbe.train tests
# ---------------------------------------------------------------------------


class TestHALPProbeTrain:
    """Spec: REQ-VERIFY-155-3, SCENARIO-VERIFY-209"""

    def test_weights_populated_after_train(self) -> None:
        """self.weights is set after train() is called."""
        # REQ-VERIFY-155-3, SCENARIO-VERIFY-209
        probe = HALPProbe(n_features=4)
        probe.train(["short q", "a longer question here"], [0, 1])
        assert probe.weights is not None
        assert "weights" in probe.weights
        assert "bias" in probe.weights

    def test_weights_length(self) -> None:
        """Returned weights list has length n_features, bias has length 1."""
        # REQ-VERIFY-155-3, SCENARIO-VERIFY-209
        probe = HALPProbe(n_features=4)
        w = probe.train(["short q", "a longer question here"], [0, 1])
        assert len(w["weights"]) == 4
        assert len(w["bias"]) == 1

    def test_train_returns_same_as_self_weights(self) -> None:
        """Return value of train() is identical to self.weights."""
        # REQ-VERIFY-155-3
        probe = HALPProbe(n_features=4)
        returned = probe.train(["hello world", "foo bar"], [1, 0])
        assert returned == probe.weights

    def test_train_multiple_samples(self) -> None:
        """train() handles larger datasets without error."""
        # REQ-VERIFY-155-3
        questions = [f"question number {i} is long enough" for i in range(20)]
        labels = [i % 2 for i in range(20)]
        probe = HALPProbe(n_features=8)
        probe.train(questions, labels)
        assert probe.weights is not None


# ---------------------------------------------------------------------------
# HALPProbe.predict tests
# ---------------------------------------------------------------------------


class TestHALPProbePredict:
    """Spec: REQ-VERIFY-155-4, SCENARIO-VERIFY-209, SCENARIO-VERIFY-210"""

    def test_predict_returns_result_type(self) -> None:
        """predict() returns a HALPProbeResult instance."""
        # REQ-VERIFY-155-4
        probe = HALPProbe(n_features=4)
        result = probe.predict("what is two plus two")
        assert isinstance(result, HALPProbeResult)

    def test_predict_question_stored(self) -> None:
        """Result.question matches the input."""
        # REQ-VERIFY-155-4
        probe = HALPProbe(n_features=4)
        result = probe.predict("test question")
        assert result.question == "test question"

    def test_predict_hidden_state_dim(self) -> None:
        """Result.hidden_state_dim equals n_features."""
        # REQ-VERIFY-155-4
        probe = HALPProbe(n_features=8)
        result = probe.predict("any question")
        assert result.hidden_state_dim == 8

    def test_predict_score_range(self) -> None:
        """hallucination_score is in [0, 1] after training."""
        # REQ-VERIFY-155-4
        probe = HALPProbe(n_features=4)
        probe.train(["hello", "world test example"], [0, 1])
        result = probe.predict("hello world")
        assert 0.0 <= result.hallucination_score <= 1.0

    def test_predict_untrained_uses_feature_mean(self) -> None:
        """Untrained probe uses feature mean as score (SCENARIO-VERIFY-210)."""
        # SCENARIO-VERIFY-210
        import jax.numpy as jnp
        probe = HALPProbe(n_features=4)
        q = "what is two plus two"
        feats = probe._extract_features(q)
        expected_score = float(jnp.mean(feats))
        result = probe.predict(q)
        assert abs(result.hallucination_score - expected_score) < 1e-5

    def test_predict_hallucinated_flag_above_threshold(self) -> None:
        """predicted_hallucinated=True when score >= threshold."""
        # REQ-VERIFY-155-4
        probe = HALPProbe(n_features=4, threshold=0.0)  # everything >= 0.0
        result = probe.predict("any question at all")
        assert result.predicted_hallucinated is True

    def test_predict_hallucinated_flag_below_threshold(self) -> None:
        """predicted_hallucinated=False when score < threshold."""
        # REQ-VERIFY-155-4
        probe = HALPProbe(n_features=4, threshold=1.1)  # nothing can reach > 1.0
        result = probe.predict("any question at all")
        assert result.predicted_hallucinated is False

    def test_predict_after_train_uses_weights(self) -> None:
        """After training, predict() uses learned weights (not feature mean)."""
        # REQ-VERIFY-155-3, REQ-VERIFY-155-4, SCENARIO-VERIFY-209
        import jax
        import jax.numpy as jnp
        probe = HALPProbe(n_features=4)
        probe.train(["hello", "world example test"], [0, 1])
        q = "test query"
        feats = probe._extract_features(q)
        w = jnp.array(probe.weights["weights"])
        b = jnp.array(probe.weights["bias"])
        expected = float(jax.nn.sigmoid(feats @ w + b[0]))
        result = probe.predict(q)
        assert abs(result.hallucination_score - expected) < 1e-5


# ---------------------------------------------------------------------------
# Export test
# ---------------------------------------------------------------------------


class TestExports:
    """Spec: REQ-VERIFY-155-6"""

    def test_exported_from_pipeline(self) -> None:
        """HALPProbe and HALPProbeResult are exported from carnot.pipeline."""
        # REQ-VERIFY-155-6
        from carnot.pipeline import HALPProbe as HP
        from carnot.pipeline import HALPProbeResult as HPR
        assert HP is HALPProbe
        assert HPR is HALPProbeResult
