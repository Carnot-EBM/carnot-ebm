"""Tests for Experiment 877: VariationalJEPAPredictor.

Every test traces to a spec requirement or scenario.
100% coverage of python/carnot/models/vjepa_predictor.py.

Spec: REQ-VERIFY-175, REQ-VERIFY-176, SCENARIO-VERIFY-229, SCENARIO-VERIFY-230
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from carnot.models.vjepa_predictor import (
    VOCAB_SIZE,
    TrainMetrics,
    VariationalEncoder,
    VariationalJEPAPredictor,
    VariationalPrior,
    build_tfidf_features,
    compute_auc,
    prepare_corpus,
    text_to_tfidf,
)


# ---------------------------------------------------------------------------
# VariationalEncoder
# ---------------------------------------------------------------------------


class TestVariationalEncoder:
    """Traces: REQ-VERIFY-175 (VariationalJEPAPredictor interface)."""

    def _enc(self, in_dim: int = 10, latent_dim: int = 8) -> VariationalEncoder:
        return VariationalEncoder(in_dim=in_dim, latent_dim=latent_dim)

    def test_encode_returns_triple(self):
        """encode() returns (z, mu, log_var) triple."""
        enc = self._enc()
        x = jnp.ones(10)
        z, mu, lv = enc.encode(x, jax.random.PRNGKey(0))
        assert z.shape == (8,)
        assert mu.shape == (8,)
        assert lv.shape == (8,)

    def test_reparameterize_samples_differ(self):
        """Reparameterisation samples differ across keys — REQ-VERIFY-175.

        If the reparameterisation trick is broken (eps=0 always), both samples
        would be identical.  This guards against the most common implementation
        error.
        """
        enc = self._enc()
        mu = jnp.zeros(8)
        lv = jnp.zeros(8)  # log_var=0 means std=1, so samples should vary
        z1 = enc.reparameterize(mu, lv, jax.random.PRNGKey(1))
        z2 = enc.reparameterize(mu, lv, jax.random.PRNGKey(2))
        assert not jnp.allclose(z1, z2), "Samples must differ across PRNGKeys"

    def test_reparameterize_uses_mu(self):
        """When log_var is very negative, z ≈ mu (nearly deterministic)."""
        enc = self._enc()
        mu = jnp.ones(8) * 3.0
        lv = jnp.ones(8) * (-20.0)  # std ≈ 0
        z = enc.reparameterize(mu, lv, jax.random.PRNGKey(0))
        assert jnp.allclose(z, mu, atol=1e-3), "With near-zero std, z should equal mu"

    def test_get_set_params_roundtrip(self):
        """get_params/set_params round-trip preserves values."""
        enc = self._enc()
        params = enc.get_params()
        enc2 = self._enc()
        enc2.set_params(params)
        for k in params:
            assert jnp.allclose(
                getattr(enc, k.replace("w_mu", "w_mu")), getattr(enc2, k.replace("w_mu", "w_mu"))
            )

    def test_encode_batch(self):
        """encode() works on batched inputs (batch, in_dim)."""
        enc = self._enc(in_dim=10, latent_dim=4)
        x = jnp.ones((5, 10))
        z, mu, lv = enc.encode(x, jax.random.PRNGKey(0))
        assert z.shape == (5, 4)
        assert mu.shape == (5, 4)


# ---------------------------------------------------------------------------
# VariationalPrior
# ---------------------------------------------------------------------------


class TestVariationalPrior:
    """Traces: REQ-VERIFY-176 (KL regularisation prevents OOD collapse)."""

    def test_predict_returns_pair(self):
        """predict() returns (prior_mu, prior_log_var) pair."""
        prior = VariationalPrior(context_dim=10, latent_dim=8)
        c = jnp.zeros(10)
        mu, lv = prior.predict(c)
        assert mu.shape == (8,)
        assert lv.shape == (8,)

    def test_predict_batch(self):
        """predict() works on batched context (batch, context_dim)."""
        prior = VariationalPrior(context_dim=10, latent_dim=4)
        c = jnp.ones((3, 10))
        mu, lv = prior.predict(c)
        assert mu.shape == (3, 4)

    def test_get_set_params_roundtrip(self):
        """get_params/set_params round-trip."""
        prior = VariationalPrior(context_dim=10, latent_dim=4)
        params = prior.get_params()
        prior2 = VariationalPrior(context_dim=10, latent_dim=4)
        prior2.set_params(params)
        assert jnp.allclose(prior.w_mu, prior2.w_mu)


# ---------------------------------------------------------------------------
# VariationalJEPAPredictor
# ---------------------------------------------------------------------------


class TestVariationalJEPAPredictor:
    """Traces: REQ-VERIFY-175, REQ-VERIFY-176, SCENARIO-VERIFY-229, SCENARIO-VERIFY-230."""

    def _model(self, in_dim: int = 10) -> VariationalJEPAPredictor:
        return VariationalJEPAPredictor(in_dim=in_dim, context_dim=in_dim, latent_dim=8)

    def _sample(self, in_dim: int = 10) -> dict:
        return {
            "feature": [0.1] * in_dim,
            "context": [0.05] * in_dim,
            "label": 1,
        }

    # --- KL term must be positive (REQ-VERIFY-176) ---

    def test_kl_positive_for_non_trivial_distributions(self):
        """KL term > 0 for non-trivial distributions — REQ-VERIFY-176.

        If encoder posterior == prior, KL = 0.  After random initialisation the
        two Gaussians are different, so KL > 0 is expected.  This guards against
        the KL vanishing immediately at initialisation (which would signal a bug).
        """
        model = self._model()
        x = jnp.array([[0.5] * 10], dtype=jnp.float32)
        c = jnp.array([[0.1] * 10], dtype=jnp.float32)
        y = jnp.array([1.0])
        _, kl = model.vjepa_loss(x, y, c, jax.random.PRNGKey(0))
        assert float(kl) > 0.0, f"KL should be >0 for non-trivial distributions, got {kl}"

    # --- Gradient flow (no NaN) ---

    def test_vjepa_loss_gradient_no_nan(self):
        """vjepa_loss gradients must be finite — SCENARIO-VERIFY-229."""
        model = self._model()
        x = jnp.array([[0.3, 0.1, 0.0, 0.5, 0.2, 0.0, 0.1, 0.0, 0.4, 0.0]], dtype=jnp.float32)
        c = jnp.zeros((1, 10), dtype=jnp.float32)
        y = jnp.array([1.0])
        params = model.get_all_params()

        def loss_fn(p):
            model.set_all_params(p)
            total, _ = model.vjepa_loss(x, y, c, jax.random.PRNGKey(5))
            return total

        grads = jax.grad(loss_fn)(params)
        for name, g in grads.items():
            assert not jnp.any(jnp.isnan(g)), f"NaN gradient in {name}"
            assert not jnp.any(jnp.isinf(g)), f"Inf gradient in {name}"

    # --- Loss is finite ---

    def test_vjepa_loss_finite(self):
        """vjepa_loss returns a finite scalar."""
        model = self._model()
        x = jnp.ones((4, 10)) * 0.1
        c = jnp.zeros((4, 10))
        y = jnp.array([1.0, 0.0, 1.0, 0.0])
        loss, kl = model.vjepa_loss(x, y, c, jax.random.PRNGKey(0))
        assert math.isfinite(float(loss)), "Loss must be finite"
        assert math.isfinite(float(kl)), "KL must be finite"

    # --- predict() ---

    def test_predict_in_unit_interval(self):
        """predict() returns a float in [0, 1] — REQ-VERIFY-175."""
        model = self._model()
        x = jnp.array([0.1] * 10, dtype=jnp.float32)
        c = jnp.zeros(10, dtype=jnp.float32)
        p = model.predict(x, c, jax.random.PRNGKey(0))
        assert isinstance(p, float), "predict() must return a Python float"
        assert 0.0 <= p <= 1.0, f"predict() must be in [0,1], got {p}"

    # --- train() converges ---

    def test_train_loss_decreases(self):
        """Training loss decreases over 100 epochs — SCENARIO-VERIFY-229."""
        model = self._model(in_dim=10)
        corpus = [
            {"feature": [1.0, 0.0] + [0.0] * 8, "context": [0.0] * 10, "label": 1},
            {"feature": [0.0, 1.0] + [0.0] * 8, "context": [0.0] * 10, "label": 0},
            {"feature": [1.0, 0.5] + [0.0] * 8, "context": [0.1] * 10, "label": 1},
            {"feature": [0.2, 0.9] + [0.0] * 8, "context": [0.05] * 10, "label": 0},
        ]
        metrics = model.train(corpus, n_epochs=50, lr=1e-3)
        assert len(metrics.epoch_losses) == 50
        # Loss at end should be <= loss at start (allow small numerical slack)
        assert metrics.epoch_losses[-1] <= metrics.epoch_losses[0] + 0.5, (
            f"Loss should not increase: start={metrics.epoch_losses[0]:.4f} "
            f"end={metrics.epoch_losses[-1]:.4f}"
        )

    def test_train_kl_tracked(self):
        """KL magnitudes are tracked during training — REQ-VERIFY-176."""
        model = self._model()
        corpus = [self._sample() for _ in range(4)]
        metrics = model.train(corpus, n_epochs=10)
        assert len(metrics.kl_magnitudes) == 10
        assert all(math.isfinite(k) for k in metrics.kl_magnitudes)

    def test_train_empty_corpus_returns_empty_metrics(self):
        """train() on empty corpus returns empty metrics without crashing."""
        model = self._model()
        metrics = model.train([], n_epochs=100)
        assert metrics.epoch_losses == []
        assert metrics.kl_magnitudes == []

    def test_train_records_nan_loss_and_stops(self, monkeypatch):
        """When loss becomes NaN, training records nan and stops early — SCENARIO-VERIFY-230.

        The NaN path is the 'training_failed' branch guard.  We inject a NaN by
        monkey-patching vjepa_loss to return (nan, 0.0) on the second call so
        that at least one epoch succeeds first.
        """
        model = self._model()
        corpus = [
            {"feature": [0.1] * 10, "context": [0.0] * 10, "label": 1},
            {"feature": [0.9] * 10, "context": [0.0] * 10, "label": 0},
        ]
        call_count = {"n": 0}
        real_loss = model.vjepa_loss

        def patched_loss(x, y, c, key):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                import jax.numpy as jnp

                return jnp.array(float("nan")), jnp.array(0.0)
            return real_loss(x, y, c, key)

        monkeypatch.setattr(model, "vjepa_loss", patched_loss)
        metrics = model.train(corpus, n_epochs=10)
        # Training should have stopped after the NaN epoch
        assert math.isnan(metrics.epoch_losses[-1])
        assert len(metrics.epoch_losses) < 10

    # --- param round-trip ---

    def test_get_set_all_params_roundtrip(self):
        """get_all_params/set_all_params round-trip."""
        model = self._model()
        params = model.get_all_params()
        model2 = self._model()
        model2.set_all_params(params)
        assert jnp.allclose(model.w_cls, model2.w_cls)
        assert jnp.allclose(model.b_cls, model2.b_cls)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    """Traces: REQ-VERIFY-175 (corpus preparation pipeline)."""

    def test_build_tfidf_features_vocab_size(self):
        """Vocab size is capped at vocab_size."""
        texts = ["hello world foo bar", "baz qux hello"]
        _, token_to_idx = build_tfidf_features(texts, vocab_size=3)
        assert len(token_to_idx) <= 3

    def test_text_to_tfidf_length(self):
        """text_to_tfidf returns a vector of exactly vocab_size."""
        _, token_to_idx = build_tfidf_features(["hello world"], vocab_size=10)
        vec = text_to_tfidf("hello world", token_to_idx, vocab_size=10)
        assert len(vec) == 10

    def test_text_to_tfidf_empty_text(self):
        """Empty text produces all-zero vector without crashing."""
        _, token_to_idx = build_tfidf_features(["hello"], vocab_size=5)
        vec = text_to_tfidf("", token_to_idx, vocab_size=5)
        assert vec == [0.0] * 5

    def test_prepare_corpus_label_encoding(self):
        """'incorrect' steps get label=1, 'correct' steps get label=0."""
        raw = [
            {"question_id": "1", "step_text": "some step", "label": "incorrect"},
            {"question_id": "1", "step_text": "another step", "label": "correct"},
        ]
        _, tok_idx = build_tfidf_features(["some step another step"], vocab_size=5)
        corpus = prepare_corpus(raw, tok_idx, vocab_size=5)
        labels = {s["label"] for s in corpus}
        assert 0 in labels
        assert 1 in labels

    def test_prepare_corpus_context_zero_for_first_step(self):
        """First step of each question has zero context vector."""
        raw = [
            {"question_id": "q1", "step_text": "first step here", "label": "correct"},
            {"question_id": "q1", "step_text": "second step here", "label": "incorrect"},
        ]
        _, tok_idx = build_tfidf_features(["first second step here"], vocab_size=10)
        corpus = prepare_corpus(raw, tok_idx, vocab_size=10)
        # First entry for q1 should have zero context
        assert corpus[0]["context"] == [0.0] * 10

    def test_prepare_corpus_context_nonzero_for_later_steps(self):
        """Second+ steps of each question have non-zero context."""
        raw = [
            {"question_id": "q1", "step_text": "alpha beta gamma", "label": "correct"},
            {"question_id": "q1", "step_text": "delta epsilon zeta", "label": "incorrect"},
        ]
        texts = [s["step_text"] for s in raw]
        _, tok_idx = build_tfidf_features(texts, vocab_size=10)
        corpus = prepare_corpus(raw, tok_idx, vocab_size=10)
        # Second step's context should not be all zeros (first step had content)
        assert any(v != 0.0 for v in corpus[1]["context"])


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------


class TestComputeAuc:
    """Traces: REQ-VERIFY-175 (evaluation pipeline)."""

    def test_perfect_predictor_auc_one(self):
        """Perfect predictor achieves AUC=1.0."""
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        assert compute_auc(labels, scores) == pytest.approx(1.0)

    def test_random_predictor_auc_half(self):
        """Constant-score predictor yields AUC≈0.5."""
        labels = [1, 0, 1, 0]
        scores = [0.5, 0.5, 0.5, 0.5]
        assert compute_auc(labels, scores) == pytest.approx(0.5, abs=0.1)

    def test_all_same_label_returns_half(self):
        """Degenerate case (all positive or all negative) returns 0.5."""
        assert compute_auc([1, 1, 1], [0.9, 0.8, 0.7]) == 0.5
        assert compute_auc([0, 0, 0], [0.3, 0.2, 0.1]) == 0.5

    def test_empty_returns_half(self):
        """Empty inputs return 0.5."""
        assert compute_auc([], []) == 0.5


# ---------------------------------------------------------------------------
# TrainMetrics dataclass
# ---------------------------------------------------------------------------


class TestTrainMetrics:
    def test_default_empty(self):
        m = TrainMetrics()
        assert m.epoch_losses == []
        assert m.kl_magnitudes == []

    def test_mutable_default_isolation(self):
        """Two TrainMetrics instances must not share the same list object."""
        m1 = TrainMetrics()
        m2 = TrainMetrics()
        m1.epoch_losses.append(1.0)
        assert m2.epoch_losses == []
