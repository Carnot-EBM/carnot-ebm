"""Tests for carnot.pipeline.otv_verifier — OTVVerifier and OTVVerificationToken.

100% coverage target for otv_verifier.py (the new code added in Exp 592).

Spec: REQ-VERIFY-120, SCENARIO-VERIFY-160, SCENARIO-VERIFY-161, SCENARIO-VERIFY-162
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.pipeline.otv_verifier import OTVVerificationToken, OTVVerifier


# ---------------------------------------------------------------------------
# OTVVerificationToken
# ---------------------------------------------------------------------------


class TestOTVVerificationToken:
    """REQ-VERIFY-120: OTVVerificationToken is a plain dataclass with the right fields."""

    def test_fields_present(self) -> None:
        # SCENARIO-VERIFY-160: dataclass holds token_logit, verification_score, is_correct_pred.
        tok = OTVVerificationToken(token_logit=1.5, verification_score=0.82, is_correct_pred=True)
        assert tok.token_logit == 1.5
        assert tok.verification_score == 0.82
        assert tok.is_correct_pred is True

    def test_false_pred(self) -> None:
        tok = OTVVerificationToken(token_logit=-2.0, verification_score=0.12, is_correct_pred=False)
        assert tok.is_correct_pred is False


# ---------------------------------------------------------------------------
# OTVVerifier.__init__
# ---------------------------------------------------------------------------


class TestOTVVerifierInit:
    """REQ-VERIFY-120: Constructor creates zero-weight linear layer of correct shape."""

    def test_default_embed_dim(self) -> None:
        v = OTVVerifier()
        assert v.embed_dim == 128
        assert v._W.shape == (1, 128)
        assert np.all(v._W == 0.0)

    def test_custom_embed_dim(self) -> None:
        v = OTVVerifier(embed_dim=64)
        assert v.embed_dim == 64
        assert v._W.shape == (1, 64)

    def test_bias_zero(self) -> None:
        v = OTVVerifier(embed_dim=32)
        assert v._b == 0.0


# ---------------------------------------------------------------------------
# OTVVerifier.score
# ---------------------------------------------------------------------------


class TestOTVVerifierScore:
    """REQ-VERIFY-120 SCENARIO-VERIFY-160: score() returns float in [0, 1]."""

    def test_zero_weights_returns_half(self) -> None:
        # Untrained model: W=0, b=0 → logit=0 → sigmoid=0.5.
        v = OTVVerifier(embed_dim=4)
        s = v.score(jnp.zeros((4,)))
        assert isinstance(s, float)
        assert abs(s - 0.5) < 1e-5

    def test_score_in_unit_interval(self) -> None:
        v = OTVVerifier(embed_dim=8)
        for _ in range(5):
            h = jnp.array(np.random.randn(8).astype(np.float32))
            s = v.score(h)
            assert 0.0 <= s <= 1.0

    def test_score_clamps_large_logit(self) -> None:
        # Force a huge positive logit — score should saturate near 1.0, not overflow.
        v = OTVVerifier(embed_dim=1)
        v._W = np.array([[1000.0]], dtype=np.float32)
        s = v.score(jnp.ones((1,)))
        assert s > 0.99

    def test_score_clamps_large_negative_logit(self) -> None:
        v = OTVVerifier(embed_dim=1)
        v._W = np.array([[-1000.0]], dtype=np.float32)
        s = v.score(jnp.ones((1,)))
        assert s < 0.01


# ---------------------------------------------------------------------------
# OTVVerifier.predict
# ---------------------------------------------------------------------------


class TestOTVVerifierPredict:
    """REQ-VERIFY-120 SCENARIO-VERIFY-161: predict() returns OTVVerificationToken."""

    def test_returns_token(self) -> None:
        v = OTVVerifier(embed_dim=4)
        tok = v.predict(jnp.zeros((4,)))
        assert isinstance(tok, OTVVerificationToken)

    def test_untrained_predicts_half(self) -> None:
        v = OTVVerifier(embed_dim=4)
        tok = v.predict(jnp.zeros((4,)))
        assert abs(tok.verification_score - 0.5) < 1e-5
        assert tok.token_logit == 0.0

    def test_is_correct_pred_threshold(self) -> None:
        v = OTVVerifier(embed_dim=1)
        # Force positive logit → is_correct_pred=True.
        v._W = np.array([[10.0]], dtype=np.float32)
        tok = v.predict(jnp.ones((1,)))
        assert tok.is_correct_pred is True

    def test_is_correct_pred_false(self) -> None:
        v = OTVVerifier(embed_dim=1)
        v._W = np.array([[-10.0]], dtype=np.float32)
        tok = v.predict(jnp.ones((1,)))
        assert tok.is_correct_pred is False

    def test_token_logit_matches_score(self) -> None:
        v = OTVVerifier(embed_dim=8)
        h = jnp.ones((8,))
        tok = v.predict(h)
        s = v.score(h)
        assert abs(tok.verification_score - s) < 1e-5


# ---------------------------------------------------------------------------
# OTVVerifier.train (CI-skip guarded)
# ---------------------------------------------------------------------------


class TestOTVVerifierTrain:
    """REQ-VERIFY-120 SCENARIO-VERIFY-162: train() skips in CI, runs in live mode."""

    def test_train_completes_in_ci(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # CARNOT_IS_CI=1: assert_live_or_ci_skip() returns without error so training runs.
        monkeypatch.setenv("CARNOT_IS_CI", "1")
        v = OTVVerifier(embed_dim=4)
        pairs = [
            (jnp.zeros((4,)), True),
            (jnp.ones((4,)), False),
        ]
        # Should not raise.
        v.train(pairs, n_epochs=5)

    def test_train_empty_pairs_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARNOT_IS_CI", "1")
        v = OTVVerifier(embed_dim=4)
        v.train([], n_epochs=10)
        # Empty pairs — weights stay zero.
        assert np.all(v._W == 0.0)

    def test_train_updates_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # CI=1 means assert_live_or_ci_skip passes; training executes.
        monkeypatch.setenv("CARNOT_IS_CI", "1")
        v = OTVVerifier(embed_dim=4)
        # zeros → correct, ones → incorrect
        pairs = [(jnp.zeros((4,)), True)] * 10 + [(jnp.ones((4,)), False)] * 10
        v.train(pairs, n_epochs=50)
        # After training, weights should be non-zero.
        assert not np.all(v._W == 0.0)

    def test_train_empty_pairs_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARNOT_IS_CI", "1")
        v = OTVVerifier(embed_dim=4)
        v.train([], n_epochs=10)
        assert np.all(v._W == 0.0)

    def test_trained_model_separates_classes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # After training on clearly separated classes, zeros should score higher than ones.
        monkeypatch.setenv("CARNOT_IS_CI", "1")
        v = OTVVerifier(embed_dim=8)
        correct_h = jnp.zeros((8,))
        incorrect_h = jnp.ones((8,))
        pairs = [(correct_h, True)] * 20 + [(incorrect_h, False)] * 20
        v.train(pairs, n_epochs=100)
        s_correct = v.score(correct_h)
        s_incorrect = v.score(incorrect_h)
        # Correct hidden state should get higher score (more likely correct).
        assert s_correct > s_incorrect


# ---------------------------------------------------------------------------
# Export check
# ---------------------------------------------------------------------------


class TestExports:
    """Verify OTVVerifier and OTVVerificationToken are exported from carnot.pipeline."""

    def test_exported_from_pipeline(self) -> None:
        from carnot.pipeline import OTVVerificationToken as T
        from carnot.pipeline import OTVVerifier as V

        assert V is OTVVerifier
        assert T is OTVVerificationToken
