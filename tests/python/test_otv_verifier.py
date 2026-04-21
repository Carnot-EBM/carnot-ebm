"""Tests for carnot.models.otv_verifier.

Spec: REQ-VERIFY-145, REQ-VERIFY-145-1..5,
      SCENARIO-VERIFY-192, SCENARIO-VERIFY-193
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from carnot.models.otv_verifier import (
    OTVTrainer,
    OTVVerificationHead,
    _bce_loss_batch,
    _forward_from_params,
    _params_from_head,
    _params_to_head,
)


# ---------------------------------------------------------------------------
# REQ-VERIFY-145-1: Weight shapes
# ---------------------------------------------------------------------------


class TestOTVVerificationHeadInit:
    """REQ-VERIFY-145-1: zero-initialised weights with correct shapes."""

    def test_default_shapes(self):
        head = OTVVerificationHead()
        assert head.W1.shape == (128, 64)
        assert head.W2.shape == (64, 1)
        assert head.b1.shape == (64,)
        assert head.b2.shape == (1,)

    def test_custom_dims(self):
        head = OTVVerificationHead(input_dim=32, hidden_dim=16)
        assert head.W1.shape == (32, 16)
        assert head.W2.shape == (16, 1)

    def test_zero_init(self):
        head = OTVVerificationHead()
        assert float(jnp.sum(jnp.abs(head.W1))) == 0.0
        assert float(jnp.sum(jnp.abs(head.W2))) == 0.0
        assert float(jnp.sum(jnp.abs(head.b1))) == 0.0
        assert float(jnp.sum(jnp.abs(head.b2))) == 0.0


# ---------------------------------------------------------------------------
# REQ-VERIFY-145-2 / SCENARIO-VERIFY-192: forward pass output
# ---------------------------------------------------------------------------


class TestOTVVerificationHeadForward:
    """REQ-VERIFY-145-2 / SCENARIO-VERIFY-192."""

    def test_zero_weights_zero_input_gives_half(self):
        # SCENARIO-VERIFY-192: sigmoid(0) = 0.5
        head = OTVVerificationHead()
        result = head.forward(jnp.zeros(128))
        assert isinstance(result, float)
        assert abs(result - 0.5) < 1e-5

    def test_output_in_unit_interval(self):
        head = OTVVerificationHead()
        for val in [-1.0, 0.0, 1.0, 100.0]:
            x = jnp.full((128,), val)
            out = head.forward(x)
            assert 0.0 <= out <= 1.0

    def test_returns_float(self):
        head = OTVVerificationHead()
        result = head.forward(jnp.ones(128))
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# REQ-VERIFY-145-3: feature_vector
# ---------------------------------------------------------------------------


class TestOTVVerificationHeadFeatureVector:
    """REQ-VERIFY-145-3: feature_vector output shape and semantics."""

    def test_shape_128(self):
        head = OTVVerificationHead()
        fv = head.feature_vector("hello world 42")
        assert fv.shape == (128,)

    def test_non_negative(self):
        head = OTVVerificationHead()
        fv = head.feature_vector("The answer is 42.")
        assert float(jnp.min(fv)) >= 0.0

    def test_length_feature(self):
        head = OTVVerificationHead()
        short_fv = head.feature_vector("hi")
        long_fv = head.feature_vector("x " * 500)
        # First feature is length/1000; longer response should score higher.
        assert float(long_fv[0]) > float(short_fv[0])

    def test_digit_density_feature(self):
        head = OTVVerificationHead()
        math_fv = head.feature_vector("1 2 3 4 5")
        text_fv = head.feature_vector("the cat sat on the mat")
        # Third feature is digit density; math response scores higher.
        assert float(math_fv[2]) > float(text_fv[2])

    def test_operator_density_feature(self):
        head = OTVVerificationHead()
        op_fv = head.feature_vector("3 + 4 = 7")
        no_op_fv = head.feature_vector("three four seven")
        # Fourth feature is operator density.
        assert float(op_fv[3]) > float(no_op_fv[3])

    def test_empty_response(self):
        head = OTVVerificationHead()
        fv = head.feature_vector("")
        assert fv.shape == (128,)
        assert float(jnp.min(fv)) >= 0.0

    def test_custom_input_dim(self):
        head = OTVVerificationHead(input_dim=32)
        fv = head.feature_vector("hello 42")
        assert fv.shape == (32,)


# ---------------------------------------------------------------------------
# REQ-VERIFY-145-4 / SCENARIO-VERIFY-193: training
# ---------------------------------------------------------------------------


class TestOTVTrainer:
    """REQ-VERIFY-145-4 / SCENARIO-VERIFY-193: trainer updates weights."""

    def _make_pairs(self) -> list[dict]:
        # Correct responses: long math with digits and operators.
        # Incorrect responses: short non-math text.
        correct = [
            {
                "response": f"Step 1: multiply {i} by 3 = {i*3}. Step 2: add 5 = {i*3+5}. Answer: {i*3+5}",
                "is_correct": True,
            }
            for i in range(1, 21)
        ]
        incorrect = [
            {"response": f"The answer is {i}.", "is_correct": False}
            for i in range(1, 21)
        ]
        return correct + incorrect

    def test_train_returns_head(self):
        # REQ-VERIFY-145-4: train() returns an OTVVerificationHead.
        head = OTVVerificationHead()
        trainer = OTVTrainer(head)
        result = trainer.train(self._make_pairs(), n_epochs=2)
        assert isinstance(result, OTVVerificationHead)
        assert result is head  # Same object, updated in place.

    def test_weights_change_after_training(self):
        # The trainer warm-starts from small random values when all weights
        # are zero, so W1 will be non-zero after training.
        head = OTVVerificationHead()
        trainer = OTVTrainer(head, lr=0.01)
        trainer.train(self._make_pairs(), n_epochs=5)
        w1_after = float(jnp.sum(jnp.abs(head.W1)))
        # Weights must have moved from zero (via warm start + gradient update).
        assert w1_after > 0.0

    def test_auc_above_random_after_training(self):
        # SCENARIO-VERIFY-193: AUC > 0.5 on training set.
        pairs = self._make_pairs()
        head = OTVVerificationHead()
        trainer = OTVTrainer(head, lr=0.05)
        trained = trainer.train(pairs, n_epochs=50)

        scores = [trained.forward(trained.feature_vector(p["response"])) for p in pairs]
        labels = [int(p["is_correct"]) for p in pairs]

        # Manual AUC calculation.
        paired = sorted(zip(scores, labels), key=lambda t: -t[0])
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        tp = fp = 0
        prev_fpr = prev_tpr = 0.0
        auc = 0.0
        for _s, lbl in paired:
            if lbl == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            prev_fpr, prev_tpr = fpr, tpr

        assert auc > 0.5, f"AUC {auc:.3f} <= 0.5 (random baseline)"

    def test_custom_lr(self):
        head = OTVVerificationHead()
        trainer = OTVTrainer(head, lr=0.1)
        assert trainer.lr == 0.1

    def test_single_pair(self):
        head = OTVVerificationHead()
        trainer = OTVTrainer(head)
        result = trainer.train([{"response": "42", "is_correct": True}], n_epochs=1)
        assert result is head


# ---------------------------------------------------------------------------
# Internal helpers coverage
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    """Cover _params_from_head, _params_to_head, _forward_from_params, _bce_loss_batch."""

    def test_params_from_head_keys(self):
        head = OTVVerificationHead()
        params = _params_from_head(head)
        assert set(params.keys()) == {"W1", "W2", "b1", "b2"}

    def test_params_roundtrip(self):
        head = OTVVerificationHead()
        head.W1 = head.W1 + 1.0
        params = _params_from_head(head)
        head2 = OTVVerificationHead()
        _params_to_head(head2, params)
        assert float(jnp.sum(jnp.abs(head2.W1 - head.W1))) < 1e-6

    def test_forward_from_params_shape(self):
        head = OTVVerificationHead()
        params = _params_from_head(head)
        x = jnp.zeros(128)
        out = _forward_from_params(params, x)
        assert out.shape == (1,)

    def test_bce_loss_batch_scalar(self):
        head = OTVVerificationHead()
        params = _params_from_head(head)
        xs = jnp.zeros((4, 128))
        labels = jnp.array([1.0, 0.0, 1.0, 0.0])
        loss = _bce_loss_batch(params, xs, labels)
        assert loss.shape == ()
        assert float(loss) > 0.0

    def test_bce_loss_batch_perfect_prediction(self):
        # All positives predicted at 0.5 (zero weights) — loss should be log(2).
        head = OTVVerificationHead()
        params = _params_from_head(head)
        xs = jnp.zeros((2, 128))
        labels = jnp.array([1.0, 1.0])
        loss = _bce_loss_batch(params, xs, labels)
        expected = -math.log(0.5 + 1e-7)
        assert abs(float(loss) - expected) < 1e-4


# ---------------------------------------------------------------------------
# REQ-VERIFY-145-5: Export from carnot.models
# ---------------------------------------------------------------------------


class TestExportFromCarnotModels:
    """REQ-VERIFY-145-5: OTVVerificationHead and OTVTrainer in carnot.models."""

    def test_importable_from_carnot_models(self):
        from carnot.models import OTVTrainer as T
        from carnot.models import OTVVerificationHead as H

        assert H is OTVVerificationHead
        assert T is OTVTrainer
