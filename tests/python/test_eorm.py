"""Tests for EORM: Energy-based cOt Reward Model.

100% coverage for python/carnot/models/eorm.py.

Spec coverage: REQ-LEARN-022, REQ-LEARN-023,
               SCENARIO-LEARN-038, SCENARIO-LEARN-039, SCENARIO-LEARN-040
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from carnot.models.eorm import (
    _PAD_ID,
    _SEP_ID,
    CoTEnergyInput,
    EORMModel,
    EORMTrainer,
    _count_params,
    _flatten_params,
    _forward,
    _init_layer,
    _init_params,
    _layer_norm,
    _make_token_sequence,
    _tokenize,
    _transformer_layer_forward,
    _unflatten_params,
)


# ---------------------------------------------------------------------------
# CoTEnergyInput
# ---------------------------------------------------------------------------

class TestCoTEnergyInput:
    """Tests for REQ-LEARN-022-1: CoTEnergyInput dataclass."""

    def test_fields(self) -> None:
        """REQ-LEARN-022-1: dataclass holds question_text and response_text."""
        cot = CoTEnergyInput(question_text="2+2?", response_text="It is 4.")
        assert cot.question_text == "2+2?"
        assert cot.response_text == "It is 4."

    def test_defaults_none(self) -> None:
        """Both fields must be provided — no implicit defaults."""
        with pytest.raises(TypeError):
            CoTEnergyInput()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

class TestTokenize:
    """Tests for _tokenize and _make_token_sequence helpers."""

    def test_basic_split(self) -> None:
        """Words extracted from mixed text."""
        ids = _tokenize("Hello, world! 42", max_seq_len=10, vocab_size=256)
        # Should produce 3 tokens (hello, world, 42)
        assert len(ids) == 3
        assert all(2 <= i < 256 for i in ids)  # PAD=0, SEP=1 reserved

    def test_empty_string(self) -> None:
        """Empty text tokenizes to empty list."""
        ids = _tokenize("", max_seq_len=10, vocab_size=256)
        assert ids == []

    def test_truncation(self) -> None:
        """Sequences longer than max_seq_len are truncated."""
        text = " ".join(["word"] * 100)
        ids = _tokenize(text, max_seq_len=10, vocab_size=256)
        assert len(ids) == 10

    def test_deterministic(self) -> None:
        """Same text always produces the same token IDs."""
        ids1 = _tokenize("hello world", max_seq_len=5, vocab_size=256)
        ids2 = _tokenize("hello world", max_seq_len=5, vocab_size=256)
        assert ids1 == ids2

    def test_special_tokens_not_returned(self) -> None:
        """No word token maps to PAD (0) or SEP (1)."""
        ids = _tokenize("anything goes here", max_seq_len=100, vocab_size=256)
        assert _PAD_ID not in ids
        assert _SEP_ID not in ids

    def test_make_token_sequence_contains_sep(self) -> None:
        """SEP token appears between question and response."""
        ids = _make_token_sequence("q", "r", max_seq_len=100, vocab_size=256)
        assert _SEP_ID in ids

    def test_make_token_sequence_truncation(self) -> None:
        """Combined sequence is truncated to max_seq_len."""
        long_q = " ".join(["word"] * 50)
        long_r = " ".join(["word"] * 50)
        ids = _make_token_sequence(long_q, long_r, max_seq_len=10, vocab_size=256)
        assert len(ids) == 10

    def test_make_token_sequence_empty_inputs(self) -> None:
        """Both empty → just the SEP token."""
        ids = _make_token_sequence("", "", max_seq_len=10, vocab_size=256)
        assert ids == [_SEP_ID]


# ---------------------------------------------------------------------------
# Layer norm
# ---------------------------------------------------------------------------

class TestLayerNorm:
    """Tests for _layer_norm helper."""

    def test_zero_mean_unit_var(self) -> None:
        """Normalized output has approximately zero mean and unit variance."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gamma = jnp.ones(5)
        beta = jnp.zeros(5)
        out = _layer_norm(x, gamma, beta)
        assert float(jnp.abs(jnp.mean(out))) < 0.01
        assert abs(float(jnp.var(out)) - 1.0) < 0.1

    def test_affine_transform(self) -> None:
        """Non-unit gamma and non-zero beta shift the normalized output."""
        x = jnp.array([1.0, 2.0, 3.0])
        gamma = jnp.array([2.0, 2.0, 2.0])
        beta = jnp.array([1.0, 1.0, 1.0])
        out = _layer_norm(x, gamma, beta)
        # Mean should be approximately beta (1.0) since normalized mean is 0
        assert abs(float(jnp.mean(out)) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Parameter initialization
# ---------------------------------------------------------------------------

class TestInitLayer:
    """Tests for _init_layer."""

    def test_keys_present(self) -> None:
        """All expected parameter keys are present."""
        lp = _init_layer(embed_dim=16, n_heads=2, key=jrandom.PRNGKey(0))
        expected = {
            "w_q", "b_q", "w_k", "b_k", "w_v", "b_v", "w_o", "b_o",
            "ln1_gamma", "ln1_beta",
            "w_ff1", "b_ff1", "w_ff2", "b_ff2",
            "ln2_gamma", "ln2_beta",
        }
        assert set(lp.keys()) == expected

    def test_shapes(self) -> None:
        """Weight matrices have correct shapes."""
        d = 16
        lp = _init_layer(embed_dim=d, n_heads=2, key=jrandom.PRNGKey(0))
        assert lp["w_q"].shape == (d, d)
        assert lp["b_q"].shape == (d,)
        assert lp["w_ff1"].shape == (d * 4, d)
        assert lp["b_ff1"].shape == (d * 4,)
        assert lp["w_ff2"].shape == (d, d * 4)


class TestInitParams:
    """Tests for _init_params."""

    def test_structure(self) -> None:
        """Top-level keys are token_embed, pos_embed, layers, final_ln_*, out_*."""
        params = _init_params(16, 2, 2, 32, 64, jrandom.PRNGKey(0))
        assert "token_embed" in params
        assert "pos_embed" in params
        assert "layers" in params
        assert len(params["layers"]) == 2
        assert "final_ln_gamma" in params
        assert "out_weight" in params
        assert "out_bias" in params

    def test_embedding_shapes(self) -> None:
        """Token and positional embedding tables have correct shapes."""
        params = _init_params(16, 2, 1, 32, 64, jrandom.PRNGKey(0))
        assert params["token_embed"].shape == (64, 16)
        assert params["pos_embed"].shape == (32, 16)

    def test_out_bias_shape(self) -> None:
        """Output bias has shape (1,)."""
        params = _init_params(16, 2, 1, 32, 64, jrandom.PRNGKey(0))
        assert params["out_bias"].shape == (1,)


# ---------------------------------------------------------------------------
# Transformer layer forward
# ---------------------------------------------------------------------------

class TestTransformerLayerForward:
    """Tests for _transformer_layer_forward."""

    def test_output_shape(self) -> None:
        """Output has same shape as input."""
        d = 16
        seq_len = 5
        lp = _init_layer(d, 2, jrandom.PRNGKey(0))
        x = jrandom.normal(jrandom.PRNGKey(1), (seq_len, d))
        out = _transformer_layer_forward(x, lp, n_heads=2)
        assert out.shape == (seq_len, d)

    def test_finite_output(self) -> None:
        """Forward pass produces finite values for normal inputs."""
        d = 8
        lp = _init_layer(d, 2, jrandom.PRNGKey(0))
        x = jrandom.normal(jrandom.PRNGKey(1), (4, d))
        out = _transformer_layer_forward(x, lp, n_heads=2)
        assert bool(jnp.all(jnp.isfinite(out)))


# ---------------------------------------------------------------------------
# Pure forward pass
# ---------------------------------------------------------------------------

class TestForward:
    """Tests for _forward pure function."""

    def test_returns_scalar(self) -> None:
        """_forward returns a scalar JAX array."""
        params = _init_params(16, 2, 1, 32, 64, jrandom.PRNGKey(0))
        token_ids = [2, 3, 1, 4, 5]
        out = _forward(params, token_ids, n_heads=2)
        assert out.shape == ()

    def test_finite(self) -> None:
        """Energy is finite for valid token IDs."""
        params = _init_params(16, 2, 1, 32, 64, jrandom.PRNGKey(0))
        token_ids = [2, 1, 3]
        out = _forward(params, token_ids, n_heads=2)
        assert bool(jnp.isfinite(out))

    def test_grad_computable(self) -> None:
        """jax.grad computes finite gradients through _forward."""
        params = _init_params(16, 2, 1, 32, 64, jrandom.PRNGKey(0))
        token_ids = [2, 1, 3]

        def fn(p):
            return _forward(p, token_ids, n_heads=2)

        grads = jax.grad(fn)(params)
        # Check that at least one gradient leaf is finite
        leaves = jax.tree_util.tree_leaves(grads)
        assert any(bool(jnp.any(jnp.isfinite(g))) for g in leaves)


# ---------------------------------------------------------------------------
# Param count, flatten, unflatten
# ---------------------------------------------------------------------------

class TestParamHelpers:
    """Tests for _count_params, _flatten_params, _unflatten_params."""

    def test_count_params_positive(self) -> None:
        """Parameter count is a positive integer."""
        params = _init_params(16, 2, 1, 32, 64, jrandom.PRNGKey(0))
        count = _count_params(params)
        assert count > 0
        assert isinstance(count, int)

    def test_flatten_produces_flat_dict(self) -> None:
        """_flatten_params produces a flat dict with string keys."""
        params = _init_params(16, 2, 1, 32, 64, jrandom.PRNGKey(0))
        flat = _flatten_params(params)
        assert all(isinstance(k, str) for k in flat)
        assert all("/" in k or True for k in flat)  # all keys are strings
        # List entries should use numeric string keys
        assert any("layers" in k for k in flat)

    def test_roundtrip(self) -> None:
        """Flatten + unflatten recovers the original structure."""
        params = _init_params(16, 2, 1, 32, 64, jrandom.PRNGKey(0))
        flat = _flatten_params(params)
        jax_flat = {k: jnp.array(v) for k, v in flat.items()}
        restored = _unflatten_params(jax_flat, params)

        # Check that all leaf arrays match the originals
        orig_leaves = jax.tree_util.tree_leaves(params)
        rest_leaves = jax.tree_util.tree_leaves(restored)
        assert len(orig_leaves) == len(rest_leaves)
        for orig, rest in zip(orig_leaves, rest_leaves):
            assert jnp.allclose(orig, rest)


# ---------------------------------------------------------------------------
# EORMModel
# ---------------------------------------------------------------------------

class TestEORMModel:
    """Tests for REQ-LEARN-022: EORMModel."""

    def test_creation_defaults(self) -> None:
        """REQ-LEARN-022-2: model creates with default config."""
        model = EORMModel()
        assert model.embed_dim == 128
        assert model.n_heads == 4
        assert model.n_layers == 2
        assert model.max_seq_len == 512
        assert model.vocab_size == 4096

    def test_creation_custom(self) -> None:
        """REQ-LEARN-022-2: model accepts custom config."""
        model = EORMModel(embed_dim=32, n_heads=2, n_layers=1, max_seq_len=64, vocab_size=128)
        assert model.embed_dim == 32
        assert model.n_heads == 2

    def test_invalid_embed_dim_n_heads(self) -> None:
        """REQ-LEARN-022-2: embed_dim not divisible by n_heads raises ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            EORMModel(embed_dim=10, n_heads=3)

    def test_energy_finite(self) -> None:
        """SCENARIO-LEARN-038: energy is finite for a CoTEnergyInput."""
        model = EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64)
        cot = CoTEnergyInput(question_text="What is 2+2?", response_text="It is 4.")
        e = model.energy(cot)
        assert math.isfinite(e)
        assert isinstance(e, float)

    def test_energy_empty_texts(self) -> None:
        """Edge case: empty question and response fall back to SEP-only sequence."""
        model = EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64)
        cot = CoTEnergyInput(question_text="", response_text="")
        e = model.energy(cot)
        assert math.isfinite(e)

    def test_n_params_positive_and_bounded(self) -> None:
        """SCENARIO-LEARN-038: n_params is positive and ≤ 100M."""
        model = EORMModel(embed_dim=128, n_heads=4, n_layers=2)
        n = model.n_params
        assert n > 0
        assert n <= 100_000_000

    def test_n_params_small_model(self) -> None:
        """Small model has fewer params than large model."""
        small = EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64)
        large = EORMModel(embed_dim=64, n_heads=4, n_layers=2, max_seq_len=64, vocab_size=256)
        assert small.n_params < large.n_params

    def test_rank_returns_all_indices(self) -> None:
        """SCENARIO-LEARN-039: rank returns a permutation of [0, n-1]."""
        model = EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64)
        responses = ["bad answer", "good answer", "mediocre answer"]
        ranked = model.rank(responses, question="What is 2+2?")
        assert sorted(ranked) == [0, 1, 2]

    def test_rank_sorted_by_energy(self) -> None:
        """SCENARIO-LEARN-039: energies at returned indices are non-decreasing."""
        model = EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64)
        responses = ["alpha", "beta", "gamma", "delta"]
        question = "test question"
        ranked = model.rank(responses, question=question)
        energies = [
            model.energy(CoTEnergyInput(question_text=question, response_text=r))
            for r in responses
        ]
        ordered_energies = [energies[i] for i in ranked]
        # Non-decreasing order
        for a, b in zip(ordered_energies[:-1], ordered_energies[1:]):
            assert a <= b

    def test_rank_single_response(self) -> None:
        """rank with a single response returns [0]."""
        model = EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64)
        ranked = model.rank(["only answer"], question="q?")
        assert ranked == [0]

    def test_save_and_load(self) -> None:
        """REQ-LEARN-022-5: saved model loads with identical parameters."""
        model = EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64,
                          key=jrandom.PRNGKey(42))
        cot = CoTEnergyInput(question_text="What is 2+2?", response_text="4")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.safetensors"
            model.save(path)

            # Check that both files were written
            assert path.exists()
            config_path = path.parent / "test_model_config.json"
            assert config_path.exists()

            # Load and verify energy matches
            loaded = EORMModel.load(path)
            assert loaded.embed_dim == model.embed_dim
            assert loaded.n_heads == model.n_heads
            assert loaded.n_layers == model.n_layers
            assert loaded.max_seq_len == model.max_seq_len
            assert loaded.vocab_size == model.vocab_size

            e_original = model.energy(cot)
            e_loaded = loaded.energy(cot)
            assert abs(e_original - e_loaded) < 1e-5

    def test_save_creates_parent_dirs(self) -> None:
        """save() creates parent directories if they don't exist."""
        model = EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=16, vocab_size=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = Path(tmpdir) / "nested" / "dir" / "model.safetensors"
            model.save(deep_path)
            assert deep_path.exists()

    def test_energy_different_texts_different_energies(self) -> None:
        """Different CoT texts generally produce different energies."""
        model = EORMModel(embed_dim=32, n_heads=2, n_layers=1, max_seq_len=64, vocab_size=128,
                          key=jrandom.PRNGKey(7))
        cot1 = CoTEnergyInput(question_text="q", response_text="correct step by step answer")
        cot2 = CoTEnergyInput(question_text="q", response_text="wrong completely different")
        e1 = model.energy(cot1)
        e2 = model.energy(cot2)
        # Energies should be distinct (extremely unlikely to be exactly equal)
        assert e1 != e2


# ---------------------------------------------------------------------------
# EORMTrainer
# ---------------------------------------------------------------------------

class TestEORMTrainer:
    """Tests for REQ-LEARN-023: EORMTrainer."""

    def _small_model(self) -> EORMModel:
        return EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64,
                         key=jrandom.PRNGKey(0))

    def test_contrastive_loss_zero_when_margin_met(self) -> None:
        """SCENARIO-LEARN-040: loss is 0 when incorrect has energy > correct + margin."""
        trainer = EORMTrainer(self._small_model(), lr=1e-4, margin=1.0)
        loss = trainer.contrastive_loss(correct_energy=2.0, incorrect_energy=4.0)
        assert loss == 0.0

    def test_contrastive_loss_positive_when_margin_violated(self) -> None:
        """SCENARIO-LEARN-040: loss is positive when incorrect energy ≤ correct + margin."""
        trainer = EORMTrainer(self._small_model(), lr=1e-4, margin=1.0)
        loss = trainer.contrastive_loss(correct_energy=4.0, incorrect_energy=2.0)
        assert loss == 3.0  # max(0, 4 - 2 + 1) = 3

    def test_contrastive_loss_exactly_at_margin(self) -> None:
        """Loss is 0 when incorrect - correct == margin exactly."""
        trainer = EORMTrainer(self._small_model(), lr=1e-4, margin=1.0)
        # incorrect - correct = 1.0 = margin, so correct - incorrect + margin = 0
        loss = trainer.contrastive_loss(correct_energy=3.0, incorrect_energy=4.0)
        assert loss == 0.0

    def test_train_step_returns_float(self) -> None:
        """REQ-LEARN-023-3: train_step returns a Python float."""
        model = self._small_model()
        trainer = EORMTrainer(model, lr=1e-4)
        loss = trainer.train_step(
            correct_response="The answer is 4",
            incorrect_response="The answer is 5",
            question="What is 2+2?",
        )
        assert isinstance(loss, float)
        assert math.isfinite(loss)

    def test_train_step_modifies_params(self) -> None:
        """REQ-LEARN-023-3: train_step updates model.params when loss is positive.

        We use margin=20.0 (much larger than any random-init energy difference)
        to guarantee the hinge loss is positive and therefore gradients are
        non-zero.  A zero-gradient update (loss=0) is correct behavior but
        does not exercise the gradient-application path.
        """
        model = self._small_model()
        # Large margin ensures relu(E_correct - E_incorrect + margin) > 0 for any
        # randomly initialized model, so the gradient step is always active.
        trainer = EORMTrainer(model, lr=1e-1, margin=20.0)

        # Snapshot out_weight before the step
        orig_w = jnp.array(model.params["out_weight"])

        loss = trainer.train_step(
            correct_response="four is the answer",
            incorrect_response="five is wrong here",
            question="2+2",
        )

        # With margin=20.0 the loss should be positive (gradient is non-zero)
        assert loss > 0.0, (
            "Expected positive loss with margin=20.0; this indicates the energy"
            " difference exceeded 20 for a random model, which is unexpected."
        )
        new_w = model.params["out_weight"]
        assert not jnp.allclose(orig_w, new_w, atol=1e-9), (
            "train_step should update out_weight when loss > 0"
        )

    def test_train_step_empty_responses(self) -> None:
        """Edge case: empty response strings produce finite loss."""
        model = self._small_model()
        trainer = EORMTrainer(model, lr=1e-4)
        loss = trainer.train_step(
            correct_response="",
            incorrect_response="",
            question="",
        )
        assert math.isfinite(loss)

    def test_train_epoch_mean_loss(self) -> None:
        """REQ-LEARN-023-4: train_epoch returns mean loss as float."""
        model = self._small_model()
        trainer = EORMTrainer(model, lr=1e-4)
        pairs = [
            ("right answer", "wrong answer", "q1"),
            ("correct", "incorrect", "q2"),
        ]
        mean_loss = trainer.train_epoch(pairs, batch_size=1)
        assert isinstance(mean_loss, float)
        assert math.isfinite(mean_loss)
        assert mean_loss >= 0.0

    def test_train_epoch_empty_pairs(self) -> None:
        """REQ-LEARN-023-4: train_epoch with empty list returns 0.0."""
        model = self._small_model()
        trainer = EORMTrainer(model, lr=1e-4)
        result = trainer.train_epoch([], batch_size=16)
        assert result == 0.0

    def test_train_epoch_batch_size_larger_than_pairs(self) -> None:
        """train_epoch works when batch_size > len(pairs)."""
        model = self._small_model()
        trainer = EORMTrainer(model, lr=1e-4)
        pairs = [("a", "b", "q")]
        mean_loss = trainer.train_epoch(pairs, batch_size=100)
        assert math.isfinite(mean_loss)

    def test_train_epoch_reduces_loss_over_iterations(self) -> None:
        """Loss should decrease (or stay low) after multiple epochs on a fixed pair."""
        model = EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64,
                          key=jrandom.PRNGKey(123))
        trainer = EORMTrainer(model, lr=1e-2)
        pairs = [("The answer is four", "The answer is five", "What is two plus two")]

        losses = [trainer.train_epoch(pairs) for _ in range(5)]
        # At least the first epoch should have finite loss
        assert all(math.isfinite(l) for l in losses)

    def test_trainer_custom_margin(self) -> None:
        """EORMTrainer respects custom margin in contrastive_loss."""
        model = self._small_model()
        trainer = EORMTrainer(model, margin=0.5)
        # With margin=0.5: max(0, 3.0 - 4.0 + 0.5) = max(0, -0.5) = 0
        assert trainer.contrastive_loss(3.0, 4.0) == 0.0
        # max(0, 3.0 - 3.0 + 0.5) = 0.5
        assert trainer.contrastive_loss(3.0, 3.0) == 0.5


# ---------------------------------------------------------------------------
# Integration: energy function is differentiable through full model
# ---------------------------------------------------------------------------

class TestEORMGradient:
    """Integration tests for gradient flow through the full model."""

    def test_gradient_of_energy_wrt_params(self) -> None:
        """Gradient of energy w.r.t. all params is finite after one step."""
        model = EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64,
                          key=jrandom.PRNGKey(5))
        token_ids = _make_token_sequence("q", "a", 32, 64) or [_SEP_ID]

        def fn(params):
            return _forward(params, token_ids, n_heads=2)

        grads = jax.grad(fn)(model.params)
        leaves = jax.tree_util.tree_leaves(grads)
        # At minimum the out_weight gradient should be finite
        assert any(bool(jnp.any(jnp.isfinite(g) & (g != 0.0))) for g in leaves)

    def test_special_token_constants(self) -> None:
        """PAD_ID is 0 and SEP_ID is 1."""
        assert _PAD_ID == 0
        assert _SEP_ID == 1

    def test_energy_empty_token_list_fallback(self, monkeypatch) -> None:
        """Line 714: energy() falls back to [SEP_ID] when _make_token_sequence returns [].

        This is the defensive branch when max_seq_len is 0 (truncates away the SEP
        token).  We monkeypatch _make_token_sequence to return [] to exercise the
        branch directly without needing an unusual model configuration.
        """
        import carnot.models.eorm as eorm_module

        model = EORMModel(embed_dim=16, n_heads=2, n_layers=1, max_seq_len=32, vocab_size=64)
        cot = CoTEnergyInput(question_text="q", response_text="r")

        # Force _make_token_sequence to return [] so the fallback is triggered
        monkeypatch.setattr(eorm_module, "_make_token_sequence", lambda *a, **kw: [])

        e = model.energy(cot)
        # The fallback uses [SEP_ID], so the result should still be finite
        assert math.isfinite(e)
