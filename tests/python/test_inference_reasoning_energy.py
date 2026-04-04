"""Tests for EBM-CoT reasoning energy verification.

Spec coverage: REQ-INFER-011, SCENARIO-INFER-012
"""

from __future__ import annotations

import jax.numpy as jnp

from carnot.inference.reasoning_energy import (
    ReasoningEnergyResult,
    ReasoningVerifierConfig,
    compute_reasoning_energy,
    generate_reasoning_training_data,
    refine_reasoning,
    text_to_reasoning_embedding,
    train_reasoning_energy,
    verify_reasoning_chain,
)
from carnot.models.gibbs import GibbsConfig, GibbsModel


class TestTextToReasoningEmbedding:
    """Tests for embedding conversion."""

    def test_correct_shape(self) -> None:
        """REQ-INFER-011: returns vector of vocab_size."""
        emb = text_to_reasoning_embedding("2 + 3 = 5", vocab_size=128)
        assert emb.shape == (128,)

    def test_deterministic(self) -> None:
        """REQ-INFER-011: same text → same embedding."""
        e1 = text_to_reasoning_embedding("test")
        e2 = text_to_reasoning_embedding("test")
        assert jnp.allclose(e1, e2)

    def test_different_text_different_embedding(self) -> None:
        """REQ-INFER-011: different text → different embedding."""
        e1 = text_to_reasoning_embedding("2 + 3 = 5")
        e2 = text_to_reasoning_embedding("7 * 8 = 56")
        assert not jnp.allclose(e1, e2)

    def test_handles_empty(self) -> None:
        """REQ-INFER-011: empty text produces valid embedding."""
        emb = text_to_reasoning_embedding("")
        assert emb.shape == (256,)

    def test_handles_malformed_token(self) -> None:
        """REQ-INFER-011: malformed text doesn't crash (TokenError path)."""
        # Unterminated string triggers TokenError
        emb = text_to_reasoning_embedding("'unterminated string")
        assert emb.shape == (256,)


class TestComputeReasoningEnergy:
    """Tests for energy computation."""

    def test_finite(self) -> None:
        """REQ-INFER-011: energy is finite."""
        model = GibbsModel(GibbsConfig(input_dim=256, hidden_dims=[32]))
        emb = text_to_reasoning_embedding("2 + 3 = 5")
        e = compute_reasoning_energy(model, emb)
        assert jnp.isfinite(e)


class TestRefineReasoning:
    """Tests for Langevin refinement."""

    def test_returns_same_shape(self) -> None:
        """REQ-INFER-011: refined embedding has same shape."""
        model = GibbsModel(GibbsConfig(input_dim=64, hidden_dims=[16]))
        x = jnp.ones(64) * 0.5
        refined = refine_reasoning(x, model, n_langevin_steps=3)
        assert refined.shape == (64,)

    def test_changes_embedding(self) -> None:
        """REQ-INFER-011: refinement modifies the embedding."""
        model = GibbsModel(GibbsConfig(input_dim=64, hidden_dims=[16]))
        x = jnp.ones(64) * 0.5
        refined = refine_reasoning(x, model, n_langevin_steps=5, step_size=0.1)
        assert not jnp.allclose(x, refined)


class TestGenerateTrainingData:
    """Tests for training data generation."""

    def test_correct_shapes(self) -> None:
        """REQ-INFER-011: both batches have (n_samples, vocab_size) shape."""
        coherent, incoherent = generate_reasoning_training_data(20, vocab_size=128)
        assert coherent.shape == (20, 128)
        assert incoherent.shape == (20, 128)

    def test_deterministic(self) -> None:
        """REQ-INFER-011: same seed → same data."""
        c1, _ = generate_reasoning_training_data(10, seed=42)
        c2, _ = generate_reasoning_training_data(10, seed=42)
        assert jnp.allclose(c1, c2)


class TestTrainReasoningEnergy:
    """Tests for NCE training."""

    def test_returns_model(self) -> None:
        """REQ-INFER-011: returns a GibbsModel."""
        config = ReasoningVerifierConfig(
            vocab_size=64, hidden_dims=[16], n_epochs=3, n_training_samples=10
        )
        coherent, incoherent = generate_reasoning_training_data(10, vocab_size=64)
        model = train_reasoning_energy(coherent, incoherent, config)
        assert isinstance(model, GibbsModel)

    def test_default_config(self) -> None:
        """SCENARIO-INFER-012: works with default config."""
        coherent, incoherent = generate_reasoning_training_data(10, vocab_size=256)
        model = train_reasoning_energy(
            coherent, incoherent, ReasoningVerifierConfig(n_epochs=2, n_training_samples=10)
        )
        assert model.input_dim == 256

    def test_none_config(self) -> None:
        """REQ-INFER-011: config=None uses defaults."""
        coherent, incoherent = generate_reasoning_training_data(5, vocab_size=256)
        model = train_reasoning_energy(coherent, incoherent, None)
        assert model.input_dim == 256


class TestVerifyReasoningChain:
    """Tests for full verification pipeline."""

    def test_returns_result(self) -> None:
        """SCENARIO-INFER-012: returns ReasoningEnergyResult."""
        model = GibbsModel(GibbsConfig(input_dim=256, hidden_dims=[32]))
        result = verify_reasoning_chain("2 + 3 = 5", model)
        assert isinstance(result, ReasoningEnergyResult)
        assert jnp.isfinite(result.chain_energy)

    def test_result_defaults(self) -> None:
        """REQ-INFER-011: default result values."""
        r = ReasoningEnergyResult()
        assert r.chain_energy == 0.0
        assert r.is_coherent is True

    def test_config_defaults(self) -> None:
        """REQ-INFER-011: config default values."""
        c = ReasoningVerifierConfig()
        assert c.vocab_size == 256
        assert c.n_epochs == 50
