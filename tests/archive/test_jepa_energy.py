"""Tests for EB-JEPA context-prediction energy function.

Spec coverage: REQ-JEPA-001, SCENARIO-JEPA-001 through SCENARIO-JEPA-005
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from carnot.embeddings.jepa_energy import (
    ContextPredictionEnergy,
    JEPAEnergyConfig,
    embedding_repair,
    generate_jepa_training_data,
    nce_loss,
    nearest_code_match,
    train_jepa_energy,
)


# --- Sample code snippets for training data generation ---
# These are real Python functions with enough structure for meaningful
# AST embeddings when split in half.
SAMPLE_SNIPPETS = [
    """\
def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b
""",
    """\
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
""",
    """\
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
""",
    """\
def factorial(n):
    if n < 0:
        raise ValueError("negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
""",
    """\
def is_palindrome(s):
    cleaned = s.lower().strip()
    left = 0
    right = len(cleaned) - 1
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    return True
""",
]


class TestJEPAEnergyConfig:
    """Tests for SCENARIO-JEPA-001: Configuration validation."""

    def test_default_values(self) -> None:
        """SCENARIO-JEPA-001: Default config has sensible values."""
        config = JEPAEnergyConfig()
        assert config.embed_dim == 64
        assert config.hidden_dims == [64, 32]
        assert config.activation == "silu"

    def test_custom_values(self) -> None:
        """SCENARIO-JEPA-001: Custom config values are respected."""
        config = JEPAEnergyConfig(embed_dim=32, hidden_dims=[16, 8], activation="relu")
        assert config.embed_dim == 32
        assert config.hidden_dims == [16, 8]
        assert config.activation == "relu"

    def test_validate_embed_dim_positive(self) -> None:
        """SCENARIO-JEPA-001: embed_dim must be > 0."""
        config = JEPAEnergyConfig(embed_dim=0)
        with pytest.raises(ValueError, match="embed_dim must be > 0"):
            config.validate()

    def test_validate_embed_dim_negative(self) -> None:
        """SCENARIO-JEPA-001: Negative embed_dim is rejected."""
        config = JEPAEnergyConfig(embed_dim=-5)
        with pytest.raises(ValueError, match="embed_dim must be > 0"):
            config.validate()

    def test_validate_empty_hidden_dims(self) -> None:
        """SCENARIO-JEPA-001: Empty hidden_dims is rejected."""
        config = JEPAEnergyConfig(hidden_dims=[])
        with pytest.raises(ValueError, match="hidden_dims must have at least one layer"):
            config.validate()

    def test_validate_negative_hidden_dim(self) -> None:
        """SCENARIO-JEPA-001: Negative values in hidden_dims are rejected."""
        config = JEPAEnergyConfig(hidden_dims=[64, -1])
        with pytest.raises(ValueError, match="all hidden_dims must be > 0"):
            config.validate()

    def test_validate_zero_hidden_dim(self) -> None:
        """SCENARIO-JEPA-001: Zero values in hidden_dims are rejected."""
        config = JEPAEnergyConfig(hidden_dims=[0])
        with pytest.raises(ValueError, match="all hidden_dims must be > 0"):
            config.validate()

    def test_validate_bad_activation(self) -> None:
        """SCENARIO-JEPA-001: Unknown activation is rejected."""
        config = JEPAEnergyConfig(activation="gelu")
        with pytest.raises(ValueError, match="Unknown activation"):
            config.validate()

    def test_validate_passes_for_valid_config(self) -> None:
        """SCENARIO-JEPA-001: Validation passes for valid config."""
        config = JEPAEnergyConfig()
        config.validate()  # should not raise

    def test_validate_all_activations(self) -> None:
        """SCENARIO-JEPA-001: All supported activations pass validation."""
        for act in ("silu", "relu", "tanh"):
            config = JEPAEnergyConfig(activation=act)
            config.validate()


class TestContextPredictionEnergy:
    """Tests for REQ-JEPA-001: ContextPredictionEnergy model."""

    def test_energy_returns_scalar(self) -> None:
        """SCENARIO-JEPA-002: Energy output is a scalar."""
        model = ContextPredictionEnergy(JEPAEnergyConfig(embed_dim=8, hidden_dims=[8]))
        x = jnp.ones(16)
        e = model.energy(x)
        assert e.shape == ()

    def test_energy_batch(self) -> None:
        """SCENARIO-JEPA-002: Batched energy returns one scalar per sample."""
        model = ContextPredictionEnergy(JEPAEnergyConfig(embed_dim=8, hidden_dims=[8]))
        xs = jnp.ones((5, 16))
        energies = model.energy_batch(xs)
        assert energies.shape == (5,)

    def test_grad_energy_shape(self) -> None:
        """SCENARIO-JEPA-002: Gradient has same shape as input."""
        model = ContextPredictionEnergy(JEPAEnergyConfig(embed_dim=8, hidden_dims=[8]))
        x = jnp.ones(16)
        g = model.grad_energy(x)
        assert g.shape == x.shape

    def test_input_dim(self) -> None:
        """SCENARIO-JEPA-002: input_dim is 2 * embed_dim."""
        model = ContextPredictionEnergy(JEPAEnergyConfig(embed_dim=32))
        assert model.input_dim == 64

    def test_energy_pair_convenience(self) -> None:
        """SCENARIO-JEPA-002: energy_pair gives same result as manual concat."""
        model = ContextPredictionEnergy(JEPAEnergyConfig(embed_dim=8, hidden_dims=[8]))
        ctx = jnp.ones(8) * 0.5
        pred = jnp.ones(8) * 0.3
        e_pair = model.energy_pair(ctx, pred)
        e_manual = model.energy(jnp.concatenate([ctx, pred]))
        np.testing.assert_allclose(float(e_pair), float(e_manual), atol=1e-6)

    def test_initial_energy_is_zero(self) -> None:
        """SCENARIO-JEPA-002: Output layer is zero-initialized, so initial energy = 0."""
        model = ContextPredictionEnergy(JEPAEnergyConfig(embed_dim=8, hidden_dims=[8]))
        x = jnp.ones(16)
        e = model.energy(x)
        np.testing.assert_allclose(float(e), 0.0, atol=1e-6)

    def test_default_key_deterministic(self) -> None:
        """SCENARIO-JEPA-002: Default key=None gives reproducible initialization."""
        m1 = ContextPredictionEnergy(JEPAEnergyConfig(embed_dim=8, hidden_dims=[8]))
        m2 = ContextPredictionEnergy(JEPAEnergyConfig(embed_dim=8, hidden_dims=[8]))
        for (w1, b1), (w2, b2) in zip(m1.layers, m2.layers):
            np.testing.assert_array_equal(np.array(w1), np.array(w2))
            np.testing.assert_array_equal(np.array(b1), np.array(b2))

    def test_custom_key_different_init(self) -> None:
        """SCENARIO-JEPA-002: Different PRNG keys produce different weights."""
        m1 = ContextPredictionEnergy(
            JEPAEnergyConfig(embed_dim=8, hidden_dims=[8]), key=jrandom.PRNGKey(0)
        )
        m2 = ContextPredictionEnergy(
            JEPAEnergyConfig(embed_dim=8, hidden_dims=[8]), key=jrandom.PRNGKey(999)
        )
        # At least one layer should differ
        w1 = np.array(m1.layers[0][0])
        w2 = np.array(m2.layers[0][0])
        assert not np.allclose(w1, w2)

    def test_multiple_hidden_layers(self) -> None:
        """SCENARIO-JEPA-002: Model works with multiple hidden layers."""
        config = JEPAEnergyConfig(embed_dim=8, hidden_dims=[16, 8, 4])
        model = ContextPredictionEnergy(config)
        assert len(model.layers) == 3
        x = jnp.ones(16)
        e = model.energy(x)
        assert e.shape == ()

    def test_relu_activation(self) -> None:
        """SCENARIO-JEPA-002: Model works with relu activation."""
        config = JEPAEnergyConfig(embed_dim=8, hidden_dims=[8], activation="relu")
        model = ContextPredictionEnergy(config)
        x = jnp.ones(16)
        e = model.energy(x)
        assert e.shape == ()

    def test_tanh_activation(self) -> None:
        """SCENARIO-JEPA-002: Model works with tanh activation."""
        config = JEPAEnergyConfig(embed_dim=8, hidden_dims=[8], activation="tanh")
        model = ContextPredictionEnergy(config)
        x = jnp.ones(16)
        e = model.energy(x)
        assert e.shape == ()


class TestNCELoss:
    """Tests for SCENARIO-JEPA-003: NCE loss computation."""

    def test_nce_loss_is_scalar(self) -> None:
        """SCENARIO-JEPA-003: NCE loss returns a scalar."""
        model = ContextPredictionEnergy(JEPAEnergyConfig(embed_dim=4, hidden_dims=[4]))
        data = jnp.ones((3, 8))
        noise = jnp.ones((3, 8)) * 2.0
        loss = nce_loss(model, data, noise)
        assert loss.shape == ()

    def test_nce_loss_positive(self) -> None:
        """SCENARIO-JEPA-003: NCE loss is positive (sum of -log(sigmoid) terms)."""
        model = ContextPredictionEnergy(JEPAEnergyConfig(embed_dim=4, hidden_dims=[4]))
        data = jnp.ones((3, 8))
        noise = jnp.ones((3, 8)) * 2.0
        loss = nce_loss(model, data, noise)
        assert float(loss) > 0.0

    def test_nce_loss_initial_value(self) -> None:
        """SCENARIO-JEPA-003: With zero-initialized output, all energies = 0,
        so NCE loss = -log(sigmoid(0)) - log(sigmoid(0)) = 2 * log(2)."""
        model = ContextPredictionEnergy(JEPAEnergyConfig(embed_dim=4, hidden_dims=[4]))
        data = jnp.ones((3, 8))
        noise = jnp.zeros((3, 8))
        loss = nce_loss(model, data, noise)
        # sigmoid(0) = 0.5, -log(0.5) = log(2) ~= 0.693
        expected = 2.0 * np.log(2.0)
        np.testing.assert_allclose(float(loss), expected, atol=1e-5)


class TestGenerateJEPATrainingData:
    """Tests for SCENARIO-JEPA-004: Training data generation."""

    def test_output_shapes(self) -> None:
        """SCENARIO-JEPA-004: Correct output shapes for data and noise pairs."""
        data_pairs, noise_pairs = generate_jepa_training_data(
            SAMPLE_SNIPPETS, embed_dim=32
        )
        n = len(SAMPLE_SNIPPETS)
        assert data_pairs.shape == (n, 64)  # 2 * embed_dim
        assert noise_pairs.shape == (n, 64)

    def test_data_pairs_differ_from_noise(self) -> None:
        """SCENARIO-JEPA-004: Data pairs and noise pairs are different."""
        data_pairs, noise_pairs = generate_jepa_training_data(
            SAMPLE_SNIPPETS, embed_dim=32
        )
        # The second halves should be shuffled in noise, so pairs differ
        assert not np.allclose(np.array(data_pairs), np.array(noise_pairs))

    def test_context_halves_same(self) -> None:
        """SCENARIO-JEPA-004: Context halves are the same in data and noise pairs."""
        data_pairs, noise_pairs = generate_jepa_training_data(
            SAMPLE_SNIPPETS, embed_dim=32
        )
        # First embed_dim columns are context — should be identical
        np.testing.assert_array_equal(
            np.array(data_pairs[:, :32]),
            np.array(noise_pairs[:, :32]),
        )

    def test_prediction_halves_shuffled(self) -> None:
        """SCENARIO-JEPA-004: Prediction halves are shuffled in noise pairs."""
        data_pairs, noise_pairs = generate_jepa_training_data(
            SAMPLE_SNIPPETS, embed_dim=32
        )
        # Second embed_dim columns are prediction — should differ due to shuffle
        data_preds = np.array(data_pairs[:, 32:])
        noise_preds = np.array(noise_pairs[:, 32:])
        assert not np.allclose(data_preds, noise_preds)

    def test_reproducible_with_same_key(self) -> None:
        """SCENARIO-JEPA-004: Same key produces same data."""
        key = jrandom.PRNGKey(123)
        d1, n1 = generate_jepa_training_data(SAMPLE_SNIPPETS[:3], embed_dim=16, key=key)
        d2, n2 = generate_jepa_training_data(SAMPLE_SNIPPETS[:3], embed_dim=16, key=key)
        np.testing.assert_array_equal(np.array(d1), np.array(d2))
        np.testing.assert_array_equal(np.array(n1), np.array(n2))

    def test_default_key(self) -> None:
        """SCENARIO-JEPA-004: Default key=None uses seed 42 deterministically."""
        d1, n1 = generate_jepa_training_data(SAMPLE_SNIPPETS[:3], embed_dim=16)
        d2, n2 = generate_jepa_training_data(SAMPLE_SNIPPETS[:3], embed_dim=16)
        np.testing.assert_array_equal(np.array(d1), np.array(d2))

    def test_single_line_snippet(self) -> None:
        """SCENARIO-JEPA-004: Single-line snippets produce valid output."""
        snippets = ["x = 1", "y = 2", "z = 3"]
        data, noise = generate_jepa_training_data(snippets, embed_dim=16)
        assert data.shape == (3, 32)
        assert noise.shape == (3, 32)


class TestTrainJEPAEnergy:
    """Tests for SCENARIO-JEPA-005: Training reduces NCE loss and learns coherence."""

    def test_training_reduces_loss(self) -> None:
        """SCENARIO-JEPA-005: NCE loss decreases over training steps."""
        data_pairs, noise_pairs = generate_jepa_training_data(
            SAMPLE_SNIPPETS, embed_dim=16
        )
        model = ContextPredictionEnergy(
            JEPAEnergyConfig(embed_dim=16, hidden_dims=[16, 8]),
            key=jrandom.PRNGKey(0),
        )

        loss_history = train_jepa_energy(
            model, data_pairs, noise_pairs,
            learning_rate=0.05, n_steps=50,
        )

        # Loss should decrease: final loss < initial loss
        assert loss_history[-1] < loss_history[0], (
            f"Loss did not decrease: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}"
        )

    def test_correct_pairs_lower_energy_than_noise(self) -> None:
        """SCENARIO-JEPA-005: After training, correct pairs get lower energy than noise."""
        data_pairs, noise_pairs = generate_jepa_training_data(
            SAMPLE_SNIPPETS, embed_dim=16
        )
        model = ContextPredictionEnergy(
            JEPAEnergyConfig(embed_dim=16, hidden_dims=[16, 8]),
            key=jrandom.PRNGKey(0),
        )

        train_jepa_energy(
            model, data_pairs, noise_pairs,
            learning_rate=0.05, n_steps=200,
        )

        data_energies = model.energy_batch(data_pairs)
        noise_energies = model.energy_batch(noise_pairs)

        mean_data_energy = float(jnp.mean(data_energies))
        mean_noise_energy = float(jnp.mean(noise_energies))

        assert mean_data_energy < mean_noise_energy, (
            f"Data energy ({mean_data_energy:.4f}) should be lower than "
            f"noise energy ({mean_noise_energy:.4f})"
        )

    def test_loss_history_length(self) -> None:
        """SCENARIO-JEPA-005: Loss history has one entry per training step."""
        data_pairs, noise_pairs = generate_jepa_training_data(
            SAMPLE_SNIPPETS[:3], embed_dim=8
        )
        model = ContextPredictionEnergy(
            JEPAEnergyConfig(embed_dim=8, hidden_dims=[8])
        )
        history = train_jepa_energy(
            model, data_pairs, noise_pairs, n_steps=10
        )
        assert len(history) == 10

    def test_loss_history_all_finite(self) -> None:
        """SCENARIO-JEPA-005: All loss values are finite (no NaN/Inf)."""
        data_pairs, noise_pairs = generate_jepa_training_data(
            SAMPLE_SNIPPETS[:3], embed_dim=8
        )
        model = ContextPredictionEnergy(
            JEPAEnergyConfig(embed_dim=8, hidden_dims=[8])
        )
        history = train_jepa_energy(
            model, data_pairs, noise_pairs, n_steps=20
        )
        assert all(np.isfinite(v) for v in history)

    def test_model_parameters_change_after_training(self) -> None:
        """SCENARIO-JEPA-005: Training modifies model parameters."""
        data_pairs, noise_pairs = generate_jepa_training_data(
            SAMPLE_SNIPPETS[:3], embed_dim=8
        )
        model = ContextPredictionEnergy(
            JEPAEnergyConfig(embed_dim=8, hidden_dims=[8])
        )

        # Record initial output weight (starts at zeros)
        ow_before = np.array(model.output_weight).copy()

        train_jepa_energy(
            model, data_pairs, noise_pairs, n_steps=10, learning_rate=0.05
        )

        ow_after = np.array(model.output_weight)
        assert not np.allclose(ow_before, ow_after), "Output weight should change"


class TestEmbeddingRepair:
    """Tests for REQ-JEPA-002, SCENARIO-JEPA-006: Embedding repair via gradient descent."""

    def _trained_model(self) -> tuple[ContextPredictionEnergy, jax.Array, jax.Array]:
        """Helper: train a small JEPA model and return (model, data_pairs, noise_pairs).

        **For engineers:**
            Builds a small model (embed_dim=16), trains it for 200 steps so it
            assigns lower energy to correct pairs than noise pairs. Returns the
            model and training data so tests can construct ctx/pred embeddings.
        """
        data_pairs, noise_pairs = generate_jepa_training_data(
            SAMPLE_SNIPPETS, embed_dim=16
        )
        model = ContextPredictionEnergy(
            JEPAEnergyConfig(embed_dim=16, hidden_dims=[16, 8]),
            key=jrandom.PRNGKey(0),
        )
        train_jepa_energy(
            model, data_pairs, noise_pairs,
            learning_rate=0.05, n_steps=200,
        )
        return model, data_pairs, noise_pairs

    def test_repair_reduces_energy(self) -> None:
        """SCENARIO-JEPA-006: Repairing a bad prediction reduces the energy."""
        model, data_pairs, noise_pairs = self._trained_model()
        embed_dim = 16

        # Take the first noise pair — a mismatched (context, prediction)
        ctx_emb = noise_pairs[0, :embed_dim]
        bad_pred = noise_pairs[0, embed_dim:]

        energy_before = float(model.energy_pair(ctx_emb, bad_pred))

        repaired = embedding_repair(
            ctx_emb, bad_pred, model, steps=100, step_size=0.05
        )

        energy_after = float(model.energy_pair(ctx_emb, repaired))

        assert energy_after < energy_before, (
            f"Repair should reduce energy: {energy_before:.4f} -> {energy_after:.4f}"
        )

    def test_repair_preserves_shape(self) -> None:
        """SCENARIO-JEPA-006: Repaired embedding has the same shape as the input."""
        model = ContextPredictionEnergy(
            JEPAEnergyConfig(embed_dim=8, hidden_dims=[8]),
            key=jrandom.PRNGKey(0),
        )
        ctx = jnp.ones(8)
        pred = jnp.ones(8) * 3.0

        repaired = embedding_repair(ctx, pred, model, steps=5, step_size=0.01)
        assert repaired.shape == pred.shape

    def test_repair_zero_steps_returns_original(self) -> None:
        """SCENARIO-JEPA-006: With steps=0, the prediction is returned unchanged."""
        model = ContextPredictionEnergy(
            JEPAEnergyConfig(embed_dim=8, hidden_dims=[8]),
            key=jrandom.PRNGKey(0),
        )
        ctx = jnp.ones(8)
        pred = jnp.ones(8) * 2.0

        repaired = embedding_repair(ctx, pred, model, steps=0, step_size=0.01)
        np.testing.assert_array_equal(np.array(repaired), np.array(pred))

    def test_repair_more_steps_lower_energy(self) -> None:
        """SCENARIO-JEPA-006: More repair steps yield lower (or equal) energy."""
        model, _data, noise_pairs = self._trained_model()
        embed_dim = 16
        ctx_emb = noise_pairs[0, :embed_dim]
        bad_pred = noise_pairs[0, embed_dim:]

        repaired_10 = embedding_repair(ctx_emb, bad_pred, model, steps=10, step_size=0.05)
        repaired_100 = embedding_repair(ctx_emb, bad_pred, model, steps=100, step_size=0.05)

        e10 = float(model.energy_pair(ctx_emb, repaired_10))
        e100 = float(model.energy_pair(ctx_emb, repaired_100))

        assert e100 <= e10, (
            f"100 steps ({e100:.4f}) should be <= 10 steps ({e10:.4f})"
        )


class TestNearestCodeMatch:
    """Tests for REQ-JEPA-002, SCENARIO-JEPA-007: Nearest codebook match."""

    def test_exact_match(self) -> None:
        """SCENARIO-JEPA-007: An embedding that matches a codebook entry exactly is found."""
        codebook_embs = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        codebook_texts = ["code_x", "code_y", "code_z"]

        # Query is exactly the second entry
        result = nearest_code_match(jnp.array([0.0, 1.0, 0.0]), codebook_embs, codebook_texts)
        assert result == "code_y"

    def test_closest_by_cosine(self) -> None:
        """SCENARIO-JEPA-007: Query closest to one entry by cosine similarity returns it."""
        codebook_embs = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        codebook_texts = ["code_x", "code_y", "code_z"]

        # Mostly in the z-direction, should match "code_z"
        query = jnp.array([0.1, 0.1, 5.0])
        result = nearest_code_match(query, codebook_embs, codebook_texts)
        assert result == "code_z"

    def test_magnitude_invariant(self) -> None:
        """SCENARIO-JEPA-007: Cosine similarity is invariant to embedding magnitude."""
        codebook_embs = jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        codebook_texts = ["horizontal", "vertical"]

        # Large-magnitude vector pointing in x-direction
        result = nearest_code_match(jnp.array([100.0, 0.01]), codebook_embs, codebook_texts)
        assert result == "horizontal"

    def test_single_codebook_entry(self) -> None:
        """SCENARIO-JEPA-007: Works with a single codebook entry."""
        codebook_embs = jnp.array([[1.0, 2.0, 3.0]])
        codebook_texts = ["only_one"]

        result = nearest_code_match(jnp.array([0.5, 1.0, 1.5]), codebook_embs, codebook_texts)
        assert result == "only_one"

    def test_near_zero_embedding_handled(self) -> None:
        """SCENARIO-JEPA-007: Near-zero query doesn't crash (epsilon prevents div-by-zero)."""
        codebook_embs = jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        codebook_texts = ["a", "b"]

        # Near-zero query — should not raise, just return some result
        result = nearest_code_match(jnp.array([1e-20, 1e-20]), codebook_embs, codebook_texts)
        assert result in codebook_texts


class TestPackageExports:
    """Tests for package-level imports."""

    def test_jepa_exports_from_embeddings(self) -> None:
        """SCENARIO-JEPA-001: JEPA symbols are accessible from carnot.embeddings."""
        from carnot.embeddings import (
            ContextPredictionEnergy as PkgModel,
            JEPAEnergyConfig as PkgConfig,
            embedding_repair as pkg_repair,
            generate_jepa_training_data as pkg_gen,
            nce_loss as pkg_nce,
            nearest_code_match as pkg_nearest,
            train_jepa_energy as pkg_train,
        )

        assert PkgModel is ContextPredictionEnergy
        assert PkgConfig is JEPAEnergyConfig
        assert pkg_gen is generate_jepa_training_data
        assert pkg_nce is nce_loss
        assert pkg_train is train_jepa_energy
        assert pkg_repair is embedding_repair
        assert pkg_nearest is nearest_code_match
