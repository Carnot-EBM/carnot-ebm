"""Tests for layer-wise activation extraction and hallucination statistics.

Spec coverage: REQ-INFER-014
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.embeddings.activation_extractor import (
    ActivationConfig,
    _activation_entropy,
    _cosine_similarity,
    _find_transformer_layers,
    compute_activation_stats,
    extract_layer_activations,
    jax_softmax,
)


class TestActivationConfig:
    """Tests for REQ-INFER-014: Activation extraction configuration."""

    def test_default_model_name(self) -> None:
        """REQ-INFER-014: Default model is Qwen3-0.6B."""
        config = ActivationConfig()
        assert config.model_name == "Qwen/Qwen3-0.6B"

    def test_default_device(self) -> None:
        """REQ-INFER-014: Default device is CPU."""
        config = ActivationConfig()
        assert config.device == "cpu"

    def test_custom_config(self) -> None:
        """REQ-INFER-014: Custom config values are respected."""
        config = ActivationConfig(model_name="custom/model", device="cuda")
        assert config.model_name == "custom/model"
        assert config.device == "cuda"


class TestExtractLayerActivationsFallback:
    """Tests for REQ-INFER-014: Graceful fallback when dependencies missing."""

    def test_returns_none_when_transformers_missing(self) -> None:
        """REQ-INFER-014: Returns None when transformers not installed."""
        import sys

        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            result = extract_layer_activations("hello world")

        assert result is None

    def test_returns_none_with_explicit_config(self) -> None:
        """REQ-INFER-014: Returns None with explicit config when deps missing."""
        import sys

        config = ActivationConfig(model_name="some/model")
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            result = extract_layer_activations("hello", config=config)

        assert result is None


class TestExtractLayerActivationsWithMock:
    """Tests for REQ-INFER-014: Activation extraction with mocked transformer."""

    def _build_mocks(
        self, num_layers: int = 3, hidden_dim: int = 16, seq_len: int = 4
    ):
        """Build mock tokenizer, model, and torch module with hookable layers."""
        mock_torch = MagicMock()

        # Create fake layers that support register_forward_hook.
        # When the model forward is called, we simulate the hooks firing.
        fake_layers = []
        hook_callbacks: list[tuple[int, MagicMock]] = []

        for i in range(num_layers):
            layer = MagicMock()
            # Track registered hooks so we can fire them during forward pass.
            registered_hooks = []

            def make_register(layer_hooks):
                def register_forward_hook(fn):
                    layer_hooks.append(fn)
                    handle = MagicMock()
                    handle.remove = MagicMock()
                    return handle

                return register_forward_hook

            layer.register_forward_hook = make_register(registered_hooks)
            fake_layers.append((layer, registered_hooks))

        # Build the mock model with model.layers attribute path.
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None

        # Set up model.model.layers (LLaMA/Qwen path).
        layer_modules = [fl[0] for fl in fake_layers]
        mock_model.model.layers = layer_modules

        # When forward is called, fire all registered hooks with fake activations.
        def fake_forward(**kwargs):
            for i, (layer_mod, hooks) in enumerate(fake_layers):
                # Create a fake output tensor for this layer.
                fake_hidden = np.random.randn(1, seq_len, hidden_dim).astype(
                    np.float32
                )

                class FakeOutput:
                    def __init__(self, data):
                        self._data = data

                    def detach(self):
                        return self

                    def cpu(self):
                        return self

                    def numpy(self):
                        return self._data

                output_tensor = FakeOutput(fake_hidden)
                # Fire each hook registered on this layer.
                for hook_fn in hooks:
                    hook_fn(layer_mod, None, (output_tensor,))

            return MagicMock()

        mock_model.__call__ = fake_forward
        mock_model.side_effect = fake_forward

        # Mock torch.no_grad context manager.
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        # Mock tokenizer.
        mock_tokenizer = MagicMock()

        class FakeTensor:
            def __init__(self, data):
                self._data = data

            def to(self, device):
                return self

        mock_tokenizer.return_value = {
            "input_ids": FakeTensor(np.zeros((1, seq_len), dtype=np.int64)),
            "attention_mask": FakeTensor(np.ones((1, seq_len), dtype=np.int64)),
        }

        # Mock transformers module.
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        return mock_torch, mock_transformers, num_layers, hidden_dim, seq_len

    def test_returns_dict_with_layer_keys(self) -> None:
        """REQ-INFER-014: Returns a dict keyed by layer index."""
        mock_torch, mock_transformers, num_layers, _, _ = self._build_mocks()

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = extract_layer_activations("test text")

        assert result is not None
        assert isinstance(result, dict)
        assert set(result.keys()) == {0, 1, 2}

    def test_activation_shapes(self) -> None:
        """REQ-INFER-014: Each activation has shape (seq_len, hidden_dim)."""
        mock_torch, mock_transformers, _, hidden_dim, seq_len = self._build_mocks(
            num_layers=2, hidden_dim=8, seq_len=5
        )

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = extract_layer_activations("test")

        assert result is not None
        for layer_idx, act in result.items():
            assert isinstance(act, jnp.ndarray)
            assert act.shape == (seq_len, hidden_dim)

    def test_default_config_used(self) -> None:
        """REQ-INFER-014: Default config is used when config=None."""
        mock_torch, mock_transformers, _, _, _ = self._build_mocks()

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = extract_layer_activations("code", config=None)

        mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "Qwen/Qwen3-0.6B"
        )
        assert result is not None

    def test_custom_config_model_name(self) -> None:
        """REQ-INFER-014: Custom model name is passed through."""
        mock_torch, mock_transformers, _, _, _ = self._build_mocks()
        config = ActivationConfig(model_name="custom/llm")

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = extract_layer_activations("code", config=config)

        mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "custom/llm"
        )
        assert result is not None

    def test_returns_none_when_layers_not_found(self) -> None:
        """REQ-INFER-014: Returns None when model architecture is unrecognized."""
        mock_torch, mock_transformers, _, _, _ = self._build_mocks()
        # Replace the model with one that has no recognizable layer path.
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        # Remove all known layer attribute paths so _find_transformer_layers returns None.
        del mock_model.model
        del mock_model.transformer
        del mock_model.encoder
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = extract_layer_activations("test")

        assert result is None

    def test_hook_handles_non_tuple_output(self) -> None:
        """REQ-INFER-014: Hook handles layers that return tensor directly (not tuple)."""
        mock_torch, mock_transformers, _, hidden_dim, seq_len = self._build_mocks(
            num_layers=1, hidden_dim=4, seq_len=2
        )

        # Override the model's forward to fire hook with a non-tuple output.
        mock_model = mock_transformers.AutoModel.from_pretrained.return_value
        original_layers = mock_model.model.layers

        def fake_forward_non_tuple(**kwargs):
            layer = original_layers[0]
            # Get the registered hooks from the layer.
            # We need to call register_forward_hook to register, then the
            # hook should already be registered by extract_layer_activations.
            # Instead, access the hooks that were registered.
            fake_hidden = np.ones((1, seq_len, hidden_dim), dtype=np.float32)

            class FakeDirectOutput:
                """Mimics a layer output that is a tensor, not a tuple."""

                def detach(self):
                    return self

                def cpu(self):
                    return self

                def numpy(self):
                    return fake_hidden

            # Fire the hooks with a direct tensor output (not wrapped in tuple).
            for hook_fn in layer.register_forward_hook._registered:
                hook_fn(layer, None, FakeDirectOutput())

            return MagicMock()

        # Track registered hooks on the layer.
        registered = []
        original_register = original_layers[0].register_forward_hook

        def tracking_register(fn):
            registered.append(fn)
            tracking_register._registered = registered
            handle = MagicMock()
            handle.remove = MagicMock()
            return handle

        tracking_register._registered = registered
        original_layers[0].register_forward_hook = tracking_register

        def fake_forward_v2(**kwargs):
            fake_hidden = np.ones((1, seq_len, hidden_dim), dtype=np.float32)

            class FakeDirectOutput:
                def detach(self):
                    return self

                def cpu(self):
                    return self

                def numpy(self):
                    return fake_hidden

            for hook_fn in registered:
                hook_fn(original_layers[0], None, FakeDirectOutput())
            return MagicMock()

        mock_model.side_effect = fake_forward_v2

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = extract_layer_activations("test")

        assert result is not None
        assert 0 in result
        # Should be all ones since we used np.ones.
        np.testing.assert_allclose(
            np.array(result[0]), np.ones((seq_len, hidden_dim)), atol=1e-6
        )


class TestFindTransformerLayers:
    """Tests for REQ-INFER-014: Transformer layer discovery."""

    def test_finds_model_layers_path(self) -> None:
        """REQ-INFER-014: Finds layers at model.layers (LLaMA/Qwen path)."""
        mock = MagicMock()
        fake_layers = [MagicMock(), MagicMock()]
        mock.model.layers = fake_layers
        result = _find_transformer_layers(mock)
        assert result == fake_layers

    def test_finds_transformer_h_path(self) -> None:
        """REQ-INFER-014: Finds layers at transformer.h (GPT-2 path)."""
        mock = MagicMock(spec=[])
        mock.transformer = MagicMock(spec=[])
        mock.transformer.h = [MagicMock(), MagicMock()]
        # Ensure model.layers path doesn't exist.
        result = _find_transformer_layers(mock)
        assert result is not None
        assert len(result) == 2

    def test_finds_encoder_layer_path(self) -> None:
        """REQ-INFER-014: Finds layers at encoder.layer (BERT path)."""
        mock = MagicMock(spec=[])
        mock.encoder = MagicMock(spec=[])
        mock.encoder.layer = [MagicMock()]
        result = _find_transformer_layers(mock)
        assert result is not None
        assert len(result) == 1

    def test_returns_none_for_unknown_architecture(self) -> None:
        """REQ-INFER-014: Returns None when no known layer path matches."""
        mock = MagicMock(spec=[])
        result = _find_transformer_layers(mock)
        assert result is None


class TestComputeActivationStats:
    """Tests for REQ-INFER-014: Per-layer activation statistics."""

    def test_single_layer_stats(self) -> None:
        """REQ-INFER-014: Stats computed correctly for a single layer."""
        # Create a known activation: all ones, shape (4, 8).
        act = jnp.ones((4, 8))
        activations = {0: act}

        stats = compute_activation_stats(activations)

        assert 0 in stats
        s = stats[0]
        assert "norm" in s
        assert "direction_change" in s
        assert "entropy" in s
        # For all-ones input, the mean vector is all-ones.
        # L2 norm = sqrt(8) ≈ 2.828
        np.testing.assert_allclose(s["norm"], np.sqrt(8), atol=1e-5)
        # First layer has direction_change = 0.0 by convention.
        assert s["direction_change"] == 0.0
        # Entropy should be a valid non-negative number.
        assert s["entropy"] >= 0.0

    def test_multi_layer_direction_change(self) -> None:
        """REQ-INFER-014: Direction change is nonzero between different layers."""
        # Layer 0: all ones. Layer 1: different direction.
        act0 = jnp.ones((4, 8))
        act1 = jnp.array(np.random.randn(4, 8).astype(np.float32))
        activations = {0: act0, 1: act1}

        stats = compute_activation_stats(activations)

        assert stats[0]["direction_change"] == 0.0
        # Direction change should be nonzero for different activations.
        assert stats[1]["direction_change"] != 0.0

    def test_identical_layers_zero_direction_change(self) -> None:
        """REQ-INFER-014: Identical consecutive layers have ~0 direction change."""
        act = jnp.ones((4, 8))
        activations = {0: act, 1: act}

        stats = compute_activation_stats(activations)

        np.testing.assert_allclose(stats[1]["direction_change"], 0.0, atol=1e-6)

    def test_orthogonal_layers_direction_change(self) -> None:
        """REQ-INFER-014: Orthogonal layers have direction change ~1.0."""
        # Create two orthogonal mean vectors.
        act0 = jnp.array([[1.0, 0.0, 0.0, 0.0]])
        act1 = jnp.array([[0.0, 1.0, 0.0, 0.0]])
        activations = {0: act0, 1: act1}

        stats = compute_activation_stats(activations)

        np.testing.assert_allclose(stats[1]["direction_change"], 1.0, atol=1e-5)

    def test_entropy_uniform_vs_peaked(self) -> None:
        """REQ-INFER-014: Uniform activations have higher entropy than peaked."""
        # Uniform: all dimensions equal magnitude.
        uniform = jnp.ones((1, 32))
        # Peaked: one dimension dominates.
        peaked = jnp.zeros((1, 32)).at[0, 0].set(100.0)

        stats_uniform = compute_activation_stats({0: uniform})
        stats_peaked = compute_activation_stats({0: peaked})

        assert stats_uniform[0]["entropy"] > stats_peaked[0]["entropy"]

    def test_empty_activations(self) -> None:
        """REQ-INFER-014: Empty dict returns empty stats."""
        stats = compute_activation_stats({})
        assert stats == {}

    def test_unsorted_layer_indices(self) -> None:
        """REQ-INFER-014: Stats are computed correctly for non-contiguous indices."""
        act0 = jnp.ones((2, 4))
        act5 = jnp.ones((2, 4)) * 2
        activations = {5: act5, 0: act0}

        stats = compute_activation_stats(activations)

        assert 0 in stats
        assert 5 in stats
        # Layer 0 is first in sorted order, so direction_change=0.
        assert stats[0]["direction_change"] == 0.0
        # Layer 5 comes after layer 0, so direction_change is computed.
        assert "direction_change" in stats[5]


class TestHelperFunctions:
    """Tests for REQ-INFER-014: Internal helper functions."""

    def test_cosine_similarity_identical(self) -> None:
        """REQ-INFER-014: Identical vectors have cosine similarity 1.0."""
        a = jnp.array([1.0, 2.0, 3.0])
        result = _cosine_similarity(a, a)
        np.testing.assert_allclose(float(result), 1.0, atol=1e-6)

    def test_cosine_similarity_orthogonal(self) -> None:
        """REQ-INFER-014: Orthogonal vectors have cosine similarity 0.0."""
        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 1.0])
        result = _cosine_similarity(a, b)
        np.testing.assert_allclose(float(result), 0.0, atol=1e-6)

    def test_cosine_similarity_opposite(self) -> None:
        """REQ-INFER-014: Opposite vectors have cosine similarity -1.0."""
        a = jnp.array([1.0, 0.0])
        b = jnp.array([-1.0, 0.0])
        result = _cosine_similarity(a, b)
        np.testing.assert_allclose(float(result), -1.0, atol=1e-6)

    def test_cosine_similarity_zero_vector(self) -> None:
        """REQ-INFER-014: Zero vector handled without NaN."""
        a = jnp.array([0.0, 0.0])
        b = jnp.array([1.0, 2.0])
        result = _cosine_similarity(a, b)
        assert not jnp.isnan(result)

    def test_jax_softmax_sums_to_one(self) -> None:
        """REQ-INFER-014: Softmax output sums to 1.0."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = jax_softmax(x)
        np.testing.assert_allclose(float(jnp.sum(result)), 1.0, atol=1e-6)

    def test_jax_softmax_2d(self) -> None:
        """REQ-INFER-014: 2D softmax normalizes along last axis."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = jax_softmax(x, axis=-1)
        row_sums = jnp.sum(result, axis=-1)
        np.testing.assert_allclose(np.array(row_sums), [1.0, 1.0], atol=1e-6)

    def test_jax_softmax_numerical_stability(self) -> None:
        """REQ-INFER-014: Softmax handles large values without overflow."""
        x = jnp.array([1000.0, 1001.0, 1002.0])
        result = jax_softmax(x)
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))
        np.testing.assert_allclose(float(jnp.sum(result)), 1.0, atol=1e-6)

    def test_activation_entropy_nonnegative(self) -> None:
        """REQ-INFER-014: Entropy is always non-negative."""
        act = jnp.array(np.random.randn(4, 8).astype(np.float32))
        result = _activation_entropy(act)
        assert float(result) >= 0.0

    def test_activation_entropy_uniform_high(self) -> None:
        """REQ-INFER-014: Uniform activations produce high entropy."""
        uniform = jnp.ones((1, 64))
        peaked = jnp.zeros((1, 64)).at[0, 0].set(1000.0)
        e_uniform = float(_activation_entropy(uniform))
        e_peaked = float(_activation_entropy(peaked))
        assert e_uniform > e_peaked


class TestPackageExports:
    """Tests for REQ-INFER-014: Package-level imports work correctly."""

    def test_activation_config_exported(self) -> None:
        """REQ-INFER-014: ActivationConfig is importable from embeddings package."""
        from carnot.embeddings import ActivationConfig as PkgConfig

        assert PkgConfig is ActivationConfig

    def test_extract_layer_activations_exported(self) -> None:
        """REQ-INFER-014: extract_layer_activations is importable from embeddings package."""
        from carnot.embeddings import (
            extract_layer_activations as pkg_extract,
        )

        assert pkg_extract is extract_layer_activations

    def test_compute_activation_stats_exported(self) -> None:
        """REQ-INFER-014: compute_activation_stats is importable from embeddings package."""
        from carnot.embeddings import (
            compute_activation_stats as pkg_compute,
        )

        assert pkg_compute is compute_activation_stats
