"""Tests for transformer-based semantic code embeddings.

Spec coverage: REQ-EMBED-001, SCENARIO-EMBED-001, SCENARIO-EMBED-002, SCENARIO-EMBED-003
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.embeddings.model_embeddings import ModelEmbeddingConfig, extract_embedding


class TestModelEmbeddingConfig:
    """Tests for REQ-EMBED-001: Embedding configuration defaults."""

    def test_default_model_name(self) -> None:
        """SCENARIO-EMBED-001: Default model is CodeBERT."""
        config = ModelEmbeddingConfig()
        assert config.model_name == "microsoft/codebert-base"

    def test_default_device(self) -> None:
        """SCENARIO-EMBED-001: Default device is CPU."""
        config = ModelEmbeddingConfig()
        assert config.device == "cpu"

    def test_default_max_length(self) -> None:
        """SCENARIO-EMBED-001: Default max_length is 512."""
        config = ModelEmbeddingConfig()
        assert config.max_length == 512

    def test_custom_config(self) -> None:
        """SCENARIO-EMBED-001: Custom config values are respected."""
        config = ModelEmbeddingConfig(
            model_name="Salesforce/codet5-small",
            device="cuda",
            max_length=256,
        )
        assert config.model_name == "Salesforce/codet5-small"
        assert config.device == "cuda"
        assert config.max_length == 256


class TestExtractEmbeddingFallback:
    """Tests for REQ-EMBED-001: Graceful fallback when transformers unavailable."""

    def test_returns_none_when_transformers_missing(self) -> None:
        """SCENARIO-EMBED-002: Returns None when transformers not installed."""
        # Remove torch and transformers from sys.modules so the lazy import
        # inside extract_embedding triggers an ImportError.
        import sys

        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            result = extract_embedding("def foo(): pass")

        assert result is None

    def test_returns_none_with_explicit_config(self) -> None:
        """SCENARIO-EMBED-002: Returns None with explicit config when transformers missing."""
        import sys

        config = ModelEmbeddingConfig(model_name="some/model")
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            result = extract_embedding("x = 1", config=config)

        assert result is None


class TestExtractEmbeddingWithMockModel:
    """Tests for REQ-EMBED-001: Embedding extraction with mocked transformer."""

    def _build_mocks(self, hidden_dim: int = 768, seq_len: int = 5):
        """Build mock tokenizer, model, and torch module.

        Creates a mock transformer pipeline that returns a predetermined
        hidden state tensor, so we can test the mean-pooling and JAX
        conversion logic without needing actual model weights.
        """
        mock_torch = MagicMock()

        # Create a realistic fake hidden state: (1, seq_len, hidden_dim)
        # Use a known pattern so we can verify the mean-pooling result.
        hidden_np = np.ones((1, seq_len, hidden_dim), dtype=np.float32)
        attention_mask_np = np.ones((1, seq_len), dtype=np.int64)

        # Mock tensor operations to behave like real PyTorch tensors
        # by using numpy under the hood.
        class FakeTensor:
            """Mimics a PyTorch tensor using numpy arrays underneath."""

            def __init__(self, data: np.ndarray):
                self._data = data

            def to(self, device: str) -> "FakeTensor":
                return self

            def unsqueeze(self, dim: int) -> "FakeTensor":
                return FakeTensor(np.expand_dims(self._data, axis=dim))

            def float(self) -> "FakeTensor":
                return FakeTensor(self._data.astype(np.float32))

            def sum(self, dim: int | None = None) -> "FakeTensor":
                if dim is not None:
                    return FakeTensor(self._data.sum(axis=dim, keepdims=False))
                return FakeTensor(np.array(self._data.sum()))

            def clamp(self, min: float = 0.0) -> "FakeTensor":
                return FakeTensor(np.clip(self._data, a_min=min, a_max=None))

            def squeeze(self, dim: int) -> "FakeTensor":
                return FakeTensor(np.squeeze(self._data, axis=dim))

            def cpu(self) -> "FakeTensor":
                return self

            def numpy(self) -> np.ndarray:
                return self._data

            def __mul__(self, other: "FakeTensor") -> "FakeTensor":
                return FakeTensor(self._data * other._data)

            def __truediv__(self, other: "FakeTensor") -> "FakeTensor":
                return FakeTensor(self._data / other._data)

        # Build the mock model output
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = FakeTensor(hidden_np)

        # Build the mock model
        mock_model = MagicMock()
        mock_model.return_value = mock_outputs
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None

        # Build the mock tokenizer that returns FakeTensors
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": FakeTensor(np.zeros((1, seq_len), dtype=np.int64)),
            "attention_mask": FakeTensor(attention_mask_np),
        }

        # Mock torch.no_grad context manager
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        # Mock transformers module
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        return mock_torch, mock_transformers, hidden_dim

    def test_returns_jax_array(self) -> None:
        """SCENARIO-EMBED-003: extract_embedding returns a JAX array."""
        mock_torch, mock_transformers, hidden_dim = self._build_mocks()

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = extract_embedding("def hello(): return 42")

        assert result is not None
        assert isinstance(result, jnp.ndarray)

    def test_embedding_shape(self) -> None:
        """SCENARIO-EMBED-003: Embedding has correct dimensionality."""
        mock_torch, mock_transformers, hidden_dim = self._build_mocks(
            hidden_dim=768, seq_len=5
        )

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = extract_embedding("x = 1 + 2")

        assert result is not None
        assert result.shape == (768,)

    def test_mean_pooling_values(self) -> None:
        """SCENARIO-EMBED-003: Mean pooling produces correct values for uniform input."""
        # With all-ones hidden states and all-ones attention mask,
        # mean pooling should produce all-ones output.
        mock_torch, mock_transformers, hidden_dim = self._build_mocks(
            hidden_dim=4, seq_len=3
        )

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = extract_embedding("test code")

        assert result is not None
        expected = jnp.ones(4)
        np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-6)

    def test_default_config_used_when_none(self) -> None:
        """SCENARIO-EMBED-003: Default config is used when config=None."""
        mock_torch, mock_transformers, _ = self._build_mocks()

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = extract_embedding("code", config=None)

        # Verify that the default model name was used
        mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "microsoft/codebert-base"
        )
        assert result is not None

    def test_custom_config_model_name(self) -> None:
        """SCENARIO-EMBED-003: Custom model name is passed to from_pretrained."""
        mock_torch, mock_transformers, _ = self._build_mocks()
        config = ModelEmbeddingConfig(model_name="custom/model")

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = extract_embedding("code", config=config)

        mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "custom/model"
        )
        mock_transformers.AutoModel.from_pretrained.assert_called_once_with(
            "custom/model"
        )
        assert result is not None


class TestExtractEmbeddingImportPath:
    """Tests for REQ-EMBED-001: Package-level imports work correctly."""

    def test_package_exports(self) -> None:
        """SCENARIO-EMBED-001: embeddings package exports expected symbols."""
        from carnot.embeddings import ModelEmbeddingConfig as PkgConfig
        from carnot.embeddings import extract_embedding as pkg_extract

        assert PkgConfig is ModelEmbeddingConfig
        assert pkg_extract is extract_embedding
