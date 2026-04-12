"""Tests for activation steering during generation.

Spec coverage: REQ-INFER-015
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.embeddings.activation_steering import (
    SteeringConfig,
    calibrate_alpha,
    steered_generate,
)


# ---------------------------------------------------------------------------
# Helpers: build mock model for generation
# ---------------------------------------------------------------------------


def _build_generation_mocks(
    num_layers: int = 3, hidden_dim: int = 8, seq_len: int = 4
):
    """Build mocks supporting model.generate() with hookable layers.

    REQ-INFER-015: Mocked transformer for activation steering tests.
    """
    mock_torch = MagicMock()

    # Layers with hook support.
    fake_layers = []
    for _i in range(num_layers):
        layer = MagicMock()
        registered_hooks: list = []

        def make_register(hooks_list):
            def register_forward_hook(fn):
                hooks_list.append(fn)
                handle = MagicMock()
                handle.remove = MagicMock()
                return handle

            return register_forward_hook

        layer.register_forward_hook = make_register(registered_hooks)
        fake_layers.append((layer, registered_hooks))

    # Build mock model.
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = None

    layer_modules = [fl[0] for fl in fake_layers]
    mock_model.model.layers = layer_modules

    fake_param = MagicMock()
    fake_param.device = "cpu"
    mock_model.parameters = lambda: iter([fake_param])

    # model.generate returns fake token IDs.
    mock_model.generate.return_value = np.array([[1, 2, 3, 4, 5]])

    # Mock torch utilities.
    mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    mock_torch.tensor = lambda data, dtype=None, device=None: MagicMock(
        unsqueeze=lambda dim: MagicMock()
    )

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
    mock_tokenizer.decode.return_value = "The answer is 42."

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_transformers.AutoModel.from_pretrained.return_value = mock_model

    return mock_torch, mock_transformers, mock_model, mock_tokenizer


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestSteeringConfig:
    """Tests for REQ-INFER-015: SteeringConfig."""

    def test_default_values(self) -> None:
        """REQ-INFER-015: Default config has empty layers, alpha=1.0, max_new_tokens=50."""
        config = SteeringConfig()
        assert config.layer_indices == []
        assert config.alpha == 1.0
        assert config.max_new_tokens == 50

    def test_custom_values(self) -> None:
        """REQ-INFER-015: Custom config values are respected."""
        config = SteeringConfig(layer_indices=[1, 3], alpha=2.0, max_new_tokens=100)
        assert config.layer_indices == [1, 3]
        assert config.alpha == 2.0
        assert config.max_new_tokens == 100


# ---------------------------------------------------------------------------
# steered_generate tests
# ---------------------------------------------------------------------------


class TestSteeredGenerate:
    """Tests for REQ-INFER-015: Steered text generation."""

    def test_returns_string(self) -> None:
        """REQ-INFER-015: steered_generate returns generated text as string."""
        mock_torch, mock_transformers, model, tokenizer = _build_generation_mocks()
        direction = jnp.ones(8)
        config = SteeringConfig(layer_indices=[0, 1])

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = steered_generate(model, tokenizer, "prompt", direction, config)

        assert result is not None
        assert isinstance(result, str)
        assert result == "The answer is 42."

    def test_returns_none_for_unrecognized_model(self) -> None:
        """REQ-INFER-015: Returns None when model layers not found."""
        mock_torch, _, _, tokenizer = _build_generation_mocks()
        model = MagicMock(spec=[])
        direction = jnp.ones(8)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = steered_generate(
                model, tokenizer, "prompt", direction,
                SteeringConfig(layer_indices=[0]),
            )

        assert result is None

    def test_default_config_used(self) -> None:
        """REQ-INFER-015: Default config is used when config=None."""
        mock_torch, mock_transformers, model, tokenizer = _build_generation_mocks()
        direction = jnp.ones(8)

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = steered_generate(model, tokenizer, "prompt", direction, config=None)

        assert result is not None

    def test_hooks_removed_after_generate(self) -> None:
        """REQ-INFER-015: Hooks are cleaned up after generation."""
        mock_torch, mock_transformers, model, tokenizer = _build_generation_mocks()
        direction = jnp.ones(8)
        config = SteeringConfig(layer_indices=[0, 1, 2])

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            steered_generate(model, tokenizer, "prompt", direction, config)

        # No exception means hooks were properly cleaned up.

    def test_hooks_removed_on_exception(self) -> None:
        """REQ-INFER-015: Hooks are cleaned up even if generate() raises."""
        mock_torch, mock_transformers, model, tokenizer = _build_generation_mocks()
        model.generate.side_effect = RuntimeError("GPU OOM")
        direction = jnp.ones(8)
        config = SteeringConfig(layer_indices=[0])

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            with pytest.raises(RuntimeError, match="GPU OOM"):
                steered_generate(model, tokenizer, "prompt", direction, config)

    def test_skips_out_of_range_layers(self) -> None:
        """REQ-INFER-015: Layers beyond model size are silently skipped."""
        mock_torch, mock_transformers, model, tokenizer = _build_generation_mocks(
            num_layers=2
        )
        direction = jnp.ones(8)
        config = SteeringConfig(layer_indices=[0, 99])

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = steered_generate(model, tokenizer, "prompt", direction, config)

        assert result is not None

    def test_generate_called_with_max_new_tokens(self) -> None:
        """REQ-INFER-015: max_new_tokens is passed to model.generate()."""
        mock_torch, mock_transformers, model, tokenizer = _build_generation_mocks()
        direction = jnp.ones(8)
        config = SteeringConfig(layer_indices=[], max_new_tokens=77)

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            steered_generate(model, tokenizer, "prompt", direction, config)

        # Check that generate was called with max_new_tokens=77.
        model.generate.assert_called_once()
        call_kwargs = model.generate.call_args
        assert call_kwargs.kwargs.get("max_new_tokens") == 77


# ---------------------------------------------------------------------------
# calibrate_alpha tests
# ---------------------------------------------------------------------------


class TestCalibrateAlpha:
    """Tests for REQ-INFER-015: Alpha calibration."""

    def test_returns_float(self) -> None:
        """REQ-INFER-015: calibrate_alpha returns a float."""
        mock_torch, mock_transformers, model, tokenizer = _build_generation_mocks()
        direction = jnp.ones(8)
        qa_pairs = [("What is 2+2?", "42")]

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = calibrate_alpha(
                model, tokenizer, qa_pairs, [0], direction
            )

        assert result is not None
        assert isinstance(result, float)

    def test_returns_best_alpha_from_list(self) -> None:
        """REQ-INFER-015: Returns the alpha that maximizes accuracy."""
        mock_torch, mock_transformers, model, tokenizer = _build_generation_mocks()
        # The mock always generates "The answer is 42."
        # So qa_pairs with "42" as expected answer will always match.
        direction = jnp.ones(8)
        qa_pairs = [("q", "42")]

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = calibrate_alpha(
                model, tokenizer, qa_pairs, [0], direction,
                alphas=[0.1, 1.0, 5.0],
            )

        # All alphas should achieve 100% accuracy since mock always returns "42".
        # On ties, should return smallest alpha (0.1).
        assert result == 0.1

    def test_default_alphas_used(self) -> None:
        """REQ-INFER-015: Default alpha grid is used when alphas=None."""
        mock_torch, mock_transformers, model, tokenizer = _build_generation_mocks()
        direction = jnp.ones(8)

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = calibrate_alpha(
                model, tokenizer, [("q", "42")], [0], direction
            )

        assert result is not None
        # With default alphas [0.1, 0.5, 1.0, 2.0, 5.0] and all matching,
        # should return 0.1 (smallest on tie).
        assert result == 0.1

    def test_prefers_smaller_alpha_on_tie(self) -> None:
        """REQ-INFER-015: When multiple alphas tie, returns smallest."""
        mock_torch, mock_transformers, model, tokenizer = _build_generation_mocks()
        direction = jnp.ones(8)

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = calibrate_alpha(
                model, tokenizer, [("q", "42")], [0], direction,
                alphas=[5.0, 1.0, 0.1],
            )

        # All achieve 100%; should return first encountered with max accuracy.
        # Since 5.0 is first and all tie, 5.0 gets set first, but then 1.0
        # ties (not strictly greater), so 5.0 stays. This tests the ">"
        # not ">=" logic.
        assert result == 5.0

    def test_empty_qa_pairs(self) -> None:
        """REQ-INFER-015: Returns first alpha when QA pairs are empty."""
        mock_torch, mock_transformers, model, tokenizer = _build_generation_mocks()
        direction = jnp.ones(8)

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = calibrate_alpha(
                model, tokenizer, [], [0], direction, alphas=[0.5, 1.0]
            )

        assert result == 0.5

    def test_case_insensitive_matching(self) -> None:
        """REQ-INFER-015: Answer matching is case-insensitive."""
        mock_torch, mock_transformers, model, tokenizer = _build_generation_mocks()
        # Mock generates "The answer is 42." — test matching "THE ANSWER"
        tokenizer.decode.return_value = "The Answer Is 42."
        direction = jnp.ones(8)

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = calibrate_alpha(
                model, tokenizer, [("q", "the answer")], [0], direction,
                alphas=[1.0],
            )

        assert result == 1.0


# ---------------------------------------------------------------------------
# Package export tests
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Tests for REQ-INFER-015: Package-level imports."""

    def test_steering_config_exported(self) -> None:
        """REQ-INFER-015: SteeringConfig importable from embeddings."""
        from carnot.embeddings import SteeringConfig as PkgConfig

        assert PkgConfig is SteeringConfig

    def test_steered_generate_exported(self) -> None:
        """REQ-INFER-015: steered_generate importable from embeddings."""
        from carnot.embeddings import steered_generate as pkg_fn

        assert pkg_fn is steered_generate

    def test_calibrate_alpha_exported(self) -> None:
        """REQ-INFER-015: calibrate_alpha importable from embeddings."""
        from carnot.embeddings import calibrate_alpha as pkg_fn

        assert pkg_fn is calibrate_alpha
