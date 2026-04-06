"""Tests for Contrastive Weight Steering (CWS).

Spec coverage: REQ-INFER-015
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.embeddings.weight_steering import (
    _find_output_projection,
    apply_cws,
    revert_cws,
    steered_model,
)


# ---------------------------------------------------------------------------
# Helpers: build mock model with weight parameters
# ---------------------------------------------------------------------------


class FakeWeightData:
    """Mimics a torch tensor with in-place operations.

    REQ-INFER-015: Provides clone/copy_/__isub__ for weight steering tests.
    """

    def __init__(self, data: np.ndarray) -> None:
        self._data = data.copy()
        self.isub_called = False
        self.isub_arg = None
        self.copy_called = False
        self.copy_arg = None

    def clone(self) -> np.ndarray:
        return self._data.copy()

    def __isub__(self, other):
        self.isub_called = True
        self.isub_arg = other
        return self

    def copy_(self, other):
        self.copy_called = True
        self.copy_arg = other


class FakeParameter:
    """Mimics a torch.nn.Parameter with mutable .data attribute.

    REQ-INFER-015: Weight parameter mock for CWS tests.
    """

    def __init__(self, hidden_dim: int = 8) -> None:
        self.data = FakeWeightData(np.eye(hidden_dim, dtype=np.float32))
        self.dtype = "float32"
        self.device = "cpu"


def _build_weight_mock(num_layers: int = 3, hidden_dim: int = 8):
    """Build a mock model with modifiable weight parameters.

    REQ-INFER-015: Mocked transformer for weight steering tests.
    """
    mock_torch = MagicMock()

    # Create layers with output projection weights.
    fake_layers = []
    params = []
    for _i in range(num_layers):
        layer = MagicMock()
        param = FakeParameter(hidden_dim)
        # Simulate LLaMA-style self_attn.o_proj.weight
        layer.self_attn = MagicMock()
        layer.self_attn.o_proj = MagicMock()
        layer.self_attn.o_proj.weight = param
        fake_layers.append(layer)
        params.append(param)

    mock_model = MagicMock()
    mock_model.model.layers = fake_layers

    # Mock torch.no_grad context manager.
    mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    # Mock torch.tensor to return numpy arrays directly.
    mock_torch.tensor = lambda data, dtype=None, device=None: data

    return mock_torch, mock_model, fake_layers, params


# ---------------------------------------------------------------------------
# _find_output_projection tests
# ---------------------------------------------------------------------------


class TestFindOutputProjection:
    """Tests for REQ-INFER-015: Output projection weight discovery."""

    def test_finds_self_attn_o_proj(self) -> None:
        """REQ-INFER-015: Finds weight at self_attn.o_proj.weight (LLaMA)."""
        layer = MagicMock()
        layer.self_attn.o_proj.weight = MagicMock()
        result = _find_output_projection(layer)
        assert result is layer.self_attn.o_proj.weight

    def test_finds_attn_c_proj(self) -> None:
        """REQ-INFER-015: Finds weight at attn.c_proj.weight (GPT-2)."""
        layer = MagicMock(spec=[])
        layer.attn = MagicMock(spec=[])
        layer.attn.c_proj = MagicMock(spec=[])
        layer.attn.c_proj.weight = MagicMock()
        result = _find_output_projection(layer)
        assert result is layer.attn.c_proj.weight

    def test_finds_output_dense(self) -> None:
        """REQ-INFER-015: Finds weight at output.dense.weight (BERT)."""
        layer = MagicMock(spec=[])
        layer.output = MagicMock(spec=[])
        layer.output.dense = MagicMock(spec=[])
        layer.output.dense.weight = MagicMock()
        result = _find_output_projection(layer)
        assert result is layer.output.dense.weight

    def test_returns_none_for_unknown(self) -> None:
        """REQ-INFER-015: Returns None for unrecognized layer structure."""
        layer = MagicMock(spec=[])
        result = _find_output_projection(layer)
        assert result is None


# ---------------------------------------------------------------------------
# apply_cws tests
# ---------------------------------------------------------------------------


class TestApplyCWS:
    """Tests for REQ-INFER-015: Applying Contrastive Weight Steering."""

    def test_returns_original_weights(self) -> None:
        """REQ-INFER-015: apply_cws returns dict with original_weight."""
        mock_torch, model, layers, params = _build_weight_mock()
        direction = jnp.ones(8)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = apply_cws(model, 0, direction, alpha=1.0)

        assert result is not None
        assert "original_weight" in result
        assert "layer_idx" in result
        assert result["layer_idx"] == 0

    def test_returns_none_for_unrecognized_model(self) -> None:
        """REQ-INFER-015: Returns None when model layers not found."""
        mock_torch, _, _, _ = _build_weight_mock()
        model = MagicMock(spec=[])
        direction = jnp.ones(8)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = apply_cws(model, 0, direction)

        assert result is None

    def test_returns_none_for_out_of_range_layer(self) -> None:
        """REQ-INFER-015: Returns None when layer_idx exceeds model layers."""
        mock_torch, model, _, _ = _build_weight_mock(num_layers=2)
        direction = jnp.ones(8)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = apply_cws(model, 99, direction)

        assert result is None

    def test_modifies_weight_in_place(self) -> None:
        """REQ-INFER-015: Weight data is modified via __isub__."""
        mock_torch, model, layers, params = _build_weight_mock()
        direction = jnp.ones(8)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = apply_cws(model, 0, direction, alpha=1.0)

        assert result is not None
        assert params[0].data.isub_called

    def test_zero_direction_no_modification(self) -> None:
        """REQ-INFER-015: Zero direction vector causes no weight modification."""
        mock_torch, model, layers, params = _build_weight_mock()
        direction = jnp.zeros(8)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = apply_cws(model, 0, direction, alpha=1.0)

        assert result is not None
        # Zero direction => d_norm_sq < 1e-12 => no modification.
        assert not params[0].data.isub_called

    def test_returns_none_when_no_output_proj(self) -> None:
        """REQ-INFER-015: Returns None when layer has no output projection."""
        mock_torch, model, layers, _ = _build_weight_mock()
        # Replace layer 0 with one that has no output projection.
        bare_layer = MagicMock(spec=[])
        model.model.layers[0] = bare_layer
        direction = jnp.ones(8)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = apply_cws(model, 0, direction)

        assert result is None


# ---------------------------------------------------------------------------
# revert_cws tests
# ---------------------------------------------------------------------------


class TestRevertCWS:
    """Tests for REQ-INFER-015: Reverting weight steering."""

    def test_revert_restores_weights(self) -> None:
        """REQ-INFER-015: revert_cws calls weight.data.copy_ with original."""
        mock_torch, model, layers, params = _build_weight_mock()
        direction = jnp.ones(8)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            saved = apply_cws(model, 0, direction)
            assert saved is not None
            result = revert_cws(model, 0, saved)

        assert result is True
        assert params[0].data.copy_called

    def test_revert_returns_false_for_bad_layer(self) -> None:
        """REQ-INFER-015: Returns False when layer_idx out of range."""
        mock_torch, model, _, _ = _build_weight_mock(num_layers=2)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = revert_cws(model, 99, {"original_weight": None})

        assert result is False

    def test_revert_returns_false_for_unknown_model(self) -> None:
        """REQ-INFER-015: Returns False when model has no known layers."""
        mock_torch, _, _, _ = _build_weight_mock()
        model = MagicMock(spec=[])

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = revert_cws(model, 0, {"original_weight": None})

        assert result is False

    def test_revert_returns_false_for_no_output_proj(self) -> None:
        """REQ-INFER-015: Returns False when layer has no output projection."""
        mock_torch, model, layers, _ = _build_weight_mock()
        bare_layer = MagicMock(spec=[])
        model.model.layers[0] = bare_layer

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = revert_cws(model, 0, {"original_weight": None})

        assert result is False


# ---------------------------------------------------------------------------
# steered_model context manager tests
# ---------------------------------------------------------------------------


class TestSteeredModelContextManager:
    """Tests for REQ-INFER-015: Context manager for temporary weight steering."""

    def test_yields_model(self) -> None:
        """REQ-INFER-015: Context manager yields the model object."""
        mock_torch, model, _, _ = _build_weight_mock()
        direction = jnp.ones(8)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with steered_model(model, [0], direction, alpha=1.0) as m:
                assert m is model

    def test_applies_cws_on_entry(self) -> None:
        """REQ-INFER-015: Weights are modified on context entry."""
        mock_torch, model, layers, params = _build_weight_mock()
        direction = jnp.ones(8)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with steered_model(model, [0, 1], direction, alpha=1.0):
                # Inside context: weights should have been modified.
                assert params[0].data.isub_called
                assert params[1].data.isub_called

    def test_reverts_on_exit(self) -> None:
        """REQ-INFER-015: Weights are reverted on context exit."""
        mock_torch, model, layers, params = _build_weight_mock()
        direction = jnp.ones(8)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with steered_model(model, [0], direction, alpha=1.0):
                pass  # Exit context.

        # After context: revert should have called copy_.
        assert params[0].data.copy_called

    def test_reverts_on_exception(self) -> None:
        """REQ-INFER-015: Weights are reverted even if exception occurs inside."""
        mock_torch, model, layers, params = _build_weight_mock()
        direction = jnp.ones(8)

        with pytest.raises(ValueError):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                with steered_model(model, [0], direction, alpha=1.0):
                    raise ValueError("test error")

        assert params[0].data.copy_called

    def test_handles_multiple_layers(self) -> None:
        """REQ-INFER-015: Applies and reverts CWS across multiple layers."""
        mock_torch, model, layers, params = _build_weight_mock(num_layers=3)
        direction = jnp.ones(8)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with steered_model(model, [0, 1, 2], direction, alpha=0.5):
                pass

        # All three layers should have had apply + revert.
        for i in range(3):
            assert params[i].data.isub_called
            assert params[i].data.copy_called

    def test_empty_layers_list(self) -> None:
        """REQ-INFER-015: Empty layers list is a no-op."""
        mock_torch, model, layers, params = _build_weight_mock()
        direction = jnp.ones(8)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with steered_model(model, [], direction, alpha=1.0) as m:
                assert m is model

        # No layers modified.
        for p in params:
            assert not p.data.isub_called


# ---------------------------------------------------------------------------
# Package export tests
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Tests for REQ-INFER-015: Package-level imports."""

    def test_apply_cws_exported(self) -> None:
        """REQ-INFER-015: apply_cws importable from embeddings."""
        from carnot.embeddings import apply_cws as pkg_fn

        assert pkg_fn is apply_cws

    def test_revert_cws_exported(self) -> None:
        """REQ-INFER-015: revert_cws importable from embeddings."""
        from carnot.embeddings import revert_cws as pkg_fn

        assert pkg_fn is revert_cws

    def test_steered_model_exported(self) -> None:
        """REQ-INFER-015: steered_model importable from embeddings."""
        from carnot.embeddings import steered_model as pkg_fn

        assert pkg_fn is steered_model
