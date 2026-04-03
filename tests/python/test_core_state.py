"""Tests for model state and serialization.

Spec coverage: REQ-CORE-003, REQ-CORE-004, SCENARIO-CORE-004
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.core.state import ModelConfig, ModelMetadata, ModelState


class TestModelConfig:
    """Tests for REQ-CORE-003: Model configuration."""

    def test_default_precision(self) -> None:
        """REQ-CORE-003: Default precision is f32."""
        config = ModelConfig(input_dim=10)
        assert config.precision == "f32"


class TestModelState:
    """Tests for REQ-CORE-003, REQ-CORE-004: Model state and serialization."""

    def test_creation(self) -> None:
        """REQ-CORE-003: ModelState holds parameters, config, metadata."""
        params = {"weight": jnp.ones(5), "bias": jnp.zeros(3)}
        config = ModelConfig(input_dim=5, hidden_dims=[3])
        state = ModelState(parameters=params, config=config)
        assert "weight" in state.parameters
        assert state.metadata.step == 0

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """SCENARIO-CORE-004: Serialize and deserialize model state."""
        params = {"weight": jnp.array([1.0, 2.0, 3.0]), "bias": jnp.array([0.1, 0.2])}
        config = ModelConfig(input_dim=3, hidden_dims=[2])
        metadata = ModelMetadata(step=42, loss_history=[1.0, 0.5, 0.25])
        state = ModelState(parameters=params, config=config, metadata=metadata)

        state.save(tmp_path)
        loaded = ModelState.load(tmp_path)

        assert loaded.config.input_dim == 3
        assert loaded.metadata.step == 42
        assert len(loaded.metadata.loss_history) == 3
        for name, arr in params.items():
            assert jnp.allclose(loaded.parameters[name], arr, atol=1e-6)
