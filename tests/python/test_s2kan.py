"""Tests for S2KAN differentiable gates."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from carnot.models.s2kan import (
    S2KANConfig,
    S2KANParams,
    S2KANLayer,
    _step,
    evaluate_primitives,
    build_experiment_1857_artifact,
    write_experiment_1857_artifact,
    DEFAULT_RESULT_PATH,
)

def test_step():
    """Test soft step function."""
    x = jnp.array([0.0, 10.0, -10.0])
    y = _step(x)
    assert float(y[0]) == 0.5
    assert float(y[1]) > 0.99
    assert float(y[2]) < 0.01

def test_evaluate_primitives():
    """Test primitive evaluations."""
    x = jnp.array([0.0])
    y = evaluate_primitives(x)
    # sin(0) = 0, exp(0) = 1, step(0) = 0.5
    assert jnp.allclose(y, jnp.array([[0.0, 1.0, 0.5]]))

def test_s2kan_layer_init():
    """Test S2KAN layer initialization."""
    config = S2KANConfig(input_dim=2)
    layer = S2KANLayer(config)
    assert layer.params.gate_logits.shape == (2, 3)
    
    # Test with custom key
    layer2 = S2KANLayer(config, key=jax.random.PRNGKey(42))
    assert layer2.params.gate_logits.shape == (2, 3)

def test_s2kan_forward():
    """Test S2KAN forward pass with hard-coded logits to test routing."""
    config = S2KANConfig(input_dim=1)
    
    # Test routing to sin (index 0)
    params_sin = S2KANParams(gate_logits=jnp.array([[100.0, 0.0, 0.0]]))
    layer_sin = S2KANLayer(config, params=params_sin)
    x = jnp.array([[np.pi / 2]])
    y_sin = layer_sin.forward(x)
    assert jnp.allclose(y_sin, jnp.array([[1.0]]), atol=1e-3)
    
    # Test routing to exp (index 1)
    params_exp = S2KANParams(gate_logits=jnp.array([[0.0, 100.0, 0.0]]))
    layer_exp = S2KANLayer(config, params=params_exp)
    y_exp = layer_exp.forward(jnp.array([[0.0]]))
    assert jnp.allclose(y_exp, jnp.array([[1.0]]), atol=1e-3)
    
    # Test routing to step (index 2)
    params_step = S2KANParams(gate_logits=jnp.array([[0.0, 0.0, 100.0]]))
    layer_step = S2KANLayer(config, params=params_step)
    y_step = layer_step.forward(jnp.array([[0.0]]))
    assert jnp.allclose(y_step, jnp.array([[0.5]]), atol=1e-3)

def test_artifact_building():
    """Test artifact payload generation."""
    artifact = build_experiment_1857_artifact()
    assert artifact["experiment_id"] == 1857
    assert artifact["status"] == "complete"
    assert "REQ-KAN-1857" in artifact["spec_traces"]

def test_artifact_writing(tmp_path):
    """Test artifact writing to disk."""
    out_file = tmp_path / "test_artifact.json"
    artifact = write_experiment_1857_artifact(output_path=out_file)
    assert out_file.exists()
    
    with open(out_file, "r") as f:
        loaded = json.load(f)
    
    assert loaded["schema"] == artifact["schema"]
    assert loaded["experiment_id"] == artifact["experiment_id"]

def test_default_result_path():
    """Ensure DEFAULT_RESULT_PATH points to right file."""
    assert DEFAULT_RESULT_PATH.name == "experiment_1857_s2kan.json"
