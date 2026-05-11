"""Tests for S2KAN Lipschitz bounds.

Spec references: REQ-KAN-1858, SCENARIO-KAN-1858.
"""

import json
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp
import pytest

from carnot.models.s2kan import (
    S2KANConfig,
    S2KANLayer,
    S2KANParams,
    build_experiment_1858_artifact,
    write_experiment_1858_artifact,
)

def test_s2kan_lipschitz_bounds():
    """Verify that empirical perturbations respect the computed local Lipschitz bounds.
    
    Traces to: REQ-KAN-1858, SCENARIO-KAN-1858.
    """
    config = S2KANConfig(input_dim=1)
    # Give reasonable gates so all primitives contribute
    params = S2KANParams(gate_logits=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32))
    layer = S2KANLayer(config, params=params)
    
    x = jnp.array([[0.0], [np.pi / 4], [1.0], [-1.0]], dtype=jnp.float32)
    radius = 0.1
    
    # Get forward pass with bounds
    y, lip = layer.forward(x, return_lipschitz=True, radius=radius)
    
    # Also verify that without return_lipschitz we get just y
    y_only = layer.forward(x, return_lipschitz=False)
    np.testing.assert_allclose(y, y_only, rtol=1e-6, atol=1e-6)
    
    # Generate some perturbations within radius
    key = jax.random.PRNGKey(42)
    noise = jax.random.uniform(key, x.shape, minval=-radius, maxval=radius)
    x_perturbed = x + noise
    
    y_perturbed = layer.forward(x_perturbed, return_lipschitz=False)
    
    # Mathematical property: |f(x + delta) - f(x)| <= lip * |delta|
    delta = jnp.abs(noise)
    diff = jnp.abs(y_perturbed - y)
    
    max_expected_diff = lip * delta
    
    # Assert bounds hold mathematically
    # We allow a very small tolerance for floating point inaccuracies
    assert jnp.all(diff <= max_expected_diff + 1e-6), "Lipschitz bound violated"

def test_experiment_1858_artifact_generation(tmp_path):
    """Test generating the artifact for 1858.
    
    Traces to: REQ-KAN-1858, SCENARIO-KAN-1858.
    """
    out_file = tmp_path / "exp1858.json"
    artifact = write_experiment_1858_artifact(output_path=out_file)
    
    assert out_file.exists()
    
    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.s2kan.experiment_1858.v1"
    assert data["status"] == "complete"
    assert data["experiment_id"] == 1858
    assert "REQ-KAN-1858" in data["spec_traces"]
    assert "SCENARIO-KAN-1858" in data["spec_traces"]

def test_s2kan_single_output_fallback():
    """Verify that returning just the output works as originally."""
    config = S2KANConfig(input_dim=1)
    layer = S2KANLayer(config)
    x = jnp.array([[0.5]], dtype=jnp.float32)
    out = layer.forward(x)
    assert out.shape == (1, 1)
