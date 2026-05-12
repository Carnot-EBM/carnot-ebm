"""Tests for S2KAN Symbolic primitives dictionary.

Spec references: REQ-KAN-1926, SCENARIO-KAN-1926.
"""

import jax.numpy as jnp
import numpy as np

from carnot.models.s2kan_symbolic import (
    PRIMITIVE_NAMES,
    S2KANSymbolicConfig,
    S2KANSymbolicLayer,
    S2KANSymbolicParams,
    build_experiment_1926_artifact,
    write_experiment_1926_artifact,
)

def test_s2kan_symbolic_forward_shape():
    """SCENARIO-KAN-1926: layer processes input correctly."""
    config = S2KANSymbolicConfig(input_dim=2)
    layer = S2KANSymbolicLayer(config)
    x = jnp.array([[1.0, 2.0], [0.5, -0.5]])
    y = layer.forward(x)
    assert y.shape == (2, 2)

def test_s2kan_symbolic_validation():
    """SCENARIO-KAN-1926: validate against a known functional form."""
    config = S2KANSymbolicConfig(input_dim=1)
    
    # Gate to select "exp"
    logits = np.zeros((1, len(PRIMITIVE_NAMES)), dtype=np.float32)
    logits[0, PRIMITIVE_NAMES.index("exp")] = 20.0
    params = S2KANSymbolicParams(gate_logits=jnp.array(logits))
    layer = S2KANSymbolicLayer(config, params=params)
    
    x = jnp.array([[0.0], [1.0], [2.0]])
    y = layer.forward(x)
    
    expected = jnp.exp(x)
    np.testing.assert_allclose(y, expected, rtol=1e-4)

def test_s2kan_symbolic_artifact(tmp_path):
    """SCENARIO-KAN-1926: artifact generation."""
    out_path = tmp_path / "result.json"
    artifact = write_experiment_1926_artifact(output_path=out_path)
    
    assert artifact["schema"] == "carnot.s2kan_symbolic.experiment_1926.v1"
    assert artifact["status"] == "complete"
    assert artifact["validation_passed"] is True
    assert out_path.exists()
