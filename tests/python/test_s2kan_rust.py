"""Tests for the Rust implementation of S2KAN.

Spec references: REQ-KAN-1857, SCENARIO-KAN-1857.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from carnot._rust_compat import RustS2KANLayer, RUST_AVAILABLE
from carnot.models.s2kan import S2KANConfig, S2KANLayer, S2KANParams


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not installed")
def test_s2kan_rust_equivalence() -> None:
    """Verify that the Rust S2KAN forward pass matches the Python/JAX implementation."""
    input_dim = 2
    temperature = 1.0
    gate_logits = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
    ], dtype=np.float32)

    # Initialize Python layer
    config = S2KANConfig(input_dim=input_dim, temperature=temperature)
    params = S2KANParams(gate_logits=jnp.array(gate_logits))
    py_layer = S2KANLayer(config, params=params)

    # Initialize Rust layer
    rust_layer = RustS2KANLayer(input_dim, float(temperature), gate_logits)

    # Generate random test data
    x = np.array([
        [0.0, np.pi / 2],
        [np.pi, 1.0],
        [-1.0, 0.0]
    ], dtype=np.float32)

    # Python forward pass
    py_output = py_layer.forward(jnp.array(x))

    # Rust forward pass
    rust_output = rust_layer.forward(x)

    # Verify equivalence
    np.testing.assert_allclose(py_output, rust_output, rtol=1e-4, atol=1e-4)
