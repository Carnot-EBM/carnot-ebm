"""Tests for KAGNN Verifier.

Spec references: REQ-KAN-2034, SCENARIO-KAN-2034.
"""
import json
import jax.numpy as jnp
from pathlib import Path

from carnot.models.ising.kagnn import KAGNNConfig, KAGNNVerifier
from carnot.models.kan.symbolic_kan import SymbolicKANParams

def test_kagnn_graph_coloring_energy():
    config = KAGNNConfig(
        node_dim=1,
        n_routes=1,
        primitives=("abs", "cos"),
    )
    
    # route_logits: make "cos" the selected primitive.
    # "cos" is at index 1.
    params = SymbolicKANParams(
        projection_weights=jnp.array([[1.0, -1.0]], dtype=jnp.float32),
        projection_bias=jnp.array([0.0], dtype=jnp.float32),
        route_logits=jnp.array([[0.0, 10.0]], dtype=jnp.float32),
        route_scales=jnp.array([1.0], dtype=jnp.float32),
        output_bias=jnp.array(0.0, dtype=jnp.float32),
    )
    
    verifier = KAGNNVerifier(config, params=params)
    
    # Graph: triangle (0-1, 1-2, 2-0)
    edge_indices = jnp.array([
        [0, 1],
        [1, 2],
        [2, 0],
    ])
    
    # Valid coloring: nodes have different colors (e.g., 0, pi/2, pi)
    valid_features = jnp.array([
        [0.0],
        [jnp.pi / 2],
        [jnp.pi],
    ], dtype=jnp.float32)
    
    # Invalid coloring: two adjacent nodes have same color
    invalid_features = jnp.array([
        [0.0],
        [0.0],
        [jnp.pi / 2],
    ], dtype=jnp.float32)
    
    valid_energy = verifier.forward(valid_features, edge_indices, hard=True, params=params)
    invalid_energy = verifier.forward(invalid_features, edge_indices, hard=True, params=params)
    
    assert valid_energy < invalid_energy
    
    # Write artifact
    artifact = {
        "schema": "carnot.kan.experiment_2034_kagnn.v1",
        "status": "complete",
        "valid_energy": float(valid_energy),
        "invalid_energy": float(invalid_energy),
        "honest_verdict": "complete: kagnn_verifier_evaluates_graph_coloring"
    }
    
    path = Path("results/experiment_2034_kagnn.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2))

def test_kagnn_empty_graph():
    config = KAGNNConfig(node_dim=1, n_routes=1, primitives=("identity",))
    verifier = KAGNNVerifier(config)
    energy = verifier.forward(
        node_features=jnp.array([[1.0]]),
        edge_indices=jnp.array([], dtype=jnp.int32).reshape((0, 2)),
    )
    assert energy == 0.0

def test_kagnn_edge_energy():
    config = KAGNNConfig(node_dim=1, n_routes=1, primitives=("identity",))
    verifier = KAGNNVerifier(config)
    
    u = jnp.array([1.0])
    v = jnp.array([2.0])
    
    # Should run without error
    energy = verifier.edge_energy(u, v)
    assert energy.shape == ()
