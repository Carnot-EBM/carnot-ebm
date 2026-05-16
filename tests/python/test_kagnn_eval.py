"""Tests for comparative evaluation logic for KAGNN vs MLP.

Spec references: REQ-KAN-2035, SCENARIO-KAN-2035.
"""

import os

import jax
import jax.numpy as jnp

from carnot.models.ising.kagnn_eval import (
    MLPVerifier,
    generate_synthetic_graphs,
    run_evaluation,
)


def test_mlp_verifier():
    """Test standard forward pass of MLPVerifier."""
    key = jax.random.PRNGKey(0)
    mlp = MLPVerifier(key, input_dim=2, hidden_dim=8)
    nf = jnp.ones((5, 1))
    edges = jnp.array([[0, 1], [1, 2]])
    energy = mlp.forward(nf, edges)
    assert energy.shape == ()


def test_mlp_verifier_empty():
    """Test forward pass of MLPVerifier with empty edge list."""
    key = jax.random.PRNGKey(0)
    mlp = MLPVerifier(key, input_dim=2, hidden_dim=8)
    nf = jnp.ones((5, 1))
    edges = jnp.zeros((0, 2), dtype=jnp.int32)
    energy = mlp.forward(nf, edges)
    assert float(energy) == 0.0


def test_generate_synthetic_graphs():
    """Test generation of synthetic constraint graphs."""
    graphs = generate_synthetic_graphs(n_graphs=2)
    assert len(graphs) == 2
    nf, edges = graphs[0]
    assert nf.shape == (5, 1)
    assert edges.shape == (6, 2)


def test_run_evaluation():
    """Test the full evaluation pipeline and JSON output generation."""
    if os.path.exists("results/exp2035_kagnn_eval.json"):
        os.remove("results/exp2035_kagnn_eval.json")
        
    results = run_evaluation()
    assert "kagnn_mean_energy" in results
    assert "mlp_mean_energy" in results
    assert "conclusion" in results
    assert "kagnn_efficiency_benefit" in results
    assert results["kagnn_efficiency_benefit"] is True
    assert os.path.exists("results/exp2035_kagnn_eval.json")
