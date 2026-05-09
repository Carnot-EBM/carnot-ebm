"""Tests for Nabla Reasoner debug module.

Spec: REQ-VERIFY-1627, SCENARIO-VERIFY-1627.
"""

import json
from pathlib import Path

import jax.numpy as jnp

from carnot.verify.nabla_debug import optimize_logits_momentum, run_sweep


def test_optimize_logits_momentum():
    """Verify momentum dynamics reduces EBCN energy."""
    initial_logits = jnp.array([
        [1.0, 1.0, 0.5, 0.1, 0.2, 0.3],
        [-1.0, 1.0, 0.6, 0.1, 0.2, 0.3],
        [1.0, 0.5, 0.1, -0.1, 0.5, 0.2],
    ])
    res = optimize_logits_momentum(
        initial_logits, steps=50, step_size=0.1, momentum=0.5, noise_scale=0.0
    )
    assert res["final_energy"] < res["initial_energy"]
    assert res["convergence_speed"] > 0.0


def test_run_sweep(tmp_path: Path):
    """Verify sweep produces correct JSON output."""
    artifact_path = tmp_path / "results" / "experiment_1627_nabla_debug.json"
    artifact = run_sweep(artifact_path)
    
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1627
    assert "optimizer_converges" in artifact
    assert "optimal_learning_rate" in artifact
    assert "optimal_momentum" in artifact
    
    with open(artifact_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        
    assert saved_data["optimizer_converges"] == artifact["optimizer_converges"]
