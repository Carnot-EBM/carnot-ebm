"""Tests for the Nabla Reasoner logit optimizer.

Spec: REQ-VERIFY-1616, SCENARIO-VERIFY-1616.
"""

import json
from pathlib import Path

import jax.numpy as jnp

from carnot.verify.nabla_reasoner import (
    differentiable_ebcn_energy,
    optimize_logits,
    run_experiment_1616,
)

def test_differentiable_ebcn_energy():
    """Verify energy computation returns a valid scalar scalar."""
    logits = jnp.ones((4, 8))
    energy = differentiable_ebcn_energy(logits)
    assert energy.shape == ()
    assert float(energy) >= 0.0

def test_optimize_logits_reduces_energy():
    """Verify Langevin dynamics reduces EBCN energy."""
    initial_logits = jnp.array([
        [1.0, 1.0, 0.5, 0.1, 0.2, 0.3],
        [-1.0, 1.0, 0.6, 0.1, 0.2, 0.3],
        [1.0, 0.5, 0.1, -0.1, 0.5, 0.2],
    ])
    res = optimize_logits(initial_logits, steps=50, step_size=0.1, noise_scale=0.0)
    assert res["final_energy"] < res["initial_energy"]
    assert res["convergence_speed"] > 0.0
    assert res["steps_run"] == 50

def test_run_experiment_1616_writes_artifact(tmp_path: Path):
    """Verify the experiment runs and writes the expected JSON artifact."""
    artifact_path = tmp_path / "results" / "experiment_1616_nabla_reasoner.json"
    artifact = run_experiment_1616(artifact_path)
    
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1616
    assert artifact["final_energy"] < artifact["initial_energy"]
    
    with open(artifact_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    
    assert saved_data["status"] == "complete"
    assert "convergence_speed" in saved_data
