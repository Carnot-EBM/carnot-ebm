"""Tests for EBRM latent trace scoring.

Spec: REQ-VERIFY-1628, SCENARIO-VERIFY-1628.
"""

import json
from pathlib import Path

import jax.numpy as jnp

from carnot.verify.ebrm_scoring import score_latent_trace


def test_score_latent_trace():
    """Verify trace scoring returns a valid float, higher for better traces."""
    # Bad trace: polarities clash heavily (first column is polarity)
    bad_logits = jnp.array([
        [1.0, 0.0, 0.5, 0.9, 0.1, 0.1],
        [-1.0, 0.0, 0.5, 0.9, 0.1, 0.1],
        [1.0, 0.0, 0.5, 0.9, 0.1, 0.1],
    ])
    
    # Good trace: polarities agree, low contradiction
    good_logits = jnp.array([
        [1.0, 0.0, 0.5, 0.9, 0.1, 0.1],
        [1.0, 0.0, 0.5, 0.9, 0.1, 0.1],
        [1.0, 0.0, 0.5, 0.9, 0.1, 0.1],
    ])
    
    bad_score = score_latent_trace(bad_logits)
    good_score = score_latent_trace(good_logits)
    
    assert float(good_score) > float(bad_score)


def test_experiment_1628_runs(tmp_path: Path):
    """Verify the script runs and evaluates accuracy > 0.8."""
    # Import inside to avoid running logic at module level if not needed,
    # but we can just import from scripts since it's in path.
    import sys
    import importlib.util
    
    script_path = Path("scripts/experiment_1628_ebrm_scoring.py").resolve()
    spec = importlib.util.spec_from_file_location("experiment_1628_ebrm_scoring", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment_1628_ebrm_scoring"] = module
    spec.loader.exec_module(module)
    
    artifact_path = tmp_path / "results" / "experiment_1628_ebrm_scoring.json"
    artifact = module.run_experiment(artifact_path)
    
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1628
    assert "scoring_accuracy" in artifact
    assert artifact["scoring_accuracy"] > 0.8
    assert artifact["honest_verdict"] == "ebrm_scoring_distinguishes_traces"
    
    with open(artifact_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    
    assert saved_data["status"] == "complete"
    assert saved_data["scoring_accuracy"] == artifact["scoring_accuracy"]
