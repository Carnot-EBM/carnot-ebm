"""Tests for the Exp 1962 NI Sampling benchmark artifact.

Spec traces: REQ-SAMPLE-1962, SCENARIO-SAMPLE-1962.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.experiment_1962_ni_sampling_token_order as exp
from carnot.inference.samplers import NISampler, RandomDiscreteDiffusionSampler

def test_scenario_sample_1962_experiment_writes_required_results_artifact(tmp_path: Path) -> None:
    """REQ-SAMPLE-1962: runner writes the required terminal JSON artifact."""
    output_path = tmp_path / "experiment_1962_ni_sampling_token_order.json"

    artifact = exp.run_experiment(output_path=output_path)
    written = json.loads(output_path.read_text())

    assert written == artifact
    assert artifact["artifact_path"] == str(output_path)
    assert artifact["experiment_id"] == "1962"
    assert artifact["status"] == "complete"
    assert "REQ-SAMPLE-1962" in artifact["spec_refs"]
    assert artifact["metrics"]["baseline_time"] > 0
    assert artifact["metrics"]["ni_time"] >= 0
    assert artifact["metrics"]["acceleration_factor"] > 0
    assert artifact["metrics"]["semantic_retention_verified"] is True

def test_ni_sampler_token_ordering():
    """Verify that NISampler resolves tokens in the correct indicator-based order."""
    
    # An indicator function that prefers evens over odds
    def indicator(seq, idx):
        return float(idx % 2)
        
    sampler = NISampler(indicator_fn=indicator)
    
    # 0 is noised, 1 is resolved
    initial_seq = [0, 0, 0, 0] 
    
    resolved_order = []
    def denoise_fn(seq, idx):
        resolved_order.append(idx)
        return 1
        
    sampler.sample(initial_seq, denoise_fn)
    
    # Evens should be first (indicator 0), then odds (indicator 1)
    assert resolved_order[0] in [0, 2]
    assert resolved_order[1] in [0, 2]
    assert resolved_order[2] in [1, 3]
    assert resolved_order[3] in [1, 3]

def test_random_discrete_diffusion_sampler():
    """Verify that the baseline resolves all tokens."""
    sampler = RandomDiscreteDiffusionSampler()
    
    initial_seq = [0, 0, 0, 0] 
    
    resolved_order = []
    def denoise_fn(seq, idx):
        resolved_order.append(idx)
        return 1
        
    sampler.sample(initial_seq, denoise_fn)
    
    assert len(resolved_order) == 4
    assert set(resolved_order) == {0, 1, 2, 3}
