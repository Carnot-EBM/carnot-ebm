"""Tests for KAN formal verification bounds.

Spec coverage: REQ-CORE-001, SCENARIO-CORE-001
"""

import json
import pytest
from pathlib import Path
from carnot.models.kan.formal_verification import compute_empirical_certified_bounds, load_telemetry

def test_load_telemetry(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"token_logprobs": [0.1, 0.2]}) + "\n" +
        json.dumps({"token_logprobs": [0.3, 0.4]}) + "\n"
    )
    examples = load_telemetry(str(manifest_path), 2)
    assert len(examples) == 2
    assert examples[0] == [0.1, 0.2]

def test_compute_empirical_certified_bounds(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for _ in range(20):
            f.write(json.dumps({"token_logprobs": [0.1] * 32}) + "\n")
            
    result = compute_empirical_certified_bounds(
        telemetry_manifest_path=str(manifest_path),
        n_examples=20,
        perturb_delta=0.1,
        random_seed=42,
        input_dim=32
    )
    
    assert "certified_coverage" in result
    assert result["certified_coverage"] >= 0.0
    assert result["certified_coverage"] <= 1.0
    assert "mean_local_lipschitz" in result
    assert result["n_eval_examples"] == 20
