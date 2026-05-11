"""Tests for Exp 1789 math benchmark.

Spec: REQ-BENCH-1789, SCENARIO-BENCH-1789
"""
import os
import json
from pathlib import Path

from scripts.experiment_1789_math_benchmark import run_experiment_1789

def test_experiment_1789_math_benchmark(tmp_path: Path):
    """Test that the benchmark script produces the required artifact.
    
    Spec: SCENARIO-BENCH-1789
    """
    out_path = tmp_path / "experiment_1789_math_benchmark.json"
    result = run_experiment_1789(str(out_path))
    
    assert os.path.exists(out_path)
    with open(out_path, "r") as f:
        data = json.load(f)
        
    assert data["status"] in ["complete", "blocked"]
    assert "machine_checkable_proof_success_rate" in data
    assert data["model"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert data["experiment_id"] == 1789
    assert "honest_verdict" in data
