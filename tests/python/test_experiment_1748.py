"""Tests for experiment 1748 benchmark."""

import json
from pathlib import Path
import pytest
import scripts.experiment_1748_benchmark as exp

def test_scenario_sample_1748_benchmark(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-1748: Benchmark sparse EqM latency."""
    output_path = tmp_path / "experiment_1748_benchmark.json"
    
    # Run with small parameters for fast testing
    artifact = exp.run_experiment(
        output_path=output_path, 
        dimension=128, 
        batch_size=2, 
        n_steps=2
    )
    
    assert output_path.exists()
    
    data = json.loads(output_path.read_text())
    
    assert data["experiment_id"] == "1748"
    assert "REQ-SAMPLE-1748" in data["spec_refs"]
    assert "metrics" in data
    assert "latency_ms" in data["metrics"]
    assert data["metrics"]["batch_size"] == 2
    assert data["metrics"]["dimension"] == 128
    assert data["metrics"]["n_steps"] == 2
    assert data["honest_verdict"] in ["success", "failed"]
