"""Tests for NRGPT-style explorer.

Spec: REQ-PIPELINE-1788
"""
import os
import json
from carnot.inference.nrgpt_explorer import NRGPTExplorer, run_experiment_1788

def test_nrgpt_explorer_explore():
    """SCENARIO-PIPELINE-1788: NRGPT Exploration scales compute based on energy."""
    def dummy_energy(x: float) -> float:
        return x ** 2

    explorer = NRGPTExplorer(base_compute=10.0, energy_scale=2.0)
    result = explorer.explore(dummy_energy, initial_state=5.0)

    assert result["scaled_compute"] == 10.0 * 2.0 * (5.0 ** 2)
    assert result["energy"] == 25.0
    assert result["success"] is True

def test_run_experiment_1788(tmp_path):
    """Test that the experiment artifact is correctly generated."""
    output_path = tmp_path / "results" / "experiment_1788_nrgpt_exploration.json"
    result = run_experiment_1788(str(output_path))

    assert os.path.exists(output_path)
    with open(output_path) as f:
        data = json.load(f)
    
    assert data["experiment_id"] == 1788
    assert data["status"] == "complete"
    assert "scaled_compute" in data
