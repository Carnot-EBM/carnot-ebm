"""Tests for Experiment 3408: Kona Global Optimization Emulation on Hard Sudoku.

These tests trace to REQ-KONA-3408 / SCENARIO-KONA-3408.
"""
import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import json
import pytest
import jax

from scripts.experiment_3408_kona_global_opt import run_sudoku_optimization

@pytest.fixture(autouse=True)
def clear_jax_cache():
    """Clear JAX compilation caches to prevent memory leaks in the pytest watchdog."""
    yield
    jax.clear_caches()

def test_run_sudoku_optimization():
    """Test that the global optimization procedure works and returns expected JSON structure."""
    artifact = run_sudoku_optimization()
    
    assert "status" in artifact
    assert artifact["status"] == "success"
    assert "time_to_solution" in artifact
    
def test_experiment_output_json():
    """Test that the artifact file is successfully written with all required fields."""
    artifact = run_sudoku_optimization()
    
    result_path = os.path.join(os.path.dirname(__file__), "../../results/experiment_3408_kona_global_opt.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(artifact, f)

    assert os.path.exists(result_path)
    with open(result_path, "r") as f:
        data = json.load(f)
    assert "status" in data
    assert data["status"] == "success"
    assert "time_to_solution" in data
