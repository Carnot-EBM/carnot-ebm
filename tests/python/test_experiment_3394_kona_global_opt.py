"""Tests for Experiment 3394: Kona Global Optimization Emulation on Hard Sudoku.

These tests trace to REQ-KONA-3394 / SCENARIO-KONA-3394.
"""
import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import json
import pytest
import jax

from scripts.experiment_3394_kona_global_opt import run_sudoku_optimization

@pytest.fixture(autouse=True)
def clear_jax_cache():
    """Clear JAX compilation caches to prevent memory leaks in the pytest watchdog."""
    yield
    jax.clear_caches()

def test_run_sudoku_optimization():
    """Test that the global optimization procedure works and returns expected JSON structure."""
    artifact = run_sudoku_optimization()
    
    assert "solved_sudoku" in artifact
    assert "time_to_solution" in artifact
    assert "honest_verdict" in artifact
    assert "SUCCESS" in artifact["honest_verdict"] or "FAILURE" in artifact["honest_verdict"]
    
def test_experiment_output_json():
    """Test that the artifact file is successfully written with all required fields."""
    artifact = run_sudoku_optimization()
    
    result_path = os.path.join(os.path.dirname(__file__), "../../results/experiment_3394_kona_global_opt.json")
    with open(result_path, "w") as f:
        json.dump(artifact, f)

    assert os.path.exists(result_path)
    with open(result_path, "r") as f:
        data = json.load(f)
    assert "solved_sudoku" in data
    assert "time_to_solution" in data
    assert "honest_verdict" in data
