import json
import pytest
from pathlib import Path
from carnot.pipeline.empirical_delta import compute_empirical_delta

def test_compute_empirical_delta(tmp_path):
    # Tests REQ-PIPELINE-EMPIRICAL-DELTA and SCENARIO-PIPELINE-EMPIRICAL-DELTA
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    mock_data = {
        "per_seed": [
            {"repair_iterations": 10, "converged": True},
            {"repair_iterations": 20, "converged": False},
            {"adaptive_repair_iterations": 5, "adaptive_converged": True}
        ]
    }
    
    with open(results_dir / "experiment_test.json", "w") as f:
        json.dump(mock_data, f)
        
    delta = compute_empirical_delta(results_dir)
    assert abs(delta - (2 / 35)) < 1e-6

def test_compute_empirical_delta_empty(tmp_path):
    # Tests REQ-PIPELINE-EMPIRICAL-DELTA
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "dir.json").mkdir()
    delta = compute_empirical_delta(results_dir)
    assert delta == 0.0

def test_compute_empirical_delta_exception(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    with open(results_dir / "invalid.json", "w") as f:
        f.write("{invalid")
    delta = compute_empirical_delta(results_dir)
    assert delta == 0.0
