import pytest
import os
import json
from carnot.eval.four_delta_bound import compute_four_delta_metrics, gather_successful_traces, run_evaluation

def test_compute_four_delta_metrics():
    # Mock traces: 2 traces.
    # Trace 1: fail, pass (2 iterations, 1 success)
    # Trace 2: fail, fail, pass (3 iterations, 1 success)
    traces = [
        [{"iteration": 0, "passed": False}, {"iteration": 1, "passed": True}],
        [{"iteration": 0, "passed": False}, {"iteration": 1, "passed": False}, {"iteration": 2, "passed": True}]
    ]
    checksum = "mock_checksum"
    
    # Total attempts = 5
    # Total successes = 2
    # delta = 2 / 5 = 0.4
    # mean_iterations = (2 + 3) / 2 = 2.5
    # predicted_bound = 4 / 0.4 = 10.0
    # 2.5 <= 10.0 -> True
    
    artifact = compute_four_delta_metrics(traces, checksum)
    assert artifact is not None
    assert artifact["schema"] == "carnot.four_delta_bound_empirical.v1"
    assert artifact["n_runs"] == 2
    assert abs(artifact["delta_empirical"] - 0.4) < 1e-6
    assert abs(artifact["mean_iterations"] - 2.5) < 1e-6
    assert abs(artifact["predicted_bound"] - 10.0) < 1e-6
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["reproducibility_checksum"] == "mock_checksum"
    assert "complete: four_delta_bound_validated_empirical_delta_0.40_mean_n_2.5_predicted_10.00" == artifact["honest_verdict"]

def test_compute_four_delta_metrics_empty():
    assert compute_four_delta_metrics([], "checksum") is None

def test_run_evaluation(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    output_path = tmp_path / "experiment_2108_four_delta_bound.json"
    
    # Write a mock JSON file
    mock_data = {
        "per_problem_results": [
            {
                "verify_repair": {
                    "iterations": [
                        {"iteration": 0, "passed": False},
                        {"iteration": 1, "passed": True}
                    ]
                }
            }
        ]
    }
    
    mock_file = results_dir / "experiment_100_mock.json"
    with open(mock_file, 'w') as f:
        json.dump(mock_data, f)
        
    success = run_evaluation(str(results_dir), str(output_path))
    assert success is True
    assert output_path.exists()
    
    with open(output_path, 'r') as f:
        output_data = json.load(f)
        
    assert output_data["schema"] == "carnot.four_delta_bound_empirical.v1"
    assert output_data["n_runs"] == 1
    assert output_data["delta_empirical"] == 0.5
    assert output_data["mean_iterations"] == 2.0
    assert output_data["predicted_bound"] == 8.0
    assert output_data["acceptance_gate_passed"] is True
