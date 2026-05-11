"""Tests for Exp 1796 Performance Ablation."""

from pathlib import Path
import json

from scripts.experiment_1796_performance_ablation import run_experiment


def test_run_experiment(tmp_path):
    """Test the benchmark generates a valid result artifact."""
    result_file = tmp_path / "experiment_1796_performance_ablation.json"
    result = run_experiment(output_path=result_file)

    assert result["experiment"] == 1796
    assert "cpu_latency_ms" in result
    assert "kv260_latency_ms" in result
    assert "speedup_factor" in result
    assert result["honest_verdict"] == "hardware_acceleration_benchmarked"

    # Assert it was written
    assert result_file.exists()
    with open(result_file) as f:
        data = json.load(f)
    assert data["experiment"] == 1796
