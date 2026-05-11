"""Tests for Exp 1746: EqM test-time sampler profiling."""

import json
from pathlib import Path

import scripts.experiment_1746_eqm_profile as exp


def test_scenario_sample_1746_exp_writes_timing_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-1746: Exp 1746 Profiler Extracts Timing Metrics.
    
    REQ-SAMPLE-1746-1: Configure MODEL_SPECS to use unsloth/Qwen3.6-35B-A3B-GGUF.
    REQ-SAMPLE-1746-2: Implement PyTorch/CUDA profiling.
    REQ-SAMPLE-1746-3: Write results/experiment_1746_profile.json.
    """
    output_path = tmp_path / "experiment_1746_profile.json"
    artifact = exp.run_experiment(output_path=output_path, dimension=64, n_steps=2)

    assert output_path.exists()
    assert output_path.is_file()

    with output_path.open() as f:
        data = json.load(f)

    assert data["experiment_id"] == "1746"
    assert "REQ-SAMPLE-1746" in data["spec_refs"]
    assert data["model_specs"][0]["hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert "total_latency_ms" in data["metrics"]
    assert "profiler_cpu_time_us" in data["metrics"]
    assert "profiler_cuda_time_us" in data["metrics"]
    assert data["honest_verdict"] == "profile_completed"
