import pytest
import json
from pathlib import Path
from carnot.pipeline.experiment_1707 import run_experiment_1707

def test_experiment_1707_full_pipeline(tmp_path: Path):
    output_path = tmp_path / "experiment_1707_full_pipeline.json"
    artifact = run_experiment_1707(output_path)
    
    assert artifact["experiment_id"] == 1707
    assert artifact["total_scenarios_evaluated"] == 100
    assert artifact["honest_verdict"] == "complete: full_pipeline_verified"
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in artifact["models_used"]
    assert "unsloth/gemma-4-31B-it-GGUF" in artifact["models_used"]
    
    # Verify the file was written
    with open(output_path, "r") as f:
        data = json.load(f)
        assert data["experiment_id"] == 1707
