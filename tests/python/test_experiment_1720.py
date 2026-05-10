import json
from pathlib import Path

from carnot.pipeline.experiment_1720 import run_experiment_1720

def test_experiment_1720_full_pipeline(tmp_path: Path):
    # Spec: REQ-EXPERIMENT-1720, SCENARIO-EXPERIMENT-1720
    output_path = tmp_path / "experiment_1720_full_pipeline.json"
    artifact = run_experiment_1720(output_path)
    
    assert artifact["experiment_id"] == 1720
    assert artifact["total_scenarios_evaluated"] == 100
    assert artifact["honest_verdict"] == "complete: full_pipeline_verified"
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in artifact["models_used"]
    assert "unsloth/gemma-4-31B-it-GGUF" in artifact["models_used"]
    assert "pruned_constraints" in artifact
    
    # Verify the file was written
    with open(output_path, "r") as f:
        data = json.load(f)
        assert data["experiment_id"] == 1720
