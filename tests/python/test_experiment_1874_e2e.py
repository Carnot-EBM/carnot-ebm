import json
import pytest
from pathlib import Path

from experiment_1874_e2e import run_experiment, build_artifact, execute_pipeline

def test_execute_pipeline():
    result = execute_pipeline()
    assert result["cross_language_equivalences"] is True
    assert result["serialization_successful"] is True
    assert result["sampling_pipelines_successful"] is True
    assert result["roce_enforced"] is True
    assert result["hiled_enforced"] is True
    assert result["continuous_learning_updates"] is True
    assert "unsloth/gemma-4-31B-it-GGUF" in result["evaluated_models"]

def test_build_artifact():
    artifact = build_artifact(duration_s=1.5)
    assert artifact["status"] == "complete"
    assert artifact["schema"] == "carnot.experiment_1874_e2e.v1"
    assert artifact["experiment_id"] == 1874
    assert artifact["cross_language_equivalences"] is True
    assert artifact["serialization_successful"] is True
    assert artifact["sampling_pipelines_successful"] is True
    assert artifact["honest_verdict"] == "complete: triple_integration_e2e_successful"
    assert artifact["duration_s"] == 1.5

def test_run_experiment(tmp_path):
    output_path = tmp_path / "experiment_1874_e2e.json"
    result = run_experiment(output_path=output_path)
    
    assert output_path.exists()
    
    with open(output_path, "r", encoding="utf-8") as f:
        saved_artifact = json.load(f)
        
    assert saved_artifact["status"] == "complete"
    assert saved_artifact["experiment_id"] == 1874
    assert saved_artifact["honest_verdict"] == "complete: triple_integration_e2e_successful"
    
    assert result == saved_artifact
