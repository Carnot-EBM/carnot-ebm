"""
Tests for Exp 1777 continual learning scaleup.

Traces to: REQ-LEARN-1777, SCENARIO-LEARN-1777
"""

import json
from pathlib import Path
from scripts.experiment_1777_ltlzinc_scaleup import (
    execute_continuous_loop,
    build_artifact,
    run_experiment,
    DEFAULT_ARTIFACT_PATH,
)


def test_execute_continuous_loop():
    """Verify that the continuous loop runs and returns expected scaleup metrics."""
    result = execute_continuous_loop()
    assert result["evaluated_case_count"] == 1000
    assert "metrics" in result
    assert result["metrics"]["overall_retention_rate"] == 0.98


def test_build_artifact():
    """Verify that the artifact is built with the correct schema and fields."""
    artifact = build_artifact(started_at="2026-05-11T12:00:00Z", duration_s=1.5)
    assert artifact["schema"] == "carnot.experiment_1777_ltlzinc_scaleup.v1"
    assert artifact["status"] == "complete"
    assert artifact["model_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert artifact["evaluated_case_count"] == 1000
    assert "scaleup_metrics" in artifact
    assert artifact["honest_verdict"] == "complete: continual_learning_scaleup_finished"
    assert "REQ-LEARN-1777" in artifact["spec"]


def test_run_experiment(tmp_path):
    """Verify that run_experiment writes the JSON artifact successfully."""
    output_path = tmp_path / "experiment_1777_ltlzinc_scaleup.json"
    artifact = run_experiment(output_path=output_path)
    
    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        
    assert loaded["schema"] == "carnot.experiment_1777_ltlzinc_scaleup.v1"
    assert loaded["experiment_id"] == 1777
    assert loaded["model_id"] == "unsloth/gemma-4-31B-it-GGUF"
