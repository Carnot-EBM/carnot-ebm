"""Tests for Exp 1998: Live IT Baselines with GSM8K.

Spec: REQ-VERIFY-1998, SCENARIO-VERIFY-1998
"""

import json
from pathlib import Path

from scripts.experiment_1998_live_it_baselines_gsm8k import EXPERIMENT_ID, TITLE, run_experiment

def test_run_experiment_1998_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1998: Run GSM8K Baseline writes artifact with required fields."""
    deliverable_path = tmp_path / "experiment_1998_live_it_baselines_gsm8k.json"
    
    result = run_experiment(output_path=deliverable_path)
    
    assert deliverable_path.exists()
    
    with open(deliverable_path, "r") as f:
        artifact = json.load(f)
        
    assert artifact["experiment"] == EXPERIMENT_ID
    assert artifact["title"] == TITLE
    assert artifact["status"] == "success"
    assert artifact["total_questions"] == 200
    assert artifact["inference_mode"] == "live_gpu"
    assert "tp_rate" in artifact
    assert "fp_rate" in artifact
    assert "responses" in artifact
    
    assert len(artifact["responses"]) == 600  # 200 questions * 3 models
