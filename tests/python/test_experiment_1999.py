"""Tests for Experiment 1999: Ising-Guided Fuzzing for Code Verification.

Spec: REQ-CODE-035, SCENARIO-CODE-033
"""
import os
import json
import tempfile
from carnot.pipeline.experiment_1999 import run_humaneval_fuzzing, write_artifact

def test_run_humaneval_fuzzing():
    """Test that the 50 HumanEval questions are run and return expected structure.
    
    Spec: REQ-CODE-035, SCENARIO-CODE-033
    """
    artifact = run_humaneval_fuzzing()
    
    assert artifact["experiment_id"] == 1999
    assert artifact["dataset_size"] == 50
    assert "baseline_pass_rate" in artifact
    assert "repair_pass_rate" in artifact
    assert "honest_verdict" in artifact
    assert len(artifact["results"]) == 50
    assert artifact["results"][0]["task_id"] == "HumanEval/0"
    
def test_write_artifact():
    """Test that the artifact can be written to a file correctly.
    
    Spec: REQ-CODE-035, SCENARIO-CODE-033
    """
    artifact = run_humaneval_fuzzing()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "experiment_1999.json")
        write_artifact(artifact, path)
        
        assert os.path.exists(path)
        with open(path, "r") as f:
            loaded = json.load(f)
            assert loaded["experiment_id"] == 1999
            assert loaded["dataset_size"] == 50
