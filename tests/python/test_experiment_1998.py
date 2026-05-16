"""Tests for Experiment 1998: Continuous reasoning generation on Sudoku."""
import os
import json
import tempfile
from carnot.pipeline.experiment_1998 import run_sudoku_evaluation, write_artifact

def test_run_sudoku_evaluation():
    """Test that the 5 Sudoku problems are run and return expected structure."""
    artifact = run_sudoku_evaluation()
    
    assert artifact["schema"] == "carnot.benchmark.v4"
    assert artifact["experiment"] == 1998
    assert artifact["model_specs"]["target_model"] == "unsloth/gemma-4-31B-it-GGUF"
    assert artifact["honest_verdict"].startswith("SUCCESS:")
    assert len(artifact["results"]) == 5
    assert artifact["results"][0]["problem_id"] == "sudoku_0"

def test_write_artifact():
    """Test that the artifact can be written to a file correctly."""
    artifact = run_sudoku_evaluation()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "experiment_1998.json")
        write_artifact(artifact, path)
        
        assert os.path.exists(path)
        with open(path, "r") as f:
            loaded = json.load(f)
            assert loaded["experiment"] == 1998
            assert loaded["schema"] == "carnot.benchmark.v4"
