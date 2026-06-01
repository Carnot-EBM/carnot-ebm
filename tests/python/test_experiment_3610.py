import os
import json
import pytest
from unittest.mock import patch
from pathlib import Path
import sys

# Import the run function directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.experiment_3610_capstone_v331 import run

@patch("scripts.experiment_3610_capstone_v331.Path")
@patch("scripts.experiment_3610_capstone_v331.glob.glob")
def test_experiment_3610_capstone_v331(mock_glob, mock_path, tmp_path):
    # Setup mock Path to return our tmp_path for results
    mock_results_dir = tmp_path / "results"
    mock_results_dir.mkdir()
    
    # We want Path("results") to return our mocked path
    mock_path.return_value = mock_results_dir
    
    # Set up some dummy upstream files
    # 3598
    exp_3598_path = mock_results_dir / "experiment_3598.json"
    with open(exp_3598_path, "w") as f:
        json.dump({"reproducibility_checksum": "a5a63d9b90e6bfe9261e70b66afec721"}, f)
    # 3600 blocked
    exp_3600_path = mock_results_dir / "experiment_3600.json"
    with open(exp_3600_path, "w") as f:
        json.dump({"status": "blocked"}, f)
        
    def glob_side_effect(pattern):
        if "3598" in pattern:
            return [str(exp_3598_path)]
        elif "3600" in pattern:
            return [str(exp_3600_path)]
        elif "3601" in pattern:
            return [] # Missing file case
        elif "3602" in pattern:
            # Test exception case (invalid JSON)
            bad_path = mock_results_dir / "bad.json"
            with open(bad_path, "w") as f:
                f.write("bad json")
            return [str(bad_path)]
        return []
        
    mock_glob.side_effect = glob_side_effect
    
    run()
    
    # Check that output is generated
    out_file = mock_results_dir / "experiment_3610_capstone_v331.json"
    assert out_file.exists()
    
    with open(out_file) as f:
        data = json.load(f)
        
    assert data["honest_verdict"] == "complete: capstone_v331_329_null_was_confirmed_verifier_value_math_only_earned_gate_cascade_fixed_paper_ready_true"
    assert data["gate_cascade_fixed"] is True
    assert data["cited_upstream_artifacts"]["3598"] == "a5a63d9b90e6bfe9261e70b66afec721"
    assert data["cited_upstream_artifacts"]["3600"] == "blocked"
    assert "3601" not in data["cited_upstream_artifacts"] # Missing
    assert "3602" not in data["cited_upstream_artifacts"] # Exception hit so no update
