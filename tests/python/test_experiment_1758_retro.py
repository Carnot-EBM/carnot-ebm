import json
import os
import tempfile
from scripts.experiment_1758_retro import run_synthesis

def test_run_synthesis():
    """Test generating the Phase 5 Synthesis Retrospective for REQ-REPORT-1758."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = tmpdir
        output_path = os.path.join(tmpdir, "experiment_1758_retro.json")
        
        # Create dummy input files mimicking 1746 to 1757
        dummy_data_1 = {"honest_verdict": "success", "metric_a": 10}
        with open(os.path.join(input_dir, "experiment_1746_profile.json"), "w") as f:
            json.dump(dummy_data_1, f)
            
        dummy_data_2 = {"honest_verdict": "partial", "error_rate": 0.05}
        with open(os.path.join(input_dir, "experiment_1757_e2e_multi_agent.json"), "w") as f:
            json.dump(dummy_data_2, f)
            
        result = run_synthesis(input_dir, output_path)
        
        assert os.path.exists(output_path)
        assert result["milestone"] == "2026.05.135"
        assert result["honest_verdict"] == "phase_5_synthesis_complete"
        assert isinstance(result["new_gaps"], list)
        assert len(result["new_gaps"]) > 0
        assert "details" in result
        assert result["details"]["parsed_files_count"] == 2
        
        # Verify the file content matches the returned result
        with open(output_path, "r") as f:
            saved_data = json.load(f)
            assert saved_data == result

def test_run_synthesis_no_files():
    """Test behavior when no relevant files are found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "experiment_1758_retro.json")
        result = run_synthesis(tmpdir, output_path)
        
        assert result["milestone"] == "2026.05.135"
        assert result["honest_verdict"] == "phase_5_synthesis_complete"
        assert result["details"]["parsed_files_count"] == 0

def test_run_synthesis_invalid_json():
    """Test behavior when an invalid JSON file is encountered."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "experiment_1758_retro.json")
        
        # Create an invalid JSON file
        invalid_file_path = os.path.join(tmpdir, "experiment_1746_invalid.json")
        with open(invalid_file_path, "w") as f:
            f.write("this is not json")
            
        result = run_synthesis(tmpdir, output_path)
        
        assert result["details"]["parsed_files_count"] == 0
        assert result["details"]["experiment_summaries"]["experiment_1746_invalid.json"] == "unknown"
