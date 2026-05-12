import json
import os
from tempfile import TemporaryDirectory
from carnot.retro_152 import generate_retro

def test_generate_retro():
    """
    Test generating the milestone 152 retrospective by creating dummy files
    and verifying the generated JSON artifact counts correctly.
    """
    with TemporaryDirectory() as tmpdir:
        files_data = [
            ("experiment_1944_blocked.json", {"honest_verdict": "blocked_gate_check_failed"}),
            ("experiment_1946_complete.json", {"status": "complete"}),
            ("experiment_1949_success.json", {"result": "success"}),
            ("experiment_1950_failed.json", {"honest_verdict": "failed_due_to_error"}),
            ("experiment_1953_blocked.json", {"status": "blocked"}),
            ("experiment_1955_ignored.json", {"status": "failed"}) # Out of range
        ]
        
        for fname, data in files_data:
            with open(os.path.join(tmpdir, fname), 'w') as f:
                json.dump(data, f)
                
        out_path = os.path.join(tmpdir, "experiment_1955_milestone_152_retro.json")
        generate_retro(out_path, results_dir=tmpdir)
        
        assert os.path.exists(out_path)
        with open(out_path, 'r') as f:
            result = json.load(f)
            
        assert result["schema"] == "carnot.milestone_retro.v1"
        assert result["milestone"] == 152
        assert result["completed_task_count"] == 2  
        assert result["blocked_task_count"] == 2    
        assert result["failed_task_count"] == 1     
        assert len(result["recommendations"]) > 0
