import json
import os
from tempfile import TemporaryDirectory
from carnot.pre_retro_audit import run_pre_retro_audit

def test_run_pre_retro_audit():
    """
    Test generating the milestone 153 pre-retro audit by creating dummy files
    and verifying the generated JSON artifact counts correctly.
    (Traces to REQ-REPORT-1967 / SCENARIO-REPORT-1967)
    """
    with TemporaryDirectory() as tmpdir:
        files_data = [
            ("experiment_1956_success.json", {"schema": "v1", "status": "complete", "logprob": 1, "format": 1, "deterministic": 1}),
            ("experiment_1957_gate_fail.json", {"honest_verdict": "gate check failed", "logprob": 1, "format": 1, "deterministic": 1}),
            ("experiment_1958_fail.json", {"status": "failed", "logprob": 1, "format": 1, "deterministic": 1}),
            ("experiment_1959_missing_checks.json", {"status": "complete", "format": 1, "deterministic": 1}), # missing logprobs
        ]
        
        for fname, data in files_data:
            with open(os.path.join(tmpdir, fname), 'w') as f:
                json.dump(data, f)
                
        out_path = os.path.join(tmpdir, "experiment_1967_milestone_153_pre_retro_audit.json")
        run_pre_retro_audit(out_path, results_dir=tmpdir)
        
        assert os.path.exists(out_path)
        with open(out_path, 'r') as f:
            result = json.load(f)
            
        assert result["schema"] == "carnot.milestone_pre_retro_audit.v1"
        assert result["milestone"] == 153
        assert len(result["missing_files"]) == 7 # 1960 to 1966
        assert result["violated_gates"] == 1
        assert result["compliant_artifacts"] == 3
        assert result["non_compliant_artifacts"] == 1

def test_run_pre_retro_audit_invalid_json():
    """
    Test with an invalid JSON file.
    """
    with TemporaryDirectory() as tmpdir:
        # Create an invalid JSON file
        with open(os.path.join(tmpdir, "experiment_1956_bad.json"), 'w') as f:
            f.write("{ bad json ")
            
        out_path = os.path.join(tmpdir, "experiment_1967_milestone_153_pre_retro_audit.json")
        run_pre_retro_audit(out_path, results_dir=tmpdir)
        
        with open(out_path, 'r') as f:
            result = json.load(f)
            
        assert result["non_compliant_artifacts"] == 1
        
def test_run_pre_retro_audit_not_dict():
    """
    Test with a JSON file that is not a dictionary.
    """
    with TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "experiment_1956_list.json"), 'w') as f:
            f.write('["not", "a", "dict"]')
            
        out_path = os.path.join(tmpdir, "experiment_1967_milestone_153_pre_retro_audit.json")
        run_pre_retro_audit(out_path, results_dir=tmpdir)
        
        with open(out_path, 'r') as f:
            result = json.load(f)
            
        assert result["non_compliant_artifacts"] == 1
