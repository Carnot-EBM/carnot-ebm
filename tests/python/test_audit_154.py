import json
import os
from tempfile import TemporaryDirectory
from carnot.audit_154 import run_pre_retro_audit_154

def test_run_pre_retro_audit_154():
    """
    Test generating the milestone 154 pre-retro audit by creating dummy files
    and verifying the generated JSON artifact counts correctly.
    
    Spec traces: REQ-REPORT-154, SCENARIO-REPORT-154
    """
    with TemporaryDirectory() as tmpdir:
        files_data = [
            ("experiment_1969_success.json", {"schema": "v1", "status": "complete", "logprob": 1, "format": 1, "zero-false-accept": 1}),
            ("experiment_1970_gate_fail.json", {"honest_verdict": "gate check failed", "logprob": 1, "format": 1, "zero-false-accept": 1}),
            ("experiment_1971_fail.json", {"status": "failed", "logprob": 1, "format": 1, "zero-false-accept": 1}),
            ("experiment_1972_missing_checks.json", {"status": "complete", "format": 1, "zero-false-accept": 1}), # missing logprobs
        ]
        
        for fname, data in files_data:
            with open(os.path.join(tmpdir, fname), 'w') as f:
                json.dump(data, f)
                
        out_path = os.path.join(tmpdir, "experiment_1980_milestone_154_pre_retro.json")
        run_pre_retro_audit_154(out_path, results_dir=tmpdir)
        
        assert os.path.exists(out_path)
        with open(out_path, 'r') as f:
            result = json.load(f)
            
        assert result["schema"] == "carnot.milestone_pre_retro_audit.v1"
        assert result["milestone"] == 154
        assert len(result["missing_files"]) == 6 # 1973 to 1978 (1979 is skipped)
        assert result["violated_gates"] == 1
        assert result["compliant_artifacts"] == 2
        assert result["non_compliant_artifacts"] == 1

def test_run_pre_retro_audit_154_invalid_json():
    """
    Test with an invalid JSON file.
    """
    with TemporaryDirectory() as tmpdir:
        # Create an invalid JSON file
        with open(os.path.join(tmpdir, "experiment_1969_bad.json"), 'w') as f:
            f.write("{ bad json ")
            
        out_path = os.path.join(tmpdir, "experiment_1980_milestone_154_pre_retro.json")
        run_pre_retro_audit_154(out_path, results_dir=tmpdir)
        
        with open(out_path, 'r') as f:
            result = json.load(f)
            
        assert result["non_compliant_artifacts"] == 1
        
def test_run_pre_retro_audit_154_not_dict():
    """
    Test with a JSON file that is not a dictionary.
    """
    with TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "experiment_1969_list.json"), 'w') as f:
            f.write('["not", "a", "dict"]')
            
        out_path = os.path.join(tmpdir, "experiment_1980_milestone_154_pre_retro.json")
        run_pre_retro_audit_154(out_path, results_dir=tmpdir)
        
        with open(out_path, 'r') as f:
            result = json.load(f)
            
        assert result["non_compliant_artifacts"] == 1
