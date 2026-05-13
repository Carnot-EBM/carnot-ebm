import os
import json
import tempfile
from carnot.audit_155 import run_pre_retro_audit_155

def test_audit_155_compliant():
    with tempfile.TemporaryDirectory() as d:
        # Create mock results
        for exp in range(1982, 1994):
            if exp in (1987, 1993):
                continue
            with open(os.path.join(d, f"experiment_{exp}_mock.json"), "w") as f:
                json.dump({"schema": "v1", "honest_verdict": "success"}, f)

        out_path = os.path.join(d, "audit.json")
        run_pre_retro_audit_155(out_path, d)
        
        with open(out_path) as f:
            data = json.load(f)
            
        assert data["milestone"] == 155
        assert len(data["missing_files"]) == 0
        assert data["violated_gates"] == 0
        assert data["compliant_artifacts"] == 10
        assert data["non_compliant_artifacts"] == 0
        assert "Audit complete" in data["honest_verdict"]

def test_audit_155_missing_files():
    with tempfile.TemporaryDirectory() as d:
        # Create mock results but missing one
        for exp in range(1982, 1993): # missing 1993 which is skipped anyway, wait, let's miss 1982
            if exp in (1982, 1987, 1993):
                continue
            with open(os.path.join(d, f"experiment_{exp}_mock.json"), "w") as f:
                json.dump({"schema": "v1", "honest_verdict": "success"}, f)

        out_path = os.path.join(d, "audit.json")
        run_pre_retro_audit_155(out_path, d)
        
        with open(out_path) as f:
            data = json.load(f)
            
        assert len(data["missing_files"]) == 1
        assert data["missing_files"][0] == "experiment_1982"
        assert "Audit failed" in data["honest_verdict"]

def test_audit_155_gate_failure():
    with tempfile.TemporaryDirectory() as d:
        for exp in range(1982, 1994):
            if exp in (1987, 1993):
                continue
            with open(os.path.join(d, f"experiment_{exp}_mock.json"), "w") as f:
                if exp == 1983:
                    json.dump({"schema": "v1", "honest_verdict": "blocked_gate_check_failed"}, f)
                else:
                    json.dump({"schema": "v1", "honest_verdict": "success"}, f)

        out_path = os.path.join(d, "audit.json")
        run_pre_retro_audit_155(out_path, d)
        
        with open(out_path) as f:
            data = json.load(f)
            
        assert data["violated_gates"] == 1
        assert data["compliant_artifacts"] == 10
        assert "Audit failed" in data["honest_verdict"]

def test_audit_155_non_compliant_format():
    with tempfile.TemporaryDirectory() as d:
        for exp in range(1982, 1994):
            if exp in (1987, 1993):
                continue
            with open(os.path.join(d, f"experiment_{exp}_mock.json"), "w") as f:
                if exp == 1984:
                    f.write("invalid json")
                elif exp == 1985:
                    json.dump([], f) # Not a dict
                elif exp == 1986:
                    json.dump({"foo": "bar"}, f) # Missing schema and honest_verdict
                else:
                    json.dump({"schema": "v1", "honest_verdict": "success"}, f)

        out_path = os.path.join(d, "audit.json")
        run_pre_retro_audit_155(out_path, d)
        
        with open(out_path) as f:
            data = json.load(f)
            
        assert data["non_compliant_artifacts"] == 3
        assert "Audit failed" in data["honest_verdict"]
