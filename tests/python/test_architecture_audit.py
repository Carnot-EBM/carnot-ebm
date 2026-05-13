import os
import json
import tempfile
import sys
from unittest.mock import patch
from carnot.phase3.architecture_audit import audit_continuous_execution, main

def test_audit_continuous_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some mock experiment results
        for i in range(12):
            data = {"experiment": 2040 + i, "title": f"Test {i}"}
            if i == 5:
                data["title"] = "Test with EqM"
            if i == 11:
                # Add an invalid json file to test the exception handling path
                with open(os.path.join(tmpdir, f"experiment_{2040+i}.json"), "w") as f:
                    f.write("invalid json")
                continue
                
            with open(os.path.join(tmpdir, f"experiment_{2040+i}.json"), "w") as f:
                json.dump(data, f)
                
        # Run audit
        result = audit_continuous_execution(tmpdir)
        
        assert result["experiment"] == 2051
        assert "run_date" in result
        assert len(result["analyzed_tasks"]) == 11
        assert len(result["divergence_conflicts"]) > 0
        
        conflict_found = any("EqM" in c["conflict"] for c in result["divergence_conflicts"])
        assert conflict_found

        # Test main method
        with patch('carnot.phase3.architecture_audit.audit_continuous_execution') as mock_audit:
            mock_audit.return_value = {"mock": "data"}
            with patch('builtins.open') as mock_open:
                main()
                mock_audit.assert_called_once_with("results")
                mock_open.assert_called_once_with(os.path.join("results", "experiment_2051_architecture_audit.json"), "w")
