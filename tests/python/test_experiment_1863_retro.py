import os
import json
import tempfile
from carnot.reporting.experiment_1863_retro import generate_retro

def test_generate_retro():
    """
    Test that REQ-REPORT-0863 is satisfied.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "experiment_1863_retro.json")
        generate_retro(out_path, 1.0, 1.0)

        assert os.path.exists(out_path)
        with open(out_path, "r") as f:
            result = json.load(f)
        
        assert result["schema"] == "carnot.milestone_research_retro.v1"
        assert result["milestone"] == "2026.05.145"
        assert result["vl_proxy_pass_rate"] == 1.0
        assert result["s2kan_pass_rate"] == 1.0
        assert result["honest_verdict"] == "milestone_145_retro_complete"
