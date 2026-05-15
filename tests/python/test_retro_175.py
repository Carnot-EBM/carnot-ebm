import os
import json
import tempfile
from carnot.retro_175 import generate_retro

def test_generate_retro_175():
    """
    Test generating the .175 operational retrospective.
    Satisfies REQ-REPORTING-001 (Operational Retrospectives).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "operational_retro_2026_05_175.json")
        artifact = generate_retro(output_path)
        
        assert os.path.exists(output_path)
        
        with open(output_path, "r") as fh:
            loaded = json.load(fh)
            
        assert loaded["schema"] == "carnot.operational_retro.v64"
        assert loaded["milestone"] == "2026.05.175"
        assert loaded["retro_type"] == "operational_skip_recovery"
        assert loaded["experiments_completed"] == 2
        assert loaded["compute_bound_experiments_count"] == 1
        assert len(loaded["slowest_experiments"]) == 2
        assert loaded["skip_recovery_rate"] > 0
        assert "bottlenecks_identified" in loaded
        assert "improvements_suggested" in loaded
        assert len(loaded["top_3_highest_leverage_actions"]) == 3
        assert loaded["honest_verdict"].startswith("complete:")
