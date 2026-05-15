import os
import json
import tempfile
from carnot.retro_1797 import generate_retro

# Spec traces: REQ-RETRO-187, SCENARIO-RETRO-187
def test_generate_retro():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "experiment_1797_milestone_187_retrospective.json")
        artifact = generate_retro(output_path)
        
        assert os.path.exists(output_path)
        
        with open(output_path, "r") as fh:
            loaded = json.load(fh)
            
        assert loaded["schema"] == "carnot.milestone_research_retro.v1"
        assert loaded["milestone"] == "2026.05.187"
        assert len(loaded["successes"]) == 1
        assert len(loaded["failures_and_blocks"]) == 10
        assert loaded["status"] == "complete"
        assert loaded["honest_verdict"].startswith("complete:")
