import os
import json
import tempfile
from carnot.reporting.experiment_1914_retro import generate_retro

def test_generate_retro():
    """
    Test that the retro for 1914 generates correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "experiment_1914_retro.json")
        generate_retro(out_path)

        assert os.path.exists(out_path)
        with open(out_path, "r") as f:
            result = json.load(f)
        
        assert result["schema"] == "carnot.milestone_research_retro.v1"
        assert result["milestone"] == "2026.05.193"
        assert result["date"] == "20260516"
        assert result["honest_verdict"].startswith("complete:")
