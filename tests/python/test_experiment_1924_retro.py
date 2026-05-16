import os
import json
import tempfile
from carnot.reporting.experiment_1924_retro import generate_retro

def test_generate_retro_1924():
    """
    Test generating the .195 retrospective artifact.
    Satisfies REQ-REPORT-1924.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "experiment_1924_retro.json")
        data = generate_retro(output_path)
        
        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            loaded_data = json.load(f)
            
        assert loaded_data["schema"] == "carnot.retro.v1"
        assert loaded_data["experiment"] == 1924
        assert loaded_data["honest_verdict"] == "complete: .195 finished"
        assert "retrospective_summary" in loaded_data
        assert data == loaded_data
