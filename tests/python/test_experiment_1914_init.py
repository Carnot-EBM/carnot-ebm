import os
import json
import tempfile
from carnot.reporting.experiment_1914_init import generate_init

def test_generate_init():
    """
    Test that the milestone init artifact is generated correctly.
    
    References REQ-REPORT-195.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "experiment_1914_init.json")
        generate_init(out_path)

        assert os.path.exists(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        
        assert result["schema"] == "carnot.init.v1"
        assert result["experiment"] == 1914
        assert result["status_updated"] is True
        assert result["honest_verdict"] == "complete: initialized .195"
