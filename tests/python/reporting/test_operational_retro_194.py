import os
import json
import tempfile
from carnot.reporting.operational_retro_194 import generate_retro_194

def test_generate_retro_194():
    """
    Test that the retro for 194 generates correctly with the specific flag-fields.
    Satisfies REQ-REPORT-194.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "operational_retro_2026_05_194.json")
        generate_retro_194(out_path)

        assert os.path.exists(out_path)
        with open(out_path, "r") as f:
            result = json.load(f)
        
        assert result["schema"] == "carnot.operational_retro.v64"
        assert result["milestone"] == "2026.05.194"
        assert "generated_at" in result
        assert result["retro_type"] == "operational_full"
        assert result["adversarial_confirmation_result"] == "confirmed"
        assert result["pypi_ship_result"] == "blocked"
        assert result["phase4_closure_result"] == "decision_rendered"
        assert result["honest_verdict"].startswith("success:")
