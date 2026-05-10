import json
import os
import tempfile
from scripts.experiment_1745_retro import run_synthesis

def test_run_synthesis():
    """Test generating the Phase 4 Synthesis Retrospective for REQ-REPORT-1745."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.json")
        output_path = os.path.join(tmpdir, "output.json")
        
        test_data = {
            "eqm_latency_overhead_ms": 150.5,
            "accuracy_gain_pct": 4.2,
            "repair_success_rate": 0.85
        }
        with open(input_path, "w") as f:
            json.dump(test_data, f)
            
        result = run_synthesis(input_path, output_path)
        
        assert os.path.exists(output_path)
        assert result["milestone"] == "2026.05.134"
        assert result["honest_verdict"] == "phase_4_synthesis_complete"
        assert "150.5 ms" in result["hardware_resolution"]
        assert "4.2%" in result["system_2_eqm_accuracy"]
        assert "0.85" in result["continuous_learning_scale_up"]
        assert len(result["gaps_for_135"]) > 0
        
        # Test error handling (missing file)
        bad_input_path = os.path.join(tmpdir, "bad.json")
        result_err = run_synthesis(bad_input_path, output_path)
        assert result_err["honest_verdict"] == "phase_4_synthesis_complete"
        assert "unknown" in result_err["continuous_learning_scale_up"]
