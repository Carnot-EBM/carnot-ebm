import os
import json
import tempfile
from carnot.retro_1679 import generate_retro

def test_generate_retro():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "experiment_1679_retro.json")
        artifact = generate_retro(output_path)
        
        assert os.path.exists(output_path)
        
        with open(output_path, "r") as fh:
            loaded = json.load(fh)
            
        assert loaded["schema"] == "carnot.milestone_research_retro.v1"
        assert loaded["milestone"] == "2026.05.168"
        assert len(loaded["tasks_summary"]) == 4
        assert loaded["gates_passed_count"] == 4
        assert loaded["gates_failed_count"] == 0
        
        assert "gemini" in loaded["actual_agent_backend_distribution"]
        assert loaded["actual_agent_backend_distribution"]["gemini"] == 3
        
        assert len(loaded["paper_v6_carryforward_items"]) == 1
        assert "GateMate" in loaded["paper_v6_carryforward_items"][0]
        
        assert len(loaded["hardware_sovereignty_data_points"]) == 1
        assert loaded["hardware_sovereignty_data_points"][0]["board"] == "Olimex GateMateA1-EVB-2M (CC GM1A1)"
        assert loaded["hardware_sovereignty_data_points"][0]["gate_passed"] is True
        
        assert loaded["adversarial_verify_flag_count"] == 5
        assert loaded["honest_verdict"].startswith("complete:")
