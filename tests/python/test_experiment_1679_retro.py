import json
import os
from carnot.experiment_1679_retro import generate_retro

def test_retro_generation_REQ_REPORT_168(tmp_path):
    """
    Test that the retro matches REQ-REPORT-168 requirements.
    """
    out_path = tmp_path / "experiment_1679_retro.json"
    retro = generate_retro(str(out_path))
    
    assert os.path.exists(out_path)
    with open(out_path, "r") as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.milestone_research_retro.v1"
    assert data["milestone"] == "2026.05.168"
    assert "tasks_summary" in data
    assert len(data["tasks_summary"]) == 4
    
    assert data["gates_passed_count"] == 4
    assert data["gates_failed_count"] == 0
    assert "gemini" in data["actual_agent_backend_distribution"]
    
    assert len(data["paper_v6_carryforward_items"]) > 0
    assert "KV260" in data["paper_v6_carryforward_items"][0]
    
    assert data["hardware_sovereignty_data_points"][0]["board"] == "Olimex GateMateA1-EVB-2M (CC GM1A1)"
    assert data["hardware_sovereignty_data_points"][0]["gate_passed"] is True
    
    assert data["adversarial_verify_flag_count"] == 4
    assert data["honest_verdict"].startswith("complete:")
