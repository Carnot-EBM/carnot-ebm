import json
import os
from carnot.experiment_1684_retro import generate_retro

def test_experiment_1684_retro(tmp_path):
    """
    Test that the retro matches REQ-REPORT-169 requirements.
    """
    out_path = tmp_path / "experiment_1684_retro.json"
    retro = generate_retro(str(out_path))

    assert os.path.exists(out_path)
    with open(out_path, "r") as f:
        data = json.load(f)

    assert data["schema"] == "carnot.milestone_research_retro.v1"
    assert data["milestone"] == "2026.05.169"
    assert "tasks_summary" in data
    assert len(data["tasks_summary"]) == 4

    assert data["gates_passed_count"] == 3
    assert data["gates_failed_count"] == 1
    assert "gemini" in data["actual_agent_backend_distribution"]

    assert len(data["paper_v6_carryforward_items"]) > 0
    assert "THRML" in data["paper_v6_carryforward_items"][0]

    assert data["phase1_ship_progress_pp_remaining"] == 8
    assert data["adversarial_verify_flag_count"] == 2
    assert data["honest_verdict"].startswith("complete:")
