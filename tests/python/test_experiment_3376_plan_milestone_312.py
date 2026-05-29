import os
import json
import sys

# REQ-AUTO-MILESTONE-PLAN: Planning script generation
scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts")
sys.path.insert(0, scripts_dir)
import experiment_3376_plan_milestone_312

def test_experiment_3376_plan_milestone_312(tmp_path, monkeypatch, capsys):
    # REQ-AUTO-MILESTONE-PLAN: The system shall generate a valid JSON deliverable
    monkeypatch.chdir(tmp_path)
    
    experiment_3376_plan_milestone_312.main()
    
    captured = capsys.readouterr()
    assert "status=success" in captured.out
    assert "proposed 13 tasks" in captured.out
    
    output_path = os.path.join("results", "experiment_3376_plan_milestone_312.json")
    assert os.path.exists(output_path)
    
    with open(output_path, "r") as f:
        data = json.load(f)
        
    assert data["experiment"] == 3376
    assert data["status"] == "success"
    assert data["honest_verdict"] == "success"
    assert data["tasks_proposed"] == 13
