import os
import json
import sys

# REQ-AUTO-3391
# SCENARIO-AUTO-3391

def test_experiment_3391_plan_milestone_313(tmp_path, monkeypatch, capsys):
    # REQ-AUTO-3391: The system shall generate a valid JSON deliverable
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts")
    sys.path.insert(0, scripts_dir)
    import experiment_3391_plan_milestone_313
    
    monkeypatch.chdir(tmp_path)
    
    experiment_3391_plan_milestone_313.main()
    
    captured = capsys.readouterr()
    assert "status=success" in captured.out
    
    output_path = os.path.join("results", "experiment_3391_plan_milestone_313.json")
    assert os.path.exists(output_path)
    
    with open(output_path, "r") as f:
        data = json.load(f)
        
    assert data["experiment"] == 3391
    assert data["status"] == "success"
    assert data["honest_verdict"] == "success"
    assert "tasks_proposed" in data
    assert isinstance(data["tasks_proposed"], int)
