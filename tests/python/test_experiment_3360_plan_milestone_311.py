import os
import json
import sys

# Add scripts directory to path to import the script directly for coverage
scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts")
sys.path.insert(0, scripts_dir)
import experiment_3360_plan_milestone_311

def test_experiment_3360_plan_milestone_311(tmp_path, monkeypatch, capsys):
    # Change current working directory to a tmp path so we don't mess up the actual results folder during test
    monkeypatch.chdir(tmp_path)
    
    # Run the main function
    experiment_3360_plan_milestone_311.main()
    
    # Capture output
    captured = capsys.readouterr()
    assert "status=success" in captured.out
    assert "proposed 13 tasks" in captured.out
    
    # Check if the deliverable was created
    output_path = os.path.join("results", "experiment_3360_plan_milestone_311.json")
    assert os.path.exists(output_path)
    
    with open(output_path, "r") as f:
        data = json.load(f)
        
    assert data["experiment"] == 3360
    assert data["status"] == "success"
    assert data["honest_verdict"] == "success"
    assert data["tasks_proposed"] == 13
