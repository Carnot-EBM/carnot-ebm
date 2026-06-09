import pytest
import runpy
from pathlib import Path

def test_experiment_3946_r11l_first_solve(monkeypatch):
    """
    REQ-ARC-R11L-FIRST-SOLVE
    SCENARIO-SOLVE-R11L-LEVEL0
    """
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "experiment_3946_r11l_first_solve.py"
    
    # We mock sys.argv to run it quickly
    monkeypatch.setattr("sys.argv", ["experiment_3946_r11l_first_solve.py", "--budget", "60"])
    
    # Run the script module
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as e:
        assert e.code == 0
        
    # Verify the JSON artifact is written
    result_path = Path(__file__).resolve().parents[2] / "results" / "experiment_3946_r11l_first_solve.json"
    assert result_path.exists()
    import json
    data = json.loads(result_path.read_text())
    assert data["ACCURACY_levels_solved"] > 0
    assert data["solved"] is True
