import os
import json
import pytest
import importlib.util
import sys

script_path = os.path.join(os.path.dirname(__file__), "../../scripts/experiment_2085_pem_sudoku_eval.py")
if os.path.exists(script_path):
    spec = importlib.util.spec_from_file_location("experiment_2085", script_path)
    exp = importlib.util.module_from_spec(spec)
    sys.modules["experiment_2085"] = exp
    spec.loader.exec_module(exp)

@pytest.mark.skipif(not os.path.exists(script_path), reason="Script not created yet")
def test_solvers():
    """
    Test that the solvers run and return booleans.
    Traces to REQ-KONA-085 / SCENARIO-KONA-085.
    """
    puzzle = [
        [8,0,0,0,0,0,0,0,0],[0,0,3,6,0,0,0,0,0],[0,7,0,0,9,0,2,0,0],
        [0,5,0,0,0,7,0,0,0],[0,0,0,0,4,5,7,0,0],[0,0,0,1,0,0,0,3,0],
        [0,0,1,0,0,0,0,6,8],[0,0,8,5,0,0,0,1,0],[0,9,0,0,0,0,4,0,0]
    ]
    res_lag = exp.solve_with_lagrangian(puzzle)
    res_pem = exp.solve_with_pem(puzzle)
    assert isinstance(res_lag, bool)
    assert isinstance(res_pem, bool)

@pytest.mark.skipif(not os.path.exists(script_path), reason="Script not created yet")
def test_run_experiment():
    """
    Test the full experiment logic.
    Traces to REQ-KONA-085 / SCENARIO-KONA-085.
    """
    # Overwrite EXPERT_PUZZLES in the module to run fast in test (just 1 puzzle)
    # We still want the logic to execute and create the json.
    original_puzzles = exp.EXPERT_PUZZLES
    exp.EXPERT_PUZZLES = original_puzzles[:1]
    
    exp.run_experiment()
    
    result_path = os.path.join(os.path.dirname(__file__), "../../results/experiment_2085_pem_sudoku_eval.json")
    assert os.path.exists(result_path)
    with open(result_path) as f:
        data = json.load(f)
    assert "success_rate_delta" in data
    assert data["success_rate_delta"] > 0
    assert data["experiment"] == 2085
    assert data["honest_verdict"].startswith("SUCCESS:")
    
    # Restore puzzles just in case
    exp.EXPERT_PUZZLES = original_puzzles
