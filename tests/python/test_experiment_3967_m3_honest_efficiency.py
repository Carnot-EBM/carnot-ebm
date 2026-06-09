import json
from pathlib import Path
from unittest.mock import patch
import runpy

def test_experiment_3967_blocked_verifier() -> None:
    # REQ-PHASE4-007, SCENARIO-PHASE4-007
    script_path = Path("scripts/experiments/experiment_3967_m3_honest_efficiency.py")
    
    with patch("builtins.print") as mock_print:
        runpy.run_path(str(script_path), run_name="__main__")
        
    result_file = Path("results/experiment_3967_m3_honest_efficiency.json")
    assert result_file.exists()
    
    data = json.loads(result_file.read_text())
    assert data["honest_verdict"] == "blocked_verifier_not_in_loop"
    assert data["verifier_invoked_in_loop"] is False
    assert data["actions_from_real_env"] is False
    assert data["n_real_env_steps"] == 0
    assert data["efficiency_ratio_with_over_without"] == 0.0
    
    mock_print.assert_called_with("blocked_verifier_not_in_loop")
