import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from scripts.experiments.experiment_3993_fourth_game_verifier_pruned import run

# REQ-PHASE4-019: The GAP-4 verifier pruner must attempt a fourth non-spatial game to prove generalization
# SCENARIO-PHASE4-019: The script runs and returns a JSON dictionary conforming to the required schema

def test_experiment_3993_schema():
    # We mock Arcade so it doesn't take 2 minutes to run in CI
    with patch("scripts.experiments.experiment_3993_fourth_game_verifier_pruned.Arcade") as mock_arcade:
        mock_arc = MagicMock()
        mock_arcade.return_value = mock_arc
        
        # Simulate exception to test offline block logic
        mock_arc.make.side_effect = Exception("No env")
        
        art = run(budget=2)
        
        assert "experiment" in art
        assert "honest_verdict" in art
        assert "game_solved" in art
        assert "games_attempted" in art
        assert "ACCURACY_levels_solved" in art
        assert "first_solve_at_action" in art
        assert "actions_vs_baseline" in art
        assert "verifier_pruner_used" in art
        assert "induced_mechanic" in art
        assert "real_env_confirmed" in art
        assert "duration_s" in art
        assert "random_seed" in art
        
        assert "blocked" in art["honest_verdict"] or "complete" in art["honest_verdict"] or "success" in art["honest_verdict"]
