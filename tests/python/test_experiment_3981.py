"""
Tests for Exp 3981: Fourth Game First Solve (su15).
Spec refs: REQ-PHASE4-019, SCENARIO-PHASE4-019.
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_3981_fourth_game_first_solve


def test_experiment_3981_run_arcade_fail():
    """SCENARIO-PHASE4-019: verifies blocked_arc_offline_env_unavailable when offline Arcade cannot load."""
    with mock.patch("experiment_3981_fourth_game_first_solve.Arcade", side_effect=Exception("mocked fail")):
        art = experiment_3981_fourth_game_first_solve.run(budget=10)
        assert "blocked_arc_offline_env_unavailable" in art["honest_verdict"]
        assert art["ACCURACY_levels_solved"] == 0
        assert art["game_solved"] == "none"


def test_experiment_3981_run_env_make_fail():
    """SCENARIO-PHASE4-019: verifies handling of environment creation failure."""
    mock_arc = mock.Mock()
    mock_arc.make.side_effect = Exception("mocked make fail")
    with mock.patch("experiment_3981_fourth_game_first_solve.Arcade", return_value=mock_arc):
        art = experiment_3981_fourth_game_first_solve.run(budget=10)
        assert "complete: fourth_game_no_solve_env_failed" in art["honest_verdict"]
        assert art["ACCURACY_levels_solved"] == 0


def test_experiment_3981_run_solve_success():
    """SCENARIO-PHASE4-019: verifies successful solve logic using mocked game env."""
    mock_arc = mock.Mock()
    mock_env = mock.Mock()
    mock_game = mock.Mock()
    mock_env._game = mock_game
    
    mock_f_start = mock.Mock()
    mock_f_start.levels_completed = 0
    mock_f_start.frame = [[[0]]] # Dummy grid
    mock_env.reset.return_value = mock_f_start
    
    mock_f_win = mock.Mock()
    mock_f_win.levels_completed = 1
    mock_f_win.frame = [[[0]]]
    mock_env.step.return_value = mock_f_win
    
    mock_arc.make.return_value = mock_env
    
    with mock.patch("experiment_3981_fourth_game_first_solve.Arcade", return_value=mock_arc):
        with mock.patch("experiment_3981_fourth_game_first_solve.objects", return_value=[(10, 10)]):
            art = experiment_3981_fourth_game_first_solve.run(budget=10)
            
    assert art["ACCURACY_levels_solved"] == 1
    assert "success:" in art["honest_verdict"]


def test_experiment_3981_run_solve_fail_budget():
    """SCENARIO-PHASE4-019: verifies failure due to exceeded budget/no solution found."""
    mock_arc = mock.Mock()
    mock_env = mock.Mock()
    mock_game = mock.Mock()
    mock_env._game = mock_game
    
    mock_f_start = mock.Mock()
    mock_f_start.levels_completed = 0
    mock_f_start.frame = [[[0]]] # Dummy grid
    mock_env.reset.return_value = mock_f_start
    
    mock_f_fail = mock.Mock()
    mock_f_fail.levels_completed = 0
    mock_f_fail.frame = [[[0]]]
    mock_env.step.return_value = mock_f_fail
    
    mock_arc.make.return_value = mock_env
    
    with mock.patch("experiment_3981_fourth_game_first_solve.Arcade", return_value=mock_arc):
        with mock.patch("experiment_3981_fourth_game_first_solve.objects", return_value=[(10, 10)]):
            art = experiment_3981_fourth_game_first_solve.run(budget=10)
            
    assert art["ACCURACY_levels_solved"] == 0
    assert "complete: fourth_game_no_solve_budget_exceeded" in art["honest_verdict"]
