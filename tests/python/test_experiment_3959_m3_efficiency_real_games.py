import json
import os
from pathlib import Path
from unittest import mock

import pytest
import math

from scripts.experiments.experiment_3959_m3_efficiency_real_games import (
    RatioConfidenceInterval,
    bootstrap_ratio_ci,
    get_objects_and_target_area,
    simulate_geometric,
    process_game,
    run
)

def test_ratio_ci():
    """Test bootstrap CI for M3 transfer test."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = bootstrap_ratio_ci(data, random_seed=42, resamples=100)
    assert isinstance(ci, RatioConfidenceInterval)
    assert 1.0 <= ci.low <= ci.high <= 5.0

def test_get_objects_and_target_area():
    import numpy as np
    grid = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [2, 2, 0, 0]
    ])
    num_objs, t_area, total = get_objects_and_target_area(grid, 1, 1)
    assert num_objs == 2
    assert t_area == 2
    assert total == 16
    
    # Target not found
    num_objs, t_area, total = get_objects_and_target_area(grid, 0, 0)
    assert num_objs == 2
    assert t_area == 1

def test_simulate_geometric():
    import random
    rng = random.Random(42)
    assert simulate_geometric(1.0, rng) == 1
    assert simulate_geometric(0.0, rng) == 10000
    
    val = simulate_geometric(0.5, rng)
    assert val >= 1

class MockArc:
    def __init__(self, *args, **kwargs):
        pass
    def make(self, game_id):
        return MockEnv()
    
class MockFrame:
    def __init__(self, grid):
        self.frame = grid

class MockEnv:
    def __init__(self):
        import numpy as np
        self.grid = np.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        self.baseline_actions = [60]
    def reset(self):
        return MockFrame(self.grid)
    def step(self, action, data):
        return MockFrame(self.grid)

def test_process_game():
    import random
    rng = random.Random(42)
    arc = MockArc()
    solve_log_r11l = [{"piece": [1, 1], "target": [2, 2]}]
    rw, rwo = process_game("r11l", solve_log_r11l, arc, [60], rng)
    assert len(rw) == 1000
    assert len(rwo) == 1000
    
    solve_log_lp85 = [{"y": 1, "x": 1}]
    rw, rwo = process_game("lp85", solve_log_lp85, arc, [60], rng)
    assert len(rw) == 1000
    assert len(rwo) == 1000

@mock.patch("scripts.experiments.experiment_3959_m3_efficiency_real_games.Path.exists")
@mock.patch("scripts.experiments.experiment_3959_m3_efficiency_real_games.Path.read_text")
@mock.patch("scripts.experiments.experiment_3959_m3_efficiency_real_games.Path.write_text")
@mock.patch("scripts.experiments.experiment_3959_m3_efficiency_real_games.Arcade")
def test_run_success(mock_arcade, mock_write, mock_read, mock_exists):
    """Test REQ-PHASE4-012: run writes M3 efficiency artifact."""
    mock_arcade.return_value = MockArc()
    
    # Setup mock to return True for files
    def exists_side_effect():
        return True
    mock_exists.return_value = True
    
    def read_text_side_effect():
        # returns r11l full solve structure
        return json.dumps({
            "ACCURACY_levels_solved": 1,
            "game_solved": "r11l-495a7899",
            "solve_log": [
                {"level": 0, "piece": [1, 1], "target": [2, 2]}
            ]
        })
    mock_read.return_value = read_text_side_effect()
    
    run()
    
    assert mock_write.called
    written_data = mock_write.call_args[0][0]
    art = json.loads(written_data)
    assert art["experiment"] == "experiment_3959_m3_efficiency_real_games"
    assert "honest_verdict" in art
    assert "m3_efficiency_real_games" in art["honest_verdict"]

@mock.patch("scripts.experiments.experiment_3959_m3_efficiency_real_games.Path.exists")
@mock.patch("scripts.experiments.experiment_3959_m3_efficiency_real_games.Path.write_text")
def test_run_blocked(mock_write, mock_exists):
    """Test REQ-PHASE4-012: blocked when no solved games."""
    mock_exists.return_value = False
    
    run()
    
    assert mock_write.called
    written_data = mock_write.call_args[0][0]
    art = json.loads(written_data)
    assert "blocked_no_solved_real_game" in art["honest_verdict"]
