"""
REQ-3955: Tests for active-collect + codex program synthesis on non-spatial games.
"""
from unittest import mock
import pytest
import sys
import numpy as np

import experiment_3955_active_codex_nonspatial_sweep
from experiment_3955_active_codex_nonspatial_sweep import run

class MockEnv:
    def __init__(self, game_id):
        self.game_id = game_id

class MockArcClient:
    def get_environments(self):
        return [MockEnv("r11l-xyz"), MockEnv("sc25-abc"), MockEnv("other-123")]

def test_run_blocked_codex():
    res = run(write=False, _codex_available=False)
    assert res["honest_verdict"] == "blocked_codex_unavailable"

def test_run_blocked_env():
    # Test Arcade throwing an exception
    mock_arc_agi = mock.MagicMock()
    mock_arc_agi.Arcade.side_effect = Exception("offline unavailable")
    mock_arc_agi.base.OperationMode.OFFLINE = "OFFLINE"
    
    with mock.patch.dict("sys.modules", {"arc_agi": mock_arc_agi, "arc_agi.base": mock_arc_agi.base, "arcengine.enums": mock.MagicMock()}):
        res = run(write=False, _codex_available=True, _arc_client=None)
        assert res["honest_verdict"] == "blocked_arc_offline_env_unavailable"

    # Test Arcade returning empty envs
    mock_arc_agi_empty = mock.MagicMock()
    mock_instance = mock_arc_agi_empty.Arcade.return_value
    mock_instance.get_environments.return_value = []
    mock_arc_agi_empty.base.OperationMode.OFFLINE = "OFFLINE"

    with mock.patch.dict("sys.modules", {"arc_agi": mock_arc_agi_empty, "arc_agi.base": mock_arc_agi_empty.base, "arcengine.enums": mock.MagicMock()}):
        res2 = run(write=False, _codex_available=True, _arc_client=None)
        assert res2["honest_verdict"] == "blocked_arc_offline_env_unavailable"

@mock.patch("experiment_3955_active_codex_nonspatial_sweep._collect")
@mock.patch("experiment_3955_active_codex_nonspatial_sweep.active_collect")
@mock.patch("experiment_3955_active_codex_nonspatial_sweep._common_test")
@mock.patch("experiment_3955_active_codex_nonspatial_sweep.codex_best_energy")
def test_run_success(mock_codex, mock_common, mock_active, mock_collect, tmp_path, monkeypatch):
    monkeypatch.setattr("experiment_3955_active_codex_nonspatial_sweep.REPO", tmp_path)
    client = MockArcClient()
    
    grid = np.array([[0]], dtype=np.uint8)
    
    # transitions are tuples of (state, action, next_state) where state is grid 
    mock_collect.return_value = [(grid, (1,), grid)]
    mock_active.return_value = [(grid, (1,), grid)]
    mock_common.return_value = [(grid, (1,), grid)]
    
    mock_codex.side_effect = [
        (0.10, [{"codex_s": 1.0}], 1.0),
        (0.20, [{"codex_s": 2.0}], 2.0),
    ]
    
    res = run(games=["r11l", "sc25"], write=True, _codex_available=True, _arc_client=client)
    
    assert res["honest_verdict"].startswith("success:")
    assert res["n_trustworthy_at_0.15"] == 1
    assert res["markov_vs_hidden_split"]["markov"] == ["r11l"]
    assert res["markov_vs_hidden_split"]["hidden_state"] == ["sc25"]
    assert res["total_codex_seconds"] == 3.0
    assert res["total_codex_calls"] == 2
    assert res["per_game_best_energy"] == {"r11l": 0.10, "sc25": 0.20}

    # Test missing energy handling
    mock_codex.side_effect = [
        (None, [{"codex_s": 1.0}], 1.0),
        (None, [{"codex_s": 2.0}], 2.0),
    ]
    res2 = run(games=["r11l", "sc25"], write=False, _codex_available=True, _arc_client=client)
    assert res2["n_trustworthy_at_0.15"] == 0
    assert res2["per_game"][0]["compare_to_vc33_baseline_0.005"] is None

@pytest.mark.skip(
    reason=(
        "2026-06-09 outer-loop: POISON TEST — quarantined to unblock the conductor pre-test gate "
        "(.366 was cascade-blocked, self-heal aborting). runpy.run_module(run_name='__main__') executes "
        "a FRESH __main__ copy of the module whose `run` is NOT the mock.patch'd one (the patch applies to "
        "the already-imported module, not the runpy copy), so the __main__ block calls the REAL run() -> a "
        "live multi-game codex synthesis sweep that HANGS the gate (>120s timeout). The other 3 tests in "
        "this file (blocked_codex, blocked_env, test_run_success) are properly mocked and still assert the "
        "experiment logic. A live experiment run does not belong in a unit-test gate."
    )
)
def test_main_execution():
    import runpy
    with mock.patch("experiment_3955_active_codex_nonspatial_sweep.run") as mock_run:
        with mock.patch.object(sys, "argv", ["prog", "--games", "r11l"]):
            runpy.run_module("experiment_3955_active_codex_nonspatial_sweep", run_name="__main__")
            mock_run.assert_called_once_with(games=["r11l"], train_budget=900, test_budget=1400, episodes=32, iters=3, seed=0)
