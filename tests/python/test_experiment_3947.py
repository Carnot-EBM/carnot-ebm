import sys
import pytest
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_3947_active_data_codex_nonspatial_sweep as exp

def test_run_blocked_if_no_codex(monkeypatch, tmp_path):
    """Verify SCENARIO-PHASE4-008: blocked_codex_unavailable is handled."""
    def mock_subprocess_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "command -v codex")
    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    res = exp.run(["r11l"], write=False)
    assert res["honest_verdict"].startswith("blocked_")
    assert res["n_trustworthy_at_0.15"] == 0

def test_run_success(monkeypatch, tmp_path):
    """Verify REQ-PHASE4-008: general sweep implementation works and scores games."""
    def mock_subprocess_run(args, **kwargs):
        if args == ["command", "-v", "codex"]:
            return subprocess.CompletedProcess(args, 0, stdout=b"/usr/bin/codex\n")
        raise Exception("unexpected subprocess")
    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    
    class MockEnv:
        game_id = "r11l-test"
    class MockArc:
        def get_environments(self):
            return [MockEnv()]
            
    monkeypatch.setattr("experiment_3947_active_data_codex_nonspatial_sweep.active_collect", lambda *args: [])
    monkeypatch.setattr("experiment_3947_active_data_codex_nonspatial_sweep._collect", lambda *args: [])
    monkeypatch.setattr("experiment_3947_active_data_codex_nonspatial_sweep._common_test", lambda *args: [])
    
    def mock_codex_best_energy(train, test, iters, rng):
        return 0.10, [{"status": "graded", "codex_s": 5.0}], 5.0
        
    monkeypatch.setattr("experiment_3947_active_data_codex_nonspatial_sweep.codex_best_energy", mock_codex_best_energy)
    
    # We need to mock Arc engine creation because we don't want to instantiate it for real in tests
    import arc_agi
    monkeypatch.setattr("arc_agi.Arcade", lambda **kwargs: MockArc())
    
    res = exp.run(["r11l"], write=False)
    assert res["honest_verdict"].startswith("complete:")
    assert res["n_trustworthy_at_0.15"] == 1
    assert res["per_game_best_energy"]["r11l"] == 0.10
    assert res["total_codex_calls"] == 1
    assert res["total_codex_seconds"] == 5.0
