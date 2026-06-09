import pytest
import sys
import shutil
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_3947_active_data_codex_nonspatial_sweep
from experiment_3947_active_data_codex_nonspatial_sweep import run

def test_run_blocked_codex_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda cmd: None if cmd == "codex" else shutil.which(cmd))
    
    # Mock write_text to prevent writing to real REQ
    writes = []
    def mock_write_text(self, data, encoding):
        writes.append(data)
    monkeypatch.setattr(Path, "write_text", mock_write_text)

    art = run(games=["r11l"], write=True)
    
    assert art["honest_verdict"] == "blocked_codex_unavailable"
    assert art["n_trustworthy_at_0.15"] == 0
    assert len(writes) == 1

def test_run_offline_env_load_failed(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "codex" if cmd == "codex" else shutil.which(cmd))
    
    # Mock write_text
    writes = []
    def mock_write_text(self, data, encoding):
        writes.append(data)
    monkeypatch.setattr(Path, "write_text", mock_write_text)

    # Force an exception inside Arcade instantiation
    def mock_arcade_init(self, *args, **kwargs):
        raise ValueError("Simulated offline env failure")
    
    import arc_agi
    monkeypatch.setattr("arc_agi.Arcade.__init__", mock_arcade_init)
    class MockOperationMode:
        OFFLINE = "OFFLINE"
    monkeypatch.setattr("arc_agi.base.OperationMode", MockOperationMode)

    art = run(games=["r11l"], write=True)

    assert art["honest_verdict"].startswith("blocked_offline_env_load_failed")
    assert len(writes) == 1

def test_run_complete(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "codex" if cmd == "codex" else shutil.which(cmd))

    # Mock Arcade and related
    class MockEnv:
        def __init__(self, game_id):
            self.game_id = game_id

    class MockArcade:
        def __init__(self, **kwargs):
            pass
        def get_environments(self):
            return [MockEnv("r11l-test"), MockEnv("sc25-test")]
        def make(self, *args, **kwargs):
            return None
            
    import arc_agi
    monkeypatch.setattr("arc_agi.Arcade", MockArcade)
    class MockOperationMode:
        OFFLINE = "OFFLINE"
    monkeypatch.setattr("arc_agi.base.OperationMode", MockOperationMode)

    # Mock enum
    class MockEnum:
        ACTION1 = 1
        WIN = "WIN"
    monkeypatch.setattr("arcengine.enums.GameAction", MockEnum)
    monkeypatch.setattr("arcengine.enums.GameState", MockEnum)
    
    # Mock internal logic
    def mock_collect(*args, **kwargs): return []
    def mock_active_collect(*args, **kwargs): return []
    def mock_common_test(*args, **kwargs): return []
    def mock_keys(*args, **kwargs): return set()
    def mock_codex_best_energy(*args, **kwargs): return 0.1, [{"iter": 0}], 1.5
    
    monkeypatch.setattr("experiment_3947_active_data_codex_nonspatial_sweep._collect", mock_collect)
    monkeypatch.setattr("experiment_3947_active_data_codex_nonspatial_sweep.active_collect", mock_active_collect)
    monkeypatch.setattr("experiment_3947_active_data_codex_nonspatial_sweep._common_test", mock_common_test)
    monkeypatch.setattr("experiment_3947_active_data_codex_nonspatial_sweep._keys", mock_keys)
    monkeypatch.setattr("experiment_3947_active_data_codex_nonspatial_sweep.codex_best_energy", mock_codex_best_energy)

    writes = []
    def mock_write_text(self, data, encoding):
        writes.append(data)
    monkeypatch.setattr(Path, "write_text", mock_write_text)

    art = run(games=["r11l", "sc25"], write=True)

    assert art["honest_verdict"] == "complete: nonspatial_sweep_trustworthy_2of2"
    assert art["n_trustworthy_at_0.15"] == 2
    assert len(writes) == 1

def test_main(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    writes = []
    def mock_write_text(self, data, encoding):
        writes.append(data)
    monkeypatch.setattr(Path, "write_text", mock_write_text)
    
    monkeypatch.setattr("sys.argv", ["script.py", "--games", "test1", "--iters", "1"])
    import importlib
    importlib.import_module("experiment_3947_active_data_codex_nonspatial_sweep")
    
    # We trigger the __main__ block manually to ensure coverage without running a subprocess
    experiment_3947_active_data_codex_nonspatial_sweep.__name__ = "__main__"
    
    # Re-evaluate the __main__ block
    with open(experiment_3947_active_data_codex_nonspatial_sweep.__file__) as f:
        code = f.read()
    
    namespace = experiment_3947_active_data_codex_nonspatial_sweep.__dict__
    namespace["__name__"] = "__main__"
    exec(code, namespace)
