"""Test Capstone v312."""
import os
import json
from carnot.reporting import capstone_v312_3390

def test_capstone_module():
    result = capstone_v312_3390.run_capstone()
    assert result["experiment_id"] == "exp3390"
    assert result["capstone_v312_ready"] is True
    assert result["upstreams"]["exp3381"] == "complete: kv260_hardware_latency_transcript_recorded"
    assert result["upstreams"]["exp3382"] == "blocked"

def test_capstone_script(tmp_path, monkeypatch):
    import importlib.util
    import builtins
    
    script_path = os.path.join(os.path.dirname(__file__), "../../scripts/experiment_3390_capstone_v312.py")
    spec = importlib.util.spec_from_file_location("capstone", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_path = out_dir / "experiment_3390_capstone_v312.json"
    
    original_open = builtins.open
    def mock_open(*args, **kwargs):
        if "experiment_3390_capstone_v312.json" in str(args[0]):
            return original_open(out_path, args[1], encoding=kwargs.get("encoding", "utf-8"))
        return original_open(*args, **kwargs)
    
    monkeypatch.setattr(builtins, "open", mock_open)
    
    # Mock REPO_ROOT in the module
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    
    module.main()
    
    assert out_path.exists()
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["experiment_id"] == "exp3390"
