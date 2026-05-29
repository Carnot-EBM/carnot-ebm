"""Test Capstone v311."""
import os
import json
import pytest
from pathlib import Path
from carnot.reporting import capstone_v311_3371

def test_capstone_module():
    result = capstone_v311_3371.run_capstone()
    assert result["experiment_id"] == "exp3371"
    assert result["capstone_v311_ready"] is True
    assert result["upstreams"]["exp3365"] == "success"

def test_capstone_script(tmp_path, monkeypatch):
    import importlib.util
    import builtins
    
    script_path = os.path.join(os.path.dirname(__file__), "../../scripts/experiment_3371_capstone_v311.py")
    spec = importlib.util.spec_from_file_location("capstone", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    out_path = out_dir / "experiment_3371_capstone_v311.json"
    
    original_open = builtins.open
    def mock_open(*args, **kwargs):
        if "experiment_3371_capstone_v311.json" in str(args[0]):
            return original_open(out_path, args[1], encoding=kwargs.get("encoding", "utf-8"))
        return original_open(*args, **kwargs)
    
    monkeypatch.setattr(builtins, "open", mock_open)
    
    # Mock REPO_ROOT in the module
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    
    module.main()
    
    assert out_path.exists()
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["experiment_id"] == "exp3371"
