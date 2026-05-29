"""
Test for experiment_3349_capstone_v309.py
"""
import os
import json
import importlib.util
import builtins

def test_capstone_generation(tmp_path, monkeypatch):
    """Test the main logic of the capstone script."""
    script_path = os.path.join(os.path.dirname(__file__), "../../scripts/experiment_3349_capstone_v309.py")
    spec = importlib.util.spec_from_file_location("capstone", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Mock output file
    results_path = tmp_path / "experiment_3349_capstone_v309.json"
    
    original_open = builtins.open
    
    def mock_open(*args, **kwargs):
        if args[0] == "results/experiment_3349_capstone_v309.json":
            return original_open(results_path, args[1], encoding=kwargs.get("encoding", "utf-8"))
        return original_open(*args, **kwargs)
    
    monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(builtins, "open", mock_open)
    
    module.main()
    
    assert os.path.exists(results_path)
    with original_open(results_path, encoding="utf-8") as f:
        data = json.load(f)
        assert data["milestone"] == "2026.05.309"
        assert "honest_verdict" in data
        assert "phase3_status" in data
