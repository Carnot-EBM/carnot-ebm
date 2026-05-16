import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_2059_activation import generate_activation_data, main

def test_generate_activation_data():
    data = generate_activation_data()
    assert data["experiment"] == 2059
    assert data["status"] == "success"
    assert data["honest_verdict"] == "activation_complete"

def test_main_execution(tmp_path, monkeypatch):
    def mock_makedirs(name, mode=0o777, exist_ok=False):
        pass
    
    monkeypatch.setattr(os, "makedirs", mock_makedirs)
    
    original_join = os.path.join
    
    def mock_join(a, *p):
        if a == "results":
            return original_join(str(tmp_path), *p)
        return original_join(a, *p)
        
    monkeypatch.setattr(os.path, "join", mock_join)
    
    main()
    
    out_file = tmp_path / "experiment_2059_activation.json"
    assert out_file.exists()
    
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["experiment"] == 2059
    assert data["status"] == "success"
    assert data["honest_verdict"] == "activation_complete"
