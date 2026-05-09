import json
import subprocess
import sys
from pathlib import Path

def test_experiment_1644_cerce_ledger_script(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "experiment_1644_cerce_ledger.py"
    
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    assert result.returncode == 0
    
    out_file = repo_root / "results" / "experiment_1644_cerce_ledger.json"
    assert out_file.exists()
    
    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["status"] == "complete"
    assert data["schema"] == "experiment_1644_cerce_ledger_v1"
    assert data["ledger_implemented"] is True
    assert data["cerce_ledger_ready"] is True
    assert "ledger_rows" in data
    assert data["honest_verdict"] == "complete: cerce_ledger_added"
