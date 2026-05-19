"""Tests for Exp 2526 KV260 SD Card Flash."""

import os
import sys
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_2526_kv260_sd_card_flash import run_experiment

def test_experiment_2526_produces_json():
    """Verify that the experiment runs and produces the correct JSON schema."""
    result_path = _REPO_ROOT / "results" / "experiment_2526_kv260_sd_card_flash.json"
    if result_path.exists():
        result_path.unlink()
        
    result = run_experiment()
    
    assert result_path.exists()
    
    with open(result_path, "r") as f:
        data = json.load(f)
        
    assert "honest_verdict" in data
    assert "blocked_" in data["honest_verdict"] or "terminal" in data["honest_verdict"]
    assert "kv260_hwh_path" in data
    assert "pynq_available" in data
    assert "sd_card_detected" in data
    assert "kv260_flash_attempted" in data
    assert "kv260_flash_documentation_complete" in data
    assert "operator_commands" in data
    assert "preconditions_checked" in data
    assert "duration_s" in data
    
    assert data["kv260_flash_attempted"] is True or data["kv260_flash_documentation_complete"] is True

if __name__ == "__main__":
    test_experiment_2526_produces_json()
    print("Test passed successfully.")
