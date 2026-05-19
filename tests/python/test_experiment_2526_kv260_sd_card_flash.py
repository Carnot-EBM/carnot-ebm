import os
import json
import pytest

# Add scripts directory to path to import experiment script
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))
from experiment_2526_kv260_sd_card_flash import generate_flash_results, main

def test_generate_flash_results():
    """Verify that the generated flash results conform to the required schema."""
    result = generate_flash_results()
    
    assert "honest_verdict" in result
    assert "kv260_hwh_path" in result
    assert "pynq_available" in result
    assert "sd_card_detected" in result
    assert "kv260_flash_attempted" in result
    assert "kv260_flash_documentation_complete" in result
    assert "operator_commands" in result
    assert "preconditions_checked" in result
    assert "duration_s" in result
    
    assert result["kv260_flash_attempted"] is False
    assert result["kv260_flash_documentation_complete"] is True
    assert isinstance(result["operator_commands"], list)

def test_main_writes_json(tmpdir, monkeypatch):
    """Verify that main() writes the results to a JSON file."""
    # Temporarily change working directory to tmpdir so results/ isn't created in the real repo during tests
    monkeypatch.chdir(tmpdir)
    os.makedirs("results", exist_ok=True)
    
    main()
    
    output_path = "results/experiment_2526_kv260_sd_card_flash.json"
    assert os.path.exists(output_path)
    
    with open(output_path, "r") as f:
        data = json.load(f)
        
    assert data["honest_verdict"] == "blocked_by_operator: Physical SD card flash requires operator intervention. PYNQ package missing. Operator commands documented."
