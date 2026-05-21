import os
import json
import pytest
from experiment_2825_paper_v6_multicorpus_table import generate_deliverable

def test_generate_deliverable_schema(tmp_path, monkeypatch):
    # Patch the current working directory to write to temp
    monkeypatch.chdir(tmp_path)
    
    data = generate_deliverable(42.5)
    
    # Assert JSON file was written
    assert os.path.exists("results/experiment_2825_paper_v6_multicorpus_table.json")
    
    with open("results/experiment_2825_paper_v6_multicorpus_table.json", "r") as f:
        saved_data = json.load(f)
        
    # Assert specific fields
    assert saved_data["honest_verdict"].startswith("complete:")
    assert saved_data["paper_v6_compile_success"] is True
    assert "FoVer" in saved_data["corpora_in_table"]
    assert "MBPP" in saved_data["corpora_in_table"]
    assert "HumanEval" in saved_data["corpora_in_table"]
    assert "TruthfulQA" in saved_data["corpora_in_table"]
    assert saved_data["submission_package_ready"] is True
    assert isinstance(saved_data["duration_s"], float)
