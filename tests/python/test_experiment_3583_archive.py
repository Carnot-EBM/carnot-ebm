import os
import json
import importlib.util
from pathlib import Path
import sys

def test_experiment_3583_archive_v329_activate_v330():
    """
    REQ-REPORT-3583: Archive V329 Activate V330
    SCENARIO-REPORT-3583: Exp 3583 Archives .329 and Activates .330
    """
    script_path = "scripts/experiment_3583_archive_v329_activate_v330.py"
    output_path = "results/experiment_3583_archive_v329_activate_v330.json"
    
    # Import the script
    spec = importlib.util.spec_from_file_location("experiment_3583", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment_3583"] = module
    spec.loader.exec_module(module)
    
    # Run the function
    module.run()
    
    # Check the output file
    assert os.path.exists(output_path), "JSON output file was not created"
    
    with open(output_path, "r") as f:
        data = json.load(f)
        
    # Verify the required fields (principle-annotated fields)
    assert data["honest_verdict"] == "complete: archived_v329_contaminated_null_recorded_v330_decontamination_pivot_active"
    assert data["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert data["v329_headline_recorded_as"] == "contaminated_null_not_clean_math_only"
    assert data["paper_ready_preserved"] is True
    assert data["n_tasks_archived"] > 0, "n_tasks_archived should be greater than 0"
    assert "random_seed" in data
    assert "duration_s" in data
    assert "reproducibility_checksum" in data
