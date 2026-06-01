import os
import json
import importlib.util
from pathlib import Path
import sys

def test_experiment_3611_archive_v331_activate_v332():
    """
    REQ-REPORT-3611: Archive V331 Activate V332
    SCENARIO-REPORT-3611: Exp 3611 Archives .331 and Activates .332
    """
    script_path = "scripts/experiment_3611_archive_v331_activate_v332.py"
    output_path = "results/experiment_3611_archive_v331_activate_v332.json"
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Import the script
    spec = importlib.util.spec_from_file_location("experiment_3611", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment_3611"] = module
    spec.loader.exec_module(module)
    
    # Run the function
    module.main()
    
    # Check the output file
    assert os.path.exists(output_path), "JSON output file was not created"
    
    with open(output_path, "r") as f:
        data = json.load(f)
        
    # Verify the required fields (principle-annotated fields)
    assert data["honest_verdict"] == "complete: archived_v331_unfinished_decontamination_facts_code_blocked_not_measured_v332_active_paper_ready_true"
    assert data["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert data["v331_outcome_recorded_as"] == "UNFINISHED de-contamination (facts/code rows BLOCKED not measured)"
    assert data["false_negative_risk_recorded"] == "asserted a null with no valid positive control"
    assert data["facts_corpus_exists_for_332"] is True
    assert data["paper_ready_preserved"] is True
    assert data["n_tasks_archived"] > 0, "n_tasks_archived should be greater than 0"
    assert "random_seed" in data
    assert "duration_s" in data
    assert "reproducibility_checksum" in data
