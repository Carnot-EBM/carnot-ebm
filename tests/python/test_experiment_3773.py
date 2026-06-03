import os
import json
import pytest

from scripts.experiment_3773_verifier_product_prm_positioning import main

def test_experiment_3773_positioning(tmp_path):
    """
    Test that experiment 3773 generates the correct JSON artifact with required fields.
    Traces to: REQ-REPORT-3773, SCENARIO-REPORT-3773
    """
    output_file = tmp_path / "exp3773.json"
    
    # Run the logic
    main(str(output_file))
    
    assert os.path.exists(str(output_file))
    
    with open(str(output_file), "r") as f:
        data = json.load(f)
        
    assert "honest_verdict" in data
    assert "inference_substrate" in data
    assert "comparison_table" in data
    assert "where_carnot_leads" in data
    assert "where_carnot_does_not_lead" in data
    assert "product_value_proposition" in data
    assert data["peer_numbers_are_as_reported_not_re_derived"] is True
    assert data["no_generalization_retest_run"] is True
    assert "random_seed" in data
    assert "reproducibility_checksum" in data
    assert "duration_s" in data
    
    assert data["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert "Carnot" in data["comparison_table"]
    assert "0.9131" in data["comparison_table"]["Carnot"]["metric"]
