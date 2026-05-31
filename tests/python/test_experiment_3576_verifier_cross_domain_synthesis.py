import os
import json
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.experiment_3576_verifier_cross_domain_synthesis import main

def test_experiment_3576_produces_valid_synthesis():
    """
    Verifies REQ-VERIFY-3576-1, REQ-VERIFY-3576-2, and SCENARIO-VERIFY-3576.
    Ensures the cross-domain synthesis script runs and produces the correct schema.
    """
    results_path = Path("results/experiment_3576_verifier_cross_domain_synthesis.json")
    
    if results_path.exists():
        results_path.unlink()
        
    main()
    
    assert results_path.exists(), "Output JSON was not created."
    
    with open(results_path) as f:
        data = json.load(f)
        
    assert "honest_verdict" in data
    assert data["honest_verdict"].startswith("complete: verifier_cross_domain_synthesis_value_generalizes_")
    assert "inference_substrate" in data
    assert data["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert "generalization_table" in data
    assert "math" in data["generalization_table"]
    assert "code" in data["generalization_table"]
    assert "facts" not in data["generalization_table"]  # Excluded due to adversarial flag
    assert "verifier_value_generalizes" in data
    assert data["verifier_value_generalizes"] == "math_only_domain_bound"
    assert "paper_safe_claims" in data
    assert isinstance(data["paper_safe_claims"], list)
    assert "paper_forbidden_claims" in data
    assert isinstance(data["paper_forbidden_claims"], list)
    assert "cited_upstream_artifacts" in data
    assert len(data["cited_upstream_artifacts"]) == 3
    assert "random_seed" in data
    assert "reproducibility_checksum" in data
    assert "duration_s" in data
