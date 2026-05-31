import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# REQ-PUBLISH-007
# SCENARIO-PUBLISH-007

def test_experiment_3581_execution():
    """Test that the G-gate synthesis script produces the correct JSON."""
    script_path = Path("scripts/experiment_3581_g_gate_status_synthesis_v329.py")
    
    # Run the script in a test mode or check its logic
    # Since it reads results/ and we don't want to rely on live files, 
    # we can mock or just run it and check if it produces a valid JSON artifact.
    # The script should just read existing json files in results/ which are part of the repository context right now.
    
    result = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    out_file = Path("results/experiment_3581_g_gate_status_synthesis_v329.json")
    assert out_file.exists()
    
    data = json.loads(out_file.read_text())
    
    assert "honest_verdict" in data
    assert data["honest_verdict"].startswith("complete: g_gate_synthesis_v329_paper_ready_")
    assert "verifier_generalization_math_only_domain_bound_paper_claim_scoped" in data["honest_verdict"]
    
    assert data["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert "g1" in data and isinstance(data["g1"], bool)
    assert "g2" in data and isinstance(data["g2"], bool)
    assert "g3" in data and isinstance(data["g3"], bool)
    assert "g4" in data and isinstance(data["g4"], bool)
    assert "paper_ready" in data and isinstance(data["paper_ready"], bool)
    assert "unmet_gates" in data and isinstance(data["unmet_gates"], list)
    assert data["verifier_generalization_scope"] == "math_only_domain_bound_paper_claim_scoped"
    assert data["p01_status"] == "honest-negative"
    assert "experiment_3574_verifier_factual_hallucination_error_detection.json" not in data["cited_upstream_artifacts"]
    assert "experiment_3573_verifier_code_bug_error_detection.json" in data["cited_upstream_artifacts"]
    assert "experiment_3575_verifier_discriminating_value.json" in data["cited_upstream_artifacts"]
    assert "experiment_3576_verifier_cross_domain_synthesis.json" in data["cited_upstream_artifacts"]
    assert data["random_seed"] == 42
    assert "reproducibility_checksum" in data
    assert "duration_s" in data
