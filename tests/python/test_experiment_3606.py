import json
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_3606_g_gate_status_synthesis_v331 import generate_artifact

def test_experiment_3606_generates_valid_artifact():
    """
    Test that the G-gate status synthesis generates the correct artifact.
    TRACES TO: REQ-PUBLISH-007, SCENARIO-PUBLISH-007
    """
    artifact = generate_artifact()
    
    assert "honest_verdict" in artifact
    assert artifact["honest_verdict"].startswith("complete: g_gate_synthesis_v331_paper_ready_true_verifier_generalization_math_only_earned_paper_scoped")
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    
    # G1-G4 must be populated
    assert "g1" in artifact
    assert "g2" in artifact
    assert "g3" in artifact
    assert "g4" in artifact
    
    assert artifact["paper_ready"] is True
    assert isinstance(artifact["unmet_gates"], list)
    
    assert artifact["verifier_generalization_scope"] == "math_only_earned_paper_scoped"
    assert artifact["p01_status"] == "honest-negative"
    
    # Check cited artifacts
    assert "experiment_3601" in artifact["cited_upstream_artifacts"]
    assert "experiment_3605" in artifact["cited_upstream_artifacts"]
    
    assert "random_seed" in artifact
    assert "reproducibility_checksum" in artifact
    assert "duration_s" in artifact
