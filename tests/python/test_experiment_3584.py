import json
import os
from unittest.mock import patch
from scripts.experiment_3584_diagnose_329_null_positive_control import run_diagnosis

def test_run_diagnosis_creates_expected_artifact(tmp_path):
    """
    REQ-VERIFY-3584
    SCENARIO-VERIFY-3584
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    expected_path = results_dir / "experiment_3584_diagnose_329_null_positive_control.json"

    # Patch open to write to our tmp path instead
    original_open = open
    def mock_open(path, *args, **kwargs):
        if "experiment_3584_diagnose_329_null_positive_control.json" in str(path):
            return original_open(expected_path, *args, **kwargs)
        return original_open(path, *args, **kwargs)

    with patch("builtins.open", mock_open):
        run_diagnosis()

    assert expected_path.exists()
    
    with open(expected_path, "r") as f:
        artifact = json.load(f)

    assert "honest_verdict" in artifact
    assert artifact["honest_verdict"]["value"] == "complete: diagnosed_329_null_contaminated_confidence_degenerate_verifiers_inert_applicable_sets_enumerated"
    assert "inference_substrate" in artifact
    assert artifact["confidence_baseline_degenerate"]["value"] is True
    assert artifact["per_verifier_inertia_confirmed"]["value"] is True
    assert "applicable_verifiers_facts" in artifact
    assert "semantic_energy.py" in artifact["applicable_verifiers_facts"]["value"]
    assert "applicable_verifiers_code" in artifact
    assert "ast_structure_verifier.py" in artifact["applicable_verifiers_code"]["value"]
    assert "positive_control_requirements" in artifact
    assert "random_seed" in artifact
    assert artifact["random_seed"]["value"] == 3584
    assert "reproducibility_checksum" in artifact
    assert "duration_s" in artifact
