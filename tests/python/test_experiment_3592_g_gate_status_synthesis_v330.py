import json
import os
import tempfile
import pytest
import subprocess
from pathlib import Path
from unittest import mock
import sys

# Add scripts directory to path to allow importing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

# Import the script
import experiment_3592_g_gate_status_synthesis_v330

def test_g_gate_synthesis_success(tmp_path, monkeypatch):
    """Test REQ-VERIFY-3592: Output schema matches and aggregates correctly."""
    
    # Mock PROJECT_ROOT inside the script
    mock_root = tmp_path
    monkeypatch.setattr(experiment_3592_g_gate_status_synthesis_v330, "PROJECT_ROOT", mock_root)
    
    # Create mock results directory
    results_dir = mock_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create fake exp3591 artifact
    exp3591_path = results_dir / "experiment_3591_cross_domain_synthesis_v2.json"
    exp3591_content = {
        "verifier_value_generalizes": {
            "value": "math_only_earned",
            "principle": "broad / code_only / math_only_earned"
        }
    }
    exp3591_path.write_text(json.dumps(exp3591_content))
    
    # Mock subprocess.run for publication_gate.py
    def mock_run(*args, **kwargs):
        class MockCompletedProcess:
            def __init__(self):
                self.returncode = 0
                self.stdout = json.dumps({
                    "paper_ready": True,
                    "gates": {
                        "G1": {"pass": True},
                        "G2": {"pass": True},
                        "G3": {"pass": True},
                        "G4": {"pass": True}
                    },
                    "unmet_gates": []
                })
        return MockCompletedProcess()
    
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    # Run main
    exit_code = experiment_3592_g_gate_status_synthesis_v330.main()
    assert exit_code == 0
    
    # Verify the output artifact
    out_path = results_dir / "experiment_3592_g_gate_status_synthesis_v330.json"
    assert out_path.exists()
    
    out_data = json.loads(out_path.read_text())
    
    assert out_data["honest_verdict"]["value"] == "complete: g_gate_synthesis_v330_paper_ready_true_verifier_generalization_math_only_earned"
    assert out_data["inference_substrate"]["value"] == "aggregation_from_upstream_artifacts"
    assert out_data["g1"]["value"] is True
    assert out_data["g2"]["value"] is True
    assert out_data["g3"]["value"] is True
    assert out_data["g4"]["value"] is True
    assert out_data["paper_ready"]["value"] is True
    assert out_data["unmet_gates"]["value"] == []
    assert out_data["verifier_generalization_scope"]["value"] == "math_only_earned"
    assert out_data["p01_status"]["value"] == "honest-negative"
    assert "experiment_3591" in out_data["cited_upstream_artifacts"]["value"][0]
    assert out_data["random_seed"]["value"] == 3592
    assert "reproducibility_checksum" in out_data
    assert "duration_s" in out_data
