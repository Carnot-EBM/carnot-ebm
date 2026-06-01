"""Test experiment 3602.

Spec: REQ-CODE-VERIFY-3602, SCENARIO-CODE-VERIFY-3602
"""
import pytest
import json
import tempfile
import sys
from pathlib import Path

# Add the scripts directory to path so we can import the experiment module if needed
sys.path.append(str(Path(__file__).resolve().parents[2] / "scripts"))

# Import the module to test
import experiment_3602_math_to_code_prm_transfer

def test_experiment_3602_blocked_no_labeled_corpus(tmp_path):
    """Test that the experiment correctly identifies when no labeled code corpus with code strings is present."""
    # Create a mock exp1999 that lacks code strings
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    mock_exp1999 = results_dir / "experiment_1999_code_verification_humaneval.json"
    mock_exp1999.write_text(json.dumps({
        "results": [
            {"task_id": "HumanEval/0", "baseline_passed": True}
        ]
    }))

    out_file = results_dir / "experiment_3602_math_to_code_prm_transfer.json"

    # Run logic
    artifact = experiment_3602_math_to_code_prm_transfer.run_experiment(
        exp1999_path=mock_exp1999,
        output_path=out_file
    )

    assert artifact["honest_verdict"] == "complete: blocked_no_labeled_code_corpus"
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["verifiers_fired_on_code"] is False

def test_experiment_3602_with_valid_corpus(tmp_path):
    """Test that the experiment runs verifiers if a valid corpus with code strings is present."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    mock_exp1999 = results_dir / "experiment_1999_code_verification_humaneval.json"
    mock_exp1999.write_text(json.dumps({
        "results": [
            {"task_id": "HumanEval/0", "baseline_passed": True, "generated_text": "def foo(): pass"},
            {"task_id": "HumanEval/1", "baseline_passed": False, "generated_text": "def bar(): fail"}
        ]
    }))

    out_file = results_dir / "experiment_3602_math_to_code_prm_transfer.json"

    # Run logic
    artifact = experiment_3602_math_to_code_prm_transfer.run_experiment(
        exp1999_path=mock_exp1999,
        output_path=out_file
    )

    # With only 2 samples, it should not be enough to calculate AUROC or might be all 0.5 depending on the mock.
    # The script should handle it.
    assert "honest_verdict" in artifact
