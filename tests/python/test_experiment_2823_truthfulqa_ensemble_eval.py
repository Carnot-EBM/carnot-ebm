import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "python"))

import experiment_2823_truthfulqa_ensemble_eval as exp2823

RESULTS_DIR = REPO_ROOT / "results"

def test_req_verify_2823_produces_valid_deliverable() -> None:
    deliverable = exp2823.run_experiment(results_dir=RESULTS_DIR, write=False)

    required_fields = {
        "corpus",
        "n_questions",
        "n_seeds",
        "condition_a_production_auroc_mean",
        "condition_a_production_auroc_std",
        "condition_b_architecture_only_auroc_mean",
        "condition_b_architecture_only_auroc_std",
        "learning_contribution",
        "per_verifier_condition_a_auroc",
        "per_verifier_condition_b_auroc",
        "scoring_method",
        "bleurt_threshold",
        "random_seeds_used",
        "reproducibility_checksum",
        "model_specs",
        "duration_s",
        "preconditions_checked",
        "fr11_state_files",
        "state_files_restored_sha_match",
    }
    assert required_fields <= set(deliverable)
    assert deliverable["corpus"] == "TruthfulQA-generation"
    assert deliverable["n_questions"] == 200
    assert deliverable["n_seeds"] == 5
    
    assert deliverable["learning_contribution"] == pytest.approx(deliverable["condition_a_production_auroc_mean"] - deliverable["condition_b_architecture_only_auroc_mean"])
    assert deliverable["scoring_method"] == "BLEURT-base-128, threshold tuned on 50-Q held-out"
    assert deliverable["random_seeds_used"] == [42, 137, 271, 314, 1729]
    assert deliverable["state_files_restored_sha_match"] is True

def test_req_verify_2823_methodology_note() -> None:
    deliverable = exp2823.run_experiment(results_dir=RESULTS_DIR, write=False)
    
    # Requirement: methodology_note is REQUIRED if either AUROC mean < 0.7
    if deliverable["condition_a_production_auroc_mean"] < 0.7 or deliverable["condition_b_architecture_only_auroc_mean"] < 0.7:
        assert "methodology_note" in deliverable
        assert "honest finding" in deliverable["methodology_note"]

def test_req_verify_2823_principles() -> None:
    deliverable = exp2823.run_experiment(results_dir=RESULTS_DIR, write=False)
    assert "_principles" in deliverable
    principles = deliverable["_principles"]
    assert "learning_contribution" in principles
    assert "= A - B." in principles["learning_contribution"]
    assert "per_verifier_condition_b_auroc" in principles
    assert "KEY finding" in principles["per_verifier_condition_b_auroc"]
