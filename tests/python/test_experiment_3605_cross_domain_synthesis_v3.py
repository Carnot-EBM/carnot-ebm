import json
from pathlib import Path
from carnot.verification.experiment_3605_cross_domain_synthesis_v3 import run_experiment

def test_experiment_3605_produces_artifact(tmp_path):
    """
    SCENARIO-VERIFY-3605: Cross-Domain Value Synthesis V3
    REQ-VERIFY-3605-1: The script SHALL explicitly correct the .330 record and synthesize the verdict.
    REQ-VERIFY-3605-2: The script SHALL output a terminal artifact with required schema fields.
    """
    result_path = tmp_path / "experiment_3605_cross_domain_synthesis_v3.json"
    
    # We will pass the paths to the upstream experiments
    results_dir = Path("results")
    upstream = {
        "3598": results_dir / "experiment_3598_diagnose_330_cascade_audit.json",
        "3599": results_dir / "experiment_3599_factual_corpus_v2_with_evidence.json",
        "3600": results_dir / "experiment_3600_real_nli_grounding_verifier.json",
        "3601": results_dir / "experiment_3601_corrected_cross_domain_remeasurement.json",
        "3602": results_dir / "experiment_3602_math_to_code_prm_transfer.json",
    }
    
    # Note: 3603 might not exist, but we pass it as None or handle gracefully if missing.
    
    run_experiment(output_path=result_path, upstream_paths=upstream)
    
    assert result_path.exists()
    with open(result_path) as f:
        data = json.load(f)
        
    required_fields = [
        "honest_verdict",
        "inference_substrate",
        "corrected_generalization_table",
        "v329_null_was",
        "verifier_value_generalizes",
        "paper_safe_claims",
        "paper_forbidden_claims",
        "cited_upstream_artifacts",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ]
    
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
        
    assert "complete: cross_domain_synthesis_v3_value_generalizes_" in data["honest_verdict"]
    assert "329_null_was_" in data["honest_verdict"]
    assert data["honest_verdict"].endswith("_paper_scoped")
