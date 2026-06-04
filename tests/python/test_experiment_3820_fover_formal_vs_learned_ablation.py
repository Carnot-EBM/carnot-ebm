import json
from pathlib import Path
from carnot.verify.experiment_3820_fover_formal_vs_learned_ablation import run_experiment

def test_experiment_3820_ablation():
    """Verify that the ablation script runs and outputs the required fields."""
    artifact = run_experiment()
    
    assert "honest_verdict" in artifact
    assert "full_ensemble_auroc" in artifact
    assert "formal_only_auroc" in artifact
    assert "learned_only_auroc" in artifact
    assert "verifier_partition" in artifact
    assert "n_candidates_scored" in artifact
    assert "preconditions_checked" in artifact
    assert "random_seed" in artifact
    assert "reproducibility_checksum" in artifact
    assert "duration_s" in artifact
    
    assert isinstance(artifact["duration_s"], float)
    assert artifact["duration_s"] >= 1.0
    
    if artifact["honest_verdict"].startswith("complete:"):
        assert "formal" in artifact["verifier_partition"]
        assert "learned" in artifact["verifier_partition"]
        
        formal_verifiers = artifact["verifier_partition"]["formal"]
        learned_verifiers = artifact["verifier_partition"]["learned"]
        
        assert "ASTStructureVerifier" in formal_verifiers
        assert "Z3MathVerifier" in formal_verifiers
        assert "SemanticConsistencyVerifier" in formal_verifiers
        
        assert "SOSKANEnergyV3" in learned_verifiers
        assert "SemEnergyProbe" in learned_verifiers
