import json
import os
import subprocess

def test_experiment_2824_artifact():
    # Run the script to generate the artifact
    result = subprocess.run(["python", "scripts/experiment_2824_cross_corpus_matrix.py"], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    artifact_path = "results/experiment_2824_cross_corpus_verifier_matrix.json"
    assert os.path.exists(artifact_path), "Artifact was not generated."
    
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    # Check required fields as per prompt
    assert "honest_verdict" in data
    assert data["honest_verdict"].startswith("complete:") or data["honest_verdict"].startswith("success:")
    
    assert "verifier_corpus_dual_matrix" in data
    matrix = data["verifier_corpus_dual_matrix"]
    assert isinstance(matrix, dict)
    
    # Check shape of matrix
    for verifier, corpora in matrix.items():
        assert isinstance(corpora, dict)
        for corpus, conds in corpora.items():
            assert "production" in conds
            assert "architecture_only" in conds
            assert "delta" in conds
            
    assert "architecture_transfer_verifiers" in data
    assert isinstance(data["architecture_transfer_verifiers"], list)
    
    assert "memory_augmented_verifiers" in data
    assert isinstance(data["memory_augmented_verifiers"], list)
    
    assert "corpus_specific_verifiers" in data
    assert isinstance(data["corpus_specific_verifiers"], list)
    
    assert "low_signal_verifiers" in data
    assert isinstance(data["low_signal_verifiers"], list)
    
    assert "diversity_gap_on_non_fover" in data
    assert isinstance(data["diversity_gap_on_non_fover"], bool)
    
    assert "duration_s" in data
    assert isinstance(data["duration_s"], float)
