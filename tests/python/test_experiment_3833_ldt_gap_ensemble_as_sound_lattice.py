import pytest
from pathlib import Path
import json
import sys

# ensure the script directory is accessible
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import experiment_3833_ldt_gap_ensemble_as_sound_lattice as exp3833

def test_run_experiment_success(tmp_path, monkeypatch):
    # Mock get_repo_root to return a temporary directory with mocked data
    repo_root = tmp_path
    data_dir = repo_root / "data"
    data_dir.mkdir()
    
    # Create fake corpus
    corpus = []
    for i in range(210):
        label = "correct" if i < 190 else "incorrect"
        corpus.append({"question_id": f"q{i}", "step_text": f"text {i}", "label": label})
        
    with open(data_dir / "fover_test_v4.json", "w") as f:
        json.dump(corpus, f)
        
    results_dir = repo_root / "results"
    results_dir.mkdir()

    # We need to mock the verifier_scores to avoid actually running models or logic
    def mock_score_text_verifiers(texts):
        return {
            "tier0r_curry_howard": [0.1 if "text" in t else 0.9 for t in texts],
            "tier0u_logical_consistency": [0.2 if "text" in t else 0.8 for t in texts]
        }
    
    monkeypatch.setattr("carnot.eval.fover_memory_leakage_v3._score_text_verifiers", mock_score_text_verifiers)
    
    def mock_load_fr11_memory_index(root):
        return {"question_ids": set(), "prompt_token_sets": []}
        
    monkeypatch.setattr("carnot.eval.fover_memory_leakage_v3._load_fr11_memory_index", mock_load_fr11_memory_index)

    result = exp3833.run_experiment(repo_root=repo_root, write=True)
    
    assert result["n_candidates"] == 210
    assert result["duration_s"] >= 1.0
    assert "honest_verdict" in result
    assert result["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert "reproducibility_checksum" in result
    
    out_json = results_dir / "experiment_3833_ldt_gap_ensemble_as_sound_lattice.json"
    assert out_json.exists()

def test_run_experiment_no_corpus(tmp_path):
    repo_root = tmp_path
    (repo_root / "data").mkdir()
    
    result = exp3833.run_experiment(repo_root=repo_root, write=False)
    
    assert "honest_verdict" in result
    assert "blocked_fover_corpus_not_available" in result["honest_verdict"]
    assert result["preconditions_checked"]["corpus_loaded"] is False

def test_run_experiment_verify_import_fails(tmp_path, monkeypatch):
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'carnot.verify':
            raise ImportError("Mocked missing carnot.verify")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    result = exp3833.run_experiment(repo_root=tmp_path, write=False)
    assert result["honest_verdict"] == "blocked_verify_module_import"

def test_run_experiment_score_fails(tmp_path, monkeypatch):
    repo_root = tmp_path
    data_dir = repo_root / "data"
    data_dir.mkdir()
    corpus = [{"question_id": "q1", "step_text": "text", "label": "correct"}] * 210
    with open(data_dir / "fover_test_v4.json", "w") as f:
        json.dump(corpus, f)
        
    def mock_score_text_verifiers(texts):
        raise ValueError("Mock score failure")
        
    monkeypatch.setattr("carnot.eval.fover_memory_leakage_v3._score_text_verifiers", mock_score_text_verifiers)
    
    result = exp3833.run_experiment(repo_root=repo_root, write=False)
    assert result["honest_verdict"] == "blocked_score_candidate_failed"
