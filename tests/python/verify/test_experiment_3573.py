"""Test for Experiment 3573: Verifier Ensemble Generalization to Code.

Spec: REQ-CODE-3573
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path to import the script
scripts_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import experiment_3573_verifier_code_bug_error_detection as exp


def test_generate_corpus():
    """Test that we generate >= 100 labeled examples with required fields."""
    corpus = exp.generate_corpus(n=100, seed=42)
    assert len(corpus) == 100
    
    n_buggy = sum(1 for c in corpus if c["is_buggy"])
    assert n_buggy == 50
    
    for item in corpus:
        assert "code" in item
        assert "is_buggy" in item
        assert "model_log_prob" in item


def test_compute_metrics():
    """Test AUROC computation and required artifact fields."""
    # Dummy corpus
    corpus = [
        {"code": "def add(a, b): return a + b", "is_buggy": False, "model_log_prob": -0.1},
        {"code": "def add(a, b): return a - b", "is_buggy": True, "model_log_prob": -0.8},
        {"code": "def sub(a, b): return a - b", "is_buggy": False, "model_log_prob": -0.2},
        {"code": "def sub(a, b): return a + b", "is_buggy": True, "model_log_prob": -0.9},
    ]
    
    # Dummy verifier responses (perfect verifier)
    ensemble_scores = [0.1, 0.9, 0.1, 0.9]
    single_scores = [0.2, 0.8, 0.2, 0.8]
    
    result = exp.compute_metrics(corpus, ensemble_scores, single_scores, seed=42, duration_s=5)
    
    assert "honest_verdict" in result
    assert result["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert "ensemble_code_error_detection_auroc" in result
    assert "best_single_verifier_auroc" in result
    assert "model_confidence_baseline_auroc" in result
    assert "ensemble_minus_best_baseline_delta" in result
    assert result["n_examples"] == 4
    assert result["n_buggy"] == 2
    assert result["n_correct"] == 2
    assert "generalizes_to_code" in result
    assert "random_seed" in result
    assert "reproducibility_checksum" in result
    assert "duration_s" in result
