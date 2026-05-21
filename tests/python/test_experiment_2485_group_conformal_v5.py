import json
from pathlib import Path
import pytest
import sys
import numpy as np

# Adjust sys.path to import the script
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / 'scripts'))

from experiment_2485_group_conformal_v5 import (
    run_experiment,
    compute_p_values,
    fisher_combine,
    normalize_label
)

def test_fisher_combine():
    p_values = np.array([[0.1, 0.2], [0.9, 0.8]])
    combined = fisher_combine(p_values)
    assert combined.shape == (2,)
    # p-values [0.1, 0.2] means strong evidence of hallucination
    # Thus the return value should be high (closer to 1.0)
    assert combined[0] > combined[1]
    assert combined[0] > 0.5
    assert combined[1] < 0.5

def test_normalize_label():
    assert normalize_label("correct") == 0
    assert normalize_label("incorrect") == 1
    assert normalize_label("0") == 0
    assert normalize_label("1") == 1

def test_experiment_runs():
    results_dir = Path("/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results")
    if not results_dir.exists():
        pytest.skip("Test requires the actual results directory to run.")
    
    deliverable = run_experiment(results_dir=results_dir)
    
    assert "group_conditional_auroc_mean" in deliverable
    assert "group_conditional_vs_isotonic_delta" in deliverable
    assert "hive_peer_breached_group_cond" in deliverable
    assert "honest_verdict" in deliverable
    assert deliverable["honest_verdict"].startswith("complete:")
    assert "results_by_seed" in deliverable
    assert len(deliverable["results_by_seed"]) == 5
    
    assert 0.0 < deliverable["group_conditional_auroc_mean"] <= 1.0
