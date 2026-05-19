import pytest
from pathlib import Path
import sys
import numpy as np

# Add scripts to path so we can import it
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'python'))

import experiment_2521_ensemble_v7 as exp2521

def test_run_experiment_creates_deliverable():
    results_dir = Path(__file__).parent.parent.parent / 'results'
    
    # Run the experiment
    deliverable = exp2521.run_experiment(results_dir)
    
    assert deliverable is not None
    assert "ensemble_v7_auroc" in deliverable
    assert "ensemble_v7_auroc_std" in deliverable
    assert "honest_verdict" in deliverable
    assert "tier0r_group_assignment" in deliverable
    assert deliverable["tier0r_group_assignment"] == "Group C"

def test_fisher_combine():
    p_vals = np.array([[0.01, 0.01], [0.5, 0.5]])
    combined = exp2521.fisher_combine(p_vals)
    assert combined.shape == (2,)
    assert combined[0] > combined[1]

def test_compute_p_values():
    X_cal = np.array([[0.1], [0.3], [0.5], [0.7]])
    X_test = np.array([[0.2], [0.6]])
    p_vals = exp2521.compute_p_values(X_cal, X_test)
    assert np.isclose(p_vals[0, 0], 0.6)
    assert np.isclose(p_vals[1, 0], 0.2)
