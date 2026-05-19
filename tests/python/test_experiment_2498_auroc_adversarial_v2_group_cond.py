import pytest
from pathlib import Path
import sys

# Add scripts to path so we can import it
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
import experiment_2498_auroc_adversarial_v2_group_cond as exp2498

def test_run_experiment_creates_deliverable():
    results_dir = Path(__file__).parent.parent.parent / 'results'
    
    # Run the experiment
    deliverable = exp2498.run_experiment(results_dir)
    
    assert deliverable is not None
    assert "group_conditional_auroc_replicated" in deliverable
    assert "cross_group_tautology_resolved" in deliverable
    assert "hive_peer_breached_replicated" in deliverable
    assert "auroc_adversarially_verified" in deliverable
    assert "honest_verdict" in deliverable
    
    # Assert specific conditions
    assert deliverable["cross_group_tautology_resolved"] is True
    assert deliverable["group_conditional_auroc_replicated"] > 0.0
    assert deliverable["hive_peer_breached_replicated"] is True
    assert deliverable["auroc_adversarially_verified"] is True
    
    assert "complete:" in deliverable["honest_verdict"]

def test_fisher_combine():
    import numpy as np
    # Small sanity check for fisher combine
    p_vals = np.array([[0.01, 0.01], [0.5, 0.5]])
    combined = exp2498.fisher_combine(p_vals)
    assert combined.shape == (2,)
    assert combined[0] > combined[1], "Smaller p-values should yield higher significance (closer to 1.0 depending on how sf returns)"

def test_compute_p_values():
    import numpy as np
    X_cal = np.array([[0.1], [0.3], [0.5], [0.7]])
    X_test = np.array([[0.2], [0.6]])
    # For X_test = 0.2: count of cal_scores >= 0.2 is 3 (0.3, 0.5, 0.7). p-value = 3 / (4+1) = 0.6
    # For X_test = 0.6: count of cal_scores >= 0.6 is 1 (0.7). p-value = 1 / 5 = 0.2
    p_vals = exp2498.compute_p_values(X_cal, X_test)
    assert np.isclose(p_vals[0, 0], 0.6)
    assert np.isclose(p_vals[1, 0], 0.2)
