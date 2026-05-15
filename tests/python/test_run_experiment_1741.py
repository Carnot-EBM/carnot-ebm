import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from run_experiment_1741 import calculate_overlap, run_scaling_experiment

def test_calculate_overlap_no_overlap():
    """REQ-1741-1: SCENARIO-1: calculate_overlap returns 0 if CIs do not overlap."""
    ci1 = [0.1, 0.2]
    ci2 = [0.3, 0.4]
    assert calculate_overlap(ci1, ci2) == 0.0

def test_calculate_overlap_full_overlap():
    """REQ-1741-2: SCENARIO-2: calculate_overlap returns correct percentage on full overlap."""
    ci1 = [0.1, 0.3]
    ci2 = [0.1, 0.3]
    assert calculate_overlap(ci1, ci2) == 100.0

def test_calculate_overlap_partial_overlap():
    """REQ-1741-3: SCENARIO-3: calculate_overlap correctly computes partial overlap."""
    ci1 = [0.1, 0.3] # width 0.2
    ci2 = [0.2, 0.4] # width 0.2
    # overlap [0.2, 0.3] -> width 0.1 -> 50%
    assert np.isclose(calculate_overlap(ci1, ci2), 50.0)

def test_run_scaling_experiment():
    """REQ-1741-4: SCENARIO-4: run_scaling_experiment returns expected schema and results."""
    artifact = run_scaling_experiment(mld_steps=2, n_seeds=2, n_boot=10)
    assert artifact["experiment"] == 1741
    assert "reproducibility_checksum" in artifact
    assert artifact["schema"] == "carnot.phase4_alpha_t_prime_scaling.v1"
    assert "acceptance_gate_passed" in artifact
    assert len(artifact["per_n_results"]) == 4
    assert artifact["random_seed"] == 172041
    assert artifact["honest_verdict"].startswith("complete:")
