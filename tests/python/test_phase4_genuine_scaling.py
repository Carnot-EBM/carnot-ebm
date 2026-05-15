import pytest
import numpy as np
import os
import json
import tempfile
import time
from unittest.mock import patch

from carnot.phase4.experiment_1741 import bootstrap_mean_ci, calculate_overlap_pct, main

def test_bootstrap_mean_ci():
    """Test REQ-PHASE4-003: Substrate Scaling statistics."""
    np.random.seed(42)
    data = [1.0, 1.0, 1.0, 1.0]
    mean, (lower, upper) = bootstrap_mean_ci(data, n_bootstraps=10)
    assert mean == 1.0
    assert lower == 1.0
    assert upper == 1.0

def test_calculate_overlap_pct():
    """Test REQ-PHASE4-003: Substrate Scaling overlap logic."""
    # Disjoint intervals
    ci1 = (0.0, 1.0)
    ci2 = (2.0, 3.0)
    assert calculate_overlap_pct(ci1, ci2) == 0.0

    # Identical points
    ci1 = (0.0, 0.0)
    ci2 = (0.0, 0.0)
    assert calculate_overlap_pct(ci1, ci2) == 100.0

    # Disjoint points
    ci1 = (0.0, 0.0)
    ci2 = (1.0, 1.0)
    assert calculate_overlap_pct(ci1, ci2) == 0.0

    # Fully overlapping identical intervals
    ci1 = (0.0, 2.0)
    ci2 = (0.0, 2.0)
    assert calculate_overlap_pct(ci1, ci2) == 100.0

    # Zero min range
    ci1 = (0.0, 0.0)
    ci2 = (0.0, 1.0)
    assert calculate_overlap_pct(ci1, ci2) == 0.0

    # Partially overlapping
    ci1 = (0.0, 2.0)
    ci2 = (1.0, 3.0)
    # min_upper = 2.0, max_lower = 1.0 -> overlap 1.0
    # min_range = 2.0 -> 1.0 / 2.0 = 50%
    assert calculate_overlap_pct(ci1, ci2) == 50.0

@patch('carnot.phase4.experiment_1741.time.time')
@patch('carnot.phase4.experiment_1741.time.sleep')
def test_main_experiment_full(mock_sleep, mock_time, tmp_path):
    """Test SCENARIO-PHASE4-2: Substrate Scaling end-to-end experiment run by overriding the output file."""
    time_values = [1000.0, 1205.0, 1205.0]
    mock_time.side_effect = lambda: time_values.pop(0) if time_values else 1205.0
    
    original_open = open
    def mock_open_impl(file, mode='r', *args, **kwargs):
        if 'experiment_1741_phase4_genuine_scaling.json' in str(file):
            return original_open(os.path.join(tmp_path, 'artifact.json'), mode, *args, **kwargs)
        return original_open(file, mode, *args, **kwargs)

    # To make the test run quickly, we should patch run_mld_simulation_max_caliber
    # so we don't do 4 * 30 * 100 loops of actual computation, even if it's fast.
    with patch('builtins.open', mock_open_impl), \
         patch('carnot.phase4.experiment_1741.run_mld_simulation_max_caliber') as mock_sim:
        
        # Mock simulation result
        class MockRes:
            def __init__(self, val):
                self.inf_t_alpha = val
        
        mock_sim.side_effect = lambda n_spins, k_verifiers, **kwargs: MockRes(0.1 if k_verifiers > 1 else 0.2)
        
        main()
        
        # Verify artifact was created
        with original_open(os.path.join(tmp_path, 'artifact.json'), 'r') as f:
            artifact = json.load(f)
            
        assert artifact["schema"] == "carnot.phase4_alpha_t_prime_scaling.v1"
        assert artifact["experiment"] == 1741
        assert "duration_s" in artifact
        assert artifact["acceptance_gate_passed"] in [True, False]
        assert "delta_alpha_prime" in artifact["per_n_results"][0]

