"""Tests for Exp 1682 bias investigation."""

import os
import json
import numpy as np
from unittest.mock import patch, mock_open

from scripts.experiment_1682_thrml_bias import (
    get_analytic_mean,
    run_carnot_mean,
    compute_verdict,
    run_sweeps,
    main
)

def test_get_analytic_mean():
    """Verify analytic mean calculation.
    
    References REQ-SAMPLE-1682-1
    """
    # High temp (beta * J <= 1) -> mean is 0
    assert get_analytic_mean(0.5, 1.0) == 0.0
    
    # Low temp (beta * J > 1) -> mean is > 0
    m = get_analytic_mean(1.5, 1.0)
    assert m > 0.0
    assert np.isclose(m, np.tanh(1.5 * m), atol=1e-3)

@patch('scripts.experiment_1682_thrml_bias.CpuBackend')
def test_run_carnot_mean(mock_backend):
    """Verify Carnot simulation wrapper.
    
    References REQ-SAMPLE-1682-1, REQ-SAMPLE-1682-2
    """
    instance = mock_backend.return_value
    # Mock return 10 samples of 128 spins (0 or 1)
    instance.sample.return_value = np.ones((10, 128))
    
    mean = run_carnot_mean(N_spins=128, beta=1.2, J=1.0, n_samples=10, seed=42)
    assert mean == 1.0

def test_compute_verdict():
    """Verify fit verdict logic.
    
    References REQ-SAMPLE-1682-3
    """
    sweep_a_N = [10000, 30000, 100000]
    sweep_b_beta = [1.05, 1.2, 1.5]
    
    # Simulate systematic (small slope, significant intercept)
    sweep_a_bias_sys = [0.04, 0.041, 0.039]
    sweep_b_bias_sys = [0.01, 0.04, 0.08]
    assert compute_verdict(sweep_a_N, sweep_a_bias_sys, sweep_b_beta, sweep_b_bias_sys) == "systematic"
    
    # Simulate finite_n (large slope, small intercept)
    x = 1.0 / np.sqrt(sweep_a_N)
    sweep_a_bias_fin = 2.0 * x
    sweep_b_bias_fin = [0.02, 0.02, 0.02]
    assert compute_verdict(sweep_a_N, sweep_a_bias_fin, sweep_b_beta, sweep_b_bias_fin) == "finite_n"
    
    # Simulate mixed (large slope, large intercept)
    sweep_a_bias_mix = 2.0 * x + 0.05
    sweep_b_bias_mix = [0.03, 0.05, 0.07]
    assert compute_verdict(sweep_a_N, sweep_a_bias_mix, sweep_b_beta, sweep_b_bias_mix) == "mixed"

@patch('scripts.experiment_1682_thrml_bias.run_carnot_mean')
def test_run_sweeps(mock_run_carnot):
    """Verify sweep orchestration.
    
    References REQ-SAMPLE-1682-1, REQ-SAMPLE-1682-2, REQ-SAMPLE-1682-3
    """
    mock_run_carnot.return_value = 0.6  # mock empirical mean
    
    res = run_sweeps()
    
    assert res["schema"] == "carnot.thrml_parity_bias_investigation.v1"
    assert res["n_spins"] == 128
    assert res["sweep_a_N"] == [10000, 30000, 100000]
    assert len(res["sweep_a_bias"]) == 3
    assert res["sweep_b_beta"] == [1.05, 1.2, 1.5]
    assert len(res["sweep_b_bias"]) == 3
    assert "bias_fit_verdict" in res
    assert res["acceptance_gate_passed"] is True
    assert "complete:" in res["honest_verdict"]
    
@patch('scripts.experiment_1682_thrml_bias.run_sweeps')
def test_main(mock_run_sweeps, tmp_path):
    """Verify main function writes to correct file.
    
    References REQ-SAMPLE-1682-4, SCENARIO-SAMPLE-1682
    """
    mock_run_sweeps.return_value = {"dummy": "data"}
    
    with patch('builtins.open', mock_open()) as mocked_file:
        with patch('os.makedirs'):
            main()
            mocked_file.assert_called_with("results/experiment_1682_thrml_bias.json", "w")
