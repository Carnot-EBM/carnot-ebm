import os
import json
import pytest
from unittest.mock import patch
import jax.numpy as jnp

from carnot.experiment_1747_diagnostic import (
    check_preconditions,
    run_mode_collapse_test,
    run_energy_boundedness_test,
    run_sign_convention_test,
    main
)

def test_check_preconditions():
    # REQ-EBT-1747-1
    pre = check_preconditions()
    assert isinstance(pre, list)
    assert "jax_import_successful" in pre

def test_run_mode_collapse_test():
    # REQ-EBT-1747-2
    def mock_energy_fn(x):
        return jnp.sum(x**2)
    mean_dist, median_dist, max_dist, mode_collapse = run_mode_collapse_test(mock_energy_fn, 16, 10, 10)
    assert isinstance(mean_dist, float)
    assert isinstance(median_dist, float)
    assert isinstance(max_dist, int)
    assert isinstance(mode_collapse, bool)

def test_run_energy_boundedness_test():
    # REQ-EBT-1747-3
    def mock_energy_fn(x):
        return jnp.sum(x**2)
    mean_e, std_e, min_e, max_e, unbounded = run_energy_boundedness_test(mock_energy_fn, 16, 50)
    assert isinstance(mean_e, float)
    assert isinstance(unbounded, bool)

def test_run_sign_convention_test():
    # REQ-EBT-1747-4
    def mock_energy_fn(x):
        return jnp.sum(x**2)
    inverted = run_sign_convention_test(mock_energy_fn, 16)
    assert isinstance(inverted, bool)

@patch("time.sleep", return_value=None)
def test_main(mock_sleep, tmp_path):
    # REQ-EBT-1747-5
    
    # We patch the path to results to write in a temporary dir
    # wait, main writes to "results/experiment_1747_ebt_mode_collapse_check.json"
    # let's just let it write there since results directory exists
    
    # We can patch n_inits to 5, energy_sample_count to 50 for faster test execution if needed
    # but the prompt requires it to run, let's just mock sleep
    main()
    
    assert os.path.exists("results/experiment_1747_ebt_mode_collapse_check.json")
    with open("results/experiment_1747_ebt_mode_collapse_check.json", "r") as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.ebt_mode_collapse_audit.v1"
    assert data["experiment"] == 1747
    assert data["n_samples"] == 1030
    assert "root_cause" in data
    assert data["acceptance_gate_passed"] is True
