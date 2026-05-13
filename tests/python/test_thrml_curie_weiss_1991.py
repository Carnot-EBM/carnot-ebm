"""Tests for Experiment 1991: Corrected Curie-Weiss Parity.

Spec traces: REQ-SAMPLE-1991, SCENARIO-SAMPLE-1991
"""

import os
from unittest import mock
import numpy as np

from carnot.samplers.thrml_carnot_parity_curie_weiss_1991 import (
    exact_curie_weiss_energy,
    run_experiment,
)


def test_exact_curie_weiss_energy_bounds():
    """Verify analytic ground truth calculation.
    
    Spec: SCENARIO-SAMPLE-1991
    """
    n_spins = 2
    beta = 1.0
    J_val = 1.0
    energy = exact_curie_weiss_energy(n_spins, beta, J_val)
    
    # E(k) = - (1.0 / 2) * (k^2 - k)
    # k=0: E=0, mult=1
    # k=1: E=0, mult=2
    # k=2: E=-1.0, mult=1
    expected_Z = 1 * np.exp(0.0) + 2 * np.exp(0.0) + 1 * np.exp(1.0)
    expected_E_mean = (0.0 * 3 + (-1.0) * np.exp(1.0)) / expected_Z
    np.testing.assert_allclose(energy, expected_E_mean, rtol=1e-5)


@mock.patch("carnot.samplers.thrml_carnot_parity_curie_weiss_1991.ThrmlSamplerBackend")
@mock.patch("carnot.samplers.thrml_carnot_parity_curie_weiss_1991.CpuBackend")
def test_run_experiment_parity(mock_cpu, mock_thrml):
    """Test parity experiment outputs required artifact.
    
    Spec: SCENARIO-SAMPLE-1991
    """
    mock_cpu_instance = mock.MagicMock()
    mock_cpu.return_value = mock_cpu_instance
    
    mock_thrml_instance = mock.MagicMock()
    mock_thrml.return_value = mock_thrml_instance
    
    n_samples = 100
    n_spins = 128
    rng = np.random.RandomState(42)
    
    mock_cpu_instance.sample.return_value = rng.binomial(1, 0.5, (n_samples, n_spins))
    mock_thrml_instance.sample.return_value = rng.binomial(1, 0.5, (n_samples, n_spins))
    
    result = run_experiment(n_samples=n_samples)
    
    assert result["schema"] == "carnot.curie_weiss_parity_sweep.v1"
    assert result["hardware_execution_claim"] is False
    assert "analytic_mean_energy" in result
    assert "carnot_mean_energy" in result
    assert "thrml_mean_energy" in result
    assert "acceptance_gate_passed" in result
    assert os.path.exists("results/experiment_1991_curie_weiss_parity_correction.json")
