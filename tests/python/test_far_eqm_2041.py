"""Tests for FAR continuous space EqM extraction (REQ-INFER-2041)."""

import pytest
import jax.numpy as jnp
from carnot.inference.far_eqm import extract_eqm_gradient
from carnot.inference.eqm_calibration import EqMCalibrator

def test_extract_eqm_gradient_reduces_energy_scenario_2041_001():
    """SCENARIO-INFER-2041-001: Optimization-driven refinement on toy constraints."""
    # Given continuous hidden states
    hidden_states = jnp.ones((10, 64)) * 0.5
    calibrator = EqMCalibrator(learning_rate=0.1)
    
    initial_energies, _ = extract_eqm_gradient(hidden_states)
    
    # When the EqM optimization step is applied
    current_state = hidden_states
    energies = initial_energies
    
    for _ in range(50):
        prev_energies = energies
        energies, gradients = extract_eqm_gradient(current_state)
        current_state = calibrator.update(current_state, gradients, energies, prev_energies)
        
    final_energies, _ = extract_eqm_gradient(current_state)
    
    # Then the energy landscape gradient strictly reduces the total energy
    assert jnp.mean(final_energies) < jnp.mean(initial_energies)
    
    # Ensure it goes towards 0
    assert jnp.mean(final_energies) < 0.1
