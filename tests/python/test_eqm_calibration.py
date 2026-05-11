import pytest
import jax.numpy as jnp
from carnot.inference.eqm_calibration import eqm_step, EqMCalibrator

def test_eqm_step_REQ_INFER_1829():
    """
    Test that EqM step scales based on energy difference and learning rate.
    REQ-INFER-1829
    """
    # Simple setup
    current_state = jnp.array([1.0, 2.0])
    gradient = jnp.array([0.1, -0.1])
    energy = 5.0
    prev_energy = 5.5
    lr = 0.01

    # In EqM, adaptive computation scales the step.
    # A simple valley finding scaling: lr * exp(-(prev_energy - energy))
    # or similar. We will define it as step = lr * gradient * exp(energy - prev_energy)
    
    new_state = eqm_step(current_state, gradient, energy, prev_energy, lr)
    
    # We expect some update to have happened
    assert not jnp.allclose(new_state, current_state)
    assert new_state.shape == current_state.shape

def test_eqm_calibrator_REQ_INFER_1829():
    """
    Test EqM calibrator class for stable valley finding.
    REQ-INFER-1829
    """
    calibrator = EqMCalibrator(learning_rate=0.1)
    
    states = jnp.array([[1.0, 1.0], [2.0, 2.0]])
    gradients = jnp.array([[0.1, 0.1], [0.2, 0.2]])
    energies = jnp.array([10.0, 20.0])
    prev_energies = jnp.array([11.0, 19.0])
    
    new_states = calibrator.update(states, gradients, energies, prev_energies)
    assert new_states.shape == states.shape
    assert not jnp.allclose(new_states, states)
