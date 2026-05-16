"""
Tests for IRED Trainer.

These tests trace to:
REQ-INFER-2099: IRED Training Loop
SCENARIO-INFER-2099: Training IRED on Synthetic Constraints
"""

import numpy as np
import jax
import jax.numpy as jnp
from carnot.inference.ired_trainer import create_train_state, train_step, get_energy_fn
from carnot.inference.ired_optimizer import IREDOptimizer

def test_ired_trainer_loop():
    """
    Verifies that we can create a train state, perform training steps, and
    obtain an energy function that correctly minimizes at valid states.
    Traces to REQ-INFER-2099 and SCENARIO-INFER-2099.
    """
    rng = jax.random.PRNGKey(0)
    input_dim = 3
    output_dim = 2
    state = create_train_state(rng, input_dim, output_dim, learning_rate=0.1)
    
    # Synthetic dataset
    constraints = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    targets = jnp.array([
        [0.5, 0.5],
        [-0.5, -0.5]
    ])
    
    # Train for a few steps
    for _ in range(100):
        state, loss = train_step(state, constraints, targets)
        
    # The loss should be low
    assert loss < 0.1, f"Loss is too high: {loss}"
    
    # Get energy function for constraint 0
    energy_fn_0 = get_energy_fn(state, np.array([1.0, 0.0, 0.0]))
    
    # Run IRED optimizer
    opt = IREDOptimizer(energy_fn=energy_fn_0, max_steps=100, learning_rate=0.5, epsilon=1e-3)
    initial_state = np.array([0.0, 0.0])
    optimized_state, steps = opt.optimize(initial_state)
    
    # Check that it converged near the target [0.5, 0.5]
    assert np.allclose(optimized_state, np.array([0.5, 0.5]), atol=0.2)
    
    # Get energy function for constraint 1
    energy_fn_1 = get_energy_fn(state, np.array([0.0, 1.0, 0.0]))
    opt2 = IREDOptimizer(energy_fn=energy_fn_1, max_steps=100, learning_rate=0.5, epsilon=1e-3)
    optimized_state2, steps2 = opt2.optimize(initial_state)
    
    # Check that it converged near the target [-0.5, -0.5]
    assert np.allclose(optimized_state2, np.array([-0.5, -0.5]), atol=0.2)
