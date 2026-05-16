"""
Tests for IRED (Iterative Refinement via Energy Descent) Optimizer.
"""
import numpy as np
from carnot.inference.ired_optimizer import IREDOptimizer

def test_ired_optimizer_adaptive_steps():
    """
    Verifies REQ-INFER-2098 and SCENARIO-INFER-2098:
    Adaptive threshold that stops optimization when the energy gradient norm falls below epsilon.
    Simpler constraints exit early while harder constraints use more steps.
    """
    def energy_fn_simple(state):
        # Steeper gradient, converges faster
        return np.sum(2.0 * state**2), 4.0 * state
        
    def energy_fn_hard(state):
        # Shallower gradient, converges slower
        return np.sum(0.5 * state**2), 1.0 * state
        
    initial_state = np.array([1.0, 1.0])
    
    opt_simple = IREDOptimizer(energy_fn=energy_fn_simple, max_steps=1000, learning_rate=0.1, epsilon=0.01)
    _, steps_simple = opt_simple.optimize(initial_state)
    
    opt_hard = IREDOptimizer(energy_fn=energy_fn_hard, max_steps=1000, learning_rate=0.1, epsilon=0.01)
    _, steps_hard = opt_hard.optimize(initial_state)
    
    assert steps_simple < steps_hard
    assert steps_simple > 0

def test_ired_optimizer_exits_early():
    """
    Verifies REQ-INFER-2098:
    Optimizer exits immediately if initial gradient norm is already below epsilon.
    """
    def energy_fn(state):
        return np.sum(state**2), 2.0 * state
        
    # Start close to minimum
    initial_state = np.array([0.001, 0.001])
    opt = IREDOptimizer(energy_fn=energy_fn, max_steps=1000, learning_rate=0.1, epsilon=0.1)
    _, steps = opt.optimize(initial_state)
    
    assert steps == 0
