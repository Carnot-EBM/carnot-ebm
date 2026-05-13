import jax.numpy as jnp
from carnot.pipeline.continuous_self_learner import ContinuousSelfLearner, CompositionalEnergyMinimizer

def test_compositional_energy_minimizer():
    # Test REQ-* or SCENARIO-* logic
    minimizer = CompositionalEnergyMinimizer(jnp.array([1.0, 1.0, 1.0]))
    scenario = jnp.array([0.0, 0.0, 0.0])
    optimized, energy = minimizer.minimize(scenario)
    
    # Ensure energy is reduced
    assert energy < 3.0

def test_continuous_self_learner():
    # SCENARIO: Unsupervised Continuous Self-Learning 
    learner = ContinuousSelfLearner("unsloth/Qwen3.6-35B-A3B-GGUF")
    
    scenarios = [
        jnp.array([0.5, 0.5, 0.5]),
        jnp.array([-0.5, -0.5, -0.5]),
        jnp.array([0.1, 0.2, 0.3]),
        jnp.array([2.0, 2.0, 2.0]),
        jnp.array([0.0, 1.0, 0.0])
    ]
    
    deltas = learner.process_scenarios(scenarios)
    assert len(deltas) == 5
    for delta in deltas:
        assert isinstance(delta, float)
        assert delta >= 0.0 # Energy should decrease or stay the same
