import jax
import jax.numpy as jnp
import optax
from carnot.models.ebt_compatibility import EBTCompatibilityModel, ebt_compatibility_loop, compare_with_log_prob

def test_ebt_compatibility_energy():
    """Test scalar compatibility energy output for sequence pairs.
    
    Spec: REQ-NRGPT-003, SCENARIO-NRGPT-003
    """
    key = jax.random.PRNGKey(0)
    model = EBTCompatibilityModel(input_dim=16, hidden_dim=32, key=key)
    
    seq_a = jax.random.normal(key, (16,))
    seq_b = jax.random.normal(jax.random.PRNGKey(1), (16,))
    
    energy = model.energy(seq_a, seq_b)
    assert isinstance(energy, jax.Array)
    assert energy.shape == ()

def test_ebt_optimization_loop():
    """Test the energy descent curve during optimization.
    
    Spec: REQ-NRGPT-003, SCENARIO-NRGPT-003
    """
    key = jax.random.PRNGKey(0)
    model = EBTCompatibilityModel(input_dim=8, hidden_dim=16, key=key)
    
    seq_a = jax.random.normal(key, (8,))
    seq_b_init = jax.random.normal(jax.random.PRNGKey(1), (8,))
    
    optimized_seq_b, energy_curve = ebt_compatibility_loop(model, seq_a, seq_b_init, steps=10, lr=0.1)
    
    assert len(energy_curve) == 10
    # Energy should generally go down
    assert energy_curve[-1] < energy_curve[0]
    assert optimized_seq_b.shape == (8,)

def test_compare_with_log_prob():
    """Compare EBT approach with traditional conditional log-probability.
    
    Spec: REQ-NRGPT-003, SCENARIO-NRGPT-003
    """
    key = jax.random.PRNGKey(0)
    seq_a = jnp.array([1.0, 0.5, -0.5, 2.0])
    seq_b = jnp.array([0.5, 0.0, -1.0, 1.5])
    
    # compare_with_log_prob should return log probability under a dummy baseline model
    log_prob = compare_with_log_prob(seq_a, seq_b)
    
    assert isinstance(log_prob, jax.Array)
    assert log_prob.shape == ()
