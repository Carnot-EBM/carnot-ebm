import jax
import jax.numpy as jnp
from carnot.samplers.casal import casal_sample

def test_casal_sample_never_violates_constraints():
    """
    Test that the CASAL sampler never violates constraints.
    References: REQ-SAMPLE-1688-1, REQ-SAMPLE-1688-2
    """
    key = jax.random.PRNGKey(42)
    
    # Simple energy function: x^2 + y^2
    def energy_fn(state):
        return jnp.sum(state**2)
        
    # Constraint: sum(state) >= 1.0
    # constraint_fn returns 0 when satisfied, >0 when violated
    # If sum(state) < 1.0, violation is 1.0 - sum(state)
    def constraint_fn(state):
        return jax.nn.relu(1.0 - jnp.sum(state))
        
    init_state = jnp.array([1.0, 1.0])
    
    # Verify initial state satisfies constraint
    assert constraint_fn(init_state) == 0.0
    
    # Run sampler
    final_state = casal_sample(
        energy_fn=energy_fn,
        constraint_fn=constraint_fn,
        init_state=init_state,
        steps=50,
        key=key,
        step_size=0.1
    )
    
    # Verify final state satisfies constraint
    violation = constraint_fn(final_state)
    assert violation <= 1e-5, f"Constraint violated: {violation}"

def test_casal_sample_rejection():
    """
    Test that the strict rejection gate works when projection fails.
    References: REQ-SAMPLE-1688-2
    """
    key = jax.random.PRNGKey(42)
    
    def energy_fn(state):
        return jnp.sum(state**2)
        
    # Impossible constraint to reach from projection in 1 step
    # x + y >= 10.0
    def constraint_fn(state):
        return jax.nn.relu(10.0 - jnp.sum(state))
        
    # Start at a valid state
    init_state = jnp.array([5.0, 5.0])
    
    # Even with large step size, it won't satisfy constraints if projection fails and should reject
    final_state = casal_sample(
        energy_fn=energy_fn,
        constraint_fn=constraint_fn,
        init_state=init_state,
        steps=10,
        key=key,
        step_size=10.0,
        proj_steps=1,   # not enough steps to project
        proj_lr=0.001   # too small lr
    )
    
    # Constraint must be satisfied (it should have rejected and stayed at init_state)
    violation = constraint_fn(final_state)
    assert violation <= 1e-5
