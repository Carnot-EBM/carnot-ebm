import jax.numpy as jnp
from carnot.phase3.lagrangian_optimizer import LagrangianOptimizer, global_lagrangian_energy

def test_global_lagrangian_energy():
    """Test REQ-KONA-071: global Lagrangian energy enforces hard bounds."""
    potentials_fn = lambda x: x**2
    x = jnp.array([2.0, -2.0])
    multipliers = jnp.array([1.0, 1.0])
    
    # Outside bounds [-1, 1], so bound penalty should be active
    # For x[0]=2.0: upper violation = 1.0. penalty = 1.0^2 * 1e4 = 1e4
    # For x[1]=-2.0: lower violation = 1.0. penalty = 1.0^2 * 1e4 = 1e4
    # Total bound penalty = 2e4
    # Constraints energy = 1.0 * 4.0 + 1.0 * 4.0 = 8.0
    # Total = 20008.0
    
    energy = global_lagrangian_energy(
        x, potentials_fn, multipliers, -1.0, 1.0, 1e4
    )
    assert jnp.allclose(energy, 20008.0)

def test_optimizer_step():
    """Test SCENARIO-KONA-071: one step primal-dual update works properly."""
    potentials_fn = lambda x: x**2
    opt = LagrangianOptimizer(potentials_fn, learning_rate=0.1, penalty_weight=1e4)
    x = jnp.array([0.5])
    m = jnp.array([1.0])
    
    # Loss = m * x^2 + penalty. Since x=0.5 in [-1, 1], penalty=0.
    # Loss = 1.0 * 0.25 = 0.25
    # grad_x = 2 * m * x = 2 * 1.0 * 0.5 = 1.0
    # grad_m = x^2 = 0.25
    # x_new = 0.5 - 0.1 * 1.0 = 0.4
    # m_new = 1.0 + 0.1 * 0.25 = 1.025
    
    x_new, m_new = opt.step(x, m)
    assert jnp.allclose(x_new, jnp.array([0.4]))
    assert jnp.allclose(m_new, jnp.array([1.025]))

def test_optimize_loop():
    """Test SCENARIO-KONA-071: multiple loop steps pushes towards bounds and feasibility."""
    potentials_fn = lambda x: x**2
    opt = LagrangianOptimizer(potentials_fn, learning_rate=0.01, penalty_weight=100.0)
    
    x_init = jnp.array([2.0]) # Outside bounds
    m_init = jnp.array([1.0])
    
    x_final, m_final = opt.optimize(x_init, m_init, steps=100)
    
    # It should be pushed inside the bounds
    assert x_final[0] <= 1.05
    assert x_final[0] >= -1.05
