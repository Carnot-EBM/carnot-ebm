import jax
import jax.numpy as jnp
from carnot.samplers.thermo_init import mpemba_init

def test_mpemba_init_accelerates_convergence():
    """
    Tests that mpemba_init provides a convergence speedup on a toy 2D energy landscape.
    REQ-SAMPLE-2605, SCENARIO-SAMPLE-2605
    """
    # Define a simple 2D energy landscape (convex bowl)
    def energy_fn(x):
        # Elliptical bowl to make the optimization non-trivial but simple
        return 0.5 * jnp.sum(jnp.array([1.0, 5.0]) * x**2)
    
    key = jax.random.PRNGKey(42)
    key1, key2, key3, key4 = jax.random.split(key, 4)
    shape = (2,)
    hot_beta = 0.1
    target_beta = 1.0
    
    # Standard pure pure-noise initialization
    x_std = jax.random.normal(key1, shape) / jnp.sqrt(hot_beta)
    
    # Mpemba-inspired initialization
    x_mpemba = mpemba_init(key2, energy_fn, shape, hot_beta, target_beta, num_optim_steps=15, step_size=0.1)
    
    # Simple Langevin sampler
    def run_langevin(x_init, num_steps, step_size=0.01):
        def step(x, k):
            grad = jax.grad(energy_fn)(x)
            noise = jax.random.normal(k, x.shape) * jnp.sqrt(2 * step_size / target_beta)
            x_new = x - step_size * grad + noise
            return x_new, energy_fn(x_new)
        
        keys = jax.random.split(key3, num_steps)
        _, energies = jax.lax.scan(step, x_init, keys)
        return energies

    num_steps = 100
    energies_std = run_langevin(x_std, num_steps)
    energies_mpemba = run_langevin(x_mpemba, num_steps)
    
    # Calculate how many steps to reach a target threshold
    threshold = 2.0
    
    # Find first step below threshold
    std_below = energies_std < threshold
    mpemba_below = energies_mpemba < threshold
    
    # If it never reaches, set to num_steps
    std_steps_to_converge = jnp.where(jnp.any(std_below), jnp.argmax(std_below), num_steps)
    mpemba_steps_to_converge = jnp.where(jnp.any(mpemba_below), jnp.argmax(mpemba_below), num_steps)
    
    # Verify the Mpemba initialized sampler reaches the threshold in fewer steps
    assert mpemba_steps_to_converge < std_steps_to_converge, \
        f"Mpemba took {mpemba_steps_to_converge} steps, standard took {std_steps_to_converge} steps"
    
    # Verify the convergence speedup is positive
    speedup = std_steps_to_converge - mpemba_steps_to_converge
    assert speedup > 0, "Speedup must be strictly positive"
