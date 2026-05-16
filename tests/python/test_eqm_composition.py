import jax
import jax.numpy as jnp
from carnot.phase3.eqm_landscape import EqMLandscape, ComposedEqMLandscape, sample_langevin

def test_composed_eqm_landscape():
    # REQ-KONA-2096: Combine multiple EqM models for joint constraint satisfaction
    def energy_fn1(theta, x):
        return jnp.sum(theta[0] * x**2)
        
    def energy_fn2(theta, x):
        return jnp.sum(theta[1] * (x - 1)**2)

    l1 = EqMLandscape(energy_fn1)
    l2 = EqMLandscape(energy_fn2)

    composed = ComposedEqMLandscape([l1, l2])
    grad_estimator = composed.get_gradient_estimator()

    theta = [jnp.array([1.0]), jnp.array([1.0])]
    x = jnp.array([0.5, 0.5])
    
    grad = grad_estimator(theta, x)
    assert jnp.allclose(grad, jnp.array([0.0, 0.0]))

def test_sample_langevin():
    # SCENARIO-KONA-2096-SAMPLE: Implement sampling from the composed landscape
    def energy_fn(theta, x):
        return jnp.sum(x**2)
    
    l1 = EqMLandscape(energy_fn)
    composed = ComposedEqMLandscape([l1])
    grad_estimator = composed.get_gradient_estimator()
    
    theta = None
    init_x = jnp.array([2.0, 2.0])
    key = jax.random.PRNGKey(0)
    
    x_final = sample_langevin(grad_estimator, theta, init_x, 0.01, 100, key)
    assert x_final.shape == init_x.shape
