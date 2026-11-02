import jax
import jax.numpy as jnp
from carnot.samplers.langevin import LangevinSampler

class DummyEnergy:
    def energy(self, x: jax.Array) -> jax.Array:
        return jnp.sum(x**2)
    def grad_energy(self, x: jax.Array) -> jax.Array:
        return 2.0 * x

sampler = LangevinSampler(step_size=0.01)
energy_fn = DummyEnergy()

def cbf_fn(x):
    return 10.0 * jax.nn.relu(5.0 - x[0])**2

init = jnp.array([10.0, 0.0])
chain = sampler.sample_chain(energy_fn, init, n_steps=500, key=jax.random.PRNGKey(42), cbf_fn=cbf_fn)
print(chain[-10:, 0])
