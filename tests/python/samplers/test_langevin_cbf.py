import jax
import jax.numpy as jnp
from carnot.samplers.langevin import LangevinSampler

class DummyEnergy:
    def energy(self, x: jax.Array) -> jax.Array:
        return jnp.sum(x**2)
    def grad_energy(self, x: jax.Array) -> jax.Array:
        return 2.0 * x

def test_langevin_cbf_sample():
    # REQ-SAMPLE-1807-1, REQ-SAMPLE-1807-2, SCENARIO-SAMPLE-1807
    def cbf_fn(x):
        return 10.0 * jax.nn.relu(5.0 - x[0])**2

    sampler = LangevinSampler(step_size=0.01)
    energy_fn = DummyEnergy()
    init = jnp.array([10.0, 0.0])
    
    final_state = sampler.sample(energy_fn, init, n_steps=500, cbf_fn=cbf_fn)
    assert final_state[0] >= 1.5

def test_langevin_cbf_sample_chain():
    def cbf_fn(x):
        return 10.0 * jax.nn.relu(5.0 - x[0])**2

    sampler = LangevinSampler(step_size=0.01)
    energy_fn = DummyEnergy()
    init = jnp.array([10.0, 0.0])
    
    chain = sampler.sample_chain(energy_fn, init, n_steps=500, cbf_fn=cbf_fn)
    assert chain.shape == (500, 2)
    assert chain[-1, 0] >= 1.5
