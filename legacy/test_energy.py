from typing import Protocol, runtime_checkable
import jax
import jax.numpy as jnp

class IsingEnergyFunction:
    def __init__(self, biases: jax.Array, couplings: jax.Array):
        self.biases = biases
        self.couplings = couplings
        self._input_dim = biases.shape[0]

    def energy(self, x: jax.Array) -> jax.Array:
        # x is (n_spins,)
        # continuous relaxation of {0, 1} spins
        return -jnp.dot(x, jnp.dot(self.couplings, x)) - jnp.dot(self.biases, x)

    def energy_batch(self, xs: jax.Array) -> jax.Array:
        # xs is (batch, n_spins)
        return jax.vmap(self.energy)(xs)

    def grad_energy(self, x: jax.Array) -> jax.Array:
        return jax.grad(self.energy)(x)

    @property
    def input_dim(self) -> int:
        return self._input_dim
