"""Langevin Dynamics sampler -- JAX implementation.

**Researcher summary:**
    Unadjusted Langevin Algorithm (ULA). Discrete-time Langevin SDE:
    x_{t+1} = x_t - (eps/2) * grad E(x_t) + sqrt(eps) * z_t, z_t ~ N(0, I).
    Converges to the Boltzmann distribution p(x) ~ exp(-E(x)) as eps -> 0 and
    n_steps -> infinity.

**Detailed explanation for engineers:**
    Langevin dynamics is a method for generating samples from a probability
    distribution defined by an energy function. If you have an EBM with energy
    E(x), the Boltzmann distribution is p(x) ~ exp(-E(x)). Langevin dynamics
    generates samples from this distribution by starting from a random point and
    iteratively:

    1. Computing the gradient of the energy at the current point (which direction
       makes energy increase fastest)
    2. Taking a step in the *opposite* direction (toward lower energy / higher
       probability)
    3. Adding random Gaussian noise (to explore and not just collapse to the
       mode)

    The update rule is:
        x_new = x - (step_size / 2) * grad_E(x) + sqrt(step_size) * noise

    The balance between the gradient step (exploitation) and the noise
    (exploration) is controlled by ``step_size``:
    - Larger step_size: faster exploration but less accurate (discretization error)
    - Smaller step_size: more accurate but slower convergence

    **Why "unadjusted"?**
    This implementation does NOT include a Metropolis-Hastings accept/reject step.
    This means it has some discretization bias — the samples are approximately
    (not exactly) from the target distribution. For exact sampling, use HMC which
    includes an accept/reject correction. ULA is simpler and often sufficient.

    **JAX implementation note:**
    The sampling loop uses ``jax.lax.scan``, which is JAX's way of expressing
    a for-loop that can be compiled (JIT'd) into efficient XLA code. Unlike a
    Python for-loop, ``jax.lax.scan`` runs entirely on the accelerator (GPU/TPU).

Spec: REQ-SAMPLE-001
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import EnergyFunction


@dataclass
class LangevinSampler:
    """Unadjusted Langevin Dynamics (ULA) sampler.

    **Researcher summary:**
        Discretized overdamped Langevin diffusion. No Metropolis correction.
        Returns either the final sample or the full chain.

    **Detailed explanation for engineers:**
        This sampler takes any ``EnergyFunction`` and generates samples from
        its Boltzmann distribution. You provide an initial state, and the
        sampler iterates the Langevin update for ``n_steps`` iterations.

        Two modes are available:
        - ``sample()``: Returns only the final state (memory-efficient).
        - ``sample_chain()``: Returns all intermediate states (useful for
          diagnostics, mixing analysis, or visualization).

    Attributes:
        step_size: The discretization step size (epsilon). Controls the
            trade-off between speed and accuracy. Typical values: 0.001 to 0.1.
            Too large causes instability; too small causes slow mixing.

    For example::

        from carnot.models.ising import IsingModel, IsingConfig
        import jax.numpy as jnp

        model = IsingModel(IsingConfig(input_dim=10))
        sampler = LangevinSampler(step_size=0.01)

        x0 = jnp.zeros(10)  # start from origin
        x_final = sampler.sample(model, x0, n_steps=1000)
        # x_final is a sample approximately from p(x) ~ exp(-E(x))

    Spec: REQ-SAMPLE-001
    """

    step_size: float = 0.01

    def sample(
        self,
        energy_fn: EnergyFunction,
        init: jax.Array,
        n_steps: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Run Langevin dynamics and return only the final sample.

        **Researcher summary:**
            Runs n_steps of ULA from init, returns x_final.

        **Detailed explanation for engineers:**
            This method runs the full Langevin chain but only returns the last
            state. The intermediate states are computed (needed for the
            sequential update) but not stored, making this memory-efficient
            for long chains.

            Internally, this uses ``jax.lax.scan`` — JAX's compiled loop
            primitive. ``scan`` takes a "step" function, an initial carry
            state, and a sequence to scan over. It applies the step function
            repeatedly, threading the carry through each iteration.

            The carry here is (x, key):
            - x: the current state vector
            - key: the PRNG key (split each iteration to get fresh randomness)

        Args:
            energy_fn: Any object satisfying the EnergyFunction protocol.
            init: Initial state vector, shape (input_dim,).
            n_steps: Number of Langevin steps to take.
            key: JAX PRNG key. If None, uses seed 0.

        Returns:
            The final state after n_steps, shape (input_dim,).

        Spec: REQ-SAMPLE-001, SCENARIO-SAMPLE-001
        """
        if key is None:
            key = jrandom.PRNGKey(0)

        # Pre-compute noise scale: sqrt(step_size). This is the standard
        # deviation of the Gaussian noise added at each step.
        noise_scale = jnp.sqrt(self.step_size)

        def step(carry: tuple[jax.Array, jax.Array], _: None) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            x, key = carry
            # Split the PRNG key: one for this step's noise, one for next step
            key, subkey = jrandom.split(key)
            # Compute gradient dE/dx at the current state
            grad = energy_fn.grad_energy(x)
            # Sample isotropic Gaussian noise
            noise = jrandom.normal(subkey, x.shape)
            # Langevin update: move against gradient (toward lower energy)
            # and add scaled noise for exploration
            x_new = x - (self.step_size * 0.5) * grad + noise_scale * noise
            return (x_new, key), x_new

        # jax.lax.scan runs `step` n_steps times, threading (x, key) through.
        # The second element of the return is the stacked outputs (the chain),
        # but we only need x_final from the carry.
        (x_final, _), chain = jax.lax.scan(step, (init, key), None, length=n_steps)
        return x_final

    def sample_chain(
        self,
        energy_fn: EnergyFunction,
        init: jax.Array,
        n_steps: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Run Langevin dynamics and return the full chain of states.

        **Researcher summary:**
            Returns all n_steps states for chain diagnostics and mixing analysis.

        **Detailed explanation for engineers:**
            Same algorithm as ``sample()``, but returns an array of shape
            (n_steps, *init.shape) containing every intermediate state.
            Useful for:
            - Visualizing how the sampler explores the energy landscape
            - Diagnosing mixing problems (e.g., the chain gets stuck)
            - Computing autocorrelation to estimate effective sample size

            Note: this stores all n_steps states in memory, so for very long
            chains, prefer ``sample()`` which only returns the final state.

        Args:
            energy_fn: Any object satisfying the EnergyFunction protocol.
            init: Initial state vector, shape (input_dim,).
            n_steps: Number of Langevin steps to take.
            key: JAX PRNG key. If None, uses seed 0.

        Returns:
            Array of shape (n_steps, *init.shape) — the full sampling chain.

        Spec: REQ-SAMPLE-001, SCENARIO-SAMPLE-001
        """
        if key is None:
            key = jrandom.PRNGKey(0)

        noise_scale = jnp.sqrt(self.step_size)

        def step(carry: tuple[jax.Array, jax.Array], _: None) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            x, key = carry
            key, subkey = jrandom.split(key)
            grad = energy_fn.grad_energy(x)
            noise = jrandom.normal(subkey, x.shape)
            x_new = x - (self.step_size * 0.5) * grad + noise_scale * noise
            # The second element of the tuple becomes one row of the output chain
            return (x_new, key), x_new

        # jax.lax.scan stacks the second elements from each step into an array
        # of shape (n_steps, *init.shape)
        (_, _), chain = jax.lax.scan(step, (init, key), None, length=n_steps)
        return chain
