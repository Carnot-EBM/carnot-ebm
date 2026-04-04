"""Hamiltonian Monte Carlo (HMC) sampler -- JAX implementation.

**Researcher summary:**
    HMC uses Hamiltonian dynamics (leapfrog integrator) with Metropolis
    accept/reject to produce exact samples from p(x) ~ exp(-E(x)). Superior
    mixing to ULA for high-dimensional, correlated distributions.

**Detailed explanation for engineers:**
    Hamiltonian Monte Carlo is a more sophisticated sampling algorithm than
    Langevin dynamics. It borrows ideas from classical mechanics to make
    *large, informed* moves through the state space while maintaining a high
    acceptance rate.

    **The physics analogy:**
    Imagine a ball rolling on a surface whose height is the energy function E(x).
    Instead of taking small noisy steps (Langevin), HMC:
    1. Gives the ball a random "kick" (samples random momentum p ~ N(0, I))
    2. Lets the ball roll for a while following Newton's laws (leapfrog integration)
    3. Checks if the ball's total energy (potential E(x) + kinetic 0.5*||p||^2)
       is conserved — if so, accepts the new position; if not, rejects it.

    **Why is HMC better than Langevin?**
    - Langevin takes small random steps, which is slow for correlated variables.
    - HMC follows the energy surface for many steps before proposing, so it can
      traverse long distances while staying in regions of reasonable probability.
    - The Metropolis accept/reject step removes discretization bias, so HMC
      samples are *exactly* from the target distribution (unlike ULA).

    **The leapfrog integrator:**
    A symplectic (energy-conserving) numerical integrator that alternates between
    half-steps in momentum and full steps in position. It is time-reversible and
    volume-preserving, which are required for the Metropolis correction to work.

    **Key parameters:**
    - ``step_size``: Size of each leapfrog step. Larger = faster but less accurate.
    - ``num_leapfrog_steps``: How many leapfrog steps per proposal. More steps =
      larger proposals but more computation per step.

Spec: REQ-SAMPLE-002
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import EnergyFunction


@dataclass
class HMCSampler:
    """Hamiltonian Monte Carlo sampler with leapfrog integration and optional gradient clipping.

    **Researcher summary:**
        Standard HMC with configurable step size and leapfrog steps.
        Uses ``jax.lax.fori_loop`` for the leapfrog and ``jax.lax.scan``
        for the outer chain. Optional gradient clipping prevents leapfrog
        divergence on steep energy landscapes.

    **Detailed explanation for engineers:**
        This sampler produces samples from p(x) ~ exp(-E(x)) using the
        Hamiltonian Monte Carlo algorithm. Each iteration:

        1. Sample fresh momentum p ~ N(0, I)
        2. Compute initial Hamiltonian: H_old = E(x) + 0.5 * ||p||^2
        3. Run leapfrog integration for num_leapfrog_steps steps
        4. Compute proposed Hamiltonian: H_new = E(x_new) + 0.5 * ||p_new||^2
        5. Accept x_new with probability min(1, exp(H_old - H_new))

        If the leapfrog integrator were perfect (infinite precision), the
        Hamiltonian would be exactly conserved and acceptance rate would be
        100%. In practice, discretization introduces some error, and the
        Metropolis step corrects for it.

        **Gradient clipping (optional):**
        When ``clip_norm`` is set, every gradient computed during leapfrog
        integration is rescaled so its L2 norm does not exceed ``clip_norm``.
        This prevents the leapfrog trajectory from "blowing up" on steep
        energy surfaces (e.g., the Rosenbrock function). Note that clipping
        breaks exact symplecticity, but this is acceptable since the
        unclipped integrator would diverge to NaN anyway.

    Attributes:
        step_size: Leapfrog step size (epsilon). Typical range: 0.01 to 0.5.
        num_leapfrog_steps: Number of leapfrog steps per HMC iteration (L).
            The "trajectory length" is step_size * num_leapfrog_steps.
        clip_norm: Maximum allowed L2 norm for gradients in leapfrog steps.
            If None (default), no clipping is applied. Typical values for
            steep landscapes: 1.0 to 100.0.

    For example::

        from carnot.models.ising import IsingModel, IsingConfig
        import jax.numpy as jnp

        model = IsingModel(IsingConfig(input_dim=10))
        sampler = HMCSampler(step_size=0.1, num_leapfrog_steps=10, clip_norm=10.0)

        x0 = jnp.zeros(10)
        x_final = sampler.sample(model, x0, n_steps=100)

    Spec: REQ-SAMPLE-002, REQ-SAMPLE-004
    """

    step_size: float = 0.1
    num_leapfrog_steps: int = 10
    clip_norm: float | None = None

    def _clip_gradient(self, grad: jax.Array) -> jax.Array:
        """Clip gradient to have at most ``clip_norm`` L2 norm.

        **Researcher summary:**
            Rescales grad so ||grad||_2 <= clip_norm, preserving direction.
            No-op when clip_norm is None.

        **Detailed explanation for engineers:**
            Gradient clipping prevents the leapfrog integrator from diverging
            on energy surfaces with very steep gradients. In HMC, large
            gradients cause the leapfrog trajectory to "blow up" — the
            position and momentum grow exponentially, leading to NaN values
            and a Metropolis acceptance probability of 0.

            By bounding the gradient norm, the leapfrog steps remain bounded,
            and the integrator can navigate steep regions (like the walls of
            the Rosenbrock banana valley) without diverging.

            Note: Gradient clipping breaks the exact symplecticity of the
            leapfrog integrator, which means the Metropolis correction is no
            longer exactly valid. In practice, this is acceptable because:
            (1) without clipping, the chain diverges entirely, and
            (2) clipping only activates in extreme gradient regions where the
            unclipped integrator would produce NaN anyway.

        Args:
            grad: The raw energy gradient, shape (input_dim,).

        Returns:
            The (possibly rescaled) gradient, same shape as input.

        Spec: REQ-SAMPLE-004
        """
        if self.clip_norm is None:
            return grad
        norm = jnp.linalg.norm(grad)
        return jnp.where(norm > self.clip_norm, grad * self.clip_norm / norm, grad)

    def _leapfrog(
        self,
        energy_fn: EnergyFunction,
        x: jax.Array,
        p: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Leapfrog integration for Hamiltonian dynamics.

        **Researcher summary:**
            Symplectic integrator: half-step p, full-step x, ..., half-step p.
            Time-reversible and volume-preserving (required for detailed balance).

        **Detailed explanation for engineers:**
            The leapfrog integrator is the standard ODE solver for HMC because
            it has two critical properties:

            1. **Time-reversible**: If you negate the momentum and run the same
               steps, you get back to where you started. This is required for
               the Metropolis acceptance to satisfy detailed balance.

            2. **Volume-preserving (symplectic)**: The Jacobian determinant of
               the leapfrog map is exactly 1, so we don't need to compute it
               in the acceptance probability.

            The algorithm:
            - Half-step momentum: p -= (eps/2) * grad_E(x)
            - For each of L steps:
                - Full-step position: x += eps * p
                - Full-step momentum (half at last step): p -= eps * grad_E(x)
            - The last momentum update is only a half-step

            ``jax.lax.fori_loop`` is JAX's compiled for-loop — it runs on the
            accelerator without Python overhead, unlike a regular ``for`` loop.

        Args:
            energy_fn: The energy function providing grad_energy.
            x: Current position, shape (input_dim,).
            p: Current momentum, shape (input_dim,).

        Returns:
            Tuple (x_new, p_new) after num_leapfrog_steps of integration.
        """
        # Initial half-step in momentum
        grad = energy_fn.grad_energy(x)
        grad = self._clip_gradient(grad)
        p = p - (self.step_size * 0.5) * grad

        def body(i: int, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            x, p = carry
            # Full position step: x moves in the direction of momentum
            x = x + self.step_size * p
            grad = energy_fn.grad_energy(x)
            # Clip gradient if clip_norm is set (prevents leapfrog divergence)
            grad = self._clip_gradient(grad)
            # Full momentum step for all iterations except the last,
            # which gets a half-step. We use jnp.where to handle this
            # without breaking JAX's tracing (no Python if/else on traced values).
            p = p - self.step_size * grad * jnp.where(i < self.num_leapfrog_steps - 1, 1.0, 0.5)
            return x, p

        # jax.lax.fori_loop: compiled loop from i=0 to i=num_leapfrog_steps-1
        # This runs entirely on GPU/TPU with no Python interpreter overhead.
        x_new, p_new = jax.lax.fori_loop(0, self.num_leapfrog_steps, body, (x, p))
        return x_new, p_new

    def sample(
        self,
        energy_fn: EnergyFunction,
        init: jax.Array,
        n_steps: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Run HMC for n_steps iterations and return the final sample.

        **Researcher summary:**
            Full HMC chain: leapfrog proposal + Metropolis accept/reject.
            Returns only the final state.

        **Detailed explanation for engineers:**
            Each HMC step:
            1. Samples random momentum p ~ N(0, I)
            2. Computes the "old" Hamiltonian H = E(x) + 0.5*||p||^2
               (potential energy + kinetic energy)
            3. Runs leapfrog integration to get (x_new, p_new)
            4. Computes the "new" Hamiltonian H_new = E(x_new) + 0.5*||p_new||^2
            5. Accepts x_new with probability min(1, exp(H_old - H_new))
               - If H_new < H_old: always accept (we found lower energy)
               - If H_new > H_old: accept with probability exp(-(H_new - H_old))

            The ``jnp.minimum(-h_new + h_old, 0.0)`` clamp ensures we don't
            compute exp of large positive numbers (which would overflow).

        Args:
            energy_fn: Any object satisfying the EnergyFunction protocol.
            init: Initial state, shape (input_dim,).
            n_steps: Number of HMC iterations (each includes a full leapfrog
                trajectory + accept/reject).
            key: JAX PRNG key. If None, uses seed 0.

        Returns:
            Final state after n_steps HMC iterations, shape (input_dim,).

        Spec: REQ-SAMPLE-002, SCENARIO-SAMPLE-002
        """
        if key is None:
            key = jrandom.PRNGKey(0)

        def step(carry: tuple[jax.Array, jax.Array], _: None) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            x, key = carry
            # Split into 3 keys: next iteration, momentum sampling, uniform for accept
            key, k1, k2 = jrandom.split(key, 3)
            # Sample random momentum from standard normal (the "kick")
            p = jrandom.normal(k1, x.shape)
            # Compute initial Hamiltonian (potential + kinetic energy)
            h_old = energy_fn.energy(x) + 0.5 * jnp.sum(p**2)
            # Run leapfrog integration to get proposal
            x_new, p_new = self._leapfrog(energy_fn, x, p)
            # Compute proposed Hamiltonian
            h_new = energy_fn.energy(x_new) + 0.5 * jnp.sum(p_new**2)
            # Metropolis acceptance probability: min(1, exp(H_old - H_new))
            # The clamp to 0 prevents exp overflow for very favorable proposals
            accept_prob = jnp.exp(jnp.minimum(-h_new + h_old, 0.0))
            # Sample uniform [0,1] to decide accept/reject
            u = jrandom.uniform(k2)
            # Accept if u < accept_prob, otherwise keep the old state
            # jnp.where is used instead of if/else for JAX traceability
            x_out = jnp.where(u < accept_prob, x_new, x)
            return (x_out, key), x_out

        # Run n_steps of HMC using jax.lax.scan (compiled loop)
        (x_final, _), _ = jax.lax.scan(step, (init, key), None, length=n_steps)
        return x_final
