"""Parallel Dense Ising sampler with inertia EMA dynamics.

**Researcher summary:**
    Implements the inertia dynamics from arXiv 2604.17109 for dense Ising
    machines. The key modification: instead of recomputing local fields h_i
    from scratch each step, we smooth them with an exponential moving average
    (EMA). This prevents oscillation in dense coupling graphs (where every
    spin is coupled to every other spin) and accelerates convergence by
    20-35x compared to standard synchronous Gibbs on FPGA hardware.

**Why inertia helps on dense graphs:**
    In standard synchronous Gibbs, all spins are updated simultaneously using
    the local field h_i = sum_j J_ij * s_j computed from the *current* state.
    On dense graphs, large coupling strengths cause a spin to flip, which
    immediately changes the local field for every other spin. This creates
    oscillation -- spins bounce back and forth because each flip triggers
    equal-and-opposite corrections from its many neighbors.

    The EMA fix:
        h_i(t+1) = alpha * h_i(t) + (1 - alpha) * sum_j(J_ij * s_j(t))

    By blending the new local field with a memory of past fields (controlled
    by alpha in [0,1]), we dampen these oscillations. Think of it as adding
    momentum (inertia) to the local field: the field resists sudden changes.

    alpha=0: pure current state (standard synchronous Gibbs, no inertia)
    alpha=0.3: 30% memory from past, 70% current (paper's recommended default)
    alpha=0.7: strong inertia, slower to react but more stable

**Hardware motivation:**
    The KV260 FPGA v2 RTL (ising_sampler_v2.v) implements checkerboard
    synchronous Gibbs. For v3, we add a register per spin to hold h_i and
    compute the EMA update before the flip probability. This adds ~1 extra
    multiply-accumulate per spin per cycle but eliminates hundreds of wasted
    cycles spent oscillating. Net: same ~50% area as v2 but 20-35x fewer
    cycles to converge.

Spec: REQ-SAMPLE-023, REQ-SAMPLE-024
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp


@dataclass
class ParallelDenseIsingConfig:
    """Configuration for the inertia Ising sampler.

    **Detailed explanation for engineers:**
        Bundles all hyperparameters in one place so they can be serialised
        and compared across runs. The defaults match the paper's recommended
        settings for 100-spin dense graphs.

    Attributes:
        n_spins: Number of spin variables. The coupling matrix J will be
            n_spins x n_spins.
        alpha: Inertia (EMA smoothing) coefficient. Range [0, 1).
            0.0 = no inertia (standard synchronous Gibbs).
            0.3 = paper default, good for dense arithmetic constraint graphs.
            Higher values stabilise more but slow adaptation.
        beta: Inverse temperature (sharpness of the Boltzmann distribution).
            Higher beta concentrates probability on lower-energy states.
            beta=1.0 is a reasonable starting point; increase for harder problems.
        n_steps: Number of Gibbs sweeps to run. 200 is enough to judge
            convergence for graphs up to 500 spins at beta=1.0.

    Spec: REQ-SAMPLE-023
    """

    n_spins: int = 100
    alpha: float = 0.3
    beta: float = 1.0
    n_steps: int = 200


class ParallelDenseIsingInertia:
    """Parallel Ising sampler with exponential moving average (EMA) local fields.

    **Researcher summary:**
        Implements arXiv 2604.17109 inertia dynamics. Each step, the local field
        h_i is updated via EMA before computing flip probabilities. This damps
        oscillation in dense graphs and accelerates convergence.

    **Detailed explanation for engineers:**
        The standard Ising energy is:
            E(s) = -0.5 * s^T J s - b^T s
        where s ∈ {-1, +1}^n, J is the coupling matrix, b is the bias vector.

        Flip probability for spin i:
            P(s_i = +1) = sigmoid(2 * beta * (h_i + b_i))

        Without inertia, h_i = (J @ s)[i] every step. With inertia:
            h_i(t+1) = alpha * h_i(t) + (1 - alpha) * (J @ s(t))[i]

        The `sample()` method runs n_steps sweeps and tracks energy each step
        so callers can measure convergence rate.

    Attributes:
        config: Hyperparameter bundle (n_spins, alpha, beta, n_steps).
        h_i: EMA-smoothed local fields, shape (n_spins,). Initialised to zero;
            updated in-place during sampling to accumulate momentum across
            calls if desired (though each `sample()` call resets h_i to the
            initial field computed from the starting spin state).

    Spec: REQ-SAMPLE-023, REQ-SAMPLE-024
    """

    def __init__(self, config: ParallelDenseIsingConfig) -> None:
        self.config = config
        # EMA local fields; shape (n_spins,). Reset at start of each sample() call.
        self.h_i: jax.Array = jnp.zeros(config.n_spins)

    def _compute_local_fields(
        self, J: jax.Array, s: jax.Array
    ) -> jax.Array:
        """Compute instantaneous local fields from coupling matrix and spin state.

        **Detailed explanation:**
            h_i = sum_j J_ij * s_j = (J @ s)[i]
            This is the "raw" field before EMA smoothing. For s ∈ {-1, +1}^n,
            J_ij > 0 means spins i and j prefer to be aligned.

        Args:
            J: Coupling matrix, shape (n_spins, n_spins). Should be symmetric
               with zero diagonal.
            s: Spin state vector, shape (n_spins,). Values ∈ {-1.0, +1.0}.

        Returns:
            Local fields vector, shape (n_spins,).

        Spec: REQ-SAMPLE-023
        """
        return J @ s

    def _update_inertia(
        self, h_i: jax.Array, local_fields: jax.Array
    ) -> jax.Array:
        """Apply EMA smoothing to dampen oscillation in dense coupling graphs.

        **Detailed explanation:**
            h_i(t+1) = alpha * h_i(t) + (1 - alpha) * local_fields(t)

            alpha=0: ignores history, reduces to standard synchronous Gibbs.
            alpha=0.3: keeps 30% memory, blends with 70% current field.

            Why this works: on dense graphs, a single spin flip changes the
            local field for ALL other spins. Without smoothing, this triggers
            a cascade of counter-flips (oscillation). The EMA acts like a
            low-pass filter, only passing through persistent field changes
            while filtering transient oscillations.

        Args:
            h_i: Current EMA field, shape (n_spins,).
            local_fields: Instantaneous field J @ s, shape (n_spins,).

        Returns:
            Updated EMA field, shape (n_spins,).

        Spec: REQ-SAMPLE-023
        """
        alpha = self.config.alpha
        return alpha * h_i + (1.0 - alpha) * local_fields

    def _flip_probabilities(
        self, h_i: jax.Array, biases: jax.Array
    ) -> jax.Array:
        """Compute probability of each spin being +1 under the Boltzmann distribution.

        **Detailed explanation:**
            P(s_i = +1) = sigmoid(2 * beta * (h_i + b_i))

            The factor of 2 comes from the ±1 spin convention. For s ∈ {-1, +1},
            the conditional log-ratio of P(s_i=+1) / P(s_i=-1) is exactly
            2 * beta * (h_i + b_i), so sigmoid gives the normalised probability.

            Using EMA h_i instead of raw J @ s means this probability is
            influenced by past spin states (through the EMA), which is what
            damps oscillation.

        Args:
            h_i: EMA-smoothed local fields, shape (n_spins,).
            biases: Per-spin external bias, shape (n_spins,).

        Returns:
            Flip probabilities in (0, 1), shape (n_spins,).

        Spec: REQ-SAMPLE-023
        """
        return jax.nn.sigmoid(2.0 * self.config.beta * (h_i + biases))

    def sample(
        self,
        J: jax.Array,
        biases: jax.Array,
        rng_key: jax.Array,
        init_state: jax.Array | None = None,
    ) -> dict:
        """Run inertia Ising dynamics and return final state with convergence stats.

        **Detailed explanation for engineers:**
            Each of the n_steps sweeps:
            1. Compute instantaneous local fields: lf = J @ s
            2. EMA update: h_i = alpha * h_i + (1 - alpha) * lf
            3. Flip probs: p = sigmoid(2 * beta * (h_i + b))
            4. Sample new spins: s_i ~ Bernoulli(p_i), mapped to {-1, +1}
            5. Record energy: E = -0.5 * s^T J s - b^T s

            The energy history lets callers check convergence: the sampler has
            converged when |E(t) - E(t-1)| / |E(t-1)| < 0.001 for several
            consecutive steps.

        Args:
            J: Coupling matrix, shape (n_spins, n_spins). Symmetric, zero diagonal.
               Should be normalised (e.g. divide by n_spins) for dense random graphs
               to prevent energy explosion.
            biases: Bias vector, shape (n_spins,). Zero means unbiased spins.
            rng_key: JAX PRNG key for stochastic spin sampling.
            init_state: Initial spin configuration ∈ {-1.0, +1.0}^n, shape (n_spins,).
                If None, initialises all spins to +1 (aligned ferromagnetic state).

        Returns:
            Dict with keys:
                'final_state': jnp.ndarray, shape (n_spins,), values ∈ {-1.0, +1.0}
                'final_energy': float, energy of the final spin configuration
                'energy_history': list[float], energy at each of the n_steps sweeps
                'n_steps': int, total steps run (equals config.n_steps)

        Spec: REQ-SAMPLE-023, REQ-SAMPLE-024
        """
        n = self.config.n_spins

        # Initialise spins: all +1 if no init_state provided.
        # All-up is a common starting point for ferromagnetic problems; for
        # antiferromagnetic or frustrated graphs, random init may be better.
        if init_state is not None:
            s = jnp.asarray(init_state, dtype=jnp.float32)
        else:
            s = jnp.ones(n, dtype=jnp.float32)

        # Bootstrap h_i from the initial spin state so the EMA starts from
        # a meaningful value rather than zero (which would cause a large jump
        # on the first step).
        h_i = self._compute_local_fields(J, s)
        self.h_i = h_i

        energy_history: list[float] = []

        for _step in range(self.config.n_steps):
            # Step 1: instantaneous field from current spins.
            local_fields = self._compute_local_fields(J, s)

            # Step 2: EMA smoothing — damps oscillation in dense graphs.
            h_i = self._update_inertia(h_i, local_fields)

            # Step 3: compute flip probabilities under current (smoothed) field.
            flip_probs = self._flip_probabilities(h_i, biases)

            # Step 4: sample new spins. Bernoulli with p -> map {0,1} to {-1,+1}.
            rng_key, subkey = jax.random.split(rng_key)
            s = jnp.where(
                jax.random.uniform(subkey, (n,)) < flip_probs,
                1.0,
                -1.0,
            )

            # Step 5: record energy for convergence tracking.
            # E = -0.5 * s^T J s - b^T s  (Ising convention with ±1 spins)
            energy = float(-0.5 * s @ J @ s - biases @ s)
            energy_history.append(energy)

        # Store final EMA field for potential inspection.
        self.h_i = h_i

        return {
            "final_state": s,
            "final_energy": energy_history[-1] if energy_history else float("nan"),
            "energy_history": energy_history,
            "n_steps": self.config.n_steps,
        }
