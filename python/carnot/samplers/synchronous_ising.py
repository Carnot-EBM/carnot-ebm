"""Synchronous p-bit Ising sampler — Python simulation of the KV260 RTL (Exp 624).

**Researcher summary:**
    Python-level simulation of the synchronous checkerboard p-bit Ising sampler
    described in arXiv 2604.01564 and implemented in hardware/kv260/ising_sampler_v2.v.
    This validates the RTL update logic before FPGA synthesis: if the Python
    simulation converges to the correct Ising distribution, the RTL should too.

**Detailed explanation for engineers:**
    The RTL (ising_sampler_v2.v) updates all spins in the current checkerboard
    phase simultaneously at each posedge of the clock.  Two clock cycles = one
    full Gibbs sweep (phase 0: even spins, phase 1: odd spins).

    This Python class mirrors that behaviour exactly:

        step() — one full sweep:
          Phase 0: compute h_eff for all even spins using current state,
                   sample all even spins in one shot.
          Phase 1: compute h_eff for all odd spins using the updated even
                   values, sample all odd spins in one shot.

    Why this differs from ParallelIsingSampler._checkerboard_update():
        ParallelIsingSampler works with JAX arrays and {0,1} boolean spins.
        SynchronousIsingSampler works with NumPy and {-1, +1} spin convention,
        matching the RTL which stores spins as 1-bit values where 1 = +1 spin
        and 0 = -1 spin.  The energy formula here is:
            E = -sum_ij J[i,j] * s_i * s_j - sum_i h[i] * s_i
        with s_i in {-1, +1}.

    The sigmoid is computed as:
        P(s_i = +1 | s_{-i}) = sigmoid(2 * beta * h_eff_i)
    where h_eff_i = sum_j J[i,j] * s_j + h[i].

    This matches the RTL's flip_prob logic (Q8.8 sigmoid LUT, beta-scaled h_eff).

Spec: REQ-SAMPLE-037, SCENARIO-SAMPLE-061, SCENARIO-SAMPLE-062
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class SynchronousIsingSampler:
    """Python simulation of the synchronous checkerboard p-bit Ising sampler.

    **Researcher summary:**
        Matches the update logic of ising_sampler_v2.v: two-phase checkerboard
        sweep where all spins in each phase update simultaneously.  Spins are
        {-1, +1}.  Energy: E = -sum_ij J[i,j] s_i s_j - sum_i h[i] s_i.

    **Detailed explanation for engineers:**
        The RTL performs a checkerboard (even/odd spin index) two-phase update
        rather than a fully synchronous all-at-once update.  This is important:
        fully synchronous updates (all spins at once) can oscillate on
        antiferromagnetic couplings, whereas the two-phase checkerboard breaks
        that symmetry and converges more reliably.

        This class reproduces that exact two-phase logic in NumPy so we can run
        small Ising instances on CPU and verify that (a) the energy decreases
        under annealing and (b) the marginal spin distributions match
        ParallelIsingSampler's checkerboard mode on the same instance.

    Parameters
    ----------
    n_spins : int
        Number of spin variables.
    couplings : np.ndarray, shape (n_spins, n_spins)
        Symmetric coupling matrix J with zero diagonal.  J[i,j] > 0 is
        ferromagnetic (prefers s_i = s_j); J[i,j] < 0 is antiferromagnetic.
    biases : np.ndarray, shape (n_spins,)
        External field h[i] per spin.  Positive h[i] biases s_i toward +1.
    beta : float
        Inverse temperature.  Higher beta = more deterministic (colder).

    Spec: REQ-SAMPLE-037
    """

    def __init__(
        self,
        n_spins: int,
        couplings: np.ndarray,
        biases: np.ndarray,
        beta: float = 1.0,
    ) -> None:
        if couplings.shape != (n_spins, n_spins):
            raise ValueError(
                f"couplings must be ({n_spins}, {n_spins}), got {couplings.shape}"
            )
        if biases.shape != (n_spins,):
            raise ValueError(
                f"biases must be ({n_spins},), got {biases.shape}"
            )
        self.n_spins = n_spins
        self.couplings = np.asarray(couplings, dtype=np.float64)
        self.biases = np.asarray(biases, dtype=np.float64)
        self.beta = float(beta)

        # Even/odd index masks — same split as the RTL phase register.
        self._even_mask = np.arange(n_spins) % 2 == 0
        self._odd_mask = ~self._even_mask

    def energy(self, state: np.ndarray) -> float:
        """Compute Ising energy E = -s^T J s - h^T s for a spin state.

        **Why this function:**
            Energy is the invariant we use to check convergence.  Lower energy
            means a better solution to the Ising optimisation problem.  We use
            it in compare_with_async() to confirm the sync sampler is reaching
            comparable quality to the async (ParallelIsingSampler) baseline.

        Args:
            state: Spin state array of shape (n_spins,) with values in {-1, +1}.

        Returns:
            Scalar float energy value.
        """
        s = np.asarray(state, dtype=np.float64)
        return float(-s @ self.couplings @ s - self.biases @ s)

    def step(self, state: np.ndarray) -> np.ndarray:
        """One synchronous checkerboard sweep: update even spins then odd spins.

        **Detailed explanation for engineers:**
            Mirrors the RTL FSM_RUNNING phase logic exactly:
              Phase 0 (phase == 0): compute h_eff for all even-indexed spins
                using the *current* spin values, then flip all even spins at once.
              Phase 1 (phase == 1): recompute h_eff for all odd-indexed spins
                using the *freshly updated* even spin values, flip all odd spins.

            The RTL does this in two clock cycles.  Here it happens in two NumPy
            matrix-vector multiplications (one per phase).

            NOT self-consistent within the phase: even spins don't see each
            other's updated values during phase 0, only the odd spins' current
            values.  This is the 'synchronous' part — contrasts with sequential
            Gibbs where each spin sees all previously updated neighbours.

        Args:
            state: Current spin state, shape (n_spins,), values in {-1, +1}.

        Returns:
            New spin state after one full two-phase sweep, shape (n_spins,).

        Spec: REQ-SAMPLE-037
        """
        s = np.array(state, dtype=np.float64)

        # Phase 0: update even spins.
        h_eff = self.couplings @ s + self.biases
        p_up = _sigmoid(2.0 * self.beta * h_eff)
        new_vals = np.where(np.random.rand(self.n_spins) < p_up, 1.0, -1.0)
        s = np.where(self._even_mask, new_vals, s)

        # Phase 1: update odd spins using the freshly updated even values.
        h_eff = self.couplings @ s + self.biases
        p_up = _sigmoid(2.0 * self.beta * h_eff)
        new_vals = np.where(np.random.rand(self.n_spins) < p_up, 1.0, -1.0)
        s = np.where(self._odd_mask, new_vals, s)

        return s

    def sample(
        self,
        n_steps: int,
        init_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run synchronous Gibbs for n_steps and return the final spin state.

        **Detailed explanation for engineers:**
            Each call to step() is one full two-phase sweep (= one Gibbs epoch).
            The final state is returned; no samples are collected during the run.
            Use compare_with_async() if you need energy statistics over multiple
            independent runs.

        Args:
            n_steps: Number of full sweeps to run.
            init_state: Initial spin state, shape (n_spins,), values {-1, +1}.
                        If None, initialises to all +1 (matching RTL reset).

        Returns:
            Final spin state array of shape (n_spins,), values {-1, +1}.

        Spec: REQ-SAMPLE-037
        """
        if init_state is None:
            # RTL reset: state_ram initialised to 0xFFFF…, i.e. all +1.
            state = np.ones(self.n_spins, dtype=np.float64)
        else:
            state = np.array(init_state, dtype=np.float64)

        for _ in range(n_steps):
            state = self.step(state)

        return state

    def compare_with_async(self, n_steps: int, n_trials: int) -> dict:
        """Compare mean final energy of sync vs async (ParallelIsingSampler) sampling.

        **Detailed explanation for engineers:**
            Runs n_trials independent chains for both samplers.  Each chain
            starts from all-+1 (RTL reset state) and runs n_steps sweeps.
            Reports the mean final energy of each set of chains and the gap
            between them.

            'Async' here means ParallelIsingSampler with use_checkerboard=True
            at the same beta, converted to the {-1,+1} spin convention.

            sync_converged = True when |energy_gap| / (|sync_mean| + 1e-6) < 0.05
            (within 5% relative difference), which is a conservative acceptance
            criterion for RTL pre-validation.

        Args:
            n_steps: Sweeps per trial chain.
            n_trials: Number of independent chains to average.

        Returns:
            Dict with keys:
                sync_mean_energy  : float — mean final energy, synchronous sampler.
                async_mean_energy : float — mean final energy, ParallelIsingSampler.
                energy_gap        : float — sync - async (near zero means agreement).
                sync_converged    : bool  — True if gap is within 5% of sync mean.

        Spec: REQ-SAMPLE-037
        """
        import jax
        import jax.random as jrandom

        from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

        # Run synchronous chains.
        sync_energies = []
        for _ in range(n_trials):
            final = self.sample(n_steps)
            sync_energies.append(self.energy(final))
        sync_mean = float(np.mean(sync_energies))

        # Run async (JAX checkerboard) chains.
        # Convert {-1,+1} couplings/biases to {0,1} JAX format used by
        # ParallelIsingSampler (which works in boolean / float {0,1} internally).
        # The energy formulas differ by a constant offset, so we convert back.
        async_sampler = ParallelIsingSampler(
            n_warmup=0,
            n_samples=1,
            steps_per_sample=n_steps,
            schedule=AnnealingSchedule(
                beta_init=self.beta, beta_final=self.beta
            ),
            use_checkerboard=True,
        )

        # Build JAX-compatible arrays (float32, {0,1} convention).
        import jax.numpy as jnp

        J_jax = jnp.array(self.couplings * 0.25, dtype=jnp.float32)
        # In {0,1} convention: E = -sum_ij J*(2si-1)(2sj-1) - sum_i h*(2si-1)
        # bias correction: b_jax[i] = h[i]/2 + sum_j J[i,j]/2
        b_jax = jnp.array(
            self.biases / 2.0 + self.couplings.sum(axis=1) / 2.0,
            dtype=jnp.float32,
        )

        async_energies = []
        for trial in range(n_trials):
            key = jrandom.PRNGKey(trial + 1000)
            samples = async_sampler.sample(key, b_jax, J_jax, beta=float(self.beta) * 4.0)
            # samples shape: (1, n_spins), boolean {False=0, True=1}
            s_pm = samples[0].astype(jnp.float32) * 2.0 - 1.0
            s_np = np.array(s_pm)
            async_energies.append(self.energy(s_np))
        async_mean = float(np.mean(async_energies))

        energy_gap = sync_mean - async_mean
        sync_converged = abs(energy_gap) / (abs(sync_mean) + 1e-6) < 0.05

        return {
            "sync_mean_energy": sync_mean,
            "async_mean_energy": async_mean,
            "energy_gap": energy_gap,
            "sync_converged": sync_converged,
        }
