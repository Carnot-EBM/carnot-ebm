"""Langevin Stochastic Boltzmann (LSB) sampler -- JAX implementation.

**Researcher summary:**
    Implements the LSB sampler from arXiv 2512.02323. Updates all spins
    simultaneously using Langevin dynamics rather than sequential Gibbs sampling.
    Naturally parallel on CPU (vectorized JAX) and FPGA (parallel MAC units).
    Includes Conditional Expectation Matching (CEM) for automatic beta estimation.

**Detailed explanation for engineers:**
    The standard parallel Gibbs sampler (``ParallelIsingSampler``) uses a
    probabilistic flip rule derived from the conditional distribution:
        P(s_i = 1 | s_{-i}) = sigmoid(2 * beta * h_i)
    where h_i = b_i + sum_j J_ij * s_j is the local magnetic field.

    LSB takes a fundamentally different approach: it treats the discrete spin
    problem as a continuous-space Langevin SDE. Spins are maintained as real
    values in R, and the update rule is:
        s_i(t+1) = s_i(t) + lr * h_i(t) + sqrt(2 * lr / beta) * noise_i

    where:
    - lr is the Langevin step size (learning rate)
    - h_i(t) = b_i + sum_j J_ij * s_j(t) is the local field at time t
    - noise_i ~ N(0, 1) is independent Gaussian noise per spin
    - beta is the inverse temperature controlling noise scale

    The Langevin update moves spins toward lower energy (the h_i term drives
    spins toward configurations that reduce E = -b^T s - s^T J s) while the
    noise term prevents collapse to a single mode. At thermal equilibrium,
    the continuous-valued spins are discretized by thresholding at 0.5.

    **Why this is better than Gibbs for non-frustrated graphs:**
    Parallel Gibbs samples all spins from their conditional distributions
    simultaneously, which ignores spin-spin correlations within the same
    update step. LSB's continuous update preserves correlations through the
    shared local field h_i and the continuous state trajectory. On
    non-frustrated constraint graphs (where the energy landscape has a clear
    global minimum), LSB converges faster because the gradient information
    in h_i is used directly rather than just to compute flip probabilities.

    **Conditional Expectation Matching (CEM):**
    CEM is a technique for automatically estimating the right inverse
    temperature beta. The idea: under the Boltzmann distribution, the
    expected spin value E[s_i] = sigmoid(2 * beta * h_i). CEM adjusts beta
    so that the average activation level of spins matches a target value.
    We use a simple heuristic: run a short warmup, compute the mean spin
    activation, and rescale beta to hit a target activation level of 0.7
    (corresponding to spins that are biased but not completely saturated).

    **No checkerboard needed:**
    Because the Langevin update uses a continuous gradient rather than
    conditional sampling, there is no need for the checkerboard decomposition
    that Gibbs requires to avoid instability from simultaneous discrete flips.

    **Decentralization note:**
    This sampler runs on CPU/GPU/TPU via JAX. No closed-source dependencies.
    Portable to FPGA (parallel MAC units compute h_i, parallel multipliers
    scale the noise).

Spec: REQ-SAMPLE-003, REQ-SAMPLE-LSB-001
Reference: arXiv 2512.02323 -- Langevin Stochastic Boltzmann sampling
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


@dataclass
class LangevinSBSampler:
    """Langevin Stochastic Boltzmann (LSB) sampler for Ising models.

    **Researcher summary:**
        All spins update simultaneously via Langevin dynamics.
        Update: s += lr * h + sqrt(2*lr/beta) * noise, where h = b + J @ s.
        No checkerboard decomposition needed.
        CEM auto-estimates good beta from warmup statistics.

    **Detailed explanation for engineers:**
        This class implements the ``SamplerBackend`` protocol (structural
        interface from ``carnot.samplers.backend``). It wraps the low-level
        LSB step function in warmup + sample collection phases.

        The continuous spin state s ∈ R^N is initialized randomly near 0.5
        (the "paramagnetic" fixed point). After warmup, spins are collected
        and binarized by thresholding: s_binary = (s > 0.5).

        Attributes:
            lr: Langevin step size (learning rate). Larger lr = faster
                exploration but more discretization error. Default 0.05 works
                well for most constraint problems.
            beta: Inverse temperature. Higher beta = lower noise = more
                exploitation. When use_cem=True, this is the initial beta
                that CEM starts from.
            n_warmup: Number of Langevin steps before collecting samples.
            n_samples: Number of samples to collect.
            steps_per_sample: Langevin steps between collected samples
                (controls decorrelation).
            use_cem: If True, run CEM after warmup to rescale beta.
            seed: Random seed for the internal JAX PRNG key.

    Spec: REQ-SAMPLE-003, REQ-SAMPLE-LSB-001
    """

    lr: float = 0.05
    beta: float = 10.0
    n_warmup: int = 500
    n_samples: int = 50
    steps_per_sample: int = 10
    use_cem: bool = True
    seed: int = 42
    _key: jax.Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._key = jrandom.PRNGKey(self.seed)

    @property
    def backend_name(self) -> str:
        """Human-readable backend name for logging and config."""
        return "lsb"

    def _next_key(self) -> jax.Array:
        """Split and advance the internal PRNG key (stateful RNG management)."""
        self._key, subkey = jrandom.split(self._key)
        return cast("jax.Array", subkey)

    def _cem_beta(
        self,
        spins_continuous: jax.Array,
        biases: jax.Array,
        J: jax.Array,
        beta_init: float,
    ) -> float:
        """Conditional Expectation Matching: estimate beta from warmup spin stats.

        **Detailed explanation for engineers:**
            CEM observes the mean spin activation after warmup and adjusts beta
            so that the expected activation under the Boltzmann distribution
            sigmoid(2 * beta * mean_field) matches a target of 0.7 (spins are
            biased toward their preferred state but not fully saturated).

            The local field for each spin is h_i = b_i + sum_j J_ij * s_j.
            The Boltzmann activation at beta is sigmoid(2 * beta * h_i).
            We solve for beta by averaging the mean-field magnitudes:
                target_activation = sigmoid(2 * beta * mean(|h_i|))
                beta = logit(target_activation) / (2 * mean(|h_i|))
            where logit(x) = log(x / (1-x)).

            If mean_field_magnitude is near zero (uniform landscape), CEM
            falls back to the initial beta to avoid division by zero.

        Args:
            spins_continuous: Continuous spin state after warmup, shape (n,).
            biases: Bias vector, shape (n,).
            J: Coupling matrix, shape (n, n).
            beta_init: Fallback beta if field magnitudes are near zero.

        Returns:
            Estimated beta (scalar float).
        """
        h = biases + J @ spins_continuous
        mean_h_magnitude = float(jnp.mean(jnp.abs(h)))

        if mean_h_magnitude < 1e-6:
            # Flat landscape — no information to estimate beta from.
            return beta_init

        # Target activation: spins should be 70% confident in their preferred
        # direction. logit(0.7) = log(0.7/0.3) ≈ 0.847.
        target_activation = 0.7
        logit_target = float(jnp.log(target_activation / (1.0 - target_activation)))
        # Solve: sigmoid(2 * beta * mean_h_magnitude) = target_activation
        # => 2 * beta * mean_h_magnitude = logit_target
        beta_cem = logit_target / (2.0 * mean_h_magnitude)

        # Clamp to a reasonable range to prevent numerical instability.
        return float(jnp.clip(beta_cem, 0.1, 100.0))

    def run_sampler(
        self,
        key: jax.Array,
        biases: jax.Array,
        J: jax.Array,
        beta: float,
    ) -> jax.Array:
        """Core LSB sampling loop: warmup + sample collection.

        **Detailed explanation for engineers:**
            Phase 1 (warmup): Run n_warmup Langevin steps from a random
            continuous initialization near 0.5. This lets the chain mix and
            find low-energy regions of the landscape.

            Phase 2 (optional CEM): If use_cem=True, estimate a better beta
            from the warmup endpoint spin statistics.

            Phase 3 (collection): Run n_samples * steps_per_sample more
            Langevin steps, collecting one binarized sample every
            steps_per_sample steps.

        Args:
            key: JAX PRNG key.
            biases: Bias vector, shape (n_spins,).
            J: Coupling matrix, shape (n_spins, n_spins).
            beta: Inverse temperature.

        Returns:
            Boolean array of shape (n_samples, n_spins).
        """
        n_spins = biases.shape[0]
        lr = jnp.float32(self.lr)

        # Initialize spins as continuous values near 0.5 (paramagnetic state).
        key, init_key = jrandom.split(key)
        spins = jrandom.uniform(init_key, (n_spins,), minval=0.3, maxval=0.7)

        # --- Phase 1: Warmup ---
        # jax.lax.scan compiles the loop to XLA — no Python overhead per step.
        def langevin_step(s: jax.Array, step_key: jax.Array) -> tuple[jax.Array, None]:
            """Single LSB step: s += lr * h + sqrt(2*lr/beta) * noise."""
            h = biases + J @ s
            noise = jrandom.normal(step_key, (n_spins,))
            # Langevin update: gradient term + noise term.
            # The factor sqrt(2*lr/beta) is the correct noise amplitude for
            # the Langevin SDE at inverse temperature beta.
            noise_scale = jnp.sqrt(2.0 * lr / jnp.float32(beta))
            s_new = s + lr * h + noise_scale * noise
            # Soft clamp to [0, 1] to keep continuous spins near the binary
            # region and prevent runaway drift.
            s_new = jnp.clip(s_new, 0.0, 1.0)
            return s_new, None

        key, warmup_key = jrandom.split(key)
        warmup_keys = jrandom.split(warmup_key, self.n_warmup)
        spins, _ = jax.lax.scan(langevin_step, spins, warmup_keys)

        # --- Phase 2: CEM beta estimation ---
        effective_beta = beta
        if self.use_cem:
            effective_beta = self._cem_beta(spins, biases, J, beta)

        # --- Phase 3: Sample collection ---
        def collect_one_sample(s: jax.Array, sample_key: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Run steps_per_sample Langevin steps, return binarized sample."""
            step_keys = jrandom.split(sample_key, self.steps_per_sample)

            def inner_step(s_inner: jax.Array, k: jax.Array) -> tuple[jax.Array, None]:
                h = biases + J @ s_inner
                noise = jrandom.normal(k, (n_spins,))
                noise_scale = jnp.sqrt(2.0 * lr / jnp.float32(effective_beta))
                s_out = s_inner + lr * h + noise_scale * noise
                return jnp.clip(s_out, 0.0, 1.0), None

            if self.steps_per_sample > 0:
                s, _ = jax.lax.scan(inner_step, s, step_keys)

            # Binarize by thresholding: continuous spin > 0.5 → True (spin=1).
            return s, s > 0.5

        key, collect_key = jrandom.split(key)
        collect_keys = jrandom.split(collect_key, self.n_samples)
        _, samples = jax.lax.scan(collect_one_sample, spins, collect_keys)

        return samples  # shape (n_samples, n_spins), bool

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Run LSB sampling to find low-energy spin configurations.

        **Detailed explanation for engineers:**
            Runs the LSB sampler with n_steps warmup iterations and collects
            n_samples binarized configurations. The returned samples are biased
            toward low-energy states by the Langevin gradient term.

        Args:
            biases: Bias vector, shape (n_spins,).
            couplings: Symmetric coupling matrix, shape (n_spins, n_spins).
            n_samples: Number of samples to return.
            n_steps: Number of Langevin warmup steps.
            beta: Inverse temperature (final effective beta after CEM).

        Returns:
            Boolean array of shape (n_samples, n_spins).

        Spec: REQ-SAMPLE-003
        """
        b = jnp.asarray(biases, dtype=jnp.float32)
        J = jnp.asarray(couplings, dtype=jnp.float32)

        # Temporarily override n_warmup and n_samples for this call.
        orig_warmup = self.n_warmup
        orig_samples = self.n_samples
        object.__setattr__(self, "n_warmup", n_steps)
        object.__setattr__(self, "n_samples", n_samples)

        try:
            samples = self.run_sampler(self._next_key(), b, J, beta)
        finally:
            object.__setattr__(self, "n_warmup", orig_warmup)
            object.__setattr__(self, "n_samples", orig_samples)

        return np.asarray(samples)

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Draw samples at fixed temperature via LSB dynamics.

        **Detailed explanation for engineers:**
            Reads beta, n_warmup, and steps_per_sample from config dict,
            then runs the LSB sampler. Unlike minimize_energy, CEM is optional
            and controlled by config["use_cem"] (defaults to self.use_cem).

        Args:
            biases: Bias vector, shape (n_spins,).
            couplings: Symmetric coupling matrix, shape (n_spins, n_spins).
            n_samples: Number of samples to draw.
            config: Backend-specific configuration. Reads:
                "beta" (float, default self.beta),
                "n_warmup" (int, default self.n_warmup),
                "steps_per_sample" (int, default self.steps_per_sample),
                "use_cem" (bool, default self.use_cem).

        Returns:
            Boolean array of shape (n_samples, n_spins).

        Spec: REQ-SAMPLE-003
        """
        beta = float(config.get("beta", self.beta))
        n_warmup = int(config.get("n_warmup", self.n_warmup))
        steps_per_sample = int(config.get("steps_per_sample", self.steps_per_sample))
        use_cem = bool(config.get("use_cem", self.use_cem))

        orig_warmup = self.n_warmup
        orig_samples = self.n_samples
        orig_sps = self.steps_per_sample
        orig_cem = self.use_cem

        object.__setattr__(self, "n_warmup", n_warmup)
        object.__setattr__(self, "n_samples", n_samples)
        object.__setattr__(self, "steps_per_sample", steps_per_sample)
        object.__setattr__(self, "use_cem", use_cem)

        b = jnp.asarray(biases, dtype=jnp.float32)
        J = jnp.asarray(couplings, dtype=jnp.float32)

        try:
            samples = self.run_sampler(self._next_key(), b, J, beta)
        finally:
            object.__setattr__(self, "n_warmup", orig_warmup)
            object.__setattr__(self, "n_samples", orig_samples)
            object.__setattr__(self, "steps_per_sample", orig_sps)
            object.__setattr__(self, "use_cem", orig_cem)

        return np.asarray(samples)


def lsb_sample(
    key: jax.Array,
    biases: jax.Array,
    J: jax.Array,
    beta: float,
    lr: float,
    n_steps: int,
    n_samples: int,
    steps_per_sample: int = 10,
) -> jax.Array:
    """Functional interface for LSB sampling (no class state).

    **Detailed explanation for engineers:**
        Thin wrapper around the LSB update loop for use in experiments
        that prefer a functional (stateless) API over a class instance.
        Useful for benchmarking and JAX JIT compilation at the call site.

    Args:
        key: JAX PRNG key.
        biases: Bias vector, shape (n_spins,).
        J: Coupling matrix, shape (n_spins, n_spins).
        beta: Inverse temperature.
        lr: Langevin step size.
        n_steps: Number of warmup Langevin steps.
        n_samples: Number of samples to collect.
        steps_per_sample: Langevin steps between samples.

    Returns:
        Boolean array of shape (n_samples, n_spins).
    """
    sampler = LangevinSBSampler(
        lr=lr,
        beta=beta,
        n_warmup=n_steps,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample,
        use_cem=False,  # caller controls beta directly
        seed=int(jrandom.randint(key, (), 0, 2**31 - 1)),
    )
    return jnp.asarray(sampler.run_sampler(key, biases, J, beta))


def is_lsb_enabled() -> bool:
    """Return True if CARNOT_USE_LSB=1 env var is set.

    **Detailed explanation for engineers:**
        This function implements the feature flag described in the experiment
        spec. When LSB is slower than the default parallel Gibbs sampler, it
        is kept as an opt-in alternative accessible via CARNOT_USE_LSB=1.
        When LSB is faster, it is set as the default (this check becomes
        redundant but is preserved for backwards compatibility).

    Returns:
        True if LSB is enabled via environment variable.
    """
    return os.environ.get("CARNOT_USE_LSB", "0") == "1"
