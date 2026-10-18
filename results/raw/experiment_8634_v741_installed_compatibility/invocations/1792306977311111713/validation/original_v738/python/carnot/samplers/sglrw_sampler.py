"""SGLRW Sampler -- Stochastic Gradient Lattice Random Walk.

**Researcher summary:**
    Implements a low-precision stochastic gradient lattice algorithm
    mirroring thermodynamic computing noise models. This prepares Carnot
    for future Extropic TSU integration by emulating physical stochastic
    hardware.

Spec: REQ-SAMPLE-2080
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.samplers.parallel_ising import AnnealingSchedule


@dataclass
class SGLRWSampler:
    """Stochastic Gradient Lattice Random Walk (SGLRW) Sampler.

    **Researcher summary:**
        A JAX-optimized hardware-ready sampler that emulates the continuous-time
        noisy dynamics of thermodynamic computing. Instead of discrete Bernoulli
        flips, it maintains a continuous state 'x' and updates it via Langevin
        dynamics with low-precision effects, finally threshholding to obtain spins.

    Attributes:
        n_warmup: Number of annealing steps before collecting samples.
        n_samples: Number of samples to collect after warmup.
        steps_per_sample: Sweeps between collected samples.
        schedule: Temperature annealing schedule for the warmup phase.
        step_size: Euler-Maruyama step size (dt).
        noise_scale: Scaling factor for the injected Gaussian noise.
    """

    n_warmup: int = 1000
    n_samples: int = 50
    steps_per_sample: int = 20
    schedule: AnnealingSchedule | None = None
    step_size: float = 0.1
    noise_scale: float = 1.0

    def sample(
        self,
        key: jax.Array,
        biases: jax.Array,
        coupling_matrix: jax.Array,
        beta: float | jax.Array = 10.0,
        init_spins: jax.Array | None = None,
    ) -> jax.Array:
        """Run SGLRW sampling on an Ising model.

        Args:
            key: JAX PRNG key.
            biases: Bias vector, shape (n_spins,).
            coupling_matrix: Symmetric coupling matrix, shape (n_spins, n_spins).
            beta: Inverse temperature (scalar).
            init_spins: Initial spin configuration, shape (n_spins,), boolean.

        Returns:
            Samples array of shape (n_samples, n_spins), boolean.
        """
        n_spins = biases.shape[0]
        beta = jnp.asarray(beta, dtype=jnp.float32)

        key, init_key = jrandom.split(key)
        if init_spins is None:
            spins = jrandom.bernoulli(init_key, 0.5, (n_spins,))
        else:
            spins = jnp.asarray(init_spins, dtype=jnp.bool_)

        # Internal continuous state x initialized to match spins: +1.0 for True, -1.0 for False
        x = jnp.where(spins, 1.0, -1.0).astype(jnp.float32)

        J = jnp.asarray(coupling_matrix, dtype=jnp.float32)
        b_base = jnp.asarray(biases, dtype=jnp.float32)

        schedule = self.schedule or AnnealingSchedule(beta_init=float(beta), beta_final=float(beta))
        dt = jnp.asarray(self.step_size, dtype=jnp.float32)
        noise_std = jnp.asarray(self.noise_scale, dtype=jnp.float32) * jnp.sqrt(dt)

        def sweep_fn(carry, step_key):
            x_curr, step = carry
            beta_t = schedule.beta_at_step(step, self.n_warmup)
            
            s = jnp.where(x_curr > 0, 1.0, 0.0)
            
            # Leaky integrator formulation:
            drift = -x_curr + beta_t * (b_base + J @ s)
            
            noise = noise_std * jrandom.normal(step_key, x_curr.shape)
            
            x_next = x_curr + drift * dt + noise
            x_next = jnp.clip(x_next, -10.0, 10.0)
            
            return (x_next, step + 1), None

        key, warmup_key = jrandom.split(key)
        warmup_keys = jrandom.split(warmup_key, self.n_warmup)
        (x_final, _), _ = jax.lax.scan(sweep_fn, (x, jnp.int32(0)), warmup_keys)

        beta_final = jnp.asarray(schedule.beta_final, dtype=jnp.float32)

        def sample_sweep_fn(x_inner, step_key):
            s = jnp.where(x_inner > 0, 1.0, 0.0)
            drift = -x_inner + beta_final * (b_base + J @ s)
            noise = noise_std * jrandom.normal(step_key, x_inner.shape)
            x_next = x_inner + drift * dt + noise
            return jnp.clip(x_next, -10.0, 10.0)

        def collect_fn(carry, sample_key):
            x_inner = carry
            sweep_keys = jrandom.split(sample_key, self.steps_per_sample)

            def decorrelate(inner_carry, k):
                return sample_sweep_fn(inner_carry, k), None

            x_inner, _ = jax.lax.scan(decorrelate, x_inner, sweep_keys)
            
            s_out = x_inner > 0
            return x_inner, s_out

        key, collect_key = jrandom.split(key)
        collect_keys = jrandom.split(collect_key, self.n_samples)
        _, samples = jax.lax.scan(collect_fn, x_final, collect_keys)

        return samples
