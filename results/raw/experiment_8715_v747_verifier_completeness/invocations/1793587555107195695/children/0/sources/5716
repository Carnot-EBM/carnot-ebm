"""Continuous-Latent FAR-Inspired Constraint Sampler -- JAX implementation.

**Researcher summary:**
    Wraps a standard Langevin step with a learned low-rank surrogate "shortcut
    head" inspired by Fast Autoregressive (FAR) generation.  The surrogate
    predicts constraint satisfaction cheaply; when ALL constraints are predicted
    satisfied below a configurable threshold, the sampler skips the full energy
    gradient and uses a scaled surrogate gradient instead — reducing expensive
    evaluations of the true energy function.

**Detailed explanation for engineers:**
    In many thermodynamic / EBM sampling problems the bottleneck is evaluating
    the energy function E(x) and its gradient, especially when E is a deep
    neural network or requires solving a constraint system.  FAR models solve
    an analogous problem in generation: they introduce a "shortcut" path through
    the token sequence that allows skipping costly transformer forward passes.

    This module adapts that idea to the continuous-latent sampling setting:

    1.  **FARSurrogateHead** — a tiny linear model (low-rank projection) that
        maps the current latent state z ∈ ℝ^d to n_constraints soft scores
        in [0,1].  Each score estimates how close the current state is to
        satisfying the corresponding soft constraint.  Training the head is
        out-of-scope for this prototype; it is initialised from a random key
        and meant to be replaced with a learned head in downstream work.

    2.  **ContinuousLatentSampler** — wraps any EnergyFunction (the same
        protocol used by LangevinSampler, HMC, etc.) together with a
        FARSurrogateHead.  At each step:
        a.  Run the surrogate: scores = sigmoid(z @ W + b).
        b.  If max(scores) < skip_threshold → all constraints "look satisfied";
            use -surrogate_gradient (cheap) as the drift signal instead of the
            true energy gradient.
        c.  Otherwise → evaluate the true energy gradient (expensive) and
            take the standard Langevin update.
        d.  Add Langevin noise scaled by step_size.

    **Why this matters for Phase-3 / sovereignty:**
    The surrogate-skip pattern lets the inner loop run at near-linear-layer
    speed for states that clearly satisfy constraints, while retaining full
    expressivity when the surrogate is uncertain.  On consumer NPU hardware
    (Intel AI Boost, Ryzen AI) the surrogate is a single matmul — mapped
    directly to the NPU — while the full EBM evaluation remains on CPU/GPU.
    This supports the CLAUDE.md decentralization rule that Carnot must run
    on the hardware users already own.

Spec: REQ-SAMPLE-1935, REQ-SAMPLE-1935-1, REQ-SAMPLE-1935-2,
      REQ-SAMPLE-1935-3, REQ-SAMPLE-1935-4, REQ-SAMPLE-1935-5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as jrandom

if TYPE_CHECKING:
    from carnot.core.energy import EnergyFunction


@dataclass
class FARSurrogateHead:
    """Low-rank linear surrogate that predicts soft constraint satisfaction.

    **Researcher summary:**
        Single-layer linear model: scores = sigmoid(z @ W + b).
        W is shape (latent_dim, n_constraints); b is shape (n_constraints,).
        The sigmoid maps raw scores to (0, 1) so they can be compared against
        skip_threshold.

    **Detailed explanation for engineers:**
        Think of this as an extremely cheap binary classifier per constraint.
        Each column of W is a "constraint direction" in latent space.  If
        z @ W[:, k] + b[k] is strongly negative, constraint k is predicted
        satisfied (sigmoid close to 0 means "low violation energy").

        The head is RANDOM by default (use `from_random_key`).  In production
        you would train W and b so that the head accurately tracks the true
        constraint satisfaction state.  The prototype demonstrates the
        *architectural pattern*; the learning loop is out-of-scope.

    Attributes:
        W: Linear projection weight matrix, shape (latent_dim, n_constraints).
        b: Bias vector, shape (n_constraints,).

    Spec: REQ-SAMPLE-1935-1
    """

    W: jax.Array
    b: jax.Array

    @staticmethod
    def from_random_key(
        key: jax.Array,
        latent_dim: int,
        n_constraints: int,
        scale: float = 0.1,
    ) -> "FARSurrogateHead":
        """Initialise W ~ N(0, scale²), b = 0.

        Args:
            key: JAX PRNG key for reproducibility.
            latent_dim: Dimensionality of the latent space z.
            n_constraints: Number of soft constraints to track.
            scale: Standard deviation for W initialisation.  Small values
                keep surrogate scores near 0.5 initially (maximum uncertainty).

        Returns:
            A randomly-initialised FARSurrogateHead.

        Spec: REQ-SAMPLE-1935-1
        """
        W = jrandom.normal(key, shape=(latent_dim, n_constraints)) * scale
        b = jnp.zeros(n_constraints)
        return FARSurrogateHead(W=W, b=b)

    def predict(self, z: jax.Array) -> jax.Array:
        """Return soft constraint scores in (0, 1).

        **For engineers:**
            scores[k] near 0  → constraint k predicted satisfied (low violation).
            scores[k] near 1  → constraint k predicted violated (skip is risky).

        Args:
            z: Current latent state, shape (latent_dim,).

        Returns:
            scores: Soft constraint predictions, shape (n_constraints,).

        Spec: REQ-SAMPLE-1935-1
        """
        raw = z @ self.W + self.b
        return jax.nn.sigmoid(raw)

    def grad_scores_sum(self, z: jax.Array) -> jax.Array:
        """Gradient of sum(scores) w.r.t. z, used as cheap surrogate drift.

        **For engineers:**
            d/dz sum(sigmoid(z @ W + b)) = W @ (scores * (1 - scores)).
            This is O(d * n_constraints) — the same as a single matmul —
            whereas the true energy gradient may involve many more operations.

        Args:
            z: Current latent state, shape (latent_dim,).

        Returns:
            grad: Shape (latent_dim,).

        Spec: REQ-SAMPLE-1935-3
        """
        scores = self.predict(z)
        sig_grad = scores * (1.0 - scores)  # elementwise sigmoid derivative
        return self.W @ sig_grad  # shape (latent_dim,)


@dataclass
class SamplerStats:
    """Mutable accumulator for per-run statistics.

    Attributes:
        total_steps: Number of steps taken.
        surrogate_skip_count: Steps where the surrogate head bypassed the
            full energy gradient call.
    """

    total_steps: int = 0
    surrogate_skip_count: int = 0

    @property
    def skip_rate(self) -> float:
        """Fraction of steps where the surrogate shortcut was used."""
        if self.total_steps == 0:
            return 0.0
        return self.surrogate_skip_count / self.total_steps


@dataclass
class ContinuousLatentSampler:
    """FAR-inspired continuous-latent sampler with surrogate constraint shortcut.

    **Researcher summary:**
        Extends Langevin dynamics with a surrogate gate: if the surrogate head
        predicts all constraints satisfied, use the cheap surrogate gradient
        instead of the expensive true energy gradient.  Tracks the skip rate
        for benchmarking.

    **Detailed explanation for engineers:**
        Per-step algorithm:
        1.  Compute surrogate scores s = sigmoid(z @ W + b).
        2.  If max(s) < skip_threshold:
                drift = -surrogate_grad_sum(z)     # cheap path
                surrogate_skip_count += 1
            Else:
                drift = -grad_energy(z)            # expensive path
        3.  z_new = z + (step_size / 2) * drift + sqrt(step_size) * noise

        The noise term ensures ergodicity (prevents the chain from collapsing
        to a deterministic fixed point).  The step_size interpretation is the
        same as LangevinSampler.

    Attributes:
        energy_fn: Any EnergyFunction with energy() and grad_energy() methods.
        surrogate: Trained or randomly-initialised FARSurrogateHead.
        step_size: Langevin discretisation step (epsilon). Default 0.01.
        skip_threshold: Surrogate score threshold for the shortcut path.
            When max(scores) < skip_threshold, skip the full energy gradient.
            Must be in (0, 1]. Default 0.5.

    For example::

        head = FARSurrogateHead.from_random_key(jrandom.PRNGKey(0), 32, 4)
        sampler = ContinuousLatentSampler(energy_fn=my_energy, surrogate=head)
        key = jrandom.PRNGKey(1)
        z0 = jrandom.normal(key, (32,))
        z_final, stats = sampler.sample(key, z0, n_steps=100)
        print(f"Skip rate: {stats.skip_rate:.2%}")

    Spec: REQ-SAMPLE-1935-2, REQ-SAMPLE-1935-3
    """

    energy_fn: "EnergyFunction"
    surrogate: FARSurrogateHead
    step_size: float = 0.01
    skip_threshold: float = 0.5

    def sample(
        self,
        key: jax.Array,
        init: jax.Array,
        n_steps: int,
    ) -> tuple[jax.Array, SamplerStats]:
        """Draw one sample by running n_steps FAR-surrogate Langevin steps.

        **For engineers:**
            Runs a Python for-loop (not jax.lax.scan) so that the surrogate
            gate condition can branch without requiring XLA's conditional
            primitives (jax.lax.cond).  This keeps the implementation simple
            and debuggable at the cost of JIT-compilation of the full chain.
            For large n_steps, switch to jax.lax.scan + jax.lax.cond.

        Args:
            key: JAX PRNG key. Consumed and split per step for fresh noise.
            init: Initial latent state, shape (latent_dim,).
            n_steps: Number of Langevin steps to take.

        Returns:
            Tuple of (final_state, stats) where final_state has the same
            shape as init and stats records the surrogate skip rate.

        Spec: REQ-SAMPLE-1935-2, REQ-SAMPLE-1935-3
        """
        z = init
        stats = SamplerStats()

        for _ in range(n_steps):
            key, noise_key = jrandom.split(key)
            scores = self.surrogate.predict(z)

            # Surrogate shortcut: if all constraints look satisfied, skip E grad.
            if float(jnp.max(scores)) < self.skip_threshold:
                drift = -self.surrogate.grad_scores_sum(z)
                stats.surrogate_skip_count += 1
            else:
                drift = -self.energy_fn.grad_energy(z)

            noise = jrandom.normal(noise_key, shape=z.shape)
            z = z + (self.step_size / 2.0) * drift + (self.step_size ** 0.5) * noise
            stats.total_steps += 1

        return z, stats

    def sample_chain(
        self,
        key: jax.Array,
        init: jax.Array,
        n_steps: int,
    ) -> tuple[jax.Array, SamplerStats]:
        """Same as sample() but also returns all intermediate states.

        Returns:
            Tuple of (chain, stats) where chain has shape (n_steps, latent_dim).

        Spec: REQ-SAMPLE-1935-2
        """
        z = init
        stats = SamplerStats()
        chain = []

        for _ in range(n_steps):
            key, noise_key = jrandom.split(key)
            scores = self.surrogate.predict(z)

            if float(jnp.max(scores)) < self.skip_threshold:
                drift = -self.surrogate.grad_scores_sum(z)
                stats.surrogate_skip_count += 1
            else:
                drift = -self.energy_fn.grad_energy(z)

            noise = jrandom.normal(noise_key, shape=z.shape)
            z = z + (self.step_size / 2.0) * drift + (self.step_size ** 0.5) * noise
            stats.total_steps += 1
            chain.append(z)

        return jnp.stack(chain), stats
