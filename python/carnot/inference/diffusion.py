"""Energy-based diffusion generation: parallel solution generation.

**Researcher summary:**
    Instead of sequential generation (which cascades errors), diffusion
    starts from pure noise and iteratively denoises toward low-energy
    configurations. Each step evaluates the COMPLETE configuration,
    enabling holistic consistency checking. From arxiv 2410.21357.

**Detailed explanation for engineers:**
    The algorithm:
    1. Start from random noise: x_0 ~ N(0, σ)
    2. For t = 0 to T (denoising steps):
       a. Compute energy gradient: g = ∇E(x_t)
       b. Denoise: x_{t+1} = x_t - α_t * g + σ_t * ε (Langevin step)
       c. Step size and noise anneal over time (high → low)
    3. Round to discrete domain
    4. Verify and certify

    This generates solutions from SCRATCH using only the energy landscape —
    no LLM needed. When combined with multi-start (P2), generates M
    candidates and selects the lowest-energy one.

Spec: REQ-INFER-012, SCENARIO-INFER-013
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.verify.constraint import ComposedEnergy, VerificationResult


@dataclass
class DiffusionConfig:
    """Configuration for diffusion generation.

    Spec: REQ-INFER-012
    """

    n_diffusion_steps: int = 100
    initial_noise_scale: float = 1.0
    step_size: float = 0.1
    step_schedule: str = "linear"  # "linear", "cosine", "constant"
    noise_schedule: str = "linear"
    n_candidates: int = 1
    seed: int = 42


@dataclass
class DiffusionResult:
    """Result of diffusion generation.

    Spec: REQ-INFER-012
    """

    final_state: Any = None  # jax.Array
    final_energy: float = 0.0
    verification: VerificationResult | None = None
    trajectory_energies: list[float] = field(default_factory=list)
    n_steps: int = 0
    candidates_generated: int = 0


def _schedule_value(
    t: int,
    total: int,
    max_val: float,
    schedule: str,
) -> float:
    """Compute scheduled value at step t.

    Spec: REQ-INFER-012
    """
    if total <= 1:
        return max_val
    frac = t / (total - 1)
    if schedule == "constant":
        return max_val
    if schedule == "cosine":
        return max_val * 0.5 * (1.0 + math.cos(math.pi * frac))
    # Default: linear decay
    return max_val * (1.0 - frac)


def _diffusion_single(
    energy: ComposedEnergy,
    config: DiffusionConfig,
    key: jax.Array,
) -> tuple[jax.Array, list[float]]:
    """Run one diffusion trajectory from noise. Returns (final_state, energies).

    Spec: REQ-INFER-012
    """
    key, init_key = jrandom.split(key)
    x = jrandom.normal(init_key, (energy.input_dim,)) * config.initial_noise_scale

    trajectory: list[float] = []

    for t in range(config.n_diffusion_steps):
        e_val = float(energy.energy(x))
        trajectory.append(e_val)

        # Compute gradient
        grad = energy.grad_energy(x)

        # Scheduled step size and noise
        step = _schedule_value(t, config.n_diffusion_steps, config.step_size, config.step_schedule)
        noise_scale = _schedule_value(
            t, config.n_diffusion_steps, config.initial_noise_scale * 0.1, config.noise_schedule
        )

        # Langevin update
        key, subkey = jrandom.split(key)
        noise = jrandom.normal(subkey, x.shape) * noise_scale
        x = x - step * grad + noise

    # Final energy
    trajectory.append(float(energy.energy(x)))

    return x, trajectory


def diffusion_generate(
    energy: ComposedEnergy,
    config: DiffusionConfig | None = None,
    round_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> DiffusionResult:
    """Generate a solution via energy-based diffusion.

    **Researcher summary:**
        Starts from noise, denoises via Langevin on energy, returns the
        lowest-energy candidate (optionally rounded to discrete).

    **Detailed explanation for engineers:**
        If n_candidates > 1, runs multiple independent diffusion
        trajectories from different random seeds and selects the one
        with minimum energy (self-verification pattern from EBT paper).

    Spec: REQ-INFER-012, SCENARIO-INFER-013
    """
    if config is None:
        config = DiffusionConfig()

    key = jrandom.PRNGKey(config.seed)

    best_state = None
    best_energy = float("inf")
    best_trajectory: list[float] = []

    for c in range(config.n_candidates):
        key, subkey = jrandom.split(key)
        state, trajectory = _diffusion_single(energy, config, subkey)

        final_e = trajectory[-1] if trajectory else float(energy.energy(state))
        if final_e < best_energy:
            best_state = state
            best_energy = final_e
            best_trajectory = trajectory

    # Round if requested
    if round_fn is not None and best_state is not None:
        best_state = round_fn(best_state)
        best_energy = float(energy.energy(best_state))

    # Verify
    verification = None
    if best_state is not None:
        verification = energy.verify(best_state)

    return DiffusionResult(
        final_state=best_state,
        final_energy=best_energy,
        verification=verification,
        trajectory_energies=best_trajectory,
        n_steps=config.n_diffusion_steps,
        candidates_generated=config.n_candidates,
    )


def diffusion_generate_sat(
    clauses: list[Any],
    n_vars: int,
    config: DiffusionConfig | None = None,
) -> DiffusionResult:
    """Generate SAT solution via diffusion.

    Spec: REQ-INFER-012
    """
    from carnot.verify.sat import build_sat_energy

    energy = build_sat_energy(clauses, n_vars)

    def round_fn(x: jax.Array) -> jax.Array:
        return jnp.where(x >= 0.5, 1.0, 0.0)

    return diffusion_generate(energy, config, round_fn=round_fn)


def diffusion_generate_coloring(
    edges: list[tuple[int, int]],
    n_nodes: int,
    n_colors: int,
    config: DiffusionConfig | None = None,
) -> DiffusionResult:
    """Generate graph coloring via diffusion.

    Spec: REQ-INFER-012
    """
    from carnot.verify.graph_coloring import build_coloring_energy

    energy = build_coloring_energy(edges, n_nodes, n_colors)

    def round_fn(x: jax.Array) -> jax.Array:
        return jnp.round(jnp.clip(x, 0.0, float(n_colors - 1)))

    return diffusion_generate(energy, config, round_fn=round_fn)
