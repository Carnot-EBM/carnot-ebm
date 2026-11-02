"""CASAL vs Langevin scaling comparison on random continuous EBMs.

**Researcher summary:**
    Experiment 1689 benchmarks the CASAL primal-dual sampler (Split Augmented
    Langevin Sampling, arXiv:2505.18017) against the baseline Langevin dynamics
    sampler from Phase 3's ContinuousEBM module on n=16 and n=32 variable
    landscapes.

    Both samplers optimise E(x) = -0.5 * x^T J x - h^T x where J is a random
    symmetric coupling matrix and h is a random bias.  CASAL additionally enforces
    a box constraint |x_i| <= 0.9 via an inner projection loop, while Langevin
    uses tanh squashing to keep x inside (-1, 1) but does not enforce the tighter
    constraint.

**What 'speedup_ratio' means here:**
    Both samplers run 1000 steps from the same random starting point (matching
    seeds).  The speedup_ratio is the ratio of CASAL's energy improvement over
    those 1000 steps to Langevin's energy improvement:

        speedup_ratio = (E_init - E_casal) / max(|E_init - E_langevin|, 1e-12)

    A ratio > 1 means CASAL achieved a larger energy drop in the same step budget.
    A ratio < 1 means Langevin achieved a larger drop (CASAL's constraint
    projection overhead reduces the effective gradient steps).

**Why constraint violations matter:**
    The box constraint |x_i| <= 0.9 is a proxy for the hard constraints a
    production verifier would enforce.  Counting how many components exceed 0.9
    after 1000 steps measures whether the sampler's projections are actually working.
    Langevin (no projection) will often exceed 0.9 due to tanh saturation near the
    boundary; CASAL should have zero or near-zero violations.

Spec: REQ-SAMPLE-1689, SCENARIO-SAMPLE-1689
"""

from __future__ import annotations

import datetime
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.phase3.continuous_ebm import ContinuousEBM, sample_langevin


def make_random_ebm(n: int, seed: int = 0) -> ContinuousEBM:
    """Create a random symmetric ContinuousEBM with n variables.

    **Construction:**
        J = (A + A^T) / (2n) where A ~ N(0, 1)^{n x n}.
        Dividing by n keeps the spectral radius bounded as n grows, so the
        energy landscape has a similar shape at n=16 and n=32.

    Args:
        n: Number of variables.
        seed: NumPy random seed for reproducibility.

    Returns:
        ContinuousEBM with symmetric coupling and random bias.

    Spec: REQ-SAMPLE-1689-1
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    J = (A + A.T) / (2.0 * n)
    h = rng.standard_normal(n) / np.sqrt(n)
    return ContinuousEBM(variables=n, coupling=J, bias=h)


def energy_value(model: ContinuousEBM, x: np.ndarray) -> float:
    """Compute the quadratic energy E(x) = -0.5 * x^T J x - h^T x.

    Args:
        model: ContinuousEBM whose coupling J and bias h are used.
        x: State vector of shape (n,).

    Returns:
        Scalar energy value.  Lower = more favourable configuration.
    """
    return float(-0.5 * x @ model.coupling @ x - model.bias @ x)


def count_violations(x: np.ndarray, limit: float = 0.9) -> int:
    """Count components with |x_i| > limit (strict constraint violations).

    The box constraint |x_i| <= limit is what CASAL enforces via inner
    projection.  Langevin only enforces |x_i| < 1 via tanh squashing, so
    components near the tanh boundary (|x_i| ~ 0.95-0.99) appear as
    violations here.

    Args:
        x: State vector of shape (n,).
        limit: Hard constraint threshold.

    Returns:
        Integer count of violating components.

    Spec: REQ-SAMPLE-1689-2
    """
    return int(np.sum(np.abs(x) > limit))


def run_casal_comparison(
    n: int,
    n_steps: int = 1000,
    seed: int = 42,
    constraint_limit: float = 0.9,
) -> dict[str, Any]:
    """Run CASAL and Langevin on a random n-variable EBM and compare results.

    Both samplers start from the same Gaussian initial state (matching seeds)
    and run for n_steps gradient steps.  Returns per-sampler final energy,
    constraint violations, wall-clock time, and the speedup ratio.

    Args:
        n: Number of EBM variables.
        n_steps: Number of sampler steps to run.
        seed: Random seed for EBM construction and sampler initialization.
        constraint_limit: Box constraint threshold for violation counting.

    Returns:
        Dict with keys: n, n_steps, initial_energy, casal_energy,
        langevin_energy, casal_violations, langevin_violations,
        casal_time_s, langevin_time_s, speedup_ratio, casal_finite,
        langevin_finite.

    Spec: REQ-SAMPLE-1689-1, REQ-SAMPLE-1689-3
    """
    from carnot.samplers.casal import casal_sample

    model = make_random_ebm(n, seed=seed)

    # Both samplers use the same initial state: standard Gaussian (matches
    # sample_langevin's internal init convention so the comparison is fair).
    init_np = np.random.default_rng(seed).standard_normal(n)
    initial_energy = energy_value(model, init_np)

    # --- Langevin dynamics ---
    t0 = time.perf_counter()
    langevin_result = sample_langevin(
        model,
        n_steps=n_steps,
        lr=0.005,
        noise_scale=0.1,
        temp_schedule="cosine",
        seed=seed,
    )
    langevin_time = time.perf_counter() - t0
    langevin_energy = energy_value(model, langevin_result)
    langevin_violations = count_violations(langevin_result, constraint_limit)

    # --- CASAL primal-dual sampler ---
    # Wrap the numpy EBM as a pure JAX energy function so CASAL can differentiate it.
    J_jax = jnp.array(model.coupling)
    h_jax = jnp.array(model.bias)
    init_jax = jnp.array(init_np)

    def energy_fn(x: jax.Array) -> jax.Array:
        # JAX-traced version of the quadratic EBM energy
        return -0.5 * x @ J_jax @ x - h_jax @ x

    def constraint_fn(x: jax.Array) -> jax.Array:
        # Box constraint: returns total soft-hinge violation for |x_i| > limit.
        # Returns 0.0 when all components satisfy the constraint.
        return jnp.sum(jnp.maximum(jnp.abs(x) - constraint_limit, 0.0))

    key = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()
    casal_result_jax = casal_sample(
        energy_fn=energy_fn,
        constraint_fn=constraint_fn,
        init_state=init_jax,
        steps=n_steps,
        key=key,
        step_size=0.005,
        proj_steps=5,
        proj_lr=0.1,
    )
    casal_time = time.perf_counter() - t0

    casal_result = np.array(casal_result_jax)
    casal_energy = energy_value(model, casal_result)
    casal_violations = count_violations(casal_result, constraint_limit)

    # Speedup ratio: how much more energy CASAL dropped vs Langevin.
    # Positive drop = improvement (energy went down).
    casal_drop = initial_energy - casal_energy
    langevin_drop = initial_energy - langevin_energy
    speedup_ratio = float(casal_drop / max(abs(langevin_drop), 1e-12))

    return {
        "n": n,
        "n_steps": n_steps,
        "initial_energy": round(initial_energy, 6),
        "casal_energy": round(casal_energy, 6),
        "langevin_energy": round(langevin_energy, 6),
        "casal_violations": casal_violations,
        "langevin_violations": langevin_violations,
        "casal_time_s": round(casal_time, 4),
        "langevin_time_s": round(langevin_time, 4),
        "speedup_ratio": round(speedup_ratio, 4),
        "casal_finite": bool(np.all(np.isfinite(casal_result))),
        "langevin_finite": bool(np.all(np.isfinite(langevin_result))),
    }


def build_casal_scaling_artifact(
    results_16: dict[str, Any],
    results_32: dict[str, Any],
) -> dict[str, Any]:
    """Build the required JSON artifact for experiment 1689.

    The 'headline' casal_energy and langevin_energy fields use n=16 results
    because the task asks for headline values alongside the per-n breakdown.
    The speedup_ratio is averaged across n=16 and n=32.

    Args:
        results_16: Output of run_casal_comparison(n=16, ...).
        results_32: Output of run_casal_comparison(n=32, ...).

    Returns:
        JSON-serialisable artifact dict with all REQ-SAMPLE-1689-2 fields.

    Spec: REQ-SAMPLE-1689-2, REQ-SAMPLE-1689-4
    """
    both_finite = (
        results_16["casal_finite"]
        and results_16["langevin_finite"]
        and results_32["casal_finite"]
        and results_32["langevin_finite"]
    )
    acceptance_gate_passed = both_finite

    # Average speedup across n=16 and n=32 as the headline metric.
    speedup_ratio = round(
        (results_16["speedup_ratio"] + results_32["speedup_ratio"]) / 2.0, 4
    )

    verdict_detail = (
        f"n16_casal={results_16['casal_energy']:.4f}_vs_langevin={results_16['langevin_energy']:.4f}; "
        f"n32_casal={results_32['casal_energy']:.4f}_vs_langevin={results_32['langevin_energy']:.4f}; "
        f"speedup_ratio={speedup_ratio:.4f}; acceptance={acceptance_gate_passed}"
    )

    return {
        "schema": "carnot.experiment_1689_casal_scaling.v1",
        "experiment": "exp1689-casal-scaling",
        "spec_refs": ["REQ-SAMPLE-1689", "SCENARIO-SAMPLE-1689"],
        "run_date": datetime.date.today().isoformat(),
        "n_steps": results_16["n_steps"],
        "constraint_limit": 0.9,
        "random_seed": 42,
        "results_n16": results_16,
        "results_n32": results_32,
        # Required artifact fields (REQ-SAMPLE-1689-2):
        "casal_energy": results_16["casal_energy"],
        "langevin_energy": results_16["langevin_energy"],
        "speedup_ratio": speedup_ratio,
        "acceptance_gate_passed": acceptance_gate_passed,
        "honest_verdict": f"complete: CASAL vs Langevin scaling comparison finished; {verdict_detail}",
    }
