#!/usr/bin/env python3
"""Exp 1727: Equilibrium Matching versus Langevin convergence.

Spec traces: REQ-SAMPLE-1727, REQ-SAMPLE-1728, SCENARIO-SAMPLE-1728.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import AutoGradMixin
from carnot.samplers.equilibrium_matching import EquilibriumMatchingSampler
from carnot.samplers.langevin import LangevinSampler


RESULTS_PATH = Path("results/experiment_1727_eqm.json")
SPEC_REFS = ["REQ-SAMPLE-1727", "REQ-SAMPLE-1728", "SCENARIO-SAMPLE-1728"]


@dataclass
class SoftConstraintEnergy(AutoGradMixin):
    """Convex target-matching energy with a mild residual smoothness penalty."""

    target: jax.Array
    smoothness: float = 0.05

    @property
    def input_dim(self) -> int:
        return int(self.target.shape[0])

    def energy(self, x: jax.Array) -> jax.Array:
        residual = x - self.target
        target_penalty = 0.5 * jnp.sum(residual**2)
        smoothness_penalty = self.smoothness * jnp.sum((residual[1:] - residual[:-1]) ** 2)
        return target_penalty + smoothness_penalty


def first_step_at_or_below(energies: Sequence[float], threshold: float) -> int | None:
    """Return the first energy-history index at or below threshold."""
    for idx, energy in enumerate(energies):
        if energy <= threshold:
            return idx
    return None


def _energy_history(energy_fn: SoftConstraintEnergy, init: jax.Array, chain: jax.Array) -> list[float]:
    values = [float(energy_fn.energy(init))]
    values.extend(float(value) for value in energy_fn.energy_batch(chain))
    return values


def _metrics(
    name: str,
    energy_fn: SoftConstraintEnergy,
    init: jax.Array,
    chain: jax.Array,
    threshold: float,
) -> dict[str, object]:
    energies = _energy_history(energy_fn, init, chain)
    first_threshold_step = first_step_at_or_below(energies, threshold)
    return {
        "sampler": name,
        "initial_energy": energies[0],
        "final_energy": energies[-1],
        "best_energy": min(energies),
        "steps_to_threshold": first_threshold_step,
        "converged": first_threshold_step is not None,
        "finite_chain": bool(jnp.all(jnp.isfinite(chain))),
        "energy_history": energies,
    }


def run_experiment(output_path: Path = RESULTS_PATH) -> dict[str, object]:
    """Run the deterministic EqM/Langevin convergence comparison."""
    n_steps = 80
    threshold = 0.02
    target = jnp.linspace(-1.0, 1.0, 16)
    init = target + 3.0
    energy_fn = SoftConstraintEnergy(target=target)

    eqm = EquilibriumMatchingSampler(
        step_size=0.35,
        learning_rate=0.6,
        matching_strength=0.8,
        momentum=0.0,
        clip_norm=20.0,
    )
    langevin = LangevinSampler(step_size=0.02, clip_norm=20.0)

    eqm_chain = eqm.sample_chain(energy_fn, init, n_steps=n_steps)
    langevin_chain = langevin.sample_chain(
        energy_fn,
        init,
        n_steps=n_steps,
        key=jrandom.PRNGKey(1727),
    )

    eqm_metrics = _metrics("equilibrium_matching", energy_fn, init, eqm_chain, threshold)
    langevin_metrics = _metrics("langevin_ula", energy_fn, init, langevin_chain, threshold)

    eqm_steps = eqm_metrics["steps_to_threshold"]
    langevin_steps = langevin_metrics["steps_to_threshold"]
    eqm_faster = eqm_steps is not None and (langevin_steps is None or eqm_steps < langevin_steps)
    speedup = None
    if eqm_steps is not None and langevin_steps is not None:
        speedup = float(langevin_steps / max(eqm_steps, 1))

    artifact = {
        "experiment_id": "1727",
        "spec_refs": SPEC_REFS,
        "problem": {
            "name": "soft_constraint_bowl",
            "dimension": int(target.shape[0]),
            "n_steps": n_steps,
            "energy_threshold": threshold,
            "initial_energy": float(energy_fn.energy(init)),
        },
        "samplers": {
            "eqm": {
                "step_size": eqm.step_size,
                "learning_rate": eqm.learning_rate,
                "matching_strength": eqm.matching_strength,
                "momentum": eqm.momentum,
                "clip_norm": eqm.clip_norm,
            },
            "langevin": {
                "step_size": langevin.step_size,
                "clip_norm": langevin.clip_norm,
                "seed": 1727,
            },
        },
        "metrics": {
            "eqm": eqm_metrics,
            "langevin": langevin_metrics,
        },
        "eqm_faster_than_langevin": bool(eqm_faster),
        "convergence_speedup": speedup,
        "honest_verdict": "eqm_converged_faster" if eqm_faster else "eqm_not_faster",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


if __name__ == "__main__":
    run_experiment()
