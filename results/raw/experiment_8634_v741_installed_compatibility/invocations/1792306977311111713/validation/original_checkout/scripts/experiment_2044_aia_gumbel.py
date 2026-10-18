#!/usr/bin/env python3
"""Exp 2044: AIA continuous Gumbel sampler simulation.

Spec traces: REQ-SAMPLE-2044, SCENARIO-SAMPLE-2044.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jrandom

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.inference.far_eqm import extract_eqm_gradient
from carnot.samplers.continuous_gumbel import ContinuousGumbelSampler
from scripts.experiment_template import ExperimentTemplate


RESULTS_PATH = Path("results/experiment_2044_aia_gumbel.json")
SPEC_REFS = ["REQ-SAMPLE-2044", "SCENARIO-SAMPLE-2044"]


@dataclass
class EqM2041Landscape:
    """Exp 2041 toy EqM landscape wrapped as an EnergyFunction."""

    n_constraints: int = 10
    dim: int = 64

    @property
    def input_dim(self) -> int:
        return self.n_constraints * self.dim

    def energy(self, x: jax.Array) -> jax.Array:
        energies, _ = extract_eqm_gradient(x)
        return jnp.mean(energies)

    def energy_batch(self, xs: jax.Array) -> jax.Array:
        return jax.vmap(self.energy)(xs)

    def grad_energy(self, x: jax.Array) -> jax.Array:
        _, gradients = extract_eqm_gradient(x)
        return gradients


def initial_eqm_state() -> jax.Array:
    """Return the same 10x64 starting state used by Exp 2041."""
    return jnp.ones((10, 64), dtype=jnp.float32) * 0.5


def first_step_at_or_below(energies: Sequence[float], threshold: float) -> int | None:
    """Return the first energy-history index at or below ``threshold``."""
    for idx, energy in enumerate(energies):
        if energy <= threshold:
            return idx
    return None


def _energy_history(energy_fn: EqM2041Landscape, init: jax.Array, chain: jax.Array) -> list[float]:
    values = [float(energy_fn.energy(init))]
    values.extend(float(energy_fn.energy(x)) for x in chain)
    return values


def _metrics(
    name: str,
    energy_fn: EqM2041Landscape,
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


def metropolis_hastings_chain(
    energy_fn: EqM2041Landscape,
    init: jax.Array,
    *,
    n_steps: int,
    key: jax.Array,
    proposal_scale: float = 0.02,
    temperature: float = 1.0,
) -> tuple[jax.Array, float]:
    """Run a symmetric Gaussian random-walk Metropolis-Hastings baseline."""
    current = init
    current_energy = energy_fn.energy(current)
    accepted = 0
    states: list[jax.Array] = []

    for _ in range(n_steps):
        key, proposal_key, accept_key = jrandom.split(key, 3)
        proposal = current + proposal_scale * jrandom.normal(proposal_key, current.shape)
        proposal_energy = energy_fn.energy(proposal)
        log_accept = -(proposal_energy - current_energy) / temperature
        accept = bool(jnp.log(jrandom.uniform(accept_key)) < jnp.minimum(0.0, log_accept))
        if accept:
            current = proposal
            current_energy = proposal_energy
            accepted += 1
        states.append(current)

    chain = jnp.stack(states) if states else jnp.empty((0, *init.shape), dtype=init.dtype)
    return chain, accepted / max(n_steps, 1)


def run_simulation(n_steps: int = 80, threshold: float = 0.05) -> dict[str, object]:
    """Run the Exp 2044 sampler comparison without writing an artifact."""
    landscape = EqM2041Landscape()
    init = initial_eqm_state()
    gumbel = ContinuousGumbelSampler(
        step_size=0.08,
        energy_temperature=0.03,
        gumbel_scale=0.0,
        hard=True,
        anneal_rate=0.98,
        curvature=2.0,
    )

    gumbel_chain = gumbel.sample_chain(
        landscape,
        init,
        n_steps=n_steps,
        key=jrandom.PRNGKey(2044),
    )
    mh_chain, mh_acceptance_rate = metropolis_hastings_chain(
        landscape,
        init,
        n_steps=n_steps,
        key=jrandom.PRNGKey(2045),
        temperature=1e-6,
    )

    gumbel_metrics = _metrics("continuous_gumbel", landscape, init, gumbel_chain, threshold)
    mh_metrics = _metrics("metropolis_hastings", landscape, init, mh_chain, threshold)
    gumbel_steps = gumbel_metrics["steps_to_threshold"]
    mh_steps = mh_metrics["steps_to_threshold"]
    gumbel_faster = gumbel_steps is not None and (mh_steps is None or gumbel_steps < mh_steps)
    if gumbel_steps is None:
        speedup = 0.0
        speedup_lower_bound = False
    elif mh_steps is None:
        speedup = float(n_steps / max(int(gumbel_steps), 1))
        speedup_lower_bound = True
    else:
        speedup = float(int(mh_steps) / max(int(gumbel_steps), 1))
        speedup_lower_bound = False

    return {
        "spec_refs": SPEC_REFS,
        "source_experiment": {
            "experiment": 2041,
            "landscape": "EqM quadratic gradient from carnot.inference.far_eqm.extract_eqm_gradient",
            "state_shape": [landscape.n_constraints, landscape.dim],
        },
        "problem": {
            "name": "exp2041_eqm_gradient_landscape",
            "n_steps": n_steps,
            "energy_threshold": threshold,
            "initial_energy": float(landscape.energy(init)),
        },
        "samplers": {
            "continuous_gumbel": {
                "step_size": gumbel.step_size,
                "move_values": list(gumbel.move_values),
                "energy_temperature": gumbel.energy_temperature,
                "gumbel_scale": gumbel.gumbel_scale,
                "hard": gumbel.hard,
                "anneal_rate": gumbel.anneal_rate,
                "curvature": gumbel.curvature,
            },
            "metropolis_hastings": {
                "proposal": "symmetric_gaussian_random_walk",
                "proposal_scale": 0.02,
                "temperature": 1e-6,
                "acceptance_rate": mh_acceptance_rate,
            },
        },
        "metrics": {
            "continuous_gumbel": gumbel_metrics,
            "metropolis_hastings": mh_metrics,
        },
        "gumbel_faster_than_metropolis_hastings": bool(gumbel_faster),
        "gumbel_speedup": speedup,
        "gumbel_speedup_lower_bound": speedup_lower_bound,
        "honest_verdict": "gumbel_converged_faster" if gumbel_faster else "gumbel_not_faster",
    }


def _template_for_output(output_path: Path) -> ExperimentTemplate:
    if output_path.is_absolute():
        return ExperimentTemplate(
            exp_id=2044,
            title="AIA Gumbel Sampler Simulator",
            deliverable=output_path.name,
            requires_gpu=False,
            repo_root=output_path.parent,
            seed=2044,
        )
    return ExperimentTemplate(
        exp_id=2044,
        title="AIA Gumbel Sampler Simulator",
        deliverable=str(output_path),
        requires_gpu=False,
        seed=2044,
    )


def run_experiment(output_path: Path = RESULTS_PATH) -> dict[str, object]:
    """Run Exp 2044 and write a schema-bearing JSON artifact."""
    tmpl = _template_for_output(output_path)
    tmpl.setup()
    data = run_simulation()
    data["deliverable"] = str(RESULTS_PATH)
    artifact = tmpl.build_result(
        data,
        status="success",
        code_files=[
            __file__,
            "python/carnot/samplers/continuous_gumbel.py",
            "python/carnot/inference/far_eqm.py",
        ],
    )

    tmpl._output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    tmpl.assert_deliverable_written()
    return json.loads(tmpl._output_path.read_text())


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
