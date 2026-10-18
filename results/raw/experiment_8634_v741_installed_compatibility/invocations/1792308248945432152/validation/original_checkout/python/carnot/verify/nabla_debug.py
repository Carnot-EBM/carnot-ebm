"""Exp 1627 Nabla Reasoner Debug.

Spec: REQ-VERIFY-1627, SCENARIO-VERIFY-1627.

Performs a learning rate and momentum sweep for the continuous latent optimizer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from carnot.verify.nabla_reasoner import differentiable_ebcn_energy

JsonDict = dict[str, Any]

RUN_DATE = "20260509"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1627_nabla_debug.json")


@jax.jit
def langevin_step_momentum(
    logits: jnp.ndarray,
    velocity: jnp.ndarray,
    step_size: float,
    momentum: float,
    noise_scale: float,
    key: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Applies one step of Langevin dynamics with momentum."""
    energy, grads = jax.value_and_grad(differentiable_ebcn_energy)(logits)
    noise = jax.random.normal(key, logits.shape)
    new_velocity = momentum * velocity + step_size * grads
    new_logits = logits - new_velocity + noise_scale * noise
    return new_logits, new_velocity, energy


def optimize_logits_momentum(
    initial_logits: jnp.ndarray,
    steps: int = 100,
    step_size: float = 0.05,
    momentum: float = 0.9,
    noise_scale: float = 0.01
) -> JsonDict:
    """Optimizes continuous logit states via Langevin dynamics with momentum."""
    key = jax.random.PRNGKey(0)
    logits = initial_logits
    velocity = jnp.zeros_like(logits)
    
    def scan_body(carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], _x: Any) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        current_logits, current_velocity, rng = carry
        rng, subkey = jax.random.split(rng)
        next_logits, next_velocity, current_energy = langevin_step_momentum(
            current_logits, current_velocity, step_size, momentum, noise_scale, subkey
        )
        return (next_logits, next_velocity, rng), current_energy
        
    (final_logits, _, _), energies = jax.lax.scan(scan_body, (logits, velocity, key), None, length=steps)
    final_energy = differentiable_ebcn_energy(final_logits)
    
    initial_energy = float(energies[0])
    final_energy_val = float(final_energy)
    convergence_speed = float((initial_energy - final_energy_val) / steps)
    
    return {
        "initial_energy": initial_energy,
        "final_energy": final_energy_val,
        "convergence_speed": convergence_speed,
        "steps_run": steps
    }


def run_sweep(output_path: Path | str = DEFAULT_ARTIFACT_PATH) -> JsonDict:
    """Run the learning rate and momentum sweep."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    learning_rates = [0.01, 0.05, 0.1, 0.5]
    momentums = [0.0, 0.5, 0.9]
    
    key = jax.random.PRNGKey(42)
    initial_logits = jax.random.normal(key, (5, 10))
    initial_logits = initial_logits.at[0, 0].set(1.0)
    initial_logits = initial_logits.at[1, 0].set(-1.0)
    initial_logits = initial_logits.at[2, 0].set(1.0)
    
    best_lr = 0.0
    best_mom = 0.0
    best_final_energy = float("inf")
    converges = False
    
    for lr in learning_rates:
        for mom in momentums:
            res = optimize_logits_momentum(
                initial_logits, steps=200, step_size=lr, momentum=mom, noise_scale=0.001
            )
            if res["final_energy"] < best_final_energy and res["final_energy"] < res["initial_energy"]:
                best_final_energy = res["final_energy"]
                best_lr = lr
                best_mom = mom
                converges = True
                
    artifact: JsonDict = {
        "status": "complete",
        "experiment_id": 1627,
        "run_date": RUN_DATE,
        "optimizer_converges": converges,
        "optimal_learning_rate": float(best_lr),
        "optimal_momentum": float(best_mom),
        "best_final_energy": float(best_final_energy)
    }
    output.write_text(json.dumps(artifact, indent=2))
    return artifact
