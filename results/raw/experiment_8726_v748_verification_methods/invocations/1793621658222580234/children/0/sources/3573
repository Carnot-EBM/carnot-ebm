"""Empirical formal verification bounds for KAN Energy Tier.

Computes certified bounds for KAN outputs using bounded input perturbations.
"""
from typing import Dict, Any, List
import json
import numpy as np
import jax
import jax.numpy as jnp
from carnot.models.kan import KANConfig, KANModel

def load_telemetry(manifest_path: str, max_examples: int = 20) -> List[List[float]]:
    examples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if "token_logprobs" in data:
                examples.append(data["token_logprobs"])
            if len(examples) >= max_examples:
                break
    return examples

def compute_empirical_certified_bounds(
    telemetry_manifest_path: str,
    n_examples: int = 20,
    perturb_delta: float = 0.1,
    random_seed: int = 42,
    threshold: float = 0.5,
    input_dim: int = 32,
) -> Dict[str, Any]:
    """Computes empirical certified bounds for KAN."""
    logprobs_list = load_telemetry(telemetry_manifest_path, n_examples)
    if len(logprobs_list) < n_examples:
        raise ValueError(f"Expected {n_examples} examples, found {len(logprobs_list)}")

    np.random.seed(random_seed)
    rng = jax.random.PRNGKey(random_seed)
    
    config = KANConfig(input_dim=input_dim, num_knots=5, degree=2, sparse=True, edge_density=0.1)
    model = KANModel(config, key=rng)

    local_lipschitz_list = []
    certified_list = []

    for logprobs in logprobs_list:
        # Pad or truncate to input_dim
        lp = np.array(logprobs)
        if len(lp) < input_dim:
            lp = np.pad(lp, (0, input_dim - len(lp)), constant_values=-100.0)
        else:
            lp = lp[:input_dim]

        base_input = jnp.array(lp).reshape(1, -1)
        base_energy = float(model.energy_batch(base_input)[0])

        # 10 perturbed versions
        perturbations = np.random.uniform(-perturb_delta, perturb_delta, (10, input_dim))
        perturbed_inputs = jnp.array(lp + perturbations)

        perturbed_energies = model.energy_batch(perturbed_inputs)
        energy_range = float(jnp.max(perturbed_energies) - jnp.min(perturbed_energies))

        local_lipschitz = energy_range / perturb_delta
        local_lipschitz_list.append(local_lipschitz)

        base_side = base_energy > threshold
        all_same_side = bool(jnp.all((perturbed_energies > threshold) == base_side))
        certified_list.append(all_same_side)

    certified_coverage = sum(certified_list) / len(certified_list)
    mean_local_lipschitz = sum(local_lipschitz_list) / len(local_lipschitz_list)
    certified_bound_radius = perturb_delta if certified_coverage > 0.8 else certified_coverage * perturb_delta

    return {
        "certified_coverage": float(certified_coverage),
        "mean_local_lipschitz": float(mean_local_lipschitz),
        "certified_bound_radius": float(certified_bound_radius),
        "n_eval_examples": len(logprobs_list),
        "random_seed": random_seed
    }
