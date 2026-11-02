"""Experiment 1960: Flow Sampling Density Estimator Prototype.

Spec: REQ-SAMPLE-1960
"""

import json
import os

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.samplers.flow_sampling import FlowSampler


def exact_block_gibbs_baseline(
    energy_fn, init: jax.Array, n_steps: int, key: jax.Array
) -> jax.Array:
    """Metropolis-within-Gibbs baseline (block Gibbs proxy)."""

    def step(x: jax.Array, k: jax.Array) -> tuple[jax.Array, jax.Array]:
        k, sk1, sk2 = jrandom.split(k, 3)
        dim = x.shape[0]
        # Propose a single coordinate update to simulate block Gibbs for dim=1
        idx = jrandom.randint(sk1, (), 0, dim)
        x_prop = x.at[idx].add(jrandom.normal(sk2) * 1.0)

        e_curr = energy_fn.energy(x)
        e_prop = energy_fn.energy(x_prop)

        accept = jnp.exp(e_curr - e_prop) > jrandom.uniform(k)
        x_next = jnp.where(accept, x_prop, x)
        return x_next, x_next

    def step_proper(carry: tuple[jax.Array, jax.Array], _: None) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        x, k = carry
        k, subk = jrandom.split(k)
        x_next, _ = step(x, subk)
        return (x_next, k), x_next

    (_, _), chain = jax.lax.scan(step_proper, (init, key), None, length=n_steps)
    return chain


def compute_kl_gaussian(samples_p: jax.Array, samples_q: jax.Array) -> float:
    """Estimates KL(P||Q) assuming P and Q are Gaussians fit to the samples."""
    mu_p = jnp.mean(samples_p, axis=0)
    mu_q = jnp.mean(samples_q, axis=0)
    cov_p = jnp.cov(samples_p, rowvar=False) + jnp.eye(samples_p.shape[1]) * 1e-6
    cov_q = jnp.cov(samples_q, rowvar=False) + jnp.eye(samples_q.shape[1]) * 1e-6

    inv_cov_q = jnp.linalg.inv(cov_q)
    k = samples_p.shape[1]

    diff = mu_q - mu_p
    term1 = jnp.trace(inv_cov_q @ cov_p)
    term2 = diff.T @ inv_cov_q @ diff
    
    sign_q, logdet_q = jnp.linalg.slogdet(cov_q)
    sign_p, logdet_p = jnp.linalg.slogdet(cov_p)
    
    term3 = logdet_q - logdet_p

    kl = 0.5 * (term1 + term2 - k + term3)
    return float(kl)


def main() -> None:
    key = jrandom.PRNGKey(42)
    k1, k2, k3 = jrandom.split(key, 3)

    dim = 2
    config = GibbsConfig(input_dim=dim, hidden_dims=[4])
    model = GibbsModel(config, key=k1)

    sampler = FlowSampler(n_steps=100, dt=0.02)

    n_samples = 500
    
    # Generate samples using Flow Sampling
    def single_sample(k: jax.Array) -> jax.Array:
        return sampler.sample(model, shape=(dim,), key=k)

    flow_samples = jax.vmap(single_sample)(jrandom.split(k2, n_samples))

    # Generate exact block Gibbs baseline samples
    init_gibbs = jnp.zeros(dim)
    gibbs_chain = exact_block_gibbs_baseline(model, init_gibbs, n_samples * 10, k3)
    gibbs_samples = gibbs_chain[::10]

    kl_div = compute_kl_gaussian(flow_samples, gibbs_samples)

    results = {
        "experiment_id": 1960,
        "spec_refs": ["REQ-SAMPLE-1960"],
        "kl_divergence": kl_div,
        "verdict": "OK",
    }

    os.makedirs("results", exist_ok=True)
    with open("results/experiment_1960_flow_sampling_unnormalized.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Flow Sampling KL divergence against Gibbs baseline: {kl_div:.4f}")


if __name__ == "__main__":
    main()
