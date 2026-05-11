#!/usr/bin/env python3
"""Exp 1748: Hardware benchmark of the sparse EqM sampler against the 100ms target.

Spec traces: REQ-SAMPLE-1748, SCENARIO-SAMPLE-1748.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from carnot.core.energy import AutoGradMixin
from carnot.samplers.equilibrium_matching import EquilibriumMatchingSampler
from carnot.models.sparse_kaem_energy import SparseKAEMEnergy

RESULTS_PATH = Path("results/experiment_1748_benchmark.json")
SPEC_REFS = ["REQ-SAMPLE-1748", "SCENARIO-SAMPLE-1748"]

class BatchedSparseEnergySum(AutoGradMixin):
    def __init__(self, model: SparseKAEMEnergy):
        self.model = model

    @property
    def input_dim(self) -> int:
        return self.model.n_vars

    def energy(self, x: jax.Array) -> jax.Array:
        return jnp.sum(jax.vmap(self.model.energy)(x))

def run_experiment(
    output_path: Path = RESULTS_PATH,
    dimension: int = 1024,
    batch_size: int = 100,
    n_steps: int = 50,
) -> dict[str, Any]:
    """Run the EqM sparse hardware benchmark."""
    try:
        device = jax.devices("gpu")[0]
        backend = "gpu"
    except RuntimeError:
        device = jax.devices("cpu")[0]
        backend = "cpu"

    sparse_energy = SparseKAEMEnergy(n_vars=dimension, n_knots=64, top_k_fraction=0.01)
    energy_fn = BatchedSparseEnergySum(model=sparse_energy)

    init = jax.device_put(jnp.ones((batch_size, dimension)), device)

    sampler = EquilibriumMatchingSampler(
        step_size=0.1,
        learning_rate=0.1,
        matching_strength=0.5,
        momentum=0.0,
        clip_norm=1.0,
        backend=backend,
        jit=True,
    )

    # Warmup
    warmup_chain = sampler.sample_chain(energy_fn, init, n_steps=2)
    warmup_chain.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    chain = sampler.sample_chain(energy_fn, init, n_steps=n_steps)
    chain.block_until_ready()
    end = time.perf_counter()

    latency_ms = (end - start) * 1000.0

    artifact = {
        "experiment_id": "1748",
        "spec_refs": SPEC_REFS,
        "metrics": {
            "latency_ms": float(latency_ms),
            "batch_size": batch_size,
            "dimension": dimension,
            "n_steps": n_steps,
        },
        "honest_verdict": "success" if latency_ms < 100.0 else "failed",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact

if __name__ == "__main__":
    run_experiment()
