#!/usr/bin/env python3
"""Exp 1746: EqM test-time sampler profiling.

Spec traces: REQ-SAMPLE-1746, SCENARIO-SAMPLE-1746.
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
import torch
import torch.profiler

from carnot.core.energy import AutoGradMixin
from carnot.samplers.equilibrium_matching import EquilibriumMatchingSampler

RESULTS_PATH = Path("results/experiment_1746_profile.json")
SPEC_REFS = ["REQ-SAMPLE-1746", "SCENARIO-SAMPLE-1746"]

# REQ-SAMPLE-1746-1
MODEL_SPECS = [
    {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}
]


class DummyEnergy(AutoGradMixin):
    def __init__(self, target: jax.Array) -> None:
        self.target = target

    @property
    def input_dim(self) -> int:
        return int(self.target.shape[0])

    def energy(self, x: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum((x - self.target) ** 2)


def run_experiment(
    output_path: Path = RESULTS_PATH,
    dimension: int = 1024,
    n_steps: int = 50,
) -> dict[str, Any]:
    """Run the EqM test-time sampler profiling and write the artifact."""
    
    try:
        device = jax.devices("gpu")[0]
        backend = "gpu"
    except RuntimeError:
        device = jax.devices("cpu")[0]
        backend = "cpu"

    target = jax.device_put(jnp.zeros(dimension), device)
    init = jax.device_put(jnp.ones(dimension), device)
    energy_fn = DummyEnergy(target=target)

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

    # Profile
    activities = [torch.profiler.ProfilerActivity.CPU]
    if backend == "gpu" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        start = time.perf_counter()
        chain = sampler.sample_chain(energy_fn, init, n_steps=n_steps)
        chain.block_until_ready()
        end = time.perf_counter()

    latency_ms = (end - start) * 1000.0

    key_averages = prof.key_averages()
    total_cpu_time = sum(evt.cpu_time_total for evt in key_averages)
    
    if backend == "gpu" and torch.cuda.is_available():
        total_cuda_time = sum(getattr(evt, "device_time_total", getattr(evt, "device_time", 0.0)) for evt in key_averages)
    else:
        total_cuda_time = 0.0

    artifact = {
        "experiment_id": "1746",
        "spec_refs": SPEC_REFS,
        "model_specs": MODEL_SPECS,
        "metrics": {
            "total_latency_ms": float(latency_ms),
            "profiler_cpu_time_us": float(total_cpu_time),
            "profiler_cuda_time_us": float(total_cuda_time),
        },
        "honest_verdict": "profile_completed",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


if __name__ == "__main__":
    run_experiment()
