#!/usr/bin/env python3
"""Exp 1740: EqM CPU versus GPU latency benchmark.

Spec traces: REQ-SAMPLE-1740, SCENARIO-SAMPLE-1740.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from carnot.core.energy import AutoGradMixin
from carnot.samplers.equilibrium_matching import EquilibriumMatchingSampler


RESULTS_PATH = Path("results/experiment_1740_eqm_gpu.json")
SPEC_REFS = ["REQ-SAMPLE-1740", "SCENARIO-SAMPLE-1740"]
PARITY_TOLERANCE = 1e-4


@dataclass
class EqMGpuBenchmarkEnergy(AutoGradMixin):
    """Smooth convex energy used to isolate EqM update latency."""

    target: jax.Array
    smoothness: float = 0.02

    @property
    def input_dim(self) -> int:
        return int(self.target.shape[0])

    def energy(self, x: jax.Array) -> jax.Array:
        residual = x - self.target
        bowl = 0.5 * jnp.sum(residual**2)
        smooth = self.smoothness * jnp.sum((residual[1:] - residual[:-1]) ** 2)
        return bowl + smooth


def _devices(platform: str) -> list[jax.Device]:
    """Return visible JAX devices for ``platform`` without raising on absence."""
    try:
        return list(jax.devices(platform))
    except RuntimeError:
        return []


def _device_metadata(device: jax.Device | None) -> dict[str, Any] | None:
    if device is None:
        return None
    return {
        "device": str(device),
        "platform": str(getattr(device, "platform", "unknown")),
        "id": getattr(device, "id", None),
    }


def _make_problem(dimension: int, device: jax.Device) -> tuple[EqMGpuBenchmarkEnergy, jax.Array]:
    target_host = np.linspace(-1.0, 1.0, dimension, dtype=np.float32)
    init_host = target_host + np.float32(2.0)
    target = jax.device_put(target_host, device)
    init = jax.device_put(init_host, device)
    return EqMGpuBenchmarkEnergy(target=target), init


def _block_until_ready(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif isinstance(value, tuple | list):
        for item in value:
            _block_until_ready(item)


def _latency_summary(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(samples_ms)),
        "min": float(np.min(samples_ms)),
        "max": float(np.max(samples_ms)),
    }


def _sampler(backend: str) -> EquilibriumMatchingSampler:
    return EquilibriumMatchingSampler(
        step_size=0.12,
        learning_rate=0.45,
        matching_strength=0.7,
        momentum=0.0,
        clip_norm=64.0,
        backend=backend,
        jit=True,
    )


def _benchmark_backend(
    backend: str,
    device: jax.Device,
    dimension: int,
    n_steps: int,
    repeats: int,
) -> tuple[dict[str, float], jax.Array]:
    energy_fn, init = _make_problem(dimension, device)
    sampler = _sampler(backend)

    warmup_chain = sampler.sample_chain(energy_fn, init, n_steps=n_steps)
    _block_until_ready(warmup_chain)

    timings_ms: list[float] = []
    chain = warmup_chain
    for _ in range(repeats):
        start = time.perf_counter()
        chain = sampler.sample_chain(energy_fn, init, n_steps=n_steps)
        _block_until_ready(chain)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    return _latency_summary(timings_ms), chain


def _derive_verdict(
    gpu_available: bool,
    cpu_latency_ms: float,
    gpu_latency_ms: float | None,
    parity_delta: float | None,
) -> str:
    if not gpu_available:
        return "gpu_unavailable"
    if parity_delta is None or parity_delta > PARITY_TOLERANCE:
        return "gpu_available_parity_failed"
    if gpu_latency_ms is not None and gpu_latency_ms < cpu_latency_ms:
        return "gpu_faster"
    return "gpu_available_but_not_faster"


def run_experiment(
    output_path: Path = RESULTS_PATH,
    dimension: int = 65536,
    n_steps: int = 64,
    repeats: int = 5,
) -> dict[str, Any]:
    """Run the EqM CPU/GPU benchmark and write the Exp 1740 artifact."""
    cpu_device = _devices("cpu")[0]
    gpu_devices = _devices("gpu")
    gpu_device = gpu_devices[0] if gpu_devices else None

    cpu_latency, cpu_chain = _benchmark_backend("cpu", cpu_device, dimension, n_steps, repeats)

    gpu_latency = None
    parity_delta = None
    speedup = None
    if gpu_device is not None:
        gpu_latency, gpu_chain = _benchmark_backend("gpu", gpu_device, dimension, n_steps, repeats)
        parity_delta = float(jnp.max(jnp.abs(jax.device_get(cpu_chain[-1]) - jax.device_get(gpu_chain[-1]))))
        speedup = float(cpu_latency["mean"] / gpu_latency["mean"]) if gpu_latency["mean"] > 0 else None

    artifact = {
        "experiment_id": "1740",
        "spec_refs": SPEC_REFS,
        "problem": {
            "name": "eqm_smooth_bowl_update_rule",
            "dimension": dimension,
            "n_steps": n_steps,
            "repeats": repeats,
        },
        "backend": {
            "jax_version": jax.__version__,
            "cpu": {
                "available": True,
                "selected_device": _device_metadata(cpu_device),
            },
            "gpu": {
                "available": gpu_device is not None,
                "selected_device": _device_metadata(gpu_device),
                "visible_device_count": len(gpu_devices),
            },
        },
        "latency_ms": {
            "cpu": cpu_latency,
            "gpu": gpu_latency,
        },
        "parity": {
            "tolerance": PARITY_TOLERANCE,
            "max_abs_delta": parity_delta,
            "within_tolerance": parity_delta is not None and parity_delta <= PARITY_TOLERANCE,
        },
        "speedup": speedup,
        "honest_verdict": _derive_verdict(
            gpu_device is not None,
            cpu_latency["mean"],
            None if gpu_latency is None else gpu_latency["mean"],
            parity_delta,
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


if __name__ == "__main__":
    run_experiment()
