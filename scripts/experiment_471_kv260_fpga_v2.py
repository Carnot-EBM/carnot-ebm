#!/usr/bin/env python3
"""Experiment 471: KV260 FPGA Bring-Up v2 — 128-spin sparsified Ising sampler.

Protocol:
    1. apply_env_autofix() FIRST — belt-and-suspenders CARNOT_FORCE_LIVE=1 injection.
    2. ExperimentTimeoutWatchdog(471, timeout_minutes=30) — hard wall-clock cap.
    3. ExperimentTemplate(471, ...) + DeliverableGuard(deliverable_path).
    4. Check CARNOT_KV260_BITFILE env var.
    5. FpgaBackend(simulation_mode=(not bitfile_present)).
    6. SparsifiedIsingConfig(n_spins=128, sparsity=0.9).
    7. Benchmark: FpgaBackend.sample() × 1000 iterations → median_ms_per_sample.
    8. Compare vs ParallelIsingSampler CPU baseline at n_spins=128.
    9. Build artifact with schema='carnot.fpga_backend.v1'.
    10. tmpl.assert_deliverable_written() — FINAL LINE.

Depends on: Exp 462 (DeliverableGuard).
Hardware: KV260 FPGA (optional — CPU simulation fallback).

arXiv references:
    2604.04606 — Quantum-Inspired FPGA Annealing (6x SA speedup, 4x scale)
    2505.02103 — How to Train Your OIM (10-bit EP coupling precision)
    2603.24183 — Digitally Optimized Initializations (Mpemba init)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# apply_env_autofix MUST be called before any JAX or CUDA import.
from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

EXP_ID = 471
TITLE = "KV260 FPGA Bring-Up v2"
DELIVERABLE = "results/experiment_471_kv260_fpga_v2.json"
TIMEOUT_MINUTES = 30


def _cpu_baseline_ms_per_sample(n_spins: int = 128, n_trials: int = 5) -> float:
    """Measure ParallelIsingSampler CPU baseline at n_spins.

    Runs the sampler (n_warmup=200, 1 sample) n_trials times and returns
    the median milliseconds per sample.  This is the baseline that FPGA
    hardware should beat by at least 6x (arXiv 2604.04606).
    """
    import jax.numpy as jnp
    import jax.random as jrandom

    from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

    sampler = ParallelIsingSampler(
        n_warmup=200,
        n_samples=1,
        steps_per_sample=5,
        schedule=AnnealingSchedule(beta_init=0.5, beta_final=5.0),
        use_checkerboard=True,
    )
    key = jrandom.PRNGKey(0)
    biases = jnp.zeros(n_spins, dtype=jnp.float32)
    J = jnp.zeros((n_spins, n_spins), dtype=jnp.float32)

    # Warm up JIT before timing.
    _ = sampler.sample(key, biases, J)

    times_ms: list[float] = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        k = jrandom.PRNGKey(i)
        _ = sampler.sample(k, biases, J)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    return times_ms[len(times_ms) // 2]


def main() -> None:
    """Entry point: benchmark FPGA (or simulation) vs CPU baseline."""
    result_path = str(_REPO_ROOT / DELIVERABLE)

    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    guard = DeliverableGuard(result_path)

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES):
        # ------------------------------------------------------------------
        # Step 1: Detect KV260 bitfile
        # ------------------------------------------------------------------

        bitfile_path = os.environ.get("CARNOT_KV260_BITFILE")
        bitfile_present = bitfile_path is not None and Path(bitfile_path).exists()
        simulation_mode = not bitfile_present

        _log.info(
            "Exp %d: bitfile_present=%s simulation_mode=%s",
            EXP_ID, bitfile_present, simulation_mode,
        )

        # ------------------------------------------------------------------
        # Step 2: Construct FpgaBackend and SparsifiedIsingConfig
        # ------------------------------------------------------------------

        from carnot.hardware.fpga_backend import FpgaBackend, SparsifiedIsingConfig

        backend = FpgaBackend(
            bitfile_path=bitfile_path if bitfile_present else None,
            simulation_mode=simulation_mode,
        )
        config = SparsifiedIsingConfig(n_spins=128, sparsity=0.9, seed=42)

        _log.info(
            "Exp %d: n_spins=%d sparsity=%.1f n_edges=%d",
            EXP_ID, config.n_spins, config.sparsity, config.n_edges(),
        )

        # ------------------------------------------------------------------
        # Step 3: Benchmark FpgaBackend
        # ------------------------------------------------------------------

        _log.info("Exp %d: benchmarking FpgaBackend (n_samples=1000)...", EXP_ID)
        fpga_ms = backend.benchmark(n_samples=1000)
        _log.info("Exp %d: FpgaBackend median_ms_per_sample=%.3f", EXP_ID, fpga_ms)

        fpga_ms_per_sample: float | None = fpga_ms if not simulation_mode else None
        if simulation_mode:
            _log.info(
                "Exp %d: simulation mode active — fpga_ms_per_sample set to null in artifact",
                EXP_ID,
            )

        # ------------------------------------------------------------------
        # Step 4: CPU baseline
        # ------------------------------------------------------------------

        _log.info("Exp %d: measuring CPU baseline (ParallelIsingSampler)...", EXP_ID)
        cpu_ms = _cpu_baseline_ms_per_sample(n_spins=128, n_trials=5)
        _log.info("Exp %d: CPU baseline median_ms_per_sample=%.3f", EXP_ID, cpu_ms)

        # ------------------------------------------------------------------
        # Step 5: Check Verilog and synthesis docs exist
        # ------------------------------------------------------------------

        verilog_path = _REPO_ROOT / "hardware" / "kv260" / "ising_sampler_128_sparse.v"
        synthesis_docs_path = _REPO_ROOT / "docs" / "kv260_synthesis.md"
        verilog_generated = verilog_path.exists()
        synthesis_docs_generated = synthesis_docs_path.exists()

        # ------------------------------------------------------------------
        # Step 6: Build artifact
        # ------------------------------------------------------------------

        honest_verdict = "fpga_executing" if bitfile_present else "rtl_ready_for_synthesis"

        artifact = tmpl.build_result(
            {
                "schema": "carnot.fpga_backend.v1",
                "verilog_generated": verilog_generated,
                "synthesis_docs_generated": synthesis_docs_generated,
                "bitfile_present": bitfile_present,
                "simulation_mode": simulation_mode,
                "n_spins": config.n_spins,
                "sparsity": config.sparsity,
                "n_edges": config.n_edges(),
                "fpga_ms_per_sample": fpga_ms_per_sample,
                "cpu_ms_per_sample": cpu_ms,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        # ------------------------------------------------------------------
        # Step 7: Write deliverable
        # ------------------------------------------------------------------

        output_path = _REPO_ROOT / DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)

        _log.info("Exp %d: wrote %s", EXP_ID, output_path)
        print(f"[Exp {EXP_ID}] honest_verdict={honest_verdict}")
        print(f"[Exp {EXP_ID}] simulation_mode={simulation_mode}")
        print(f"[Exp {EXP_ID}] cpu_ms_per_sample={cpu_ms:.3f}")
        if fpga_ms_per_sample is not None:
            print(f"[Exp {EXP_ID}] fpga_ms_per_sample={fpga_ms_per_sample:.3f}")
        print(f"[Exp {EXP_ID}] verilog_generated={verilog_generated}")
        print(f"[Exp {EXP_ID}] synthesis_docs_generated={synthesis_docs_generated}")
        print(f"[Exp {EXP_ID}] Deliverable: {output_path}")

    # FINAL LINE: assert deliverable was written.
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
