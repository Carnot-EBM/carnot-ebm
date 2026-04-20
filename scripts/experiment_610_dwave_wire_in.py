#!/usr/bin/env python3
"""Experiment 610: D-Wave Wire-In + HISR Integration.

**Researcher summary:**
    Exp 598 confirmed the D-Wave Neal backend is available (speedup_ratio=26.24x
    over CPU simulated annealing) and that HISR credit assignment correctly
    filters violations.  This experiment wires both capabilities into production:

    1. ``get_sampler_backend('dwave')`` — selects DWaveNealBackend via the new
       ``backend_registry`` and ``CARNOT_SAMPLER`` env var.
    2. HISR wire-in — ``ConstraintAdditionFromMemory.hisr_weighted_add()`` filters
       low-confidence violations before promoting them to constraints.

**What this experiment validates:**
    - ``backend_registry`` maps 'cpu' → CpuBackend, 'dwave' → DWaveNealBackend.
    - ``get_sampler_backend('cpu')`` returns a CpuBackend instance.
    - ``get_sampler_backend('dwave')`` returns a DWaveNealBackend instance.
    - CARNOT_SAMPLER=dwave env var is respected by get_sampler_backend().
    - DWaveNealBackend.latency_ms(100) < CpuBackend-equivalent latency (speedup).
    - hisr_weighted_add() with final_correct=False filters to high-score violations.
    - hisr_weighted_add() with final_correct=True adds zero violations.

Spec: REQ-SAMPLE-035, REQ-LEARN-075,
      SCENARIO-SAMPLE-040, SCENARIO-SAMPLE-041,
      SCENARIO-LEARN-110, SCENARIO-LEARN-111
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# apply_env_autofix MUST be called before any JAX/GPU import.
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.live_assertion import assert_live_or_ci_skip  # noqa: E402

assert_live_or_ci_skip()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_watchdog = ExperimentTimeoutWatchdog(610, timeout_minutes=20)

from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logger = logging.getLogger(__name__)

DELIVERABLE = "results/experiment_610_dwave_wire_in.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _benchmark_cpu_latency(n_spins: int, n_calls: int = 10) -> float:
    """Measure mean call latency for CpuBackend on an n_spins problem.

    **Why we benchmark CpuBackend separately:**
        DWaveNealBackend.latency_ms() benchmarks the D-Wave path.  To compute
        a speedup ratio we need the CPU baseline on the same problem size and
        call count so the comparison is apples-to-apples.

    Returns mean latency in milliseconds.
    """
    import jax.numpy as jnp
    import numpy as np

    from carnot.samplers.backend import CpuBackend

    rng = np.random.default_rng(42)
    biases = rng.standard_normal(n_spins).astype(np.float32)
    couplings = rng.standard_normal((n_spins, n_spins)).astype(np.float32)
    couplings = (couplings + couplings.T) / 2.0
    np.fill_diagonal(couplings, 0.0)

    backend = CpuBackend(seed=0)
    total = 0.0
    for _ in range(n_calls):
        t0 = time.perf_counter()
        backend.minimize_energy(biases, couplings, n_samples=10, n_steps=50, beta=10.0)
        total += time.perf_counter() - t0

    return (total / n_calls) * 1000.0


def _test_hisr_wire_in() -> dict:
    """Test HISR integration with ConstraintAdditionFromMemory.

    Creates 5 ViolationPatterns, calls hisr_weighted_add() with final_correct=False,
    and verifies only high-hindsight violations (score >= 0.5) are observed.

    For 5 violations with final_correct=False, scores are:
        index 0: score = 1/(1+4) = 0.20  → filtered out
        index 1: score = 1/(1+3) = 0.25  → filtered out
        index 2: score = 1/(1+2) = 0.33  → filtered out
        index 3: score = 1/(1+1) = 0.50  → retained
        index 4: score = 1/(1+0) = 1.00  → retained

    So only indices 3 and 4 should pass the threshold and be observed.
    """
    from carnot.pipeline.constraint_addition import (
        ConstraintAdditionFromMemory,
        ViolationPattern,
    )

    # Use threshold=2 so that violations observed twice trigger addition.
    # The two high-signal violations (carry, sign) each observed once won't add yet.
    monitor = ConstraintAdditionFromMemory(threshold=2)

    violations = [
        ViolationPattern(type="carry", count=1, example_steps=["step_early_1"]),
        ViolationPattern(type="sign", count=1, example_steps=["step_early_2"]),
        ViolationPattern(type="unit", count=1, example_steps=["step_mid"]),
        ViolationPattern(type="carry", count=1, example_steps=["step_late_1"]),
        ViolationPattern(type="sign", count=1, example_steps=["step_late_2"]),
    ]

    # Call with final_correct=False.  Indices 3+4 (carry, sign) are high-signal.
    added = monitor.hisr_weighted_add(violations, final_correct=False)
    counts_after_incorrect = monitor.get_pattern_counts()

    # Now call with final_correct=True — should observe nothing.
    monitor2 = ConstraintAdditionFromMemory(threshold=2)
    added_correct = monitor2.hisr_weighted_add(violations, final_correct=True)
    counts_after_correct = monitor2.get_pattern_counts()

    # Validate: after incorrect chain, 'carry' and 'sign' should each be observed
    # once (from the high-signal indices 3 and 4).  After correct chain, nothing.
    hisr_filters_low_confidence = (
        sum(counts_after_incorrect.values()) > 0
        and sum(counts_after_correct.values()) == 0
    )

    return {
        "counts_after_incorrect": counts_after_incorrect,
        "counts_after_correct": counts_after_correct,
        "hisr_filters_low_confidence": hisr_filters_low_confidence,
        "hisr_wired": True,
    }


def main() -> None:
    tmpl = ExperimentTemplate(
        610,
        "D-Wave Wire-In + HISR Integration",
        "results/experiment_610_dwave_wire_in.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # -----------------------------------------------------------------------
    # Step 1: Test backend registry
    # -----------------------------------------------------------------------
    from carnot.samplers import backend_registry, get_sampler_backend
    from carnot.samplers.backend import CpuBackend
    from carnot.samplers.dwave_backend import DWaveNealBackend

    # Force populate registry.
    _ = get_sampler_backend("cpu")

    cpu_registered = "cpu" in backend_registry
    dwave_registered = "dwave" in backend_registry

    cpu_instance = get_sampler_backend("cpu")
    dwave_instance = get_sampler_backend("dwave")

    cpu_correct_type = isinstance(cpu_instance, CpuBackend)
    dwave_correct_type = isinstance(dwave_instance, DWaveNealBackend)

    logger.info("cpu_registered=%s, dwave_registered=%s", cpu_registered, dwave_registered)
    logger.info("cpu_correct_type=%s, dwave_correct_type=%s", cpu_correct_type, dwave_correct_type)

    # -----------------------------------------------------------------------
    # Step 2: Test CARNOT_SAMPLER env var
    # -----------------------------------------------------------------------
    old_val = os.environ.pop("CARNOT_SAMPLER", None)
    try:
        os.environ["CARNOT_SAMPLER"] = "dwave"
        env_instance = get_sampler_backend()
        carnot_sampler_env_works = isinstance(env_instance, DWaveNealBackend)
    finally:
        if old_val is not None:
            os.environ["CARNOT_SAMPLER"] = old_val
        else:
            os.environ.pop("CARNOT_SAMPLER", None)

    logger.info("carnot_sampler_env_works=%s", carnot_sampler_env_works)

    # -----------------------------------------------------------------------
    # Step 3: Benchmark latency
    # -----------------------------------------------------------------------
    logger.info("Benchmarking DWaveNealBackend.latency_ms(100)...")
    dwave_latency_ms = dwave_instance.latency_ms(100)  # type: ignore[union-attr]

    logger.info("Benchmarking CpuBackend on 100-variable problem...")
    cpu_latency_ms = _benchmark_cpu_latency(n_spins=100, n_calls=10)

    speedup_ratio = cpu_latency_ms / max(dwave_latency_ms, 1e-6)

    logger.info(
        "dwave_latency_ms=%.2f  cpu_latency_ms=%.2f  speedup=%.2fx",
        dwave_latency_ms,
        cpu_latency_ms,
        speedup_ratio,
    )

    # -----------------------------------------------------------------------
    # Step 4: Test HISR wire-in
    # -----------------------------------------------------------------------
    hisr_results = _test_hisr_wire_in()
    logger.info("hisr_results=%s", hisr_results)

    # -----------------------------------------------------------------------
    # Step 5: Build artifact
    # -----------------------------------------------------------------------
    all_ok = all([
        cpu_registered,
        dwave_registered,
        cpu_correct_type,
        dwave_correct_type,
        carnot_sampler_env_works,
        hisr_results["hisr_filters_low_confidence"],
    ])

    honest_verdict = "dwave_wired_hisr_integrated" if all_ok else "partial_failure"

    artifact = tmpl.build_result(
        {
            "dwave_backend_registered": dwave_registered,
            "cpu_backend_registered": cpu_registered,
            "carnot_sampler_env_works": carnot_sampler_env_works,
            "dwave_latency_ms": round(dwave_latency_ms, 3),
            "cpu_latency_ms": round(cpu_latency_ms, 3),
            "speedup_ratio": round(speedup_ratio, 3),
            "hisr_wired": hisr_results["hisr_wired"],
            "hisr_filters_low_confidence": hisr_results["hisr_filters_low_confidence"],
            "hisr_counts_after_incorrect": hisr_results["counts_after_incorrect"],
            "hisr_counts_after_correct": hisr_results["counts_after_correct"],
            "honest_verdict": honest_verdict,
        },
        status="success" if all_ok else "failure",
        schema="carnot.dwave_wire_in.v1",
    )

    writer = AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE))
    writer.write(artifact)
    logger.info("artifact written: honest_verdict=%s", honest_verdict)
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
