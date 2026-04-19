#!/usr/bin/env python3
"""Experiment 483: KAEM Large n_vars Profile — find KAEM vs MCMC crossover point.

**Context (RETRO-031):**
    Exp 447 measured KAEM exact-sampling speedup at n_vars <= 100 and found
    mean_speedup=1.29x — well below the 5x production-viability threshold.
    arXiv 2506.14167 predicts the KAEM advantage grows with n_vars because
    MCMC mixing time scales as O(n^2) while KAEM inverse-transform sampling
    is O(n log n).  The crossover should occur between n_vars=100 and 500.

**What this experiment does:**
    Profiles benchmark_kaem_vs_mcmc() at n_vars=(100, 200, 300, 500, 1000)
    with n_samples=500 each, records per-n_vars speedup ratios, then uses
    KAEMCrossoverResult to determine whether a crossover was found.

**Deliverable:**
    results/experiment_483_kaem_profile_large_nvars.json
    Schema: carnot.kaem_crossover.v1

Spec: REQ-SAMPLE-019, SCENARIO-SAMPLE-032
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Apply env autofix FIRST — before any other imports that might trigger GPU init.
# This resolves RETRO-022: CARNOT_FORCE_LIVE=1 not propagating into subprocesses.
from carnot.pipeline.env_autofix import apply_env_autofix

_autofix = apply_env_autofix()

import time  # noqa: E402 — after env autofix

from carnot.models.kaem_crossover import KAEMCrossoverResult
from carnot.models.kaem_energy import benchmark_kaem_vs_mcmc
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 483
TITLE = "KAEM Large n_vars Profile"
DELIVERABLE = "results/experiment_483_kaem_profile_large_nvars.json"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULT_PATH = str(_REPO_ROOT / DELIVERABLE)

# The n_vars values to profile.  Range chosen based on RETRO-031 analysis:
# crossover predicted between 100 and 500.  1000 included to confirm trend.
N_VARS_LIST = [100, 200, 300, 500, 1000]

# Samples per benchmark run.  500 provides stable timing without excessive runtime.
N_SAMPLES = 500

# Timeout: 45 minutes is the standard conductor budget (REQ-INFRA-024).
# Each n_vars benchmark should complete in < 5 minutes on CPU at n_samples=500.
TIMEOUT_MINUTES = 45


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 483: profile KAEM vs MCMC at n_vars=(100,200,300,500,1000)."""
    _log.info("Exp %d: %s — starting", EXP_ID, TITLE)
    _log.info("env_autofix verdict: gpu_detected=%s, auto_fix=%s",
              _autofix.gpu_detected, _autofix.auto_fix_applied)

    # Deliverable guard: raises FileNotFoundError at end of main() if the
    # result JSON was never written — closes the RETRO-032/033/036 hole.
    guard = DeliverableGuard(_RESULT_PATH)

    # Watchdog: exits the process with a partial artifact if we exceed 45 min.
    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=TIMEOUT_MINUTES,
        result_path=_RESULT_PATH,
    )

    started_at = _utc_now()
    t0 = time.perf_counter()

    speedups: list[float] = []
    per_n_vars_results: list[dict] = []

    with watchdog:
        for n_vars in N_VARS_LIST:
            _log.info("Benchmarking n_vars=%d, n_samples=%d ...", n_vars, N_SAMPLES)
            bench = benchmark_kaem_vs_mcmc(n_vars=n_vars, n_samples=N_SAMPLES)
            speedup = bench["speedup_ratio"]
            speedups.append(speedup)
            per_n_vars_results.append({
                "n_vars": n_vars,
                "n_samples": N_SAMPLES,
                "kaem_latency_ms": bench["kaem_latency_ms"],
                "mcmc_latency_ms": bench["ising_mcmc_latency_ms"],
                "speedup_ratio": speedup,
            })
            _log.info(
                "n_vars=%d: kaem=%.1f ms, mcmc=%.1f ms, speedup=%.2fx",
                n_vars,
                bench["kaem_latency_ms"],
                bench["ising_mcmc_latency_ms"],
                speedup,
            )

    duration_s = time.perf_counter() - t0

    # Compute crossover using KAEMCrossoverResult
    crossover = KAEMCrossoverResult(N_VARS_LIST, speedups)

    honest_verdict = (
        "crossover_found" if crossover.kaem_viable_for_production
        else "no_crossover_at_1000"
    )

    _log.info(
        "Crossover analysis: crossover_n_vars=%s, max_speedup=%.2fx, "
        "kaem_viable=%s, verdict=%s",
        crossover.crossover_n_vars,
        crossover.max_speedup,
        crossover.kaem_viable_for_production,
        honest_verdict,
    )

    # Build artifact — schema carnot.kaem_crossover.v1
    artifact = {
        "schema": "carnot.kaem_crossover.v1",
        "experiment": EXP_ID,
        "title": TITLE,
        "run_date": _run_date(),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_s": round(duration_s, 3),
        "status": "success",
        "n_vars_tested": N_VARS_LIST,
        "n_samples_per_size": N_SAMPLES,
        "speedups": speedups,
        "per_n_vars_results": per_n_vars_results,
        "crossover_n_vars": crossover.crossover_n_vars,
        "max_speedup": round(crossover.max_speedup, 4),
        "kaem_viable_for_production": crossover.kaem_viable_for_production,
        "retro_031_closed": crossover.crossover_n_vars is not None,
        "honest_verdict": honest_verdict,
        "env_autofix": {
            "gpu_detected": _autofix.gpu_detected,
            "auto_fix_applied": _autofix.auto_fix_applied,
        },
    }

    # Write result atomically
    result_path = Path(_RESULT_PATH)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = result_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(artifact, indent=2))
    tmp_path.replace(result_path)

    _log.info("Deliverable written: %s", _RESULT_PATH)

    # FINAL LINE: assert the deliverable was actually written.
    # Raises FileNotFoundError if the file is absent (closes RETRO-032/036 hole).
    guard.assert_written()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_date() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")


if __name__ == "__main__":
    main()
