#!/usr/bin/env python3
"""Experiment 498: KAEM Extended Profile n=5000 — close RETRO-031.

**Context (RETRO-031):**
    Exp 483 profiled KAEM at n_vars=(100, 200, 300, 500, 1000) and found no 5x
    speedup crossover.  The theoretical prediction (O(n^2) MCMC mixing time vs
    O(n log n) KAEM inverse-transform sampling) suggests crossover between
    n=1000 and n=5000.

**What this experiment does:**
    Profiles benchmark_kaem_vs_mcmc() at n_vars=(1000, 2000, 3000, 5000) with
    n_samples=200 each.  Stops early at the first n_vars where speedup >= 5x.
    Uses KAEMExtendedResult to classify the outcome and close RETRO-031.

**Resolution rule:**
    - If crossover found at any tested n_vars: RETRO-031 closed, KAEM viable on CPU.
    - If no crossover at n=5000: RETRO-031 closed as 'FPGA path recommended'.

**Deliverable:**
    results/experiment_498_kaem_extended_profile.json
    Schema: carnot.kaem_crossover.v2

Spec: REQ-SAMPLE-020, REQ-SAMPLE-021,
      SCENARIO-SAMPLE-033, SCENARIO-SAMPLE-034
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

# apply_env_autofix() FIRST — before any other imports that trigger GPU init.
# Ensures CARNOT_FORCE_LIVE=1 is set before JAX or CUDA init runs.
from carnot.pipeline.env_autofix import apply_env_autofix

_autofix = apply_env_autofix()

from carnot.models.kaem_energy import benchmark_kaem_vs_mcmc  # noqa: E402
from carnot.models.kaem_extended_result import KAEMExtendedResult  # noqa: E402
from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 498
TITLE = "KAEM Extended n=5000"
DELIVERABLE = "results/experiment_498_kaem_extended_profile.json"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULT_PATH = str(_REPO_ROOT / DELIVERABLE)

# Extended n_vars list per RETRO-031: start where Exp 483 left off (n=1000).
# 200 samples per size: enough timing stability without excessive CPU time.
N_VARS_LIST = [1000, 2000, 3000, 5000]
N_SAMPLES = 200
PRIOR_MAX_N = 1000

# 60 minutes is the hard budget for this CPU-only profiling pass.
TIMEOUT_MINUTES = 60


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 498: profile KAEM vs MCMC at n_vars=(1000, 2000, 3000, 5000)."""
    _log.info("Exp %d: %s — starting", EXP_ID, TITLE)
    _log.info(
        "env_autofix verdict: gpu_detected=%s, auto_fix=%s",
        _autofix.gpu_detected,
        _autofix.auto_fix_applied,
    )

    # DeliverableGuard: raises FileNotFoundError at end of main() if the
    # result JSON was never written to disk — closes the RETRO-032/033/036 hole.
    guard = DeliverableGuard(_RESULT_PATH)

    # ExperimentTimeoutWatchdog: exits the process with a partial artifact
    # if we exceed the wall-clock budget (Exp 425 pattern).
    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=TIMEOUT_MINUTES,
        result_path=_RESULT_PATH,
    )

    started_at = _utc_now()
    t0 = time.perf_counter()

    speedups: list[float] = []
    n_vars_tested: list[int] = []
    per_n_vars_results: list[dict] = []
    early_stop = False

    with watchdog:
        for n_vars in N_VARS_LIST:
            _log.info("Benchmarking n_vars=%d, n_samples=%d ...", n_vars, N_SAMPLES)
            bench = benchmark_kaem_vs_mcmc(n_vars=n_vars, n_samples=N_SAMPLES)
            speedup = bench["speedup_ratio"]

            n_vars_tested.append(n_vars)
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

            # Early stopping: crossover found — no need to benchmark larger sizes.
            if speedup >= 5.0:
                _log.info(
                    "Crossover found at n_vars=%d (speedup=%.2fx >= 5x) — stopping early.",
                    n_vars, speedup,
                )
                early_stop = True
                break

    duration_s = time.perf_counter() - t0

    # Classify the outcome using KAEMExtendedResult.
    extended = KAEMExtendedResult(
        n_vars_tested=n_vars_tested,
        speedups=speedups,
        prior_max_n=PRIOR_MAX_N,
    )

    _log.info(
        "RETRO-031 verdict: crossover_n_vars=%s, kaem_viable_for_cpu=%s, "
        "fpga_path_recommended=%s, vs_prior=%s",
        extended.crossover_n_vars,
        extended.kaem_viable_for_cpu,
        extended.fpga_path_recommended,
        extended.vs_prior,
    )

    # Build artifact — schema carnot.kaem_crossover.v2
    artifact = {
        "schema": "carnot.kaem_crossover.v2",
        "experiment": EXP_ID,
        "title": TITLE,
        "run_date": _run_date(),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_s": round(duration_s, 3),
        "status": "success",
        "n_vars_tested": n_vars_tested,
        "n_samples_per_size": N_SAMPLES,
        "speedups": speedups,
        "per_n_vars_results": per_n_vars_results,
        "early_stop": early_stop,
        "prior_max_n": PRIOR_MAX_N,
        "crossover_n_vars": extended.crossover_n_vars,
        "max_speedup": round(extended.max_speedup, 4),
        "kaem_viable_for_cpu": extended.kaem_viable_for_cpu,
        "fpga_path_recommended": extended.fpga_path_recommended,
        "retro_031_closed": True,
        "honest_verdict": (
            "crossover_found" if extended.kaem_viable_for_cpu
            else "fpga_path_recommended"
        ),
        "env_autofix": {
            "gpu_detected": _autofix.gpu_detected,
            "auto_fix_applied": _autofix.auto_fix_applied,
        },
    }

    # Write result atomically — same pattern as Exp 483.
    result_path = Path(_RESULT_PATH)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = result_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(artifact, indent=2))
    tmp_path.replace(result_path)

    _log.info("Deliverable written: %s", _RESULT_PATH)

    # FINAL LINE: assert the deliverable was actually written to disk.
    # Raises FileNotFoundError if absent — closes RETRO-032/033/036 regression.
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
