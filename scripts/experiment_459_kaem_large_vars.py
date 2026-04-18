#!/usr/bin/env python3
"""Experiment 459: KAEM Large-Variable Crossover Profiling.

**Researcher summary (RETRO-031):**
    Exp 447 showed only 1.29x KAEM speedup at n_vars ≤ 100. The 5x target was not
    met because MCMC mixing time at n_vars ≤ 100 is still manageable. KAEM's advantage
    is expected to emerge at n_vars > 200, where MCMC mixing time grows super-linearly
    due to the curse of dimensionality in the Ising lattice.

    This experiment profiles KAEMEnergy vs IsingEBM MCMC at n_vars in
    [50, 100, 200, 500, 1000] to empirically locate the crossover point.
    CPU-only. Always produces a result JSON.

**What SpeedupProfile does:**
    Stores per-size (kaem_time, mcmc_time) measurements and provides three
    queries:
    - speedup_at(n_vars): mcmc_time / kaem_time at a specific size
    - crossover_n_vars(): first n_vars where KAEM beats MCMC (or None)
    - max_speedup(): (n_vars, speedup) for the best observed performance point

**Why median of 3 runs:**
    Single-run timing is noisy on a shared CPU (OS scheduling jitter, JAX JIT
    compilation on the first call, memory allocation warm-up). Taking the median
    of 3 runs filters one-sided outliers without requiring many repetitions.

Spec: REQ-KAEM-005, REQ-KAEM-006,
      SCENARIO-KAEM-010, SCENARIO-KAEM-011
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Apply env fix FIRST before any other imports (belt-and-suspenders for GPU env gate).
# This is a CPU-only experiment, but the autofix is still called per convention.
from carnot.pipeline.env_autofix import apply_env_autofix

_env_fix = apply_env_autofix()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo root and path wiring
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.models.kaem_energy import KAEMEnergy  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 459
TITLE = "KAEM Large-Variable Crossover Profiling"
DELIVERABLE = "results/experiment_459_kaem_large_vars.json"
TIMEOUT_MINUTES = 30

# n_vars range chosen to span the expected crossover point (RETRO-031 hypothesis:
# crossover at ~200 where MCMC mixing time grows super-linearly)
N_VARS_LIST = [50, 100, 200, 500, 1000]

# 1000 samples gives stable timing while remaining tractable on CPU.
# For large n_vars this is the most expensive axis, so we keep it fixed.
N_SAMPLES = 1000

# Median of 3 runs filters JIT warm-up noise without many repetitions.
N_RUNS = 3


# ---------------------------------------------------------------------------
# SpeedupProfile
# ---------------------------------------------------------------------------


@dataclass
class SpeedupProfile:
    """Summary of KAEM vs MCMC speedup ratios across multiple n_vars sizes.

    Stores per-size timing results and provides three queries:
    - speedup_at(n_vars): mcmc_time / kaem_time at a specific profiled size
    - crossover_n_vars(): first n_vars where KAEM is faster than MCMC (or None)
    - max_speedup(): (n_vars, speedup) for the best observed speedup point

    **Why speedup > 1.0 means KAEM wins:**
        speedup = mcmc_time / kaem_time. When KAEM is faster, kaem_time < mcmc_time,
        so the ratio is > 1.0. This convention matches benchmark_kaem_vs_mcmc() in
        kaem_energy.py and the Exp 447 results schema.

    Parameters
    ----------
    n_vars_list : list[int]
        Problem sizes, in order (must be same length as the time lists).
    kaem_times : list[float]
        Median KAEM wall-clock time in milliseconds for each n_vars.
    mcmc_times : list[float]
        Median MCMC wall-clock time in milliseconds for each n_vars.

    Spec: REQ-KAEM-005, REQ-KAEM-006
    """

    n_vars_list: list[int]
    kaem_times: list[float]
    mcmc_times: list[float]

    def __post_init__(self) -> None:
        """Validate list lengths are consistent and non-empty."""
        if len(self.n_vars_list) == 0:
            raise ValueError("SpeedupProfile requires non-empty n_vars_list")
        if not (len(self.n_vars_list) == len(self.kaem_times) == len(self.mcmc_times)):
            raise ValueError(
                "n_vars_list, kaem_times, and mcmc_times must be the same length; "
                f"got {len(self.n_vars_list)}, {len(self.kaem_times)}, {len(self.mcmc_times)}"
            )

    def speedup_at(self, n_vars: int) -> float:
        """Return mcmc_time / kaem_time for the given n_vars.

        Values > 1.0 mean KAEM is faster. Values < 1.0 mean MCMC is faster.
        The ratio is exactly mcmc_time / kaem_time (no clamping or rounding).

        Parameters
        ----------
        n_vars : int
            Must be in n_vars_list; raises KeyError otherwise.

        Spec: REQ-KAEM-005
        """
        if n_vars not in self.n_vars_list:
            raise KeyError(
                f"n_vars={n_vars} not in profiled sizes {self.n_vars_list}"
            )
        idx = self.n_vars_list.index(n_vars)
        kaem_ms = self.kaem_times[idx]
        mcmc_ms = self.mcmc_times[idx]
        return mcmc_ms / kaem_ms if kaem_ms > 0 else float("inf")

    def crossover_n_vars(self) -> Optional[int]:
        """Return first n_vars where speedup > 1.0, or None if KAEM never wins.

        Iterates n_vars_list in order and returns the first entry where
        mcmc_time > kaem_time (i.e., speedup_at(n) > 1.0). Returns None when
        KAEM is never faster than MCMC in the profiled range.

        **Why first, not maximum:**
            The crossover is a threshold — once KAEM is faster, it is expected
            to remain faster as n_vars grows (MCMC mixing time grows faster).
            Returning the first point gives the smallest safe threshold for
            sampler switching.

        Spec: REQ-KAEM-006, SCENARIO-KAEM-010, SCENARIO-KAEM-011
        """
        for n_vars, kaem_ms, mcmc_ms in zip(
            self.n_vars_list, self.kaem_times, self.mcmc_times
        ):
            if kaem_ms > 0 and mcmc_ms / kaem_ms > 1.0:
                return n_vars
        return None

    def max_speedup(self) -> tuple[int, float]:
        """Return (n_vars, speedup) for the entry with the highest speedup ratio.

        When multiple entries tie, the first one (smallest n_vars) is returned
        for determinism. This is the best observed KAEM advantage point.

        Spec: REQ-KAEM-005
        """
        best_n = self.n_vars_list[0]
        best_speedup = self.speedup_at(self.n_vars_list[0])
        for n_vars in self.n_vars_list[1:]:
            s = self.speedup_at(n_vars)
            if s > best_speedup:
                best_speedup = s
                best_n = n_vars
        return best_n, best_speedup


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_kaem_sample(n_vars: int, n_samples: int) -> float:
    """Return wall-clock time in ms for KAEMEnergy.sample(n_samples) at n_vars.

    One fresh KAEMEnergy instance is created per call (no weight-sharing between
    runs). A warm-up call of sample(1) is issued BEFORE the timed window to absorb
    JAX JIT compilation and CDF table build costs, so the timed result reflects
    steady-state throughput, not cold-start overhead.
    """
    import jax.random as jrandom

    key = jrandom.PRNGKey(42 + n_vars)
    model = KAEMEnergy(n_vars=n_vars, n_hidden=8, key=key)
    # Warm-up: build CDF tables and trigger any JIT compilation
    _ = model.sample(1)
    t0 = time.perf_counter()
    _ = model.sample(n_samples)
    return (time.perf_counter() - t0) * 1000.0


def _time_mcmc_sample(n_vars: int, n_samples: int) -> float:
    """Return wall-clock time in ms for ParallelIsingSampler.sample() at n_vars.

    Uses the same ring-topology coupling matrix as benchmark_kaem_vs_mcmc() in
    kaem_energy.py for a fair comparison. A warm-up call is issued BEFORE the
    timed window.
    """
    import jax.numpy as jnp
    import jax.random as jrandom
    import numpy as np
    from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

    key = jrandom.PRNGKey(99 + n_vars)
    k1, k2, k3 = jrandom.split(key, 3)

    biases = np.zeros(n_vars, dtype=np.float32)
    J = np.zeros((n_vars, n_vars), dtype=np.float32)
    for idx in range(n_vars):
        J[idx, (idx + 1) % n_vars] = 0.5
        J[(idx + 1) % n_vars, idx] = 0.5
    J_jax = jnp.array(J)
    b_jax = jnp.array(biases)

    schedule = AnnealingSchedule(beta_init=0.5, beta_final=2.0)
    sampler = ParallelIsingSampler(
        n_warmup=50,
        n_samples=n_samples,
        steps_per_sample=5,
        schedule=schedule,
    )
    init_spins = jnp.ones(n_vars, dtype=jnp.float32)

    # Warm-up: trigger JIT compilation
    _ = sampler.sample(k2, b_jax, J_jax, 2.0, init_spins)

    t0 = time.perf_counter()
    _ = sampler.sample(k3, b_jax, J_jax, 2.0, init_spins)
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 459: KAEM large-variable crossover profiling."""

    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=TIMEOUT_MINUTES,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
        repo_root=_REPO_ROOT,
    )
    tmpl.setup()

    with watchdog:
        _log.info("Experiment %d: KAEM large-variable crossover profiling", EXP_ID)
        _log.info("n_vars_list=%s, n_samples=%d, n_runs=%d", N_VARS_LIST, N_SAMPLES, N_RUNS)

        kaem_medians: list[float] = []
        mcmc_medians: list[float] = []
        profile_rows: list[dict] = []

        for n_vars in N_VARS_LIST:
            _log.info("Profiling n_vars=%d ...", n_vars)

            # --- KAEM: median of N_RUNS ---
            kaem_runs = []
            for r in range(N_RUNS):
                ms = _time_kaem_sample(n_vars, N_SAMPLES)
                kaem_runs.append(ms)
                _log.info("  KAEM run %d/%d: %.1f ms", r + 1, N_RUNS, ms)
            kaem_med = statistics.median(kaem_runs)

            # --- MCMC: median of N_RUNS ---
            mcmc_runs = []
            for r in range(N_RUNS):
                ms = _time_mcmc_sample(n_vars, N_SAMPLES)
                mcmc_runs.append(ms)
                _log.info("  MCMC run %d/%d: %.1f ms", r + 1, N_RUNS, ms)
            mcmc_med = statistics.median(mcmc_runs)

            speedup = mcmc_med / kaem_med if kaem_med > 0 else float("inf")
            _log.info(
                "  n_vars=%d → KAEM=%.1f ms (med), MCMC=%.1f ms (med), speedup=%.2fx",
                n_vars, kaem_med, mcmc_med, speedup,
            )

            kaem_medians.append(kaem_med)
            mcmc_medians.append(mcmc_med)
            profile_rows.append({
                "n_vars": n_vars,
                "kaem_ms": round(kaem_med, 3),
                "mcmc_ms": round(mcmc_med, 3),
                "speedup": round(speedup, 4),
            })

        # --- Build SpeedupProfile and queries ---
        sp = SpeedupProfile(
            n_vars_list=N_VARS_LIST,
            kaem_times=kaem_medians,
            mcmc_times=mcmc_medians,
        )
        crossover = sp.crossover_n_vars()
        max_n, max_s = sp.max_speedup()

        if crossover is not None:
            honest_verdict = f"crossover_found_at_{crossover}"
        else:
            honest_verdict = "no_crossover_in_range"

        retro_031_resolved = crossover is not None

        _log.info(
            "crossover_n_vars=%s, max_speedup=%.2fx at n_vars=%d",
            crossover, max_s, max_n,
        )
        _log.info("honest_verdict=%s, retro_031_resolved=%s", honest_verdict, retro_031_resolved)

        # --- Build artifact ---
        artifact = tmpl.build_result(
            {
                "schema": "carnot.kaem_profiler.v1",
                "profile": profile_rows,
                "crossover_n_vars": crossover,
                "max_speedup_n_vars": max_n,
                "max_speedup": round(max_s, 4),
                "retro_031_resolved": retro_031_resolved,
                "honest_verdict": honest_verdict,
                "env_fix": {
                    "gpu_detected": _env_fix.gpu_detected,
                    "auto_fix_applied": _env_fix.auto_fix_applied,
                },
            },
            status="success",
        )

        output_path = _REPO_ROOT / DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)

        _log.info("Artifact written to %s", output_path)


if __name__ == "__main__":
    main()
