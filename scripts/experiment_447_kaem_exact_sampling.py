#!/usr/bin/env python3
"""Experiment 447: KAEM Exact Sampling Latency Benchmark.

**Researcher summary:**
    Validates the KAEM (Kolmogorov-Arnold Energy Model) claim from arXiv 2506.14167
    that inverse-transform sampling is faster than MCMC for small constraint problems.
    Benchmarks KAEM exact sampling vs IsingEBM MCMC at n_vars in {10, 25, 50, 100}.

**CPU-only. Always produces a result JSON.**

Spec: REQ-SAMPLE-015, REQ-SAMPLE-016,
      SCENARIO-SAMPLE-027, SCENARIO-SAMPLE-028, SCENARIO-SAMPLE-029
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Apply env fix FIRST before any other imports (belt-and-suspenders for GPU env gate)
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
from carnot.models.kaem_energy import benchmark_kaem_vs_mcmc  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 447
TITLE = "KAEM Exact Sampling Latency Benchmark vs IsingEBM MCMC"
DELIVERABLE = "results/experiment_447_kaem_exact_sampling.json"
TIMEOUT_MINUTES = 20

# Problem sizes to benchmark — covers the sub-100-variable constraint regime
# that KAEM targets (arXiv 2506.14167 claims 10-100x speedup in this range).
N_VARS_SIZES = [10, 25, 50, 100]
N_SAMPLES = 100


def main() -> None:
    """Run Experiment 447: KAEM vs MCMC latency benchmark."""

    # ------------------------------------------------------------------
    # Scaffolding: watchdog + template
    # ------------------------------------------------------------------

    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=TIMEOUT_MINUTES,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,  # CPU-only; inverse-transform is pure arithmetic
        repo_root=_REPO_ROOT,
    )
    tmpl.setup()

    with watchdog:
        _log.info("Experiment %d starting — CPU-only KAEM benchmark", EXP_ID)
        _log.info("Sizes: %s, n_samples=%d", N_VARS_SIZES, N_SAMPLES)

        # ------------------------------------------------------------------
        # Benchmarks
        # ------------------------------------------------------------------

        per_size_benchmarks = []
        speedup_values = []

        for n_vars in N_VARS_SIZES:
            _log.info("Benchmarking n_vars=%d ...", n_vars)
            result = benchmark_kaem_vs_mcmc(n_vars=n_vars, n_samples=N_SAMPLES)

            entry = {
                "n_vars": result["n_vars"],
                "kaem_ms": result["kaem_latency_ms"],
                "ising_ms": result["ising_mcmc_latency_ms"],
                "speedup": result["speedup_ratio"],
            }
            per_size_benchmarks.append(entry)
            speedup_values.append(result["speedup_ratio"])

            _log.info(
                "  n_vars=%d: KAEM=%.1f ms, MCMC=%.1f ms, speedup=%.2fx",
                n_vars,
                result["kaem_latency_ms"],
                result["ising_mcmc_latency_ms"],
                result["speedup_ratio"],
            )

        # ------------------------------------------------------------------
        # Aggregate verdict
        # ------------------------------------------------------------------

        mean_speedup = sum(speedup_values) / len(speedup_values)

        if mean_speedup > 5.0:
            honest_verdict = "kaem_faster"
        elif mean_speedup > 1.5:
            honest_verdict = "modest_speedup"
        else:
            honest_verdict = "no_speedup"

        _log.info(
            "Mean speedup: %.2fx → honest_verdict=%s", mean_speedup, honest_verdict
        )

        # ------------------------------------------------------------------
        # Write artifact
        # ------------------------------------------------------------------

        artifact = tmpl.build_result(
            {
                "schema": "carnot.kaem_exact.v1",
                "per_size_benchmarks": per_size_benchmarks,
                "mean_speedup": mean_speedup,
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
