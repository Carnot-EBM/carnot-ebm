#!/usr/bin/env python3
"""Experiment 508 — KAEM Distribution Family Benchmark.

Tests KAEM exact sampling vs MCMC on three distribution families:
- GaussianMixture: multimodal, MCMC gets trapped in one mode
- StudentT: heavy-tailed, MCMC over-concentrates near mode
- PiecewiseUniform: non-smooth, MCMC cannot cross zero-density gaps

Closes RETRO-031 if any family shows KAEM advantage (kaem_wins=True).

Deliverable: results/experiment_508_kaem_distribution_family.json
Spec: REQ-SAMPLE-022, REQ-SAMPLE-023, REQ-SAMPLE-024,
      SCENARIO-SAMPLE-035, SCENARIO-SAMPLE-036, SCENARIO-SAMPLE-037
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.models.kaem_distribution_benchmark import KAEMDistributionBenchmark
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("experiment_508")

DELIVERABLE = "results/experiment_508_kaem_distribution_family.json"
EXP_ID = 508
TITLE = "KAEM Distribution Family Benchmark"


def main() -> None:
    # RETRO-022: apply env autofix FIRST before any other setup
    env_result = apply_env_autofix()
    _log.info(
        "env_autofix: gpu_detected=%s carnot_force_live_was_set=%s auto_fix=%s",
        env_result.gpu_detected,
        env_result.carnot_force_live_was_set,
        env_result.auto_fix_applied,
    )

    deliverable_path = str(_REPO_ROOT / DELIVERABLE)
    guard = DeliverableGuard(deliverable_path)
    tmpl = ExperimentTemplate(EXP_ID, TITLE, deliverable_path)
    tmpl.setup()

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=20, result_path=deliverable_path):
        bench = KAEMDistributionBenchmark(n_vars=10, n_samples=200)

        _log.info("Running benchmark_gaussian_mixture...")
        gm_result = bench.benchmark_gaussian_mixture()
        _log.info(
            "gaussian_mixture: kaem_l2=%.4f mcmc_l2=%.4f advantage=%.4f kaem_wins=%s",
            gm_result.kaem_mean_l2,
            gm_result.mcmc_mean_l2,
            gm_result.kaem_advantage,
            gm_result.kaem_wins,
        )

        _log.info("Running benchmark_student_t...")
        st_result = bench.benchmark_student_t(nu=2.0)
        _log.info(
            "student_t: kaem_l2=%.4f mcmc_l2=%.4f advantage=%.4f kaem_wins=%s",
            st_result.kaem_mean_l2,
            st_result.mcmc_mean_l2,
            st_result.kaem_advantage,
            st_result.kaem_wins,
        )

        _log.info("Running benchmark_piecewise_uniform...")
        pu_result = bench.benchmark_piecewise_uniform(n_pieces=5)
        _log.info(
            "piecewise_uniform: kaem_l2=%.4f mcmc_l2=%.4f advantage=%.4f kaem_wins=%s",
            pu_result.kaem_mean_l2,
            pu_result.mcmc_mean_l2,
            pu_result.kaem_advantage,
            pu_result.kaem_wins,
        )

        all_results = [gm_result, st_result, pu_result]
        best = bench.best_family(results=all_results)
        retro_031_closed = best != "none"
        honest_verdict = (
            "kaem_advantage_found" if retro_031_closed else "kaem_no_advantage_on_any_family"
        )

        _log.info(
            "best_family=%s retro_031_closed=%s honest_verdict=%s",
            best,
            retro_031_closed,
            honest_verdict,
        )

        artifact = tmpl.build_result(
            {
                "gaussian_mixture_result": gm_result.to_dict(),
                "student_t_result": st_result.to_dict(),
                "piecewise_uniform_result": pu_result.to_dict(),
                "best_family": best,
                "retro_031_closed": retro_031_closed,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )
        artifact["schema"] = "carnot.kaem_distribution.v1"

        output_path = Path(deliverable_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2))
        _log.info("Deliverable written: %s", deliverable_path)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
