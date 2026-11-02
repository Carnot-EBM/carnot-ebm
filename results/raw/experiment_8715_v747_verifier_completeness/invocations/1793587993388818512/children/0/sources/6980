#!/usr/bin/env python3
"""Experiment 2359: self-adaptive Lagrangian Ising benchmark.

This CPU-only benchmark compares the new NumPy `SelfAdaptiveIsingSampler`
against a fixed-penalty baseline on two constrained Ising problems:

1. A 16-spin ferromagnet with the equality constraint `sum(s) == 4`.
2. A 32-spin random-coupling problem with one linear equality constraint.

Spec: REQ-SAMPLE-2359, SCENARIO-SAMPLE-2359
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from carnot.samplers.adaptive_ising import run_self_adaptive_ising_benchmark  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402


EXP_ID = 2359
RANDOM_SEED = 42
DELIVERABLE = "results/experiment_2359_self_adaptive_ising.json"


def main() -> None:
    """Run the benchmark and write the complete Exp 2359 JSON artifact."""
    output_path = _REPO_ROOT / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title="Self-Adaptive Lagrangian Ising Benchmark",
        deliverable=DELIVERABLE,
        requires_gpu=False,
        repo_root=_REPO_ROOT,
        seed=RANDOM_SEED,
    )
    tmpl.setup()

    with tmpl.phase("self_adaptive_lagrangian_ising_benchmark", n_problems=2):
        payload = run_self_adaptive_ising_benchmark(random_seed=RANDOM_SEED)

    artifact = tmpl.build_result(
        payload,
        status="complete",
        metrics_used=[
            "adaptive_speedup",
            "final_constraint_violation",
            "iterations_to_feasibility",
        ],
        code_files=[
            str(Path(__file__).relative_to(_REPO_ROOT)),
            "python/carnot/samplers/adaptive_ising.py",
        ],
        cost_usd=0.0,
        decision_class="verify",
    )
    output_path.write_text(json.dumps(artifact, indent=2))
    print(
        json.dumps(
            {
                "adaptive_ising_validated": artifact["adaptive_ising_validated"],
                "adaptive_speedup": artifact["adaptive_speedup"],
                "final_constraint_violation": artifact["final_constraint_violation"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
