#!/usr/bin/env python3
"""Experiment 1385: self-adaptive Ising machine on FoVer arithmetic constraints.

This CPU-only probe implements the arXiv:2501.04971 Lagrange update
``lambda_{k+1} = lambda_k + eta * g(x_k)`` for FoVer arithmetic equations.
Each equation is encoded as a small binary answer-state Ising problem and
measured against a weak static-penalty baseline.

Spec: REQ-VERIFY-1385, SCENARIO-VERIFY-1385
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from carnot.samplers.self_adaptive_ising import run_self_adaptive_ising_probe  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402


EXP_ID = 1385
RUN_DATE = "20260505"
DELIVERABLE = "results/experiment_1385_self_adaptive_ising_machine_probe.json"


def _write_in_progress(path: Path) -> None:
    """Write the bootstrap artifact before loading data or running the probe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment": EXP_ID,
                "run_date": RUN_DATE,
                "status": "in_progress",
                "title": "Self-Adaptive Ising Machine FoVer Probe",
            },
            indent=2,
        )
    )


def main() -> None:
    """Run Exp 1385 and write the complete JSON artifact."""
    output_path = _REPO_ROOT / DELIVERABLE
    _write_in_progress(output_path)

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title="Self-Adaptive Ising Machine FoVer Probe",
        deliverable=DELIVERABLE,
        requires_gpu=False,
        repo_root=_REPO_ROOT,
        seed=1385,
    )
    tmpl.setup()

    with tmpl.phase("fover_self_adaptive_ising_probe", problem_limit=8):
        payload = run_self_adaptive_ising_probe(
            repo_root=_REPO_ROOT,
            limit=8,
            run_date=RUN_DATE,
        )

    artifact = tmpl.build_result(
        payload,
        status="complete",
        metrics_used=[
            "convergence_speedup",
            "constraint_violation_reduction",
            "penalty_tuning_iterations_saved",
        ],
        code_files=[
            str(Path(__file__).relative_to(_REPO_ROOT)),
            "python/carnot/samplers/self_adaptive_ising.py",
        ],
        data_path="data/fover_train_v4.json",
        cost_usd=0.0,
        decision_class="verify",
    )
    output_path.write_text(json.dumps(artifact, indent=2))
    print(
        artifact["convergence_speedup"],
        artifact["adaptive_ising_viable"],
        artifact["honest_verdict"],
    )
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
