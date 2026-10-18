#!/usr/bin/env python3
"""Exp 1597: inertial dSB Ising CPU ablation against sequential Gibbs.

This experiment compares the inertia-augmented dSB simulator against Carnot's
sequential Gibbs baseline on local FoVer-derived dense Ising problems. It is a
CPU/simulator-only ablation and records no RTL, synthesis, accelerator, or
board evidence.

Spec: REQ-ISING-029, SCENARIO-ISING-039
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.samplers.discrete_simulated_bifurcation import (  # noqa: E402
    run_fover_inertial_ising_probe,
)


EXP_ID = 1597
RUN_DATE = "20260509"
DEFAULT_RESULT_PATH = REPO_ROOT / "results/experiment_1597_inertial_ising.json"


def write_in_progress_artifact(path: Path = DEFAULT_RESULT_PATH) -> dict:
    """Write the bootstrap artifact before the CPU ablation starts."""

    marker = {
        "status": "in_progress",
        "experiment_id": EXP_ID,
        "cpu_only": True,
        "simulator_only": True,
        "hardware_execution_performed": False,
        "hardware_claim_allowed": False,
        "kv260_claim_allowed": False,
        "honest_verdict": "in_progress",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return marker


def run_experiment(
    *,
    output_path: Path = DEFAULT_RESULT_PATH,
    n_problems: int = 5,
    n_variable_schedule: Sequence[int] | None = None,
    max_steps: int = 128,
    seeds: Sequence[int] = (0, 1, 2),
    inertia_coefficient: float = 0.6,
) -> dict:
    """Run the CPU-only inertial dSB ablation and write the artifact."""

    artifact = run_fover_inertial_ising_probe(
        repo_root=REPO_ROOT,
        limit=n_problems,
        n_variable_schedule=n_variable_schedule,
        max_steps=max_steps,
        seeds=seeds,
        inertia_coefficient=inertia_coefficient,
        run_date=RUN_DATE,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    """CLI entry point used by the research conductor."""

    write_in_progress_artifact(DEFAULT_RESULT_PATH)
    artifact = run_experiment()
    print(
        artifact.get("convergence_speedup_inertial_ising"),
        artifact.get("hardware_claim_allowed"),
        artifact.get("honest_verdict"),
    )


if __name__ == "__main__":
    main()
