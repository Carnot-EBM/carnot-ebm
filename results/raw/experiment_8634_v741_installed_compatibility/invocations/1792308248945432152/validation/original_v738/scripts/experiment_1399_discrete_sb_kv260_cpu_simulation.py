#!/usr/bin/env python3
"""Exp 1399: Discrete SB CPU simulation plus KV260 BRAM/LUT estimate.

This experiment compares a simplified discrete simulated bifurcation update
against a Gibbs Ising baseline on local FoVer-derived dense Ising problems. It
then records the KV260 arithmetic estimate for a 256-variable int8 coupling
matrix and one dSB update unit. No Vivado synthesis, bitfile generation, or
KV260 board execution is performed.

Spec: REQ-ISING-022, SCENARIO-ISING-032
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
    run_fover_discrete_sb_probe,
)


EXP_ID = 1399
RUN_DATE = "20260506"
DEFAULT_RESULT_PATH = REPO_ROOT / "results/experiment_1399_discrete_sb_kv260_cpu_simulation.json"


def _write_in_progress(path: Path) -> None:
    """Write the bootstrap artifact before the CPU simulation starts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "in_progress"}, indent=2) + "\n")


def run_experiment(
    *,
    output_path: Path = DEFAULT_RESULT_PATH,
    n_problems: int = 5,
    n_variable_schedule: Sequence[int] | None = None,
    max_steps: int = 128,
    seeds: Sequence[int] = (0, 1, 2),
) -> dict:
    """Run the CPU probe and write the complete Exp 1399 artifact."""

    artifact = run_fover_discrete_sb_probe(
        repo_root=REPO_ROOT,
        limit=n_problems,
        n_variable_schedule=n_variable_schedule,
        max_steps=max_steps,
        seeds=seeds,
        run_date=RUN_DATE,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main() -> None:
    """CLI entry point used by the research conductor."""

    _write_in_progress(DEFAULT_RESULT_PATH)
    artifact = run_experiment()
    print(
        artifact.get("bram_budget_feasible"),
        artifact.get("hardware_claim_allowed"),
        artifact.get("convergence_speedup_discrete_sb"),
        artifact.get("honest_verdict"),
    )


if __name__ == "__main__":
    main()
