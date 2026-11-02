#!/usr/bin/env python3
"""Exp 1387: 2D parallel tempering CPU probe plus KV260 LUT estimate.

This experiment runs only on local CPU data structures. It compares a
single-temperature checkerboard Ising baseline with a 15-replica temperature
ladder on FoVer-derived constraint problems, then records the KV260 LUT budget
arithmetic for a possible future RTL design. It does not run Vivado, generate a
bitfile, or execute on KV260 hardware.

Spec: REQ-ISING-021, SCENARIO-ISING-031
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

from carnot.samplers.two_dimensional_parallel_tempering import (  # noqa: E402
    run_fover_2d_parallel_tempering_probe,
)


EXP_ID = 1387
RUN_DATE = "20260505"
DEFAULT_RESULT_PATH = (
    REPO_ROOT / "results/experiment_1387_2d_parallel_tempering_kv260_fpga_estimate.json"
)


def _write_in_progress(path: Path) -> None:
    """Write the bootstrap artifact before any CPU simulation starts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "in_progress"}, indent=2) + "\n")


def run_experiment(
    *,
    output_path: Path = DEFAULT_RESULT_PATH,
    n_problems: int = 5,
    n_spins: int = 128,
    max_steps: int = 96,
    seeds: Sequence[int] = (0, 1, 2),
) -> dict:
    """Run the CPU probe and write the complete Exp 1387 artifact."""

    artifact = run_fover_2d_parallel_tempering_probe(
        repo_root=REPO_ROOT,
        limit=n_problems,
        n_spins=n_spins,
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
        artifact.get("convergence_speedup_2d_pt"),
        artifact.get("lut_budget_feasible"),
        artifact.get("hardware_claim_allowed"),
        artifact.get("honest_verdict"),
    )


if __name__ == "__main__":
    main()
