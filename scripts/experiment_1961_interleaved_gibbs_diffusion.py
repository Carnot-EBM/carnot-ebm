#!/usr/bin/env python3
"""Exp 1961: IGD mixed-variable MAX-3-SAT benchmark.

Spec traces: REQ-IGD-1961-5, SCENARIO-IGD-1961.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.samplers.interleaved_gibbs_diffusion import BenchmarkConfig, run_max3sat_benchmark

RESULTS_PATH = Path("results/experiment_1961_interleaved_gibbs_diffusion.json")
RUN_DATE = "20260512"


def run_experiment(output_path: Path = RESULTS_PATH) -> dict[str, Any]:
    """Run the deterministic benchmark and write the terminal results artifact."""
    artifact = run_max3sat_benchmark(BenchmarkConfig())
    artifact = {
        "artifact_path": str(output_path),
        "run_date": RUN_DATE,
        "status": "complete",
        **artifact,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def main() -> None:
    """CLI entrypoint for conductor execution."""
    run_experiment()


if __name__ == "__main__":
    main()
