#!/usr/bin/env python3
"""Experiment 1782: Benchmark hardware accounting for KAN LUTs."""

import json
from pathlib import Path
from typing import Any

from carnot.hardware.kan_benchmark import compute_bops, compute_nabs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_1782_kan_benchmark.json"

def main() -> None:
    """Run the benchmark to measure BOPs and NABS for KAN LUTs."""
    print("Running Experiment 1782: KAN LUT Hardware Benchmark...")

    num_points = 256
    num_edges = 16
    
    bops = compute_bops(num_points, num_edges)
    nabs = compute_nabs(num_points, num_edges)
    
    artifact: dict[str, Any] = {
        "schema": "carnot.kan.benchmark.v1",
        "bops": bops,
        "nabs": nabs,
        "hardware_execution_claim": False,
        "num_points": num_points,
        "num_edges": num_edges,
    }

    DELIVERABLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(DELIVERABLE_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"Success! Artifact written to {DELIVERABLE_PATH}")

if __name__ == "__main__":
    main()
