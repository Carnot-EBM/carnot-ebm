#!/usr/bin/env python3
"""Experiment 1781: KANelE Look-Up Table (LUT) evaluations."""

import json
from pathlib import Path
from typing import Any

from carnot.hardware.kan_lut import convert_kan_to_lut

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_1781_kan_lut.json"


def main() -> None:
    """Run the experiment to transform a small KAN tier to LUT format."""
    print("Running Experiment 1781: KANelE LUT evaluations...")

    # Define a mock KAN edge function (e.g. an activation)
    def mock_edge(x: float) -> float:
        # Example simple nonlinear function
        return x**3 - x

    # Transform to LUT
    lut = convert_kan_to_lut(mock_edge, domain=(-1.0, 1.0), num_points=64)

    # Prepare artifact
    artifact: dict[str, Any] = {
        "schema": "carnot.kan.lut.v1",
        "lut_conversion_success": True,
        "lut_points": len(lut),
        "domain": [-1.0, 1.0],
        "sample_lut_values": lut[:5], # Store just a few for inspection
    }

    # Ensure results directory exists
    DELIVERABLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write artifact
    with open(DELIVERABLE_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"Success! Artifact written to {DELIVERABLE_PATH}")


if __name__ == "__main__":
    main()
