"""REQ-REPORT-8806: run a constructed consensus challenge with CPU numerical training."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.consensus_energy_runner_8806 import main

if __name__ == "__main__":
    raise SystemExit(main())
