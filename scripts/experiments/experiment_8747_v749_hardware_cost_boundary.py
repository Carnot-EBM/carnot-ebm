"""REQ-REPORT-8747: run the sealed model-free evidence mapper directly."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.hardware_cost_boundary_8747 import main

if __name__ == "__main__":
    raise SystemExit(main())
