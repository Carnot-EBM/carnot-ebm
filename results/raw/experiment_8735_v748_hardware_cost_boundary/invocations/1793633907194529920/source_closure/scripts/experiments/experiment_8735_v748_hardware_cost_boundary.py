"""REQ-REPORT-8735: direct execution uses the sealed reader without model loads."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.hardware_cost_boundary_8735 import main

if __name__ == "__main__":
    raise SystemExit(main())
