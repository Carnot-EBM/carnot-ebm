"""REQ-REPORT-8738: preserve script-path imports for fresh-process replay."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.deadline_cost_8738 import main

if __name__ == "__main__":
    raise SystemExit(main())
