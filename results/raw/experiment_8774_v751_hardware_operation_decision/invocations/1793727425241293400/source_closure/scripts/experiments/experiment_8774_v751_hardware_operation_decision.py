"""REQ-REPORT-8774: run the bounded hardware operation evidence reader."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.hardware_operation_decision_8774 import main

if __name__ == "__main__":
    raise SystemExit(main())
