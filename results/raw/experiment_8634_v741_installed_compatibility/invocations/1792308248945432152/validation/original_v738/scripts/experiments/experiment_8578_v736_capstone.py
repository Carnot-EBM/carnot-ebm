"""REQ-REPORT-8578: a thin standalone CLI preserves the registered capstone interface."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v736_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
