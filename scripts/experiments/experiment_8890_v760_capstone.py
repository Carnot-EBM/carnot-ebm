"""REQ-REPORT-8890: reconcile V760 evidence without activating a milestone."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v760_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
