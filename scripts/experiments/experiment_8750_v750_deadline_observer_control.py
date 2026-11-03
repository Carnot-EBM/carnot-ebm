"""REQ-REPORT-8750: bootstrap both import roots for external direct CLI replay."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.deadline_observer_8750 import main

if __name__ == "__main__":
    raise SystemExit(main())
