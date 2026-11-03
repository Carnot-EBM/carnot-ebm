"""REQ-REPORT-8762: invoke sealed reconciliation with the unchanged publication guards."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v750_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
