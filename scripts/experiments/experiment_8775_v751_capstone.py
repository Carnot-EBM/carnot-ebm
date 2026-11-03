"""REQ-REPORT-8775: run the sealed reconciliation through existing publication guards."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v751_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
