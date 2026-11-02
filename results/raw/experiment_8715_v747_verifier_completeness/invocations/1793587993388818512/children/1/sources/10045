"""REQ-REPORT-8708: standalone entry point for sealed V746 terminal reconciliation."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v746_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
