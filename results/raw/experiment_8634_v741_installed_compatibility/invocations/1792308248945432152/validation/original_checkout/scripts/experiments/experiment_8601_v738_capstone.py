"""REQ-REPORT-8601: expose a standalone CPU reconciliation without PYTHONPATH."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v738_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
