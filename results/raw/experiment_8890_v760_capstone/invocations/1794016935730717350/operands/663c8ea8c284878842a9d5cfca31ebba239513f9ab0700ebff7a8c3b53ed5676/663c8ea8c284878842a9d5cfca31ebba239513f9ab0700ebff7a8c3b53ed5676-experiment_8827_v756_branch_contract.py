"""REQ-REPORT-8827: bind the scheduled producer without changing activation."""

from pathlib import Path
import os
import sys

os.environ.setdefault("COVERAGE_CORE", "ctrace")
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v756_branch_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
