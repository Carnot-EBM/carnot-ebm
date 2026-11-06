"""REQ-REPORT-8859: permit direct execution and private replay without caller PYTHONPATH."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.split_certificate_evaluation_runner_8859 import main

if __name__ == "__main__":
    raise SystemExit(main())
