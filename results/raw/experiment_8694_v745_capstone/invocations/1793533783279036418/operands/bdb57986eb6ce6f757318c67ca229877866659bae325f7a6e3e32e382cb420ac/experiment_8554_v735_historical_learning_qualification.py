"""REQ-REPORT-8554: expose measured qualification through the actual dated CLI."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.historical_learning_runner_8554 import main

if __name__ == "__main__":
    raise SystemExit(main())
