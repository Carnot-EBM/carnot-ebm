"""REQ-REPORT-8642: standalone execution must also work outside the checkout."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.streaming_execution_runner_8642 import main

if __name__ == "__main__":
    raise SystemExit(main())
