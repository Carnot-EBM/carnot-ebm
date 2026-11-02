"""REQ-REPORT-8567: direct entry point for the bounded CPU validator correction."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.count_validator_runner_8567 import main

if __name__ == "__main__":
    raise SystemExit(main())
