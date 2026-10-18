"""REQ-REPORT-8580: execute the bounded current count-validator qualification."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.current_validator_runner_8580 import main

if __name__ == "__main__":
    raise SystemExit(main())
