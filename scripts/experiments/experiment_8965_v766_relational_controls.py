"""REQ-REPORT-8965: execute qualified structural controls and cold replay."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.relational_controls_runner_8965 import main

if __name__ == "__main__":
    raise SystemExit(main())
