"""REQ-REPORT-8894: qualify buffered clocks or replay retained raw operands."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.buffered_clock_runner_8894 import main

if __name__ == "__main__":
    raise SystemExit(main())
