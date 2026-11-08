"""REQ-REPORT-8921: run and cold-replay from outside the checkout without PYTHONPATH."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.inline_clock_runner_8921 import main

if __name__ == "__main__":
    raise SystemExit(main())
