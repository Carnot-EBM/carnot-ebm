"""REQ-REPORT-8885: dated direct and foreign-directory runs share the same producer."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.sequential_label_cost_runner_8885 import main

if __name__ == "__main__":
    raise SystemExit(main())
