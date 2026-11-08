"""REQ-REPORT-8920: direct CLI works outside checkout without PYTHONPATH."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.label_demand_runner_8920 import main

if __name__ == "__main__":
    raise SystemExit(main())
