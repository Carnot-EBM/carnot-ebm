"""REQ-REPORT-8855: direct private execution and replay without caller PYTHONPATH."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.selective_ceiling_runner_8855 import main

if __name__ == "__main__":
    raise SystemExit(main())
