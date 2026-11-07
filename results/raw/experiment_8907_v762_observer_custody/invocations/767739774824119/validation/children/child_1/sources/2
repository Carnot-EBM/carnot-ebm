"""REQ-REPORT-8907: run the custody consumer outside checkout without PYTHONPATH."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.observer_custody_runner_8907 import main

if __name__ == "__main__":
    raise SystemExit(main())
