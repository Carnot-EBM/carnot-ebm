"""REQ-REPORT-8832: execute or cold-replay the fixed finite-support study."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.selective_availability_runner_8832 import main

if __name__ == "__main__":
    raise SystemExit(main())
