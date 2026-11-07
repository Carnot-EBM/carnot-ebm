"""REQ-REPORT-8874: run complete costs or authenticate an immutable cold replay."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.size_aware_complete_runner_8874 import main

if __name__ == "__main__":
    raise SystemExit(main())
