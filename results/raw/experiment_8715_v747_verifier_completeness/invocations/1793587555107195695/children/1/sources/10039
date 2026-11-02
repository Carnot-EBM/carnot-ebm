"""REQ-REPORT-8698: run independent symbolic controls without loading a model."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.relational_runner_8698 import main

if __name__ == "__main__":
    raise SystemExit(main())
