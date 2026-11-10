"""REQ-REPORT-8973: run the dated freeze or replay from any working directory."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.shared_feature_runner_8973 import main

if __name__ == "__main__":
    raise SystemExit(main())
