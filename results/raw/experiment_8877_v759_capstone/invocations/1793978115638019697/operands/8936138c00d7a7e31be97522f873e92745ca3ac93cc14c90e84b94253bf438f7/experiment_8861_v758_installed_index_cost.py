"""REQ-REPORT-8861: run or replay the complete installed verifier cost study."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.installed_cost_runner_8861 import main

if __name__ == "__main__":
    raise SystemExit(main())
