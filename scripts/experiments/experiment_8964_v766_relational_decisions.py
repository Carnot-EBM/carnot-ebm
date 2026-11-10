"""REQ-REPORT-8964: make checked publication and cold replay directly executable."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.relational_decisions_runner_8964 import main

if __name__ == "__main__":
    raise SystemExit(main())
