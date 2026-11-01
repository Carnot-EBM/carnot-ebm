"""REQ-REPORT-8616: qualify source-bound deadline statements without model loads."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.deadline_statement_8616 import main

if __name__ == "__main__":
    raise SystemExit(main())
