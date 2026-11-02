"""REQ-REPORT-8592: qualify runner clock isolation with no language model load."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.learning_clock_8592 import main

if __name__ == "__main__":
    raise SystemExit(main())
