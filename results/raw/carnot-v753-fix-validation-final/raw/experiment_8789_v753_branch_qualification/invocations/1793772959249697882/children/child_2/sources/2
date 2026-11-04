"""REQ-REPORT-8789: thin CPU qualification entry point; no model is loaded."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v753_qualification_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
