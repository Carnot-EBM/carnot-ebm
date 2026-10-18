"""REQ-REPORT-8558: direct script imports work from a private current directory."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.nine_input_runner_8558 import main

if __name__ == "__main__":
    raise SystemExit(main())
