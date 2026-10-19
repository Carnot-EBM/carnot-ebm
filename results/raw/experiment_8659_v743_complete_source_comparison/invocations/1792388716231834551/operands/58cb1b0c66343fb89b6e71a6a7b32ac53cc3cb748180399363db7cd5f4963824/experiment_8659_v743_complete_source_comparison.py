"""REQ-REPORT-8659: locate this checkout for private, standalone execution."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.complete_source_runner_8659 import main

if __name__ == "__main__":
    raise SystemExit(main())
