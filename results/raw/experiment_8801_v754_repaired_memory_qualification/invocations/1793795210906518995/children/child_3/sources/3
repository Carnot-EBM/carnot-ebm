"""REQ-REPORT-8801: direct CPU qualification CLI; no model load or generation."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v754_memory_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
