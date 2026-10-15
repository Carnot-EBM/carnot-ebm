"""REQ-REPORT-8527: keep the dated CLI thin so custody checks stay reusable."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v733_source_consumer_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
