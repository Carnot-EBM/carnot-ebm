"""REQ-REPORT-8839: reconcile thirteen current scopes without activating a milestone."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v756_capstone_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
