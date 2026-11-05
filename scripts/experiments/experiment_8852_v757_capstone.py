"""REQ-REPORT-8852: reconcile V757 without milestone activation or external publication."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v757_capstone_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
