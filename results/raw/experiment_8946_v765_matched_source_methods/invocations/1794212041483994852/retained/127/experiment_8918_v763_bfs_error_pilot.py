"""REQ-REPORT-8918: direct pilot CLI also works outside the checkout."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.bfs_error_pilot_8918 import main

if __name__ == "__main__":
    raise SystemExit(main())
