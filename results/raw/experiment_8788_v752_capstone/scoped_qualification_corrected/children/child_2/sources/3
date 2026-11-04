"""REQ-REPORT-8788: run bounded V752 reconciliation without loading a model."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v752_capstone_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
