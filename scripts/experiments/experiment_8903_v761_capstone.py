"""REQ-REPORT-8903: reconcile V761 without granting publication or activation authority."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v761_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
