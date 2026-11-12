"""REQ-REPORT-9014-SCOPED: run the bounded consumer from any working directory."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.scoped_provenance_9014 import main

if __name__ == "__main__":
    raise SystemExit(main())
