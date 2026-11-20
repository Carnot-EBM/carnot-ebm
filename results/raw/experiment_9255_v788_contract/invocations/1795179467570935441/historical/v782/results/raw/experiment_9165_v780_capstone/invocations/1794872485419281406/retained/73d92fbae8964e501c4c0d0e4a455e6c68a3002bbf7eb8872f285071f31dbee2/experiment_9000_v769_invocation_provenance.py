"""REQ-REPORT-9000: run invocation qualification from any working directory."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.invocation_provenance_9000 import main

if __name__ == "__main__":
    raise SystemExit(main())
