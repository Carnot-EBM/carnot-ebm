"""REQ-REPORT-8590: expose a standalone CLI without requiring PYTHONPATH."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v737_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
