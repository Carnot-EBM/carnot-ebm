"""REQ-REPORT-8621: bootstrap the actual CLI from outside the checkout."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.lean_runner_8621 import main

if __name__ == "__main__":
    raise SystemExit(main())
