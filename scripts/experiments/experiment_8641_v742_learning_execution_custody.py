"""REQ-REPORT-8641: preserve an executable entrypoint outside the checkout."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.learning_execution_custody_8641 import main

if __name__ == "__main__":
    raise SystemExit(main())
