"""REQ-REPORT-8641: keep the direct runner usable outside the checkout for cold replay."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.learning_execution_8641 import main

if __name__ == "__main__":
    raise SystemExit(main())
